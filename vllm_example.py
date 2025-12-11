"""
This is a example of using vllm to translate multiple speech audio files.
"""
import soundfile
import os
import json
import argparse
import numpy as np
from utils.split_wav import split_wav_file
from vllm import LLM, SamplingParams
from uniss.utils.file import load_input_data, load_config
from uniss import process_input, process_output_vllm
from uniss import UniSSTokenizer

def init_vllm_model(model_path, tokenizer, tp=1, gpu_memory_utilization=0.9):
    '''
    Initialize the VLLM model.
    Args:
        model_path: the path to the model
        tokenizer: the tokenizer path, usually is the same as the model path
        tp: the tensor parallel size
    Returns:
        model: the VLLM model
    '''
    print(f"Initializing VLLM model from {model_path} with tp={tp} and gpu_memory_utilization={gpu_memory_utilization}")
    model = LLM(
        model_path, 
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_memory_utilization,
        tokenizer=tokenizer,
        dtype="bfloat16"
    )
    return model


def uniss_infer(inputs, vllm_model, speech_tokenizer, samples, output_wav_dir, tgt_lang, device, sampling_params, args):
    '''
    Translate the speech audio files, and save the results to the output wav directory.
    Args:
        inputs: the inputs
        vllm_model: the VLLM model
        speech_tokenizer: the speech tokenizer
        samples: the samples
        output_wav_dir: the output wav directory
        tgt_lang: the target language
        device: the device
        sampling_params: the sampling parameters
        args: the arguments
    '''
    # infer with vllm
    infer_results = vllm_model.generate(inputs, sampling_params)

    # Group results by original source
    from collections import defaultdict
    grouped_results = defaultdict(list)
    
    for infer_result, sample in zip(infer_results, samples):
        audio, translation, transcription = process_output_vllm(infer_result, speech_tokenizer, args.task, device)
        
        grouped_results[sample['original_source_path']].append({
            'sample': sample,
            'audio': audio,
            'translation': translation,
            'transcription': transcription
        })
        
    # Process each group
    for source_path, items in grouped_results.items():
        # Sort by chunk index
        items.sort(key=lambda x: x['sample']['chunk_index'])
        
        # Get original duration
        try:
            info = soundfile.info(source_path)
            original_duration_samples = int(info.duration * 16000)
        except Exception as e:
            print(f"Error reading original audio {source_path}: {e}")
            max_end = max(item['sample']['chunk_end'] for item in items)
            original_duration_samples = int(max_end * 16000)

        target_len = original_duration_samples
        
        # Collect segments
        segments = []
        for item in items:
            if item['audio'] is not None and len(item['audio']) > 0:
                segments.append({
                    'audio': item['audio'],
                    'orig_start': int(item['sample']['chunk_start'] * 16000),
                })
        
        # 1. Place segments, shifting right to avoid overlap
        # Ensure at least 0.2s silence at start
        min_silence_samples = int(0.2 * 16000)
        
        current_pos = min_silence_samples
        placed_segments = []
        for seg in segments:
            start = max(seg['orig_start'], current_pos)
            placed_segments.append({'start': start, 'audio': seg['audio']})
            current_pos = start + len(seg['audio'])
            
        total_len = current_pos
        
        # 2. If total length exceeds target (minus end silence), compress
        # We want at least min_silence_samples at the end too
        end_limit = max(min_silence_samples, target_len - min_silence_samples)
        
        if total_len > end_limit:
            overflow = total_len - end_limit
            
            # Calculate gaps
            # Gap 0 is the extra space before the first segment (beyond min_silence_samples)
            gaps = []
            if placed_segments:
                first_gap = placed_segments[0]['start'] - min_silence_samples
                gaps.append(first_gap)
                
                prev_end = placed_segments[0]['start'] + len(placed_segments[0]['audio'])
                for i in range(1, len(placed_segments)):
                    gap = placed_segments[i]['start'] - prev_end
                    gaps.append(gap)
                    prev_end = placed_segments[i]['start'] + len(placed_segments[i]['audio'])
            
            total_gap_len = sum(gaps)
            
            if total_gap_len >= overflow:
                # Reduce gaps
                ratio = (total_gap_len - overflow) / total_gap_len if total_gap_len > 0 else 0
                
                current_pos = min_silence_samples
                if placed_segments:
                    # Handle first segment
                    new_first_gap = int(gaps[0] * ratio)
                    placed_segments[0]['start'] = current_pos + new_first_gap
                    current_pos = placed_segments[0]['start'] + len(placed_segments[0]['audio'])
                    
                    # Handle subsequent segments
                    for i in range(1, len(placed_segments)):
                        new_gap = int(gaps[i] * ratio)
                        placed_segments[i]['start'] = current_pos + new_gap
                        current_pos = placed_segments[i]['start'] + len(placed_segments[i]['audio'])
            else:
                # Remove all gaps and trim audio
                current_pos = min_silence_samples
                for seg in placed_segments:
                    seg['start'] = current_pos
                    current_pos += len(seg['audio'])
                
                overflow = current_pos - end_limit
                
                # Trim silence from start/end of segments
                for seg in placed_segments:
                    if overflow <= 0: break
                    
                    audio = seg['audio']
                    mask = np.abs(audio) > 0.01
                    if not np.any(mask):
                        trim = len(audio)
                        seg['audio'] = np.array([])
                        overflow -= trim
                        continue
                        
                    # Trim from end
                    last_idx = np.where(mask)[0][-1]
                    silence_at_end = len(audio) - 1 - last_idx
                    if silence_at_end > 0:
                        take = min(silence_at_end, overflow)
                        seg['audio'] = audio[:-take]
                        overflow -= take
                        
                    if overflow <= 0: break
                    
                    # Trim from start
                    first_idx = np.where(mask)[0][0]
                    silence_at_start = first_idx
                    if silence_at_start > 0:
                        take = min(silence_at_start, overflow)
                        seg['audio'] = seg['audio'][take:]
                        overflow -= take
                
                # Re-layout
                current_pos = min_silence_samples
                for seg in placed_segments:
                    seg['start'] = current_pos
                    current_pos += len(seg['audio'])

        # 3. Construct final audio
        final_audio = np.zeros(target_len)
        for seg in placed_segments:
            if len(seg['audio']) == 0: continue
            
            start = seg['start']
            end = start + len(seg['audio'])
            
            if start >= target_len: break
            if end > target_len:
                seg['audio'] = seg['audio'][:target_len - start]
                end = target_len
                
            final_audio[start:end] = seg['audio']
            
        full_transcription = []
        full_translation = []
        
        for item in items:
            if item['transcription']:
                full_transcription.append(item['transcription'])
            if item['translation']:
                full_translation.append(item['translation'])
        
        wav_basename = os.path.splitext(os.path.basename(source_path))[0]
        output_wav_path = os.path.join(output_wav_dir, f"{wav_basename}_{args.task}.wav")
        
        if final_audio is not None:
            soundfile.write(output_wav_path, final_audio, 16000)
        
        result = {
            "transcription": " ".join(full_transcription),
            "translation": " ".join(full_translation),
            "source_path": source_path,
            "infer_wav": output_wav_path,
            "tgt_lang": tgt_lang,
        }
        
        with open(os.path.join(output_wav_dir, f"results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main(args, config):
    device = "cuda:0"

    # load the model
    uniss_path = config['model']['path']
    vllm_model = init_vllm_model(uniss_path, uniss_path, tp=1, gpu_memory_utilization=0.8)     # change tp and gpu_memory_utilization as needed
    vllm_model.get_tokenizer().chat_template = None

    # load the sampling parameters, higher temperature and top_p will generate more creative results
    temperature = config['infer_param']['temperature']
    top_p = config['infer_param']['top_p']

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=1500,    # around (less than) 30 seconds
        spaces_between_special_tokens=False,
        repetition_penalty=config['infer_param']['repetition_penalty']
    )

    # create the output wav root
    output_wav_root = args.output_path
    if not os.path.exists(output_wav_root):
        os.makedirs(output_wav_root, exist_ok=True)
        print(f"Create output wav root: {output_wav_root}")
    else:
        print(f'Note: output path {output_wav_root} already exists, will overwrite it')
    
    # load speech tokenizers
    speech_tokenizer = UniSSTokenizer.from_pretrained(uniss_path)

    # load the input data
    samples = load_input_data(args.input_path)

    inputs = []
    chunked_samples = []
    
    for idx, sample in enumerate(samples):
        tgt_lang = "<|eng|>" if args.target_language == "en" else "<|cmn|>"
        
        # Split audio
        try:
            chunks = split_wav_file(sample['source_path'])
        except Exception as e:
            print(f"Error splitting {sample['source_path']}: {e}")
            chunks = []
            
        if not chunks:
            continue

        for chunk_idx, chunk in enumerate(chunks):
            # Save chunk to temp file
            temp_wav_path = os.path.join(output_wav_root, "temp", f"{idx}_{chunk_idx}.wav")
            os.makedirs(os.path.dirname(temp_wav_path), exist_ok=True)
            soundfile.write(temp_wav_path, chunk['audio'], 16000)
            
            linguistic_token_ids, speaker_token_ids = speech_tokenizer.tokenize(temp_wav_path)
            input_text = process_input(linguistic_token_ids, speaker_token_ids, args.task, tgt_lang, speed=1)
            inputs.append(input_text)
            
            new_sample = sample.copy()
            new_sample['chunk_index'] = chunk_idx
            new_sample['chunk_start'] = chunk['start']
            new_sample['chunk_end'] = chunk['end']
            new_sample['original_source_path'] = sample['source_path']
            chunked_samples.append(new_sample)

    uniss_infer(inputs, vllm_model, speech_tokenizer, chunked_samples, output_wav_root, tgt_lang, device, sampling_params, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, default='Quality', choices=['Quality', 'Performance', 'S2TT', 'S2ST', 'ASR'], help="Task name")
    parser.add_argument("--target_language", "-l", type=str, required=True, choices=['zh', 'en'], help="Target language")
    parser.add_argument("--input_path", "-i", type=str, required=True, help="Path to input data, should be .jsonl, .tsv, or a folder with speech files")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="Path to output folder")
    parser.add_argument("--config_path", "-c", type=str, default='configs/uniss.yaml', help="Path to config file")

    args = parser.parse_args()

    config = load_config(args.config_path)

    main(args, config)
