"""
This is a example of using vllm to translate multiple speech audio files.
"""
import soundfile
import os
import json
import argparse
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

    results = []

    # process the results and save
    for infer_result, sample in zip(infer_results, samples):
        audio, translation, transcription = process_output_vllm(infer_result, speech_tokenizer, args.task, device)
        
        wav_basename = os.path.splitext(os.path.basename(sample['source_path']))[0]
        output_wav_path = os.path.join(output_wav_dir, f"{wav_basename}_{args.task}.wav")
        
        if audio is not None:
            soundfile.write(output_wav_path, audio, 16000)
        
        result = {
            "index": sample['index'],
            "transcription": transcription,
            "translation": translation,
            "source_path": sample['source_path'],
            "infer_wav": output_wav_path,
            "tgt_lang": tgt_lang,
        }
        results.append(result)
        
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
    
    for idx, sample in enumerate(samples):
        tgt_lang = "<|eng|>" if args.target_language == "en" else "<|cmn|>"
        
        linguistic_token_ids, speaker_token_ids = speech_tokenizer.tokenize(sample['source_path'])
        input_text = process_input(linguistic_token_ids, speaker_token_ids, args.task, tgt_lang)
        inputs.append(input_text)

    uniss_infer(inputs, vllm_model, speech_tokenizer, samples, output_wav_root, tgt_lang, device, sampling_params, args)


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
