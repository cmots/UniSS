import re
import torch

'''
Although we support many tasks, note out weight is optimized for TTS quality and performance modes.
'''
TASK_TOKEN_MAP = {
    "Quality": "<|task_s2s_translation|><|slow_mode|>",
    "Performance": "<|task_s2s_translation|><|balance_mode|>",
    "TTS": "<|task_tts|>",
    "TTS-cross": "<|task_tts|>",
    "ASR": "<|task_asr|>",
    "S2TT": "<|task_s2t_translation|>",
    "S2ST": "<|task_s2s_translation|><|fast_mode|>",
}

def process_input(linguistic_token_ids, speaker_token_ids, task, tgt_lang, speed=1.0):
    '''
    Process the input.
    Args:
        linguistic_token_ids: the linguistic token ids
        speaker_token_ids: the speaker token ids
        task: the task
        tgt_lang: the target language
        speed: the speed ratio of source and target audio, now we only support speed = 1.0
    Returns:
        input_prompt_text: the input prompt text, including task, target language, speed, and the source speech tokens
    '''
    ## For speaker tokens, only first 32 tokens is global tokens in BiCodec
    global_token_ids = speaker_token_ids[:32]
    # bicodec_semantic_token_ids = bicodec_src_pt[32:]

    global_tokens = [f"<|bicodec_global_{token}|>" for token in global_token_ids]
    global_tokens_text = "".join(global_tokens)
    
    speaker_tokens_text = f"<|start_global_token|>{global_tokens_text}<|end_global_token|>"
    
    linguistic_tokens = [f"<|glm_semantic_{token}|>" for token in linguistic_token_ids]
    linguistic_tokens_text = "".join(linguistic_tokens)

    task_token = TASK_TOKEN_MAP[task]

    # Speed control is not supported for now.
    assert speed == 1.0, "Speed control is not supported for now, please set speed to 1.0 to keep duration consistent"
    speed_x = int((speed - 0.1) / 0.1)
    speed_token = f"<|speed_{speed_x}|>"

    src_lang = "<|cmn|>" if tgt_lang == "<|eng|>" else "<|eng|>"

    if task == "Performance":
        prompt = f"{task_token}{tgt_lang}{speaker_tokens_text}"
        input_prompt_text = f"{prompt}{linguistic_tokens_text}<|write_generate|><|task_s2t_translation|>{tgt_lang}{speed_token}<|start_content|>"
    elif task == "Quality":
        prompt = f"{task_token}{tgt_lang}{speaker_tokens_text}"
        input_prompt_text = f"{prompt}{linguistic_tokens_text}<|write_generate|><|task_asr|>{src_lang}{speed_token}<|start_content|>"
    # elif task == "TTS-cross":
    #     prompt = f"{task_token}{tgt_lang}{speaker_tokens_text}"
    #     input_prompt_text = f"{prompt}<|start_content|>{sample['trans_text']}<|end_content|><|write_generate|>{tgt_lang}{speed_token}<|start_semantic_token|>"
    # elif task == "TTS":
    #     prompt = f"{task_token}{src_lang}{speaker_tokens_text}"
    #     input_prompt_text = f"{prompt}<|start_content|>{sample['text']}<|end_content|><|write_generate|>{src_lang}{speed_token}<|start_semantic_token|>"
    elif task == "ASR":
        prompt = f"{task_token}{src_lang}{speaker_tokens_text}"
        input_prompt_text = f"{prompt}{linguistic_tokens_text}<|write_generate|>{src_lang}<|start_content|>"
    elif task == "S2ST":
        prompt = f"{task_token}{tgt_lang}{speaker_tokens_text}"
        input_prompt_text = f"{prompt}{linguistic_tokens_text}<|write_generate|><|fast_mode|>{tgt_lang}{speed_token}<|start_semantic_token|>"
    elif task == "S2TT":
        prompt = f"{task_token}{tgt_lang}{speaker_tokens_text}"
        input_prompt_text = f"{prompt}{linguistic_tokens_text}<|write_generate|>{tgt_lang}<|start_content|>"
    else:
        raise NotImplementedError(f"Unsupported task: {task}")
    
    return input_prompt_text

def process_output(output, prompt, uniss_tokenizer, task, device):
    '''
    Process the output.
    Args:
        output: the output in text format
        prompt: the prompt in text format
        uniss_tokenizer: the uniss tokenizer
        task: the task
        device: the device
    Returns:
        audio: the audio
        translation: the translation
        transcription: the transcription
    '''

    if task == "Quality":
        translation, transcription = _extract_text_without_special_tokens(output, True)
    else:
        translation, transcription = _extract_text_without_special_tokens(output)

    
    if task == "TTS" or task == "S2ST":
        # TTS mode or direct S2ST mode
        global_token_ids, out_semantic_tokens = _extract_bicodec_ids_from_text(prompt, output, device)

        semantic_tokens = torch.cat([global_token_ids, out_semantic_tokens]).to(device)
        
        try:
            audio = uniss_tokenizer.decode(semantic_tokens)
        except:
            print(f"Error in cli.prompt.process_output: {task}\nYou cannot get audio output.")
            return None, translation
        return audio, translation, transcription
    else:
        if task == "Performance" or task == "Quality":
            global_token_ids, out_semantic_tokens = _extract_bicodec_ids_from_text(prompt, output, device)

            semantic_tokens = torch.cat([global_token_ids, out_semantic_tokens]).to(device)
            
            try:
                audio = uniss_tokenizer.decode(semantic_tokens)
            except:
                print(f"Error in cli.prompt.process_output: {task}\nYou cannot get audio output.")
                return None, translation
            return audio, translation, transcription
        else:
            # ASR or S2TT mode
            return None, translation, transcription

def process_output_vllm(result, uniss_tokenizer, task, device):
    '''
    Process the output.
    Args:
        result: the result from vllm
        uniss_tokenizer: the uniss tokenizer
        task: the task
        device: the device
    Returns:    
        audio: the audio
        translation: the translation
        transcription: the transcription
    '''
    output = result.outputs[0].text
    prompt = result.prompt
    return process_output(output, prompt, uniss_tokenizer, task, device)

def _extract_text_without_special_tokens(text, quality_mode=False):
    if quality_mode:
        end_content_count = text.count('<|end_content|>')
        if end_content_count == 2:
            transcription = text.split('<|end_content|>')[0]
            translation = text.split('<|end_content|>')[1]
            transcription = re.sub(r'<\|.*?\|>', '', transcription).strip()
            translation = re.sub(r'<\|.*?\|>', '', translation).strip()
            return translation, transcription
        else:
            translation = text
            translation = re.sub(r'<\|.*?\|>', '', translation).strip()
            return translation, ""

    text = re.sub(r'<\|.*?\|>', '', text).strip()
    return text, ""

def _extract_bicodec_ids_from_text(prompt, output, device):
    global_numbers = re.findall(r'<\|bicodec_global_(\d+)\|>', prompt)
    global_tokens = [int(num) for num in global_numbers]
    global_token_ids = torch.tensor(global_tokens).to(device)

    semantic_numbers = re.findall(r'<\|bicodec_semantic_(\d+)\|>', output)
    semantic_tokens = [int(num) for num in semantic_numbers]
    semantic_token_ids = torch.tensor(semantic_tokens).to(device)

    return global_token_ids, semantic_token_ids