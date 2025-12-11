import soundfile
import os
import numpy as np
import shutil
import librosa
from utils.split_wav import split_wav_file
from uniss import UniSSTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from uniss import process_input, process_output

# 1. Set the device, wav path, model path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wav_path = "prompt_audio.wav"
model_path = "pretrained_models/UniSS"

# 2. Set the mode and target language
mode = 'Quality'    # 'Quality' or 'Performance'
# tgt_lang = "<|eng|>"    # for English output
tgt_lang = "<|cmn|>"  # for Chinese output

# 3. load the model, text tokenizer, and speech tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

speech_tokenizer = UniSSTokenizer.from_pretrained(model_path, device=device)

# Read original audio
try:
    original_wav, sr = soundfile.read(wav_path)
    if len(original_wav.shape) > 1:
        original_wav = original_wav.mean(axis=1)
    
    if sr != 16000:
        print(f"Resampling original audio from {sr} to 16000 Hz...")
        original_wav = librosa.resample(original_wav, orig_sr=sr, target_sr=16000)
        sr = 16000
except Exception as e:
    print(f"Error reading original audio {wav_path}: {e}")
    exit(1)

# Split audio
chunks = split_wav_file(wav_path)
if not chunks:
    print("No chunks found.")
    exit(1)

final_audio = original_wav.copy()

# Zero out original speech segments
for chunk in chunks:
    start_sample = int(chunk['start'] * 16000)
    end_sample = int(chunk['end'] * 16000)
    start_sample = max(0, start_sample)
    end_sample = min(len(final_audio), end_sample)
    final_audio[start_sample:end_sample] = 0

full_transcription = []
full_translation = []

os.makedirs("temp_infer", exist_ok=True)

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}...")
    
    temp_chunk_path = os.path.join("temp_infer", f"chunk_{i}.wav")
    soundfile.write(temp_chunk_path, chunk['audio'], 16000)
    
    # 4. extract speech tokens
    glm4_tokens, bicodec_tokens = speech_tokenizer.tokenize(temp_chunk_path)

    # 5. process the input
    input_text = process_input(glm4_tokens, bicodec_tokens, mode, tgt_lang)
    input_token_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # 6. translate the speech
    output = model.generate(
        input_token_ids,
        max_new_tokens=1500,
        temperature=0.7,
        top_p=0.8,
        repetition_penalty=1.1
    )

    # 7. decode the output
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

    # 8. process the output
    audio_chunk, translation_chunk, transcription_chunk = process_output(output_text[0], input_text, speech_tokenizer, mode, device)
    
    # Stitching
    start_sample = int(chunk['start'] * 16000)
    
    if audio_chunk is not None:
        gen_len = len(audio_chunk)
        if start_sample + gen_len > len(final_audio):
             padding = np.zeros(start_sample + gen_len - len(final_audio))
             final_audio = np.concatenate([final_audio, padding])
        
        final_audio[start_sample:start_sample+gen_len] = audio_chunk

    if transcription_chunk:
        full_transcription.append(transcription_chunk)
    if translation_chunk:
        full_translation.append(translation_chunk)

shutil.rmtree("temp_infer")

# 9. save and show the results
soundfile.write("output_audio.wav", final_audio, 16000)

if mode == 'Quality':
    print("Transcription:\n", " ".join(full_transcription))
print("Translation:\n", " ".join(full_translation))
