import soundfile
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
tgt_lang = "<|eng|>"    # for English output
# tgt_lang = "<|cmn|>"  # for Chinese output

# 3. load the model, text tokenizer, and speech tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

speech_tokenizer = UniSSTokenizer.from_pretrained(model_path, device=device)

# 4. extract speech tokens
glm4_tokens, bicodec_tokens = speech_tokenizer.tokenize(wav_path)


# 5. process the input
input_text = process_input(glm4_tokens, bicodec_tokens, mode, tgt_lang)
input_token_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# 6. translate the speech
output = model.generate(
    input_token_ids,
    max_new_tokens=1500,
    temperature=0.8,
    top_p=0.8,
    repetition_penalty=1.1
)

# 7. decode the output
output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

# 8. process the output
audio, translation, transcription = process_output(output_text[0], input_text, speech_tokenizer, mode, device)

# 9. save and show the results
soundfile.write("output_audio.wav", audio, 16000)

if mode == 'Quality':
    print("Transcription:\n", transcription)
print("Translation:\n", translation)
