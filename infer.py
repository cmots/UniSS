import soundfile
from uniss import UniSSTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from uniss import process_input, process_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wav_path = "prompt_audio.wav"

model_path = "cmots/UniSS"

# load the model, text tokenizer, and speech tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
speech_tokenizer = UniSSTokenizer.from_pretrained(model_path)

# extract speech tokens
glm4_tokens, bicodec_tokens = speech_tokenizer.tokenize(wav_path)

tgt_lang = "<|eng|>"

# process the input
input_text = process_input(glm4_tokens, bicodec_tokens, "Quality", tgt_lang)

# translate the speech
output = model.generate(
    glm4_tokens,
    bicodec_tokens,
    max_new_tokens=1500,
    temperature=0.8,
    top_p=0.8,
    repetition_penalty=1.1
)
output_text = tokenizer.decode(output, skip_special_tokens=True)

audio, translation, transcription = process_output(output_text, input_text, speech_tokenizer, "Quality", device)

soundfile.write("output_audio.wav", audio, 16000)
print(translation)
print(transcription)
