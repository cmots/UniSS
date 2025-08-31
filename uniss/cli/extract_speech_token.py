import torch
from uniss.speech_tokenizer.glm4.glm4_tokenizer import Glm4Tokenizer
from uniss.speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer

def preprocess_speech_token(tokens: torch.Tensor) -> list:
    '''
    Preprocess the speech tokens to a list.
    Args:
        tokens: the speech tokens
    Returns:
        tokens: the preprocessed speech tokens
    '''
    if not isinstance(tokens, list):
        if tokens.ndim == 2:
            tokens = tokens.squeeze(0)
        tokens = tokens.tolist()
    else:
        if len(tokens) == 1:
            tokens = tokens[0]
    return tokens

def load_speech_token(linguistic_token_path: str, speaker_token_path: str) -> tuple[list, list]:
    '''
    Load the speech tokens from the path.
    Args:
        linguistic_token_path: the path to the linguistic tokens
        speaker_token_path: the path to the speaker tokens
    Returns:
        glm_src_pt: the linguistic tokens
    '''
    # linguistic tokens
    glm_src_pt = torch.load(linguistic_token_path)

    # speaker tokens
    bicodec_src_pt = torch.load(speaker_token_path)

    glm_src_pt = preprocess_speech_token(glm_src_pt)

    bicodec_src_pt = preprocess_speech_token(bicodec_src_pt)

    return glm_src_pt, bicodec_src_pt

def tokenize_speech(wav_path: str, glm4_tokenizer: Glm4Tokenizer, bicodec_tokenizer: BiCodecTokenizer, device: str) -> tuple[list, list]:
    '''
    Tokenize the speech.
    Args:
        wav_path: the path to the wav file
        glm4_tokenizer: the glm4 tokenizer
        bicodec_tokenizer: the bicodec tokenizer
    Returns:
        glm4_tokens: the glm4 tokens
        bicodec_tokens: the bicodec tokens
    '''
    glm4_tokens = glm4_tokenizer.tokenize(audio_path=wav_path)
    bicodec_tokens = bicodec_tokenizer.encode_wav_to_tokens(wav_path)
    glm4_tokens = preprocess_speech_token(glm4_tokens)
    bicodec_tokens = preprocess_speech_token(bicodec_tokens)
    return glm4_tokens, bicodec_tokens