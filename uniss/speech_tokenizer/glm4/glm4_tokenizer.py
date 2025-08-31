import torch
import librosa
import os
import concurrent.futures
from typing import List, Dict, Any

from transformers import WhisperFeatureExtractor
# from .modeling_whisper import WhisperVQEncoder
from .utils import extract_speech_token, load_quantize_encoder
from torch import nn
from tqdm import tqdm
import torchaudio

class Glm4Tokenizer(nn.Module):
    def __init__(self, tokenizer_path, device="cuda:0"):
        super().__init__()
        self.whisper_model = load_quantize_encoder(tokenizer_path).to(device)
        # self.whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)

    def tokenize(self, speech=None, audio_path=None, sr=16000):
        if audio_path:
            speech, sr = torchaudio.load(audio_path)
            if len(speech.shape) == 1:
                speech = speech.unsqueeze(0)
            audio_info = (speech, sr)
        else:
            assert speech is not None
            assert sr
            if isinstance(speech, list):
                speech = torch.tensor(speech).unsqueeze(0)
            if len(speech.shape) == 1:
                speech = speech.unsqueeze(0)
            audio_info = (speech, sr)

        audio_tokens = extract_speech_token(
            self.whisper_model, self.feature_extractor, [audio_info]
        )[0]
        audio_tokens = torch.tensor(audio_tokens).unsqueeze(0)
        return audio_tokens
    
    def bacth_tokenize(self, audio_paths):
        return extract_speech_token(self.whisper_model, self.feature_extractor, audio_paths)
    
    def save_token_to_file(self, tokens, save_file_paths):
        assert len(tokens) == len(save_file_paths)
        for token, save_file_path in zip(tokens, save_file_paths):
            torch.save(token, save_file_path)

def read_jsonl(file_path, blacklist=None):
    """Read a JSONL file."""
    import json
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            index = item["index"]
            if blacklist is not None and index in blacklist:
                continue
            items.append(item)
    return items


def read_blacklist(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        items = [x.strip() for x in lines]
    items = set(items)
    return items

def process_item(item: Dict[str, Any], save_path: str) -> tuple:
    """process a single item"""
    try:
        wav_path = item["wav_path"]
        save_file_path = os.path.join(save_path, f"{item['index']}.pt")
        return wav_path, save_file_path
    except KeyError as e:
        print(f"KeyError: {e} in item: {item}")
        return None

def batchify(items: List[Dict[str, Any]], batch_size: int, save_path: str, max_workers: int = 8) -> List[List[Dict[str, Any]]]:
    """batch process items"""
    batch_items = []
    single_batch_audio_paths = []
    single_batch_save_file_paths = []
    count = 0
    
    # 使用线程池处理数据
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_item = {
            executor.submit(process_item, item, save_path): item 
            for item in items
        }
        
        # 使用tqdm显示进度
        for future in tqdm(
            concurrent.futures.as_completed(future_to_item),
            total=len(items),
            desc=f"Processing items with {max_workers} workers"
        ):
            result = future.result()
            if result is not None:
                wav_path, save_file_path= result
                single_batch_audio_paths.append(wav_path)
                single_batch_save_file_paths.append(save_file_path)
                count += 1
                
                if count % batch_size == 0:
                    batch_items.append((single_batch_audio_paths, single_batch_save_file_paths))
                    single_batch_audio_paths = []
                    single_batch_save_file_paths = []
    
    if single_batch_audio_paths:
        batch_items.append((single_batch_audio_paths, single_batch_save_file_paths))
    
    assert len(batch_items) > 0, "No valid items found."
    return batch_items

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--blacklist_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--total_workers", type=int, default=8)

    args = parser.parse_args()

    # 根据 rank 设置设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Process rank {args.rank} using device: {device}")
    tokenizer = Glm4Tokenizer("/path/to/glm-4-voice-tokenizer", device=device)

    items = read_jsonl(args.jsonl_path, args.blacklist_file)
    
    # 先进行数据分片，再进行批处理
    items = items[args.rank::args.total_workers]
    batch_items = batchify(items, args.batch_size, args.save_root)
    
    for batch_item in tqdm(batch_items, desc=f"Processing batch: rank {args.rank}/{args.total_workers}"):
        try:
            audio_paths, save_file_paths = batch_item
            tokens = tokenizer.bacth_tokenize(audio_paths)
            tokenizer.save_token_to_file(tokens, save_file_paths)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue