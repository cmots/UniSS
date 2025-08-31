import json
import pandas as pd
import os
import yaml

__all__ = ["read_jsonl", "write_jsonl", "read_tsv", "load_source_wav_from_single_folder", "load_input_data", "load_config"]
def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_jsonl(file_path, items):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def read_tsv(file_path):
    return pd.read_csv(file_path, sep='\t')

def load_source_wav_from_single_folder(folder_path, exts=['wav', 'mp3', 'flac']):
    wav_files = []
    list_files = os.listdir(folder_path)
    for _file in list_files:
        for ext in exts:
            if _file.endswith(ext):
                wav_files.append(os.path.join(folder_path, _file))
                break
    print(f"Found {len(wav_files)} files in folder: {folder_path}")
    if len(wav_files) == 0:
        raise ValueError(f"No wav, mp3, or flac files found in folder: {folder_path}")
    items = []
    for wav_file in wav_files:
        item = {
            'index': wav_file.split('/')[-1].split('.')[0],
            'source_path': wav_file
        }
        items.append(item)  
    return items

def load_input_data(input_path: str) -> list:
    '''
    Load the input data from the path.
    Args:
        input_path: the path to the input data
    Returns:
        items: the input data
    '''
    if os.path.isfile(input_path):
        if input_path.endswith('.jsonl'):
            items = read_jsonl(input_path)
            test_item = items[0]
            if 'index' not in test_item:
                raise ValueError(f"Input file {input_path} must contain 'index' field.")
            if 'source_path' not in test_item:
                raise ValueError(f"Input file {input_path} must contain 'source_path' field.")
            return items
        elif input_path.endswith('.tsv'):
            items = read_tsv(input_path)
            test_item = items.iloc[0]
            if 'index' not in test_item:
                raise ValueError(f"Input file {input_path} must contain 'index' field.")
            if 'source_path' not in test_item:
                raise ValueError(f"Input file {input_path} must contain 'source_path' field.")
            return items
        elif input_path.endswith('.txt'):
            with open(input_path, 'r') as f:
                lines = f.readlines()
            items = []
            if not lines:
                raise ValueError(f"Input file {input_path} is empty.")
            for line in lines:
                filepath = line.strip()
                items.append({
                    'index': filepath.split('/')[-1].split('.')[0],
                    'source_path': filepath
                })
        else:
            raise ValueError(f"Unsupported file type: {input_path}")
    elif os.path.isdir(input_path):
        print(f"Loading input data from folder: {input_path}, output filename will be the same as the source audio.")
        return load_source_wav_from_single_folder(input_path)
    
def load_config(config_path: str) -> dict:
    '''
    Load the config from the path.
    Args:
        config_path: the path to the config.yaml file.
    Returns:
        config: the config
    '''
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)