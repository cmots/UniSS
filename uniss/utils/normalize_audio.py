import librosa
import soundfile as sf
from pathlib import Path
import numpy as np
import argparse

def normalize_audio_files(folder_path, target_amplitude=0.6):
    """
    Normalize the audio files in the current directory.
    Args:
        folder_path: the path to the folder containing audio files
        target_amplitude: the target maximum amplitude, default is 0.6
    """
    
    audio_dir = Path(folder_path)
    
    if not audio_dir.exists():
        print("Directory does not exist")
        return
    
    # find all .wav files
    wav_files = list(audio_dir.glob('**/*.wav'))
    
    if not wav_files:
        print("No .wav files found")
        return
    
    print(f"Found {len(wav_files)} .wav files")
    
    for wav_path in wav_files:
        try:
            # read audio file
            print(f"Processing {wav_path}...")
            audio, sr = librosa.load(str(wav_path), sr=None)
            
            normalized_audio = normalize_audio(audio, target_amplitude)
            if normalized_audio is None:
                continue
            # save normalized audio
            sf.write(str(wav_path), normalized_audio, sr)
            print(f"  ✓ {wav_path} normalized")
            
        except Exception as e:
            print(f"  ✗ Error processing {wav_path}: {e}")
    
    print("\nAll audio files normalized")

def normalize_audio(audio, target_amplitude=0.6):
    """
    Normalize the audio.
    Args:
        audio: the audio to normalize
        target_amplitude: the target maximum amplitude, default is 0.6
    """
    # calculate current maximum amplitude
    current_max = np.max(np.abs(audio))
    print(f"  Current maximum amplitude: {current_max:.4f}")
    
    if current_max == 0:
        print(f"  Warning: audio has 0 amplitude, skipping")
        return audio
    
    # calculate normalization factor
    scale_factor = target_amplitude / current_max
    print(f"  Normalization factor: {scale_factor:.4f}")
    
    # apply normalization
    normalized_audio = audio * scale_factor
    
    # verify normalization result
    new_max = np.max(np.abs(normalized_audio))
    print(f"  Normalized maximum amplitude: {new_max:.4f}")
    return normalized_audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True, help="Path to folder containing audio files")
    parser.add_argument("--target_amplitude", type=float, default=0.6, help="Target maximum amplitude")
    args = parser.parse_args()
    
    normalize_audio_files(args.folder_path, args.target_amplitude)
    