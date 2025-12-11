import os
import numpy as np
import torch
from typing import List, Union

try:
    from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
except ImportError:
    print("Warning: silero_vad not found. Please install it.")

class VADAudioSplitter:
    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000):
        """
        Initialize the VAD splitter with a Silero VAD model.
        """
        self.model = load_silero_vad()
        self.sampling_rate = sampling_rate
        self.threshold = threshold

    def split_audio(self, 
                    audio_path: str, 
                    target_len_min: float = 8.0, 
                    target_len_max: float = 16.0, 
                    fallback_min: float = 5.0, 
                    fallback_max: float = 30.0) -> List[dict]:
        """
        Split audio into chunks based on VAD silence detection.
        
        Args:
            audio_path: Path to the audio file.
            target_len_min: Preferred minimum length of a chunk in seconds.
            target_len_max: Preferred maximum length of a chunk in seconds.
            fallback_min: Absolute minimum length allowed (if perfect split not possible).
            fallback_max: Absolute maximum length allowed (if perfect split not possible).
            
        Returns:
            A list of dictionaries, each containing 'audio' (numpy array), 'start', and 'end'.
        """
        
        # Read audio
        try:
            wav = read_audio(audio_path, sampling_rate=self.sampling_rate)
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            return []

        total_duration = len(wav) / self.sampling_rate

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            return_seconds=True
        )

        if not speech_timestamps:
            return []

        segments = []
        current_segment = {'start': speech_timestamps[0]['start'], 'end': speech_timestamps[0]['end']}
        
        for i in range(1, len(speech_timestamps)):
            ts = speech_timestamps[i]
            gap = ts['start'] - current_segment['end']
            duration = ts['end'] - current_segment['start']
            
            # Merge logic:
            # 1. If gap is small (< 2.0s) and total duration is within max limit
            # 2. If current segment is too short (< min limit) and merging keeps it within max limit (allow slightly larger gap)
            
            if (gap < 1.0 and duration <= target_len_max) or \
               ((current_segment['end'] - current_segment['start']) < target_len_min and duration <= target_len_max and gap < 5.0):
                current_segment['end'] = ts['end']
            else:
                segments.append(current_segment)
                current_segment = {'start': ts['start'], 'end': ts['end']}
        
        segments.append(current_segment)

        chunks = []
        for seg in segments:
            start_sample = int(seg['start'] * self.sampling_rate)
            end_sample = int(seg['end'] * self.sampling_rate)
            # Ensure boundaries
            start_sample = max(0, start_sample)
            end_sample = min(len(wav), end_sample)
            
            chunk = wav[start_sample:end_sample]
            if len(chunk) > 0:
                chunks.append({
                    'audio': chunk.numpy(),
                    'start': seg['start'],
                    'end': seg['end']
                })
            
        return chunks

# Helper function for easy import and usage
def split_wav_file(audio_path: str) -> List[dict]:
    splitter = VADAudioSplitter()
    return splitter.split_audio(audio_path)

if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        chunks = split_wav_file(path)
        print(f"Split {path} into {len(chunks)} chunks.")
        for i, c in enumerate(chunks):
            print(f"Chunk {i}: {len(c['audio'])/16000:.2f}s, Start: {c['start']:.2f}s, End: {c['end']:.2f}s")
