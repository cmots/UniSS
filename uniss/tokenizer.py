"""
Unified Speech Tokenizer for UniSS
==================================

This module provides a unified interface for GLM4 and BiCodec speech tokenizers
while maintaining compatibility with transformers and other tools.
"""

import torch
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np

from .speech_tokenizer.glm4.glm4_tokenizer import Glm4Tokenizer
from .speech_tokenizer.bicodec.bicodec_tokenizer import BiCodecTokenizer
from .cli.extract_speech_token import tokenize_speech


class UniSSTokenizer:
    """
    Unified Speech Tokenizer that combines GLM4 and BiCodec tokenizers.
    
    This class provides a simple interface for speech tokenization while keeping
    the underlying tokenizers accessible for advanced use cases.
    """
    
    def __init__(
        self,
        glm4_tokenizer: Glm4Tokenizer,
        bicodec_tokenizer: BiCodecTokenizer,
        device: Optional[torch.device] = None
    ):
        """
        Initialize with GLM4 and BiCodec tokenizers.
        
        Args:
            glm4_tokenizer: Initialized GLM4 tokenizer
            bicodec_tokenizer: Initialized BiCodec tokenizer
            device: Device to use (default: auto-detect)
        """
        self.glm4_tokenizer = glm4_tokenizer
        self.bicodec_tokenizer = bicodec_tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Create UniSSTokenizer from a unified model path.
        
        Args:
            model_path: Path to the unified UniSS model directory containing:
                       - uniss/ (main model)
                       - glm4_tokenizer/ (GLM4 tokenizer)
                       - bicodec/ (BiCodec tokenizer)
            device: Device to use ("auto", "cuda", "cpu", or torch.device)
        
        Returns:
            UniSSTokenizer instance
        """
        if device == "auto" or device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        model_path = Path(model_path)
        
        # Auto-detect paths within the unified directory
        glm4_path = model_path / "glm4_tokenizer"
        bicodec_path = model_path / "bicodec"
        
        if not glm4_path.exists():
            raise ValueError(f"GLM4 tokenizer not found at {glm4_path}")
        if not bicodec_path.exists():
            raise ValueError(f"BiCodec tokenizer not found at {bicodec_path}")
        
        print(f"Loading GLM4 tokenizer from: {glm4_path}")
        glm4_tokenizer = Glm4Tokenizer(
            model_dir=str(glm4_path),
            device=device
        )
        
        print(f"Loading BiCodec tokenizer from: {bicodec_path}")
        bicodec_tokenizer = BiCodecTokenizer(
            model_dir=str(bicodec_path),
            device=device
        )
        
        return cls(glm4_tokenizer, bicodec_tokenizer, device)
    
    @classmethod
    def from_separate_paths(
        cls,
        glm4_path: Union[str, Path],
        bicodec_path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Create UniSSTokenizer from separate paths (legacy method).
        
        Args:
            glm4_path: Path to GLM4 tokenizer
            bicodec_path: Path to BiCodec tokenizer
            device: Device to use ("auto", "cuda", "cpu", or torch.device)
        
        Returns:
            UniSSTokenizer instance
        """
        if device == "auto" or device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        
        print(f"Loading GLM4 tokenizer from: {glm4_path}")
        glm4_tokenizer = Glm4Tokenizer(
            model_dir=str(glm4_path),
            device=device
        )
        
        print(f"Loading BiCodec tokenizer from: {bicodec_path}")
        bicodec_tokenizer = BiCodecTokenizer(
            model_dir=str(bicodec_path),
            device=device
        )
        
        return cls(glm4_tokenizer, bicodec_tokenizer, device)
    
    def tokenize(
        self, 
        audio_input: Union[str, Path, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize speech input using both tokenizers.
        
        Args:
            audio_input: Audio file path or audio tensor
            **kwargs: Additional arguments passed to tokenize_speech
        
        Returns:
            Tuple of (glm4_tokens, bicodec_tokens)
        """
        if isinstance(audio_input, (str, Path)):
            return tokenize_speech(
                str(audio_input),
                self.glm4_tokenizer,
                self.bicodec_tokenizer,
                self.device,
                **kwargs
            )
        else:
            # Assume it's already tokenized
            return audio_input
    
    def decode(self, tokens: torch.Tensor) -> np.array:
        """
        Decode tokens back to audio using BiCodec tokenizer.
        
        Args:
            tokens: Generated tokens from the model
        
        Returns:
            Audio tensor
        """
        return self.bicodec_tokenizer.decode_tokens_to_audio(tokens)
    
    def save_audio(
        self, 
        audio: torch.Tensor, 
        output_path: Union[str, Path],
        sample_rate: int = 16000
    ):
        """
        Save audio tensor to file.
        
        Args:
            audio: Audio tensor to save
            output_path: Output file path
            sample_rate: Sample rate (default: 16000)
        """
        import soundfile
        soundfile.write(str(output_path), audio, sample_rate)
    
    # Expose underlying tokenizers for advanced use
    @property
    def glm4(self) -> Glm4Tokenizer:
        """Access to GLM4 tokenizer."""
        return self.glm4_tokenizer
    
    @property
    def bicodec(self) -> BiCodecTokenizer:
        """Access to BiCodec tokenizer."""
        return self.bicodec_tokenizer
    
    def __call__(self, audio_input, **kwargs):
        """Convenience method for tokenization."""
        return self.tokenize(audio_input, **kwargs)
