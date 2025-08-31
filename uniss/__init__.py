"""
UniSS: Unified Speech-to-Speech Translation System
==================================================

A Chinese-English speech-to-speech translation system that effectively transfers 
translation capabilities from large language models (LLMs) to the speech domain.

This package provides unified speech tokenizers while maintaining compatibility
with transformers and other tools like vllm.

Example usage:
    >>> from uniss import UniSSTokenizer
    >>> from transformers import AutoModelForCausalLM
    >>> 
    >>> # Initialize unified tokenizer
    >>> tokenizer = UniSSTokenizer.from_pretrained(
    ...     glm4_path="/path/to/glm4",
    ...     bicodec_path="/path/to/bicodec"
    ... )
    >>> 
    >>> # Use with transformers
    >>> model = AutoModelForCausalLM.from_pretrained("/path/to/model")
    >>> 
    >>> # Tokenize speech
    >>> glm4_tokens, bicodec_tokens = tokenizer.tokenize("input.wav")
    >>> 
    >>> # Generate with model
    >>> output = model.generate(glm4_tokens, bicodec_tokens)
    >>> 
    >>> # Decode to audio
    >>> audio = tokenizer.decode(output)
"""

from .tokenizer import UniSSTokenizer
from .cli import process_input, process_output, process_output_vllm

__version__ = "0.1.0"
__all__ = ["UniSSTokenizer", "process_input", "process_output", "process_output_vllm"]
