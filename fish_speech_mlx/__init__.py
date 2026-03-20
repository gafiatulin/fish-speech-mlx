"""Fish Audio S2 Pro — MLX inference package."""

from fish_speech_mlx.load_weights import load_model, resolve_model_path, DEFAULT_MODEL
from fish_speech_mlx.generate import generate, GenerationConfig
from fish_speech_mlx.model import DualARModel, SlowARConfig, FastARConfig
