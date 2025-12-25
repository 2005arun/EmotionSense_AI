"""
Configuration settings for EmotionSense
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    
    # Face emotion detection
    face_model: str = "deepface"  # Options: 'deepface', 'fer', 'custom'
    face_backend: str = "opencv"  # For DeepFace
    
    # Voice emotion detection
    voice_model: str = "transformer"  # Options: 'transformer', 'cnn', 'rule_based'
    voice_sample_rate: int = 16000
    
    # Text emotion detection
    text_model: str = "auto"  # Options: 'auto', 'bert', 'roberta', 'rule_based'
    
    # Music generation
    music_model: str = "facebook/musicgen-small"  # Options: small, medium, large
    music_use_gpu: bool = False


@dataclass
class FusionConfig:
    """Configuration for emotion fusion"""
    
    method: str = "confidence_based"  # Options: weighted_average, voting, confidence_based, rule_based
    
    # Modality weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'face': 0.4,
        'voice': 0.35,
        'text': 0.25
    })
    
    # Minimum confidence threshold
    min_confidence: float = 0.3
    
    # Enable context awareness
    use_context: bool = True


@dataclass
class MusicConfig:
    """Configuration for music generation"""
    
    default_duration: int = 10  # seconds
    max_duration: int = 30
    min_duration: int = 5
    
    sample_rate: int = 32000
    
    # Generation parameters
    temperature: float = 1.0
    top_k: int = 250
    guidance_scale: float = 3.0


@dataclass
class UIConfig:
    """Configuration for UI"""
    
    theme: str = "light"  # Options: light, dark
    show_confidence: bool = True
    show_all_emotions: bool = True
    enable_history: bool = True
    max_history_entries: int = 50


@dataclass
class AppConfig:
    """Main application configuration"""
    
    # Sub-configurations
    models: ModelConfig = field(default_factory=ModelConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    music: MusicConfig = field(default_factory=MusicConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Paths
    data_dir: str = "data"
    output_dir: str = "output"
    models_dir: str = "models"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Features
    enable_face: bool = True
    enable_voice: bool = True
    enable_text: bool = True
    enable_music: bool = True


# Default configuration instance
default_config = AppConfig()


def get_config() -> AppConfig:
    """Get application configuration"""
    return default_config


def update_config(**kwargs) -> AppConfig:
    """Update configuration with new values"""
    global default_config
    
    for key, value in kwargs.items():
        if hasattr(default_config, key):
            setattr(default_config, key, value)
    
    return default_config


# Emotion mappings
EMOTION_LABELS = [
    'happy', 'sad', 'angry', 'fear', 
    'surprise', 'disgust', 'neutral', 'calm'
]

EMOTION_DESCRIPTIONS = {
    'happy': 'Feeling joyful, pleased, or content',
    'sad': 'Feeling unhappy, sorrowful, or melancholic',
    'angry': 'Feeling annoyed, irritated, or enraged',
    'fear': 'Feeling afraid, anxious, or worried',
    'surprise': 'Feeling astonished, amazed, or shocked',
    'disgust': 'Feeling revulsion, distaste, or aversion',
    'neutral': 'No strong emotional state detected',
    'calm': 'Feeling peaceful, relaxed, or serene'
}

# Music genre mappings
EMOTION_TO_GENRE = {
    'happy': ['pop', 'dance', 'funk', 'disco'],
    'sad': ['ballad', 'classical', 'blues', 'acoustic'],
    'angry': ['rock', 'metal', 'punk', 'industrial'],
    'fear': ['ambient', 'dark ambient', 'horror score'],
    'surprise': ['electronic', 'experimental', 'orchestral'],
    'disgust': ['industrial', 'noise', 'experimental'],
    'neutral': ['ambient', 'easy listening', 'lofi'],
    'calm': ['ambient', 'new age', 'classical', 'meditation']
}
