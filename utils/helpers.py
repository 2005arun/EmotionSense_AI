"""
Utility functions for EmotionSense
Common helper functions used across modules
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Create directories
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# Emotion constants
STANDARD_EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral', 'calm']

EMOTION_COLORS = {
    'happy': '#FFD700',
    'sad': '#4169E1',
    'angry': '#DC143C',
    'fear': '#800080',
    'surprise': '#FF8C00',
    'disgust': '#228B22',
    'neutral': '#808080',
    'calm': '#87CEEB'
}

EMOTION_EMOJIS = {
    'happy': '😄',
    'sad': '😢',
    'angry': '😡',
    'fear': '😨',
    'surprise': '😲',
    'disgust': '🤢',
    'neutral': '😐',
    'calm': '😌'
}


def get_emoji(emotion: str) -> str:
    """Get emoji for emotion"""
    return EMOTION_EMOJIS.get(emotion.lower(), '😐')


def get_color(emotion: str) -> str:
    """Get color for emotion"""
    return EMOTION_COLORS.get(emotion.lower(), '#808080')


def normalize_emotion(emotion: str) -> str:
    """
    Normalize emotion label to standard format
    
    Args:
        emotion: Input emotion label
        
    Returns:
        Normalized emotion label
    """
    emotion_lower = emotion.lower().strip()
    
    # Mapping of alternative labels
    mapping = {
        'joy': 'happy',
        'happiness': 'happy',
        'excited': 'happy',
        'sadness': 'sad',
        'sorrow': 'sad',
        'grief': 'sad',
        'anger': 'angry',
        'rage': 'angry',
        'mad': 'angry',
        'fearful': 'fear',
        'afraid': 'fear',
        'scared': 'fear',
        'anxious': 'fear',
        'surprised': 'surprise',
        'shocked': 'surprise',
        'amazed': 'surprise',
        'disgust': 'disgust',
        'disgusted': 'disgust',
        'contempt': 'disgust'
    }
    
    return mapping.get(emotion_lower, emotion_lower)


def validate_confidence(confidence: float) -> float:
    """
    Validate and clip confidence score
    
    Args:
        confidence: Input confidence score
        
    Returns:
        Validated confidence between 0 and 1
    """
    return max(0.0, min(1.0, float(confidence)))


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize emotion scores to sum to 1.0
    
    Args:
        scores: Dictionary of emotion scores
        
    Returns:
        Normalized scores
    """
    total = sum(scores.values())
    if total > 0:
        return {k: v / total for k, v in scores.items()}
    return scores


class EmotionHistory:
    """Track emotion detection history"""
    
    def __init__(self, max_entries: int = 100):
        self.history: List[Dict] = []
        self.max_entries = max_entries
    
    def add(self, modality: str, emotion: str, confidence: float, 
            metadata: Optional[Dict] = None):
        """Add emotion detection to history"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'modality': modality,
            'emotion': emotion,
            'confidence': confidence,
            'metadata': metadata or {}
        }
        
        self.history.append(entry)
        
        # Trim if needed
        if len(self.history) > self.max_entries:
            self.history = self.history[-self.max_entries:]
    
    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get most recent entries"""
        return self.history[-n:]
    
    def get_by_modality(self, modality: str) -> List[Dict]:
        """Get entries for a specific modality"""
        return [e for e in self.history if e['modality'] == modality]
    
    def get_emotion_counts(self) -> Dict[str, int]:
        """Get count of each emotion"""
        counts = {}
        for entry in self.history:
            emotion = entry['emotion']
            counts[emotion] = counts.get(emotion, 0) + 1
        return counts
    
    def save(self, file_path: str):
        """Save history to file"""
        with open(file_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, file_path: str):
        """Load history from file"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.history = json.load(f)
    
    def clear(self):
        """Clear history"""
        self.history = []


class AudioProcessor:
    """Audio processing utilities"""
    
    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to -1 to 1 range"""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        
        return audio
    
    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling (linear interpolation)"""
        if orig_sr == target_sr:
            return audio
        
        duration = len(audio) / orig_sr
        new_length = int(duration * target_sr)
        
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio)
    
    @staticmethod
    def trim_silence(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from beginning and end"""
        mask = np.abs(audio) > threshold
        if not np.any(mask):
            return audio
        
        start = np.argmax(mask)
        end = len(mask) - np.argmax(mask[::-1])
        
        return audio[start:end]


class ImageProcessor:
    """Image processing utilities"""
    
    @staticmethod
    def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
        """Resize image to target size"""
        import cv2
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to 0-1 range"""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert to grayscale"""
        import cv2
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage string"""
    return f"{confidence:.1%}"


def format_emotion_result(emotion: str, confidence: float) -> str:
    """Format emotion result for display"""
    emoji = get_emoji(emotion)
    return f"{emoji} {emotion.upper()} ({format_confidence(confidence)})"


def create_timestamp() -> str:
    """Create formatted timestamp"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_json_dumps(obj: Any) -> str:
    """Safely convert object to JSON string"""
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return str(o)
    
    return json.dumps(obj, default=default, indent=2)


def log_emotion_detection(modality: str, emotion: str, confidence: float):
    """Log emotion detection event"""
    logger.info(f"[{modality.upper()}] Detected: {emotion} ({confidence:.1%})")


def log_fusion_result(final_emotion: str, confidence: float, method: str):
    """Log fusion result"""
    logger.info(f"[FUSION] Final: {final_emotion} ({confidence:.1%}) via {method}")


def log_music_generation(emotion: str, duration: float, file_path: str):
    """Log music generation event"""
    logger.info(f"[MUSIC] Generated {duration:.1f}s {emotion} music: {file_path}")


if __name__ == "__main__":
    # Test utilities
    print("EmotionSense Utilities")
    print("=" * 40)
    
    # Test emotion normalization
    test_emotions = ['joy', 'HAPPY', 'Sadness', 'anger', 'fear']
    for em in test_emotions:
        normalized = normalize_emotion(em)
        print(f"{em} -> {normalized} {get_emoji(normalized)}")
    
    # Test history
    history = EmotionHistory()
    history.add('face', 'happy', 0.85)
    history.add('voice', 'calm', 0.72)
    history.add('text', 'sad', 0.65)
    
    print(f"\nHistory entries: {len(history.history)}")
    print(f"Emotion counts: {history.get_emotion_counts()}")
