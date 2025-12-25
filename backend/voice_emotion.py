"""
Voice Emotion Detection Module
Detects emotions from audio using speech emotion recognition
"""

import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import os
import tempfile
import wave
import struct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VoiceEmotionResult:
    """Data class for voice emotion detection results"""
    modality: str
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    features: Optional[Dict] = None


class VoiceEmotionDetector:
    """
    Voice Emotion Detector using audio features and ML models
    Supports detection from audio files and live recordings
    """
    
    # Emotion labels for speech
    EMOTIONS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    # Simplified emotion mapping
    EMOTION_MAP = {
        'angry': 'angry',
        'calm': 'calm',
        'disgust': 'disgust',
        'fearful': 'fear',
        'happy': 'happy',
        'neutral': 'neutral',
        'sad': 'sad',
        'surprised': 'surprise'
    }
    
    # Emoji mapping
    EMOJI_MAP = {
        'angry': '😡',
        'calm': '😌',
        'disgust': '🤢',
        'fear': '😨',
        'fearful': '😨',
        'happy': '😄',
        'neutral': '😐',
        'sad': '😢',
        'surprise': '😲',
        'surprised': '😲'
    }
    
    def __init__(self, model_type: str = "transformer"):
        """
        Initialize Voice Emotion Detector
        
        Args:
            model_type: Type of model to use ('transformer', 'cnn', 'rule_based')
        """
        self.model_type = model_type
        self._initialized = False
        self._librosa = None
        self._model = None
        self._processor = None
        
    def _lazy_init(self):
        """Lazy initialization of models and libraries"""
        if self._initialized:
            return
            
        # Try to import librosa
        try:
            import librosa
            self._librosa = librosa
            logger.info("Librosa loaded successfully")
        except ImportError:
            logger.warning("Librosa not available")
            self._librosa = None
            
        # Try to load transformer model for speech emotion recognition
        try:
            from transformers import pipeline
            self._model = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=-1  # CPU
            )
            logger.info("Speech emotion recognition model loaded")
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
            self._model = None
            
        self._initialized = True
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int = 22050) -> Dict:
        """
        Extract audio features using librosa
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            Dictionary of extracted features
        """
        self._lazy_init()
        
        if self._librosa is None:
            return {}
            
        features = {}
        
        try:
            # MFCC features
            mfcc = self._librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
            features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
            
            # Chroma features
            chroma = self._librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
            features['chroma_mean'] = np.mean(chroma, axis=1).tolist()
            
            # Spectral features
            spectral_centroids = self._librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)
            features['spectral_centroid'] = float(np.mean(spectral_centroids))
            
            spectral_rolloff = self._librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            # Zero crossing rate
            zcr = self._librosa.feature.zero_crossing_rate(audio_data)
            features['zero_crossing_rate'] = float(np.mean(zcr))
            
            # RMS energy
            rms = self._librosa.feature.rms(y=audio_data)
            features['rms_energy'] = float(np.mean(rms))
            
            # Tempo
            tempo, _ = self._librosa.beat.beat_track(y=audio_data, sr=sample_rate)
            features['tempo'] = float(tempo)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            
        return features
    
    def _rule_based_emotion(self, features: Dict) -> Tuple[str, float, Dict[str, float]]:
        """
        Rule-based emotion detection from audio features
        
        Args:
            features: Extracted audio features
            
        Returns:
            Tuple of (emotion, confidence, all_emotions)
        """
        # Initialize scores
        scores = {e: 0.1 for e in ['angry', 'calm', 'happy', 'sad', 'neutral', 'fear', 'surprise']}
        
        if not features:
            return 'neutral', 0.5, scores
            
        # Energy-based rules
        energy = features.get('rms_energy', 0.1)
        zcr = features.get('zero_crossing_rate', 0.1)
        tempo = features.get('tempo', 120)
        spectral_centroid = features.get('spectral_centroid', 2000)
        
        # High energy + high tempo = excited/angry/happy
        if energy > 0.15 and tempo > 120:
            scores['angry'] += 0.3
            scores['happy'] += 0.3
            scores['surprise'] += 0.2
        
        # Low energy + slow tempo = sad/calm
        if energy < 0.05 and tempo < 100:
            scores['sad'] += 0.4
            scores['calm'] += 0.3
        
        # High ZCR often indicates excitement or anger
        if zcr > 0.1:
            scores['angry'] += 0.2
            scores['surprise'] += 0.2
        
        # High spectral centroid = brighter tone (often happy)
        if spectral_centroid > 3000:
            scores['happy'] += 0.2
        elif spectral_centroid < 1500:
            scores['sad'] += 0.2
        
        # Medium energy = neutral
        if 0.05 <= energy <= 0.15:
            scores['neutral'] += 0.3
            scores['calm'] += 0.2
        
        # Normalize scores
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}
        
        # Get dominant emotion
        dominant = max(scores, key=scores.get)
        confidence = scores[dominant]
        
        return dominant, confidence, scores
    
    def analyze_emotion(self, audio_data: np.ndarray, sample_rate: int = 16000) -> VoiceEmotionResult:
        """
        Analyze emotion from audio data
        
        Args:
            audio_data: Audio signal as numpy array
            sample_rate: Sample rate of audio
            
        Returns:
            VoiceEmotionResult with detected emotion
        """
        self._lazy_init()
        
        # Normalize audio data
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Extract features
        features = self.extract_features(audio_data, sample_rate)
        
        # Try transformer model first
        if self._model is not None:
            try:
                # Save to temp file for model
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                    self._save_wav(audio_data, sample_rate, temp_path)
                
                # Run prediction
                predictions = self._model(temp_path)
                
                # Clean up temp file
                os.unlink(temp_path)
                
                # Process results
                all_emotions = {}
                for pred in predictions:
                    label = pred['label'].lower()
                    score = pred['score']
                    mapped_emotion = self.EMOTION_MAP.get(label, label)
                    all_emotions[mapped_emotion] = score
                
                # Get dominant emotion
                dominant = max(all_emotions, key=all_emotions.get)
                confidence = all_emotions[dominant]
                
                return VoiceEmotionResult(
                    modality='voice',
                    emotion=dominant,
                    confidence=confidence,
                    all_emotions=all_emotions,
                    features=features
                )
                
            except Exception as e:
                logger.warning(f"Transformer model failed: {e}")
        
        # Fall back to rule-based
        emotion, confidence, all_emotions = self._rule_based_emotion(features)
        
        return VoiceEmotionResult(
            modality='voice',
            emotion=emotion,
            confidence=confidence,
            all_emotions=all_emotions,
            features=features
        )
    
    def _save_wav(self, audio_data: np.ndarray, sample_rate: int, path: str):
        """Save audio data to WAV file"""
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
    
    def analyze_from_file(self, file_path: str) -> VoiceEmotionResult:
        """
        Analyze emotion from audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            VoiceEmotionResult with detected emotion
        """
        self._lazy_init()
        
        if self._librosa is not None:
            audio_data, sample_rate = self._librosa.load(file_path, sr=16000)
        else:
            # Basic WAV file reading
            with wave.open(file_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        
        return self.analyze_emotion(audio_data, sample_rate)
    
    def analyze_from_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> VoiceEmotionResult:
        """
        Analyze emotion from audio bytes
        
        Args:
            audio_bytes: Raw audio bytes (16-bit PCM)
            sample_rate: Sample rate of audio
            
        Returns:
            VoiceEmotionResult with detected emotion
        """
        # Convert bytes to numpy array
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        return self.analyze_emotion(audio_data, sample_rate)
    
    def get_emoji(self, emotion: str) -> str:
        """Get emoji for emotion"""
        return self.EMOJI_MAP.get(emotion.lower(), '😐')


# Singleton instance
_detector_instance = None

def get_detector() -> VoiceEmotionDetector:
    """Get singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = VoiceEmotionDetector()
    return _detector_instance


def analyze_voice_emotion(audio_data: np.ndarray, sample_rate: int = 16000) -> Dict:
    """
    Convenience function to analyze voice emotion
    
    Args:
        audio_data: Audio signal as numpy array
        sample_rate: Sample rate of audio
        
    Returns:
        Dictionary with emotion results
    """
    detector = get_detector()
    result = detector.analyze_emotion(audio_data, sample_rate)
    return {
        'modality': result.modality,
        'emotion': result.emotion,
        'confidence': result.confidence,
        'all_emotions': result.all_emotions
    }


if __name__ == "__main__":
    print("Voice Emotion Detection Module")
    print("=" * 40)
    
    # Test with synthetic audio
    duration = 3  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate test tone
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    test_audio = test_audio.astype(np.float32)
    
    detector = VoiceEmotionDetector()
    result = detector.analyze_emotion(test_audio, sample_rate)
    
    print(f"Detected Emotion: {result.emotion}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"All Emotions: {result.all_emotions}")
