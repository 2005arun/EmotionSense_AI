"""
Face Emotion Detection Module
Detects emotions from facial expressions using DeepFace and OpenCV
With temporal smoothing to reduce neutral bias
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from collections import deque
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmotionResult:
    """Data class for emotion detection results"""
    modality: str
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    raw_emotion: str = None  # Before smoothing
    smoothed: bool = False


class EmotionSmoother:
    """
    Temporal smoothing for emotion detection
    Reduces neutral bias by averaging over multiple frames
    """
    
    def __init__(self, window_size: int = 10, neutral_penalty: float = 0.7):
        """
        Args:
            window_size: Number of frames to average
            neutral_penalty: Penalty multiplier for neutral emotion (0-1)
        """
        self.window_size = window_size
        self.neutral_penalty = neutral_penalty
        self.history: deque = deque(maxlen=window_size)
        
    def add_and_smooth(self, emotions: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Add emotion scores and return smoothed result
        
        Args:
            emotions: Dictionary of emotion -> confidence scores
            
        Returns:
            Tuple of (smoothed_emotion, confidence, smoothed_all_emotions)
        """
        # Add to history
        self.history.append(emotions.copy())
        
        if len(self.history) < 2:
            # Not enough history, return as-is but penalize neutral
            adjusted = self._adjust_neutral(emotions)
            dominant = max(adjusted, key=adjusted.get)
            return dominant, adjusted[dominant], adjusted
        
        # Average across history
        smoothed = {}
        for emotion in emotions.keys():
            values = [h.get(emotion, 0) for h in self.history]
            smoothed[emotion] = sum(values) / len(values)
        
        # Apply neutral penalty
        smoothed = self._adjust_neutral(smoothed)
        
        # Get dominant emotion
        dominant = max(smoothed, key=smoothed.get)
        
        return dominant, smoothed[dominant], smoothed
    
    def _adjust_neutral(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """Apply penalty to neutral emotion to reduce bias"""
        adjusted = emotions.copy()
        if 'neutral' in adjusted:
            adjusted['neutral'] *= self.neutral_penalty
        return adjusted
    
    def clear(self):
        """Clear history"""
        self.history.clear()


class FaceEmotionDetector:
    """
    Face Emotion Detector using DeepFace library
    With temporal smoothing and improved detection
    """
    
    # Emotion labels
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Emotion to emoji mapping
    EMOJI_MAP = {
        'angry': '😡',
        'disgust': '🤢',
        'fear': '😨',
        'happy': '😄',
        'sad': '😢',
        'surprise': '😲',
        'neutral': '😐'
    }
    
    def __init__(self, model_name: str = "emotion", use_smoothing: bool = True,
                 smoothing_window: int = 10, neutral_penalty: float = 0.7):
        """
        Initialize the Face Emotion Detector
        
        Args:
            model_name: Model to use for emotion detection
            use_smoothing: Whether to use temporal smoothing
            smoothing_window: Number of frames for smoothing
            neutral_penalty: Penalty for neutral emotion (reduces bias)
        """
        self.model_name = model_name
        self.use_smoothing = use_smoothing
        self._deepface = None
        self._face_cascade = None
        self._dnn_net = None
        self._initialized = False
        
        # Initialize smoother
        self.smoother = EmotionSmoother(
            window_size=smoothing_window,
            neutral_penalty=neutral_penalty
        ) if use_smoothing else None
        
    def _lazy_init(self):
        """Lazy initialization of models"""
        if self._initialized:
            return
            
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            logger.info("[FaceEmotion] DeepFace loaded successfully")
        except ImportError:
            logger.warning("[FaceEmotion] DeepFace not available, using fallback detector")
            self._deepface = None
        
        # Load DNN face detector (more accurate than Haar Cascade)
        try:
            # Try to use DNN-based face detector
            prototxt_path = cv2.data.haarcascades + '../dnn/face_detector/deploy.prototxt'
            model_path = cv2.data.haarcascades + '../dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
            
            import os
            if os.path.exists(prototxt_path) and os.path.exists(model_path):
                self._dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                logger.info("[FaceEmotion] DNN face detector loaded")
        except Exception as e:
            logger.debug(f"[FaceEmotion] DNN detector not available: {e}")
            
        # Load face cascade as fallback
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            logger.info("[FaceEmotion] Haar Cascade loaded as fallback")
        except Exception as e:
            logger.error(f"[FaceEmotion] Failed to load face cascade: {e}")
            
        self._initialized = True
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better emotion detection
        - Improve lighting/contrast
        - Ensure proper size
        """
        # Convert to RGB if needed (DeepFace expects RGB)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Apply CLAHE for better contrast (helps with lighting issues)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in image using DNN or Haar Cascade
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (x, y, w, h) or None if no face detected
        """
        self._lazy_init()
        
        # Try DNN detector first (more accurate)
        if self._dnn_net is not None:
            try:
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
                self._dnn_net.setInput(blob)
                detections = self._dnn_net.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        return (x1, y1, x2 - x1, y2 - y1)
            except Exception as e:
                logger.debug(f"DNN detection failed: {e}")
        
        # Fallback to Haar Cascade
        if self._face_cascade is None:
            return None
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None
            
        # Return the largest face
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        return tuple(largest_face)
    
    def analyze_emotion(self, image: np.ndarray, enforce_detection: bool = False) -> EmotionResult:
        """
        Analyze emotion from facial expression with smoothing
        
        Args:
            image: Input image as numpy array (BGR format)
            enforce_detection: If True, raises exception if no face detected
            
        Returns:
            EmotionResult with detected emotion and confidence
        """
        self._lazy_init()
        
        # Preprocess image for better detection
        enhanced_image = self._preprocess_image(image)
        
        raw_emotion = 'neutral'
        emotions = {e: 0.14 for e in self.EMOTIONS}
        
        if self._deepface is not None:
            try:
                # Use DeepFace with better settings
                result = self._deepface.analyze(
                    enhanced_image,
                    actions=['emotion'],
                    enforce_detection=enforce_detection,
                    detector_backend='opencv',  # Fast and reliable
                    silent=True
                )
                
                # Handle both single and multiple face results
                if isinstance(result, list):
                    result = result[0]
                
                emotions = result.get('emotion', {})
                raw_emotion = result.get('dominant_emotion', 'neutral')
                
                # Normalize emotions to 0-1 range
                emotions = {k: v / 100.0 for k, v in emotions.items()}
                
                logger.debug(f"[FaceEmotion] Raw detection: {raw_emotion} ({emotions.get(raw_emotion, 0):.2%})")
                
            except Exception as e:
                logger.warning(f"[FaceEmotion] DeepFace analysis failed: {e}")
        
        # Apply temporal smoothing if enabled
        if self.smoother is not None:
            smoothed_emotion, smoothed_conf, smoothed_all = self.smoother.add_and_smooth(emotions)
            
            logger.debug(f"[FaceEmotion] Smoothed: {raw_emotion} -> {smoothed_emotion}")
            
            return EmotionResult(
                modality='face',
                emotion=smoothed_emotion,
                confidence=smoothed_conf,
                all_emotions=smoothed_all,
                raw_emotion=raw_emotion,
                smoothed=True
            )
        
        # No smoothing - return raw result with neutral penalty
        dominant = max(emotions, key=emotions.get)
        if dominant == 'neutral' and len(emotions) > 1:
            # Check if any other emotion is close
            sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_emotions) > 1:
                second_best = sorted_emotions[1]
                # If second best is within 15% of neutral, prefer it
                if second_best[1] > emotions['neutral'] * 0.85:
                    dominant = second_best[0]
        
        return EmotionResult(
            modality='face',
            emotion=dominant,
            confidence=emotions.get(dominant, 0.5),
            all_emotions=emotions,
            raw_emotion=raw_emotion,
            smoothed=False
        )
    
    def analyze_from_webcam(self, frame: np.ndarray) -> Tuple[EmotionResult, np.ndarray]:
        """
        Analyze emotion from webcam frame and draw annotations
        
        Args:
            frame: Webcam frame as numpy array
            
        Returns:
            Tuple of (EmotionResult, annotated frame)
        """
        result = self.analyze_emotion(frame, enforce_detection=False)
        annotated = self._annotate_frame(frame.copy(), result)
        return result, annotated
    
    def _annotate_frame(self, frame: np.ndarray, result: EmotionResult) -> np.ndarray:
        """Add emotion annotation to frame"""
        # Detect face location for drawing box
        face_loc = self.detect_face(frame)
        
        if face_loc:
            x, y, w, h = face_loc
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add emotion label
            label = f"{result.emotion.upper()} ({result.confidence:.0%})"
            cv2.putText(
                frame, label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2
            )
        
        return frame
    
    def get_emoji(self, emotion: str) -> str:
        """Get emoji for emotion"""
        return self.EMOJI_MAP.get(emotion.lower(), '😐')
    
    def analyze_image_file(self, image_path: str) -> EmotionResult:
        """
        Analyze emotion from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            EmotionResult with detected emotion
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self.analyze_emotion(image)


# Singleton instance for easy access
_detector_instance = None

def get_detector() -> FaceEmotionDetector:
    """Get singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceEmotionDetector()
    return _detector_instance


def analyze_face_emotion(image: np.ndarray) -> Dict:
    """
    Convenience function to analyze face emotion
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Dictionary with emotion results
    """
    detector = get_detector()
    result = detector.analyze_emotion(image)
    return {
        'modality': result.modality,
        'emotion': result.emotion,
        'confidence': result.confidence,
        'all_emotions': result.all_emotions
    }


if __name__ == "__main__":
    # Test the module
    print("Face Emotion Detection Module")
    print("=" * 40)
    
    # Test with webcam if available
    cap = cv2.VideoCapture(0)
    detector = FaceEmotionDetector()
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            result = detector.analyze_emotion(frame)
            print(f"Detected Emotion: {result.emotion}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"All Emotions: {result.all_emotions}")
        cap.release()
    else:
        print("No webcam available for testing")
