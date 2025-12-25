"""
Text Sentiment/Emotion Analysis Module
Detects emotions from text using NLP models
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextEmotionResult:
    """Data class for text emotion detection results"""
    modality: str
    emotion: str
    confidence: float
    all_emotions: Dict[str, float]
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None


class TextEmotionDetector:
    """
    Text Emotion Detector using NLP models
    Supports emotion detection and sentiment analysis from text
    """
    
    # Standard emotion labels
    EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'neutral']
    
    # Mapping from model labels to standard emotions
    EMOTION_MAP = {
        'anger': 'angry',
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'fearful': 'fear',
        'joy': 'happy',
        'happy': 'happy',
        'happiness': 'happy',
        'love': 'happy',
        'sadness': 'sad',
        'sad': 'sad',
        'surprise': 'surprise',
        'surprised': 'surprise',
        'neutral': 'neutral'
    }
    
    # Emoji mapping
    EMOJI_MAP = {
        'angry': '😡',
        'disgust': '🤢',
        'fear': '😨',
        'happy': '😄',
        'sad': '😢',
        'surprise': '😲',
        'neutral': '😐'
    }
    
    # Emotion keywords for rule-based fallback
    EMOTION_KEYWORDS = {
        'happy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'love', 'fantastic', 
                  'awesome', 'glad', 'pleased', 'delighted', 'cheerful', 'thrilled', 'blessed'],
        'sad': ['sad', 'unhappy', 'depressed', 'lonely', 'miserable', 'hopeless', 'gloomy', 
                'heartbroken', 'down', 'blue', 'disappointed', 'grief', 'sorrow', 'hurt'],
        'angry': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'irritated', 'outraged',
                  'hate', 'rage', 'upset', 'pissed', 'livid', 'enraged'],
        'fear': ['afraid', 'scared', 'fearful', 'anxious', 'worried', 'nervous', 'terrified',
                 'panic', 'dread', 'frightened', 'uneasy', 'alarmed'],
        'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'wow',
                     'stunning', 'incredible', 'unbelievable', 'startled'],
        'disgust': ['disgusted', 'gross', 'revolting', 'sick', 'nauseating', 'repulsive',
                    'horrible', 'awful', 'terrible', 'yuck'],
        'neutral': ['okay', 'fine', 'alright', 'normal', 'usual', 'regular']
    }
    
    def __init__(self, model_name: str = "auto"):
        """
        Initialize Text Emotion Detector
        
        Args:
            model_name: Model to use ('auto', 'bert', 'roberta', 'rule_based')
        """
        self.model_name = model_name
        self._initialized = False
        self._emotion_model = None
        self._sentiment_model = None
        
    def _lazy_init(self):
        """Lazy initialization of models"""
        if self._initialized:
            return
            
        # Try to load emotion classification model
        try:
            from transformers import pipeline
            
            # Try emotion model
            try:
                self._emotion_model = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None,
                    device=-1
                )
                logger.info("Emotion classification model loaded")
            except Exception as e:
                logger.warning(f"Could not load emotion model: {e}")
                
            # Try sentiment model as backup
            try:
                self._sentiment_model = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1
                )
                logger.info("Sentiment analysis model loaded")
            except Exception as e:
                logger.warning(f"Could not load sentiment model: {e}")
                
        except ImportError:
            logger.warning("Transformers not available, using rule-based analysis")
            
        self._initialized = True
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for analysis
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?\'"-]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _rule_based_emotion(self, text: str) -> Dict[str, float]:
        """
        Rule-based emotion detection using keywords
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of emotion scores
        """
        text_lower = text.lower()
        words = set(text_lower.split())
        
        scores = {emotion: 0.1 for emotion in ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']}
        
        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[emotion] += 0.2
                if keyword in words:
                    scores[emotion] += 0.1
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def analyze_emotion(self, text: str) -> TextEmotionResult:
        """
        Analyze emotion from text
        
        Args:
            text: Input text
            
        Returns:
            TextEmotionResult with detected emotion
        """
        self._lazy_init()
        
        if not text or not text.strip():
            return TextEmotionResult(
                modality='text',
                emotion='neutral',
                confidence=0.5,
                all_emotions={'neutral': 0.5}
            )
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Try transformer model first
        if self._emotion_model is not None:
            try:
                results = self._emotion_model(processed_text)
                
                # Process results
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], list):
                        results = results[0]
                    
                    all_emotions = {}
                    for result in results:
                        label = result['label'].lower()
                        score = result['score']
                        mapped = self.EMOTION_MAP.get(label, label)
                        if mapped in all_emotions:
                            all_emotions[mapped] = max(all_emotions[mapped], score)
                        else:
                            all_emotions[mapped] = score
                    
                    # Get dominant emotion
                    dominant = max(all_emotions, key=all_emotions.get)
                    confidence = all_emotions[dominant]
                    
                    # Get sentiment
                    sentiment = None
                    sentiment_score = None
                    if self._sentiment_model is not None:
                        try:
                            sent_result = self._sentiment_model(processed_text)[0]
                            sentiment = sent_result['label'].lower()
                            sentiment_score = sent_result['score']
                        except:
                            pass
                    
                    return TextEmotionResult(
                        modality='text',
                        emotion=dominant,
                        confidence=confidence,
                        all_emotions=all_emotions,
                        sentiment=sentiment,
                        sentiment_score=sentiment_score
                    )
                    
            except Exception as e:
                logger.warning(f"Transformer model failed: {e}")
        
        # Fall back to rule-based
        all_emotions = self._rule_based_emotion(processed_text)
        dominant = max(all_emotions, key=all_emotions.get)
        confidence = all_emotions[dominant]
        
        # Simple sentiment from emotion
        positive_emotions = ['happy', 'surprise']
        negative_emotions = ['sad', 'angry', 'fear', 'disgust']
        
        if dominant in positive_emotions:
            sentiment = 'positive'
        elif dominant in negative_emotions:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return TextEmotionResult(
            modality='text',
            emotion=dominant,
            confidence=confidence,
            all_emotions=all_emotions,
            sentiment=sentiment,
            sentiment_score=confidence
        )
    
    def analyze_batch(self, texts: List[str]) -> List[TextEmotionResult]:
        """
        Analyze emotions from multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of TextEmotionResult objects
        """
        return [self.analyze_emotion(text) for text in texts]
    
    def get_emoji(self, emotion: str) -> str:
        """Get emoji for emotion"""
        return self.EMOJI_MAP.get(emotion.lower(), '😐')
    
    def get_emotion_intensity(self, text: str) -> str:
        """
        Get intensity level of detected emotion
        
        Args:
            text: Input text
            
        Returns:
            Intensity level ('low', 'medium', 'high')
        """
        result = self.analyze_emotion(text)
        
        if result.confidence < 0.4:
            return 'low'
        elif result.confidence < 0.7:
            return 'medium'
        else:
            return 'high'


# Singleton instance
_detector_instance = None

def get_detector() -> TextEmotionDetector:
    """Get singleton detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = TextEmotionDetector()
    return _detector_instance


def analyze_text_emotion(text: str) -> Dict:
    """
    Convenience function to analyze text emotion
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with emotion results
    """
    detector = get_detector()
    result = detector.analyze_emotion(text)
    return {
        'modality': result.modality,
        'emotion': result.emotion,
        'confidence': result.confidence,
        'all_emotions': result.all_emotions,
        'sentiment': result.sentiment,
        'sentiment_score': result.sentiment_score
    }


if __name__ == "__main__":
    print("Text Emotion Detection Module")
    print("=" * 40)
    
    # Test texts
    test_texts = [
        "I'm so happy today! Everything is going great!",
        "I feel lonely and sad, nothing seems to work out.",
        "This is absolutely infuriating! I can't believe this happened!",
        "I'm a bit nervous about the upcoming presentation.",
        "Wow, I didn't expect that at all! What a surprise!",
        "Today was just a normal day, nothing special."
    ]
    
    detector = TextEmotionDetector()
    
    for text in test_texts:
        result = detector.analyze_emotion(text)
        emoji = detector.get_emoji(result.emotion)
        print(f"\nText: {text}")
        print(f"Emotion: {emoji} {result.emotion.upper()} ({result.confidence:.0%})")
        print(f"Sentiment: {result.sentiment}")
