"""
Multi-Modal Emotion Fusion Engine
Combines emotions from face, voice, and text modalities using intelligent reasoning
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModalityInput:
    """Input from a single modality"""
    modality: str  # 'face', 'voice', or 'text'
    emotion: str
    confidence: float
    all_emotions: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[datetime] = None


@dataclass
class FusionResult:
    """Result of emotion fusion"""
    final_emotion: str
    final_confidence: float
    reasoning: Dict[str, str]
    modality_weights: Dict[str, float]
    all_emotions: Dict[str, float]
    fusion_method: str
    timestamp: datetime = field(default_factory=datetime.now)


class EmotionFusionEngine:
    """
    Multi-Modal Emotion Fusion Engine
    Intelligently combines emotions from multiple modalities
    """
    
    # Standard emotion labels
    EMOTIONS = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral', 'calm']
    
    # Neutral penalty - reduces influence of neutral when other emotions are detected
    # This helps avoid the common issue where face detection defaults to neutral
    NEUTRAL_PENALTY = 0.6  # Multiply neutral scores by this factor
    
    # Default modality weights
    DEFAULT_WEIGHTS = {
        'face': 0.4,
        'voice': 0.35,
        'text': 0.25
    }
    
    # Emotion compatibility matrix (which emotions can co-occur)
    EMOTION_COMPATIBILITY = {
        'happy': ['surprise', 'calm', 'neutral'],
        'sad': ['fear', 'neutral', 'calm'],
        'angry': ['disgust', 'fear', 'surprise'],
        'fear': ['sad', 'angry', 'surprise'],
        'surprise': ['happy', 'fear', 'angry'],
        'disgust': ['angry', 'sad', 'fear'],
        'neutral': ['calm', 'happy', 'sad'],
        'calm': ['neutral', 'happy', 'sad']
    }
    
    # Emoji mapping
    EMOJI_MAP = {
        'happy': '😄',
        'sad': '😢',
        'angry': '😡',
        'fear': '😨',
        'surprise': '😲',
        'disgust': '🤢',
        'neutral': '😐',
        'calm': '😌'
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None, 
                 fusion_method: str = 'weighted_average'):
        """
        Initialize Fusion Engine
        
        Args:
            weights: Custom weights for modalities
            fusion_method: Fusion method ('weighted_average', 'voting', 'confidence_based', 'rule_based')
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.fusion_method = fusion_method
        self._normalize_weights()
        
    def _normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
    
    def _apply_neutral_penalty(self, emotion_scores: Dict[str, float], inputs: List['ModalityInput']) -> Dict[str, float]:
        """
        Apply neutral penalty to reduce neutral bias in fusion
        
        When face detection shows neutral but other modalities detect real emotions,
        we penalize neutral to let the true emotion shine through.
        """
        # Check if any non-neutral emotion was detected with decent confidence
        has_non_neutral = any(
            inp.emotion.lower() != 'neutral' and inp.confidence > 0.3 
            for inp in inputs
        )
        
        if has_non_neutral and 'neutral' in emotion_scores:
            # Apply penalty to neutral
            emotion_scores['neutral'] *= self.NEUTRAL_PENALTY
            logger.debug(f"Applied neutral penalty: {emotion_scores['neutral']:.3f}")
        
        return emotion_scores
    
    def set_weights(self, weights: Dict[str, float]):
        """Update modality weights"""
        self.weights.update(weights)
        self._normalize_weights()
    
    def fuse_emotions(self, inputs: List[ModalityInput]) -> FusionResult:
        """
        Fuse emotions from multiple modalities
        
        Args:
            inputs: List of ModalityInput from different modalities
            
        Returns:
            FusionResult with final emotion and reasoning
        """
        if not inputs:
            return FusionResult(
                final_emotion='neutral',
                final_confidence=0.5,
                reasoning={'note': 'No input provided'},
                modality_weights=self.weights,
                all_emotions={'neutral': 1.0},
                fusion_method=self.fusion_method
            )
        
        # Select fusion method
        if self.fusion_method == 'weighted_average':
            return self._weighted_average_fusion(inputs)
        elif self.fusion_method == 'voting':
            return self._voting_fusion(inputs)
        elif self.fusion_method == 'confidence_based':
            return self._confidence_based_fusion(inputs)
        elif self.fusion_method == 'rule_based':
            return self._rule_based_fusion(inputs)
        else:
            return self._weighted_average_fusion(inputs)
    
    def _weighted_average_fusion(self, inputs: List[ModalityInput]) -> FusionResult:
        """
        Weighted average fusion method
        Combines all emotion scores with modality weights
        """
        # Initialize emotion scores
        emotion_scores = {e: 0.0 for e in self.EMOTIONS}
        total_weight = 0.0
        reasoning = {}
        
        for inp in inputs:
            weight = self.weights.get(inp.modality, 0.25)
            total_weight += weight
            
            # Add weighted emotion scores
            if inp.all_emotions:
                for emotion, score in inp.all_emotions.items():
                    emotion_key = emotion.lower()
                    if emotion_key in emotion_scores:
                        emotion_scores[emotion_key] += score * weight
            else:
                # If only dominant emotion is available
                emotion_key = inp.emotion.lower()
                if emotion_key in emotion_scores:
                    emotion_scores[emotion_key] += inp.confidence * weight
            
            # Add to reasoning
            emoji = self.EMOJI_MAP.get(inp.emotion.lower(), '😐')
            reasoning[inp.modality] = f"{emoji} {inp.emotion} ({inp.confidence:.0%})"
        
        # Normalize scores
        if total_weight > 0:
            emotion_scores = {k: v / total_weight for k, v in emotion_scores.items()}
        
        # Apply neutral penalty to avoid neutral bias from face detection
        emotion_scores = self._apply_neutral_penalty(emotion_scores, inputs)
        
        # Get final emotion
        final_emotion = max(emotion_scores, key=emotion_scores.get)
        final_confidence = emotion_scores[final_emotion]
        
        # Add fusion reasoning
        reasoning['method'] = 'Weighted average across modalities'
        reasoning['final'] = f"Combined {len(inputs)} modalities with weights"
        
        return FusionResult(
            final_emotion=final_emotion,
            final_confidence=final_confidence,
            reasoning=reasoning,
            modality_weights={inp.modality: self.weights.get(inp.modality, 0.25) for inp in inputs},
            all_emotions=emotion_scores,
            fusion_method='weighted_average'
        )
    
    def _voting_fusion(self, inputs: List[ModalityInput]) -> FusionResult:
        """
        Voting fusion method
        Each modality votes for its detected emotion
        """
        votes = {}
        reasoning = {}
        
        for inp in inputs:
            emotion = inp.emotion.lower()
            if emotion not in votes:
                votes[emotion] = {'count': 0, 'total_confidence': 0.0}
            votes[emotion]['count'] += 1
            votes[emotion]['total_confidence'] += inp.confidence
            
            emoji = self.EMOJI_MAP.get(emotion, '😐')
            reasoning[inp.modality] = f"{emoji} {inp.emotion} ({inp.confidence:.0%})"
        
        # Apply neutral penalty if other emotions have votes with good confidence
        has_non_neutral = any(
            e != 'neutral' and votes[e]['total_confidence'] > 0.3 
            for e in votes
        )
        if has_non_neutral and 'neutral' in votes:
            votes['neutral']['total_confidence'] *= self.NEUTRAL_PENALTY
        
        # Get winning emotion
        final_emotion = max(votes, key=lambda x: (votes[x]['count'], votes[x]['total_confidence']))
        vote_info = votes[final_emotion]
        final_confidence = vote_info['total_confidence'] / vote_info['count']
        
        # Create all_emotions dict
        all_emotions = {e: 0.0 for e in self.EMOTIONS}
        for emotion, info in votes.items():
            if emotion in all_emotions:
                all_emotions[emotion] = info['total_confidence'] / len(inputs)
        
        reasoning['method'] = 'Majority voting with confidence tiebreaker'
        reasoning['votes'] = f"{final_emotion}: {vote_info['count']} vote(s)"
        
        return FusionResult(
            final_emotion=final_emotion,
            final_confidence=final_confidence,
            reasoning=reasoning,
            modality_weights={inp.modality: 1.0 / len(inputs) for inp in inputs},
            all_emotions=all_emotions,
            fusion_method='voting'
        )
    
    def _confidence_based_fusion(self, inputs: List[ModalityInput]) -> FusionResult:
        """
        Confidence-based fusion method
        Dynamically adjusts weights based on confidence scores
        """
        reasoning = {}
        emotion_scores = {e: 0.0 for e in self.EMOTIONS}
        
        # Calculate dynamic weights based on confidence
        confidence_sum = sum(inp.confidence for inp in inputs)
        if confidence_sum == 0:
            confidence_sum = 1.0
        
        dynamic_weights = {}
        for inp in inputs:
            # Combine static and dynamic weights
            base_weight = self.weights.get(inp.modality, 0.25)
            confidence_weight = inp.confidence / confidence_sum
            dynamic_weights[inp.modality] = 0.5 * base_weight + 0.5 * confidence_weight
        
        # Normalize
        total = sum(dynamic_weights.values())
        if total > 0:
            dynamic_weights = {k: v / total for k, v in dynamic_weights.items()}
        
        # Apply fusion
        for inp in inputs:
            weight = dynamic_weights[inp.modality]
            
            if inp.all_emotions:
                for emotion, score in inp.all_emotions.items():
                    if emotion.lower() in emotion_scores:
                        emotion_scores[emotion.lower()] += score * weight
            else:
                if inp.emotion.lower() in emotion_scores:
                    emotion_scores[inp.emotion.lower()] += inp.confidence * weight
            
            emoji = self.EMOJI_MAP.get(inp.emotion.lower(), '😐')
            reasoning[inp.modality] = f"{emoji} {inp.emotion} ({inp.confidence:.0%}) [w={weight:.2f}]"
        
        # Apply neutral penalty to avoid neutral bias from face detection
        emotion_scores = self._apply_neutral_penalty(emotion_scores, inputs)
        
        # Get final emotion
        final_emotion = max(emotion_scores, key=emotion_scores.get)
        final_confidence = emotion_scores[final_emotion]
        
        reasoning['method'] = 'Confidence-based dynamic weighting'
        
        return FusionResult(
            final_emotion=final_emotion,
            final_confidence=final_confidence,
            reasoning=reasoning,
            modality_weights=dynamic_weights,
            all_emotions=emotion_scores,
            fusion_method='confidence_based'
        )
    
    def _rule_based_fusion(self, inputs: List[ModalityInput]) -> FusionResult:
        """
        Rule-based fusion with expert knowledge
        Uses emotion compatibility and contextual rules
        """
        reasoning = {}
        
        # Get emotions from each modality
        modality_emotions = {}
        for inp in inputs:
            modality_emotions[inp.modality] = {
                'emotion': inp.emotion.lower(),
                'confidence': inp.confidence
            }
            emoji = self.EMOJI_MAP.get(inp.emotion.lower(), '😐')
            reasoning[inp.modality] = f"{emoji} {inp.emotion} ({inp.confidence:.0%})"
        
        # Apply rules
        emotions_detected = [v['emotion'] for v in modality_emotions.values()]
        confidences = [v['confidence'] for v in modality_emotions.values()]
        
        # Filter out neutral if other emotions are present with good confidence
        non_neutral_emotions = [e for e in emotions_detected if e != 'neutral']
        non_neutral_confidences = [
            v['confidence'] for v in modality_emotions.values() 
            if v['emotion'] != 'neutral' and v['confidence'] > 0.3
        ]
        
        # Rule 0: If neutral is detected but other emotions have good confidence, ignore neutral
        if non_neutral_emotions and non_neutral_confidences:
            # Replace neutral with the strongest non-neutral emotion for voting
            emotions_for_voting = non_neutral_emotions if len(non_neutral_emotions) > 0 else emotions_detected
        else:
            emotions_for_voting = emotions_detected
        
        # Rule 1: If all modalities agree, use that emotion with high confidence
        if len(set(emotions_for_voting)) == 1:
            final_emotion = emotions_for_voting[0]
            final_confidence = min(1.0, sum(confidences) / len(confidences) + 0.1)
            reasoning['rule'] = 'All modalities agree'
        
        # Rule 2: If face and text agree but voice differs, prefer face+text
        elif 'face' in modality_emotions and 'text' in modality_emotions:
            face_em = modality_emotions['face']['emotion']
            text_em = modality_emotions['text']['emotion']
            
            # If face is neutral but text has a real emotion, prefer text
            if face_em == 'neutral' and text_em != 'neutral':
                final_emotion = text_em
                final_confidence = modality_emotions['text']['confidence']
                reasoning['rule'] = 'Text emotion preferred over neutral face'
            elif face_em == text_em:
                final_emotion = face_em
                final_confidence = (modality_emotions['face']['confidence'] + 
                                   modality_emotions['text']['confidence']) / 2
                reasoning['rule'] = 'Face and text agree'
            else:
                # Use highest confidence, but penalize neutral
                candidates = []
                for modality, data in modality_emotions.items():
                    conf = data['confidence']
                    if data['emotion'] == 'neutral':
                        conf *= self.NEUTRAL_PENALTY
                    candidates.append((modality, data['emotion'], conf))
                
                best = max(candidates, key=lambda x: x[2])
                final_emotion = best[1]
                final_confidence = modality_emotions[best[0]]['confidence']
                reasoning['rule'] = f'Conflict resolved by highest confidence ({best[0]})'
        
        # Rule 3: Check emotion compatibility
        else:
            # Calculate weighted score with compatibility bonus
            emotion_scores = {e: 0.0 for e in self.EMOTIONS}
            
            for inp in inputs:
                emotion = inp.emotion.lower()
                weight = self.weights.get(inp.modality, 0.25)
                
                if emotion in emotion_scores:
                    emotion_scores[emotion] += inp.confidence * weight
                
                # Add bonus for compatible emotions
                compatible = self.EMOTION_COMPATIBILITY.get(emotion, [])
                for other in emotions_detected:
                    if other in compatible and other != emotion:
                        emotion_scores[emotion] += 0.05
            
            # Apply neutral penalty
            emotion_scores = self._apply_neutral_penalty(emotion_scores, inputs)
            
            final_emotion = max(emotion_scores, key=emotion_scores.get)
            final_confidence = emotion_scores[final_emotion]
            reasoning['rule'] = 'Compatibility-weighted scoring'
        
        # Generate all_emotions
        all_emotions = {e: 0.0 for e in self.EMOTIONS}
        all_emotions[final_emotion] = final_confidence
        for inp in inputs:
            if inp.emotion.lower() != final_emotion and inp.emotion.lower() in all_emotions:
                all_emotions[inp.emotion.lower()] = inp.confidence * 0.5
        
        # Normalize
        total = sum(all_emotions.values())
        if total > 0:
            all_emotions = {k: v / total for k, v in all_emotions.items()}
        
        reasoning['method'] = 'Rule-based expert fusion'
        
        return FusionResult(
            final_emotion=final_emotion,
            final_confidence=min(final_confidence, 1.0),
            reasoning=reasoning,
            modality_weights=self.weights,
            all_emotions=all_emotions,
            fusion_method='rule_based'
        )
    
    def get_emoji(self, emotion: str) -> str:
        """Get emoji for emotion"""
        return self.EMOJI_MAP.get(emotion.lower(), '😐')
    
    def explain_fusion(self, result: FusionResult) -> str:
        """
        Generate human-readable explanation of fusion
        
        Args:
            result: FusionResult to explain
            
        Returns:
            Explanation string
        """
        lines = [
            f"🧠 Final Emotion: {self.get_emoji(result.final_emotion)} {result.final_emotion.upper()}",
            f"📊 Confidence: {result.final_confidence:.0%}",
            f"🔄 Fusion Method: {result.fusion_method}",
            "",
            "📝 Modality Analysis:"
        ]
        
        for modality, detail in result.reasoning.items():
            if modality not in ['method', 'rule', 'final', 'votes']:
                lines.append(f"  • {modality.capitalize()}: {detail}")
        
        if 'rule' in result.reasoning:
            lines.append(f"\n🔍 Rule Applied: {result.reasoning['rule']}")
        
        return "\n".join(lines)


# Convenience function
def fuse_emotions(face_result: Optional[Dict] = None,
                  voice_result: Optional[Dict] = None,
                  text_result: Optional[Dict] = None,
                  method: str = 'confidence_based') -> Dict:
    """
    Convenience function to fuse emotions from all modalities
    
    Args:
        face_result: Face emotion detection result
        voice_result: Voice emotion detection result
        text_result: Text emotion detection result
        method: Fusion method to use
        
    Returns:
        Dictionary with fusion results
    """
    engine = EmotionFusionEngine(fusion_method=method)
    inputs = []
    
    if face_result:
        inputs.append(ModalityInput(
            modality='face',
            emotion=face_result.get('emotion', 'neutral'),
            confidence=face_result.get('confidence', 0.5),
            all_emotions=face_result.get('all_emotions', {})
        ))
    
    if voice_result:
        inputs.append(ModalityInput(
            modality='voice',
            emotion=voice_result.get('emotion', 'neutral'),
            confidence=voice_result.get('confidence', 0.5),
            all_emotions=voice_result.get('all_emotions', {})
        ))
    
    if text_result:
        inputs.append(ModalityInput(
            modality='text',
            emotion=text_result.get('emotion', 'neutral'),
            confidence=text_result.get('confidence', 0.5),
            all_emotions=text_result.get('all_emotions', {})
        ))
    
    result = engine.fuse_emotions(inputs)
    
    return {
        'final_emotion': result.final_emotion,
        'final_confidence': result.final_confidence,
        'reasoning': result.reasoning,
        'modality_weights': result.modality_weights,
        'all_emotions': result.all_emotions,
        'fusion_method': result.fusion_method,
        'explanation': engine.explain_fusion(result)
    }


if __name__ == "__main__":
    print("Multi-Modal Emotion Fusion Engine")
    print("=" * 50)
    
    # Test data
    face_result = {'emotion': 'sad', 'confidence': 0.82, 'all_emotions': {'sad': 0.82, 'neutral': 0.1, 'fear': 0.08}}
    voice_result = {'emotion': 'calm', 'confidence': 0.74, 'all_emotions': {'calm': 0.74, 'neutral': 0.2, 'sad': 0.06}}
    text_result = {'emotion': 'sad', 'confidence': 0.68, 'all_emotions': {'sad': 0.68, 'fear': 0.2, 'neutral': 0.12}}
    
    # Test fusion
    result = fuse_emotions(face_result, voice_result, text_result, method='confidence_based')
    
    print(f"\nInput:")
    print(f"  Face: {face_result['emotion']} ({face_result['confidence']:.0%})")
    print(f"  Voice: {voice_result['emotion']} ({voice_result['confidence']:.0%})")
    print(f"  Text: {text_result['emotion']} ({text_result['confidence']:.0%})")
    
    print(f"\n{result['explanation']}")
