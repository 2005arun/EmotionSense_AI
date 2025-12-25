"""
Music Generator Module
Generates personalized music based on detected emotions using MusicGen
"""

import os
import logging
import tempfile
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
from datetime import datetime
import wave
import struct

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class MusicConfig:
    """Configuration for music generation"""
    emotion: str
    prompt: str
    duration: int = 10  # seconds
    temperature: float = 1.0
    top_k: int = 250
    top_p: float = 0.0
    guidance_scale: float = 3.0


@dataclass
class GeneratedMusic:
    """Result of music generation"""
    file_path: str
    emotion: str
    prompt: str
    duration: float
    sample_rate: int
    genre: str
    tempo: str
    timestamp: datetime


class MusicGenerator:
    """
    Music Generator using Meta's MusicGen or fallback methods
    Generates emotion-appropriate music
    """
    
    # Emotion to music prompt mapping
    EMOTION_MUSIC_MAP = {
        'happy': {
            'prompt': 'upbeat cheerful music, major key, bright melody, energetic pop, acoustic guitar, happy vibes',
            'genre': 'pop',
            'tempo': 'fast',
            'bpm_range': (110, 140)
        },
        'sad': {
            'prompt': 'melancholic piano music, slow tempo, minor key, emotional strings, soft and gentle',
            'genre': 'classical',
            'tempo': 'slow',
            'bpm_range': (60, 80)
        },
        'angry': {
            'prompt': 'intense aggressive music, heavy drums, distorted guitars, powerful and driving rhythm',
            'genre': 'rock',
            'tempo': 'fast',
            'bpm_range': (120, 160)
        },
        'fear': {
            'prompt': 'tense suspenseful music, minor key, eerie atmosphere, dark ambient, mysterious',
            'genre': 'ambient',
            'tempo': 'moderate',
            'bpm_range': (80, 100)
        },
        'surprise': {
            'prompt': 'exciting dynamic music, unexpected changes, playful melody, orchestra hits, dramatic',
            'genre': 'orchestral',
            'tempo': 'variable',
            'bpm_range': (100, 130)
        },
        'disgust': {
            'prompt': 'dissonant experimental music, unconventional sounds, industrial, dark electronic',
            'genre': 'experimental',
            'tempo': 'moderate',
            'bpm_range': (90, 110)
        },
        'neutral': {
            'prompt': 'calm ambient background music, peaceful and relaxing, soft synthesizers, easy listening',
            'genre': 'ambient',
            'tempo': 'moderate',
            'bpm_range': (90, 110)
        },
        'calm': {
            'prompt': 'peaceful relaxing music, nature sounds, soft piano, gentle melody, meditation music',
            'genre': 'ambient',
            'tempo': 'slow',
            'bpm_range': (60, 80)
        }
    }
    
    # Extended prompts for more variety
    EMOTION_VARIATIONS = {
        'happy': [
            'joyful ukulele and whistling, sunny day music, feel-good indie pop',
            'uplifting electronic dance music, euphoric synths, celebration vibes',
            'cheerful jazz with saxophone, swing rhythm, happy brass section'
        ],
        'sad': [
            'heartbroken ballad, solo piano, raindrops, emotional and tender',
            'melancholic cello solo, orchestral strings, tear-jerking melody',
            'soft acoustic guitar, gentle vocals, bittersweet love song'
        ],
        'angry': [
            'thrash metal, aggressive drums, angry vocals, distorted bass',
            'intense electronic dubstep, heavy bass drops, chaotic energy',
            'powerful orchestral battle music, dramatic percussion, fierce'
        ],
        'calm': [
            'spa music, flowing water sounds, peaceful flute, zen garden',
            'lo-fi hip hop beats, relaxing study music, mellow vibes',
            'classical guitar, spanish romantic, sunset on the beach'
        ]
    }
    
    def __init__(self, model_name: str = "facebook/musicgen-small", use_gpu: bool = False):
        """
        Initialize Music Generator
        
        Args:
            model_name: MusicGen model to use
            use_gpu: Whether to use GPU for generation
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._model = None
        self._processor = None
        self._initialized = False
        self.sample_rate = 32000
        self._load_error = None  # Track loading errors
        
    def _lazy_init(self):
        """Lazy initialization of the model with robust error handling"""
        if self._initialized:
            return
        
        import os
        
        # Check for DISABLE_MUSICGEN environment variable
        if os.environ.get('DISABLE_MUSICGEN', '').lower() in ['1', 'true', 'yes']:
            logger.info("[MusicGen] Disabled via environment variable, using procedural fallback")
            self._model = None
            self._initialized = True
            return
            
        try:
            import torch
            
            # Check GPU availability first
            cuda_available = torch.cuda.is_available()
            logger.info(f"[MusicGen] CUDA available: {cuda_available}")
            
            # Check available memory
            if self.use_gpu and cuda_available:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"[MusicGen] GPU memory: {gpu_memory:.1f} GB")
                if gpu_memory < 4:
                    logger.warning("[MusicGen] Low GPU memory, falling back to CPU")
                    self.use_gpu = False
            
            # Try to import transformers
            try:
                from transformers import AutoProcessor, MusicgenForConditionalGeneration
            except ImportError as e:
                logger.warning(f"[MusicGen] Transformers not available: {e}")
                logger.info("[MusicGen] Will use procedural music generation as fallback")
                self._model = None
                self._initialized = True
                return
            
            logger.info(f"[MusicGen] Loading model: {self.model_name}")
            logger.info("[MusicGen] This may take 1-2 minutes on first run (downloading ~1GB model)...")
            
            # Set environment variables for offline mode if model already cached
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
            model_cache = os.path.join(cache_dir, "hub", f"models--facebook--musicgen-small")
            
            if os.path.exists(model_cache):
                logger.info("[MusicGen] Using cached model (no internet needed)")
            else:
                logger.info("[MusicGen] Model not cached, will download from HuggingFace...")
            
            # Load with explicit timeout and error handling
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    local_files_only=os.path.exists(model_cache)  # Use cache if available
                )
                self._model = MusicgenForConditionalGeneration.from_pretrained(
                    self.model_name,
                    local_files_only=os.path.exists(model_cache),
                    torch_dtype=torch.float32  # Use float32 for CPU stability
                )
            except Exception as download_error:
                if "resolve" in str(download_error).lower() or "connection" in str(download_error).lower():
                    logger.error(f"[MusicGen] Network error downloading model: {download_error}")
                    logger.info("[MusicGen] Falling back to procedural generation")
                    self._model = None
                    self._load_error = "Network error - using procedural fallback"
                    self._initialized = True
                    return
                raise
            
            # Move to appropriate device
            if self.use_gpu and cuda_available:
                try:
                    self._model = self._model.to("cuda")
                    logger.info("[MusicGen] Model loaded on GPU")
                except RuntimeError as gpu_error:
                    logger.warning(f"[MusicGen] Failed to load on GPU: {gpu_error}")
                    logger.info("[MusicGen] Using CPU instead")
                    self._model = self._model.to("cpu")
            else:
                logger.info("[MusicGen] Model loaded on CPU")
                
            self.sample_rate = self._model.config.audio_encoder.sampling_rate
            logger.info(f"[MusicGen] Model ready! Sample rate: {self.sample_rate}")
            
        except ImportError as e:
            logger.warning(f"[MusicGen] Dependencies not available: {e}")
            logger.info("[MusicGen] Will use procedural music generation as fallback")
            self._model = None
            self._load_error = str(e)
            
        except Exception as e:
            logger.error(f"[MusicGen] Failed to load: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._model = None
            self._load_error = str(e)
            
        self._initialized = True
    
    def get_music_config(self, emotion: str, custom_prompt: str = None, 
                         duration: int = 10, variation: int = 0) -> MusicConfig:
        """
        Get music configuration for an emotion
        
        Args:
            emotion: Detected emotion
            custom_prompt: Custom prompt override
            duration: Duration in seconds
            variation: Which variation to use (0 = default)
            
        Returns:
            MusicConfig for generation
        """
        emotion_lower = emotion.lower()
        config = self.EMOTION_MUSIC_MAP.get(emotion_lower, self.EMOTION_MUSIC_MAP['neutral'])
        
        # Get prompt
        if custom_prompt:
            prompt = custom_prompt
        elif variation > 0 and emotion_lower in self.EMOTION_VARIATIONS:
            variations = self.EMOTION_VARIATIONS[emotion_lower]
            prompt = variations[(variation - 1) % len(variations)]
        else:
            prompt = config['prompt']
        
        return MusicConfig(
            emotion=emotion_lower,
            prompt=prompt,
            duration=duration
        )
    
    def generate(self, emotion: str, duration: int = 10, 
                 custom_prompt: str = None, variation: int = 0) -> GeneratedMusic:
        """
        Generate music based on emotion
        
        Args:
            emotion: Emotion to generate music for
            duration: Duration in seconds
            custom_prompt: Custom prompt override
            variation: Which variation to use
            
        Returns:
            GeneratedMusic with file path and metadata
        """
        self._lazy_init()
        
        config = self.get_music_config(emotion, custom_prompt, duration, variation)
        emotion_info = self.EMOTION_MUSIC_MAP.get(config.emotion, self.EMOTION_MUSIC_MAP['neutral'])
        
        if self._model is not None:
            audio_data = self._generate_with_musicgen(config)
        else:
            audio_data = self._generate_procedural(config, emotion_info)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"music_{config.emotion}_{timestamp}.wav"
        file_path = os.path.join(OUTPUT_DIR, filename)
        
        self._save_audio(audio_data, file_path)
        
        return GeneratedMusic(
            file_path=file_path,
            emotion=config.emotion,
            prompt=config.prompt,
            duration=len(audio_data) / self.sample_rate,
            sample_rate=self.sample_rate,
            genre=emotion_info['genre'],
            tempo=emotion_info['tempo'],
            timestamp=datetime.now()
        )
    
    def _generate_with_musicgen(self, config: MusicConfig) -> np.ndarray:
        """Generate music using MusicGen model"""
        import torch
        
        logger.info(f"Generating music with MusicGen: {config.prompt}")
        
        # Prepare inputs
        inputs = self._processor(
            text=[config.prompt],
            padding=True,
            return_tensors="pt"
        )
        
        if self.use_gpu and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Calculate max new tokens for desired duration
        # MusicGen generates ~50 tokens per second of audio
        max_new_tokens = int(config.duration * 50)
        
        # Generate
        audio_values = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            guidance_scale=config.guidance_scale,
            temperature=config.temperature,
            do_sample=True
        )
        
        # Convert to numpy
        audio_data = audio_values[0, 0].cpu().numpy()
        
        return audio_data
    
    def _generate_procedural(self, config: MusicConfig, emotion_info: Dict) -> np.ndarray:
        """
        Generate procedural music as fallback
        Creates simple music using wave synthesis
        """
        logger.info(f"Generating procedural music for emotion: {config.emotion}")
        
        duration = config.duration
        sample_rate = 32000
        self.sample_rate = sample_rate
        
        # Get BPM from emotion
        bpm_range = emotion_info.get('bpm_range', (90, 120))
        bpm = (bpm_range[0] + bpm_range[1]) // 2
        
        # Generate based on emotion
        if config.emotion in ['happy', 'surprise']:
            audio = self._generate_happy_music(duration, sample_rate, bpm)
        elif config.emotion in ['sad', 'fear']:
            audio = self._generate_sad_music(duration, sample_rate, bpm)
        elif config.emotion == 'angry':
            audio = self._generate_angry_music(duration, sample_rate, bpm)
        elif config.emotion in ['calm', 'neutral']:
            audio = self._generate_calm_music(duration, sample_rate, bpm)
        else:
            audio = self._generate_ambient_music(duration, sample_rate)
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8
        
        return audio.astype(np.float32)
    
    def _generate_happy_music(self, duration: float, sr: int, bpm: int) -> np.ndarray:
        """Generate upbeat happy music"""
        t = np.linspace(0, duration, int(sr * duration))
        
        # Major scale frequencies (C major)
        major_scale = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        
        audio = np.zeros_like(t)
        beat_duration = 60 / bpm
        
        # Create melody
        for i, freq in enumerate(major_scale * int(duration)):
            start = i * beat_duration * 0.5
            if start >= duration:
                break
            
            mask = (t >= start) & (t < start + beat_duration * 0.4)
            envelope = np.exp(-3 * (t[mask] - start))
            audio[mask] += 0.3 * np.sin(2 * np.pi * freq * t[mask]) * envelope
        
        # Add bass
        bass_freqs = [130.81, 164.81, 196.00, 174.61]  # C, E, G, F
        for i, freq in enumerate(bass_freqs * int(duration)):
            start = i * beat_duration
            if start >= duration:
                break
            mask = (t >= start) & (t < start + beat_duration * 0.8)
            audio[mask] += 0.2 * np.sin(2 * np.pi * freq * t[mask])
        
        # Add rhythm
        beat_times = np.arange(0, duration, beat_duration)
        for bt in beat_times:
            mask = (t >= bt) & (t < bt + 0.05)
            noise = np.random.randn(np.sum(mask)) * 0.1
            audio[mask] += noise * np.exp(-50 * (t[mask] - bt))
        
        return audio
    
    def _generate_sad_music(self, duration: float, sr: int, bpm: int) -> np.ndarray:
        """Generate melancholic sad music"""
        t = np.linspace(0, duration, int(sr * duration))
        
        # Minor scale frequencies (A minor)
        minor_scale = [220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 392.00, 440.00]
        
        audio = np.zeros_like(t)
        beat_duration = 60 / bpm
        
        # Slow, sustained notes
        for i in range(int(duration / beat_duration)):
            freq = minor_scale[i % len(minor_scale)]
            start = i * beat_duration
            if start >= duration:
                break
            
            mask = (t >= start) & (t < start + beat_duration * 1.5)
            # Soft attack, slow decay
            time_in_note = t[mask] - start
            envelope = (1 - np.exp(-2 * time_in_note)) * np.exp(-0.5 * time_in_note)
            
            # Main tone with slight vibrato
            vibrato = 1 + 0.02 * np.sin(2 * np.pi * 5 * time_in_note)
            audio[mask] += 0.4 * np.sin(2 * np.pi * freq * vibrato * time_in_note) * envelope
        
        # Add reverb-like effect
        audio = np.convolve(audio, np.exp(-np.linspace(0, 3, int(sr * 0.3))), mode='same')
        
        return audio
    
    def _generate_angry_music(self, duration: float, sr: int, bpm: int) -> np.ndarray:
        """Generate intense angry music"""
        t = np.linspace(0, duration, int(sr * duration))
        
        audio = np.zeros_like(t)
        beat_duration = 60 / bpm
        
        # Power chords (root + fifth)
        power_chord_roots = [82.41, 98.00, 73.42, 110.00]  # E, G, D, A
        
        for i in range(int(duration / (beat_duration * 2))):
            root = power_chord_roots[i % len(power_chord_roots)]
            fifth = root * 1.5
            
            start = i * beat_duration * 2
            if start >= duration:
                break
            
            mask = (t >= start) & (t < start + beat_duration * 1.5)
            time_in_note = t[mask] - start
            envelope = np.exp(-1 * time_in_note)
            
            # Distorted power chord
            chord = np.sin(2 * np.pi * root * time_in_note)
            chord += 0.8 * np.sin(2 * np.pi * fifth * time_in_note)
            chord = np.tanh(chord * 3)  # Distortion
            
            audio[mask] += 0.4 * chord * envelope
        
        # Heavy drums
        beat_times = np.arange(0, duration, beat_duration)
        for i, bt in enumerate(beat_times):
            # Kick on beats 1, 3
            if i % 2 == 0:
                mask = (t >= bt) & (t < bt + 0.1)
                kick = np.sin(2 * np.pi * 60 * (t[mask] - bt)) * np.exp(-30 * (t[mask] - bt))
                audio[mask] += 0.3 * kick
            
            # Snare on beats 2, 4
            if i % 2 == 1:
                mask = (t >= bt) & (t < bt + 0.08)
                snare = np.random.randn(np.sum(mask)) * np.exp(-20 * (t[mask] - bt))
                audio[mask] += 0.25 * snare
        
        return audio
    
    def _generate_calm_music(self, duration: float, sr: int, bpm: int) -> np.ndarray:
        """Generate peaceful calm music"""
        t = np.linspace(0, duration, int(sr * duration))
        
        audio = np.zeros_like(t)
        
        # Ambient pads with slow modulation
        base_freqs = [220.00, 277.18, 329.63, 440.00]  # Am7 chord
        
        for i, freq in enumerate(base_freqs):
            # Slow sine wave with gentle amplitude modulation
            mod_freq = 0.1 + i * 0.05
            amplitude = 0.15 * (1 + 0.3 * np.sin(2 * np.pi * mod_freq * t))
            audio += amplitude * np.sin(2 * np.pi * freq * t)
        
        # Add subtle high-frequency shimmer
        shimmer_freq = 880 + 220 * np.sin(2 * np.pi * 0.2 * t)
        audio += 0.05 * np.sin(2 * np.pi * shimmer_freq * t)
        
        # Soft fade in/out
        fade_samples = int(sr * 2)
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
        
        return audio
    
    def _generate_ambient_music(self, duration: float, sr: int) -> np.ndarray:
        """Generate generic ambient music"""
        t = np.linspace(0, duration, int(sr * duration))
        
        # Layered sine waves
        audio = np.zeros_like(t)
        freqs = [110, 165, 220, 330]
        
        for freq in freqs:
            mod = 1 + 0.1 * np.sin(2 * np.pi * 0.1 * t)
            audio += 0.15 * np.sin(2 * np.pi * freq * mod * t)
        
        # Gentle noise
        audio += 0.02 * np.random.randn(len(t))
        
        # Smooth it
        from scipy.ndimage import uniform_filter1d
        try:
            audio = uniform_filter1d(audio, size=100)
        except:
            pass
        
        return audio
    
    def _save_audio(self, audio_data: np.ndarray, file_path: str):
        """Save audio data to WAV file"""
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        logger.info(f"Saved audio to: {file_path}")
    
    def get_emotion_info(self, emotion: str) -> Dict:
        """Get music information for an emotion"""
        emotion_lower = emotion.lower()
        info = self.EMOTION_MUSIC_MAP.get(emotion_lower, self.EMOTION_MUSIC_MAP['neutral'])
        return {
            'emotion': emotion_lower,
            'genre': info['genre'],
            'tempo': info['tempo'],
            'description': info['prompt']
        }


# Singleton instance
_generator_instance = None

def get_generator() -> MusicGenerator:
    """Get singleton generator instance"""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = MusicGenerator()
    return _generator_instance


def generate_music(emotion: str, duration: int = 10, 
                   custom_prompt: str = None) -> Dict:
    """
    Convenience function to generate music
    
    Args:
        emotion: Emotion to generate music for
        duration: Duration in seconds
        custom_prompt: Custom prompt override
        
    Returns:
        Dictionary with generation results
    """
    generator = get_generator()
    result = generator.generate(emotion, duration, custom_prompt)
    
    return {
        'file_path': result.file_path,
        'emotion': result.emotion,
        'prompt': result.prompt,
        'duration': result.duration,
        'sample_rate': result.sample_rate,
        'genre': result.genre,
        'tempo': result.tempo
    }


if __name__ == "__main__":
    print("Music Generator Module")
    print("=" * 40)
    
    # Test generation for different emotions
    emotions = ['happy', 'sad', 'calm', 'angry']
    
    generator = MusicGenerator()
    
    for emotion in emotions:
        print(f"\nGenerating music for: {emotion}")
        result = generator.generate(emotion, duration=5)
        print(f"  File: {result.file_path}")
        print(f"  Genre: {result.genre}")
        print(f"  Tempo: {result.tempo}")
        print(f"  Duration: {result.duration:.1f}s")
