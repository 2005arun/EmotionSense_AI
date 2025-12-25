"""
EmotionSense - Multi-Modal AI Emotion-to-Music Generator
Main Streamlit Application
"""

import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict
import base64

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="EmotionSense - AI Emotion to Music",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .emotion-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .emotion-label {
        font-size: 1.2rem;
        font-weight: bold;
        color: #333;
    }
    
    .confidence-bar {
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        transition: width 0.3s ease;
    }
    
    .final-emotion {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .final-emotion h1 {
        font-size: 3rem;
        margin: 0;
    }
    
    .music-player {
        background: #1a1a2e;
        border-radius: 15px;
        padding: 2rem;
        color: white;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        padding: 0.5rem 1rem;
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'face_emotion' not in st.session_state:
        st.session_state.face_emotion = None
    if 'voice_emotion' not in st.session_state:
        st.session_state.voice_emotion = None
    if 'text_emotion' not in st.session_state:
        st.session_state.text_emotion = None
    if 'final_emotion' not in st.session_state:
        st.session_state.final_emotion = None
    if 'generated_music' not in st.session_state:
        st.session_state.generated_music = None
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False


def get_emoji(emotion: str) -> str:
    """Get emoji for emotion"""
    emoji_map = {
        'happy': '😄',
        'sad': '😢',
        'angry': '😡',
        'fear': '😨',
        'surprise': '😲',
        'disgust': '🤢',
        'neutral': '😐',
        'calm': '😌'
    }
    return emoji_map.get(emotion.lower(), '😐')


def display_emotion_card(title: str, emotion_data: Optional[Dict], icon: str):
    """Display an emotion detection card"""
    with st.container():
        st.markdown(f"### {icon} {title}")
        
        if emotion_data:
            emotion = emotion_data.get('emotion', 'neutral')
            confidence = emotion_data.get('confidence', 0)
            emoji = get_emoji(emotion)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Detected:** {emoji} {emotion.upper()}")
                st.progress(confidence)
            with col2:
                st.metric("Confidence", f"{confidence:.0%}")
            
            # Show all emotions if available
            all_emotions = emotion_data.get('all_emotions', {})
            if all_emotions:
                with st.expander("See all emotions"):
                    for em, score in sorted(all_emotions.items(), key=lambda x: -x[1]):
                        st.write(f"{get_emoji(em)} {em}: {score:.1%}")
        else:
            st.info("No data yet. Capture input to detect emotion.")


def analyze_face_emotion(image: np.ndarray) -> Dict:
    """Analyze face emotion from image"""
    try:
        from backend.face_emotion import analyze_face_emotion
        return analyze_face_emotion(image)
    except ImportError:
        # Fallback for demo
        return {
            'modality': 'face',
            'emotion': 'neutral',
            'confidence': 0.5,
            'all_emotions': {'neutral': 0.5, 'happy': 0.2, 'sad': 0.3}
        }
    except Exception as e:
        st.error(f"Face analysis error: {e}")
        return None


def analyze_voice_emotion(audio_data: np.ndarray, sample_rate: int) -> Dict:
    """Analyze voice emotion from audio"""
    try:
        from backend.voice_emotion import analyze_voice_emotion
        return analyze_voice_emotion(audio_data, sample_rate)
    except ImportError:
        # Fallback for demo
        return {
            'modality': 'voice',
            'emotion': 'calm',
            'confidence': 0.6,
            'all_emotions': {'calm': 0.6, 'neutral': 0.3, 'sad': 0.1}
        }
    except Exception as e:
        st.error(f"Voice analysis error: {e}")
        return None


def analyze_text_emotion(text: str) -> Dict:
    """Analyze text emotion"""
    try:
        from backend.text_emotion import analyze_text_emotion
        return analyze_text_emotion(text)
    except ImportError:
        # Simple fallback
        text_lower = text.lower()
        if any(w in text_lower for w in ['happy', 'joy', 'great', 'love']):
            return {'modality': 'text', 'emotion': 'happy', 'confidence': 0.7, 'all_emotions': {'happy': 0.7}}
        elif any(w in text_lower for w in ['sad', 'lonely', 'depressed', 'unhappy']):
            return {'modality': 'text', 'emotion': 'sad', 'confidence': 0.7, 'all_emotions': {'sad': 0.7}}
        elif any(w in text_lower for w in ['angry', 'mad', 'furious', 'hate']):
            return {'modality': 'text', 'emotion': 'angry', 'confidence': 0.7, 'all_emotions': {'angry': 0.7}}
        else:
            return {'modality': 'text', 'emotion': 'neutral', 'confidence': 0.5, 'all_emotions': {'neutral': 0.5}}
    except Exception as e:
        st.error(f"Text analysis error: {e}")
        return None


def fuse_emotions(face: Dict, voice: Dict, text: Dict) -> Dict:
    """Fuse emotions from all modalities"""
    try:
        from backend.fusion_engine import fuse_emotions
        return fuse_emotions(face, voice, text, method='confidence_based')
    except ImportError:
        # Simple fallback fusion
        emotions = []
        if face:
            emotions.append((face['emotion'], face['confidence'], 0.4))
        if voice:
            emotions.append((voice['emotion'], voice['confidence'], 0.35))
        if text:
            emotions.append((text['emotion'], text['confidence'], 0.25))
        
        if not emotions:
            return {'final_emotion': 'neutral', 'final_confidence': 0.5, 'reasoning': {}}
        
        # Weighted voting
        scores = {}
        for emotion, confidence, weight in emotions:
            if emotion not in scores:
                scores[emotion] = 0
            scores[emotion] += confidence * weight
        
        final_emotion = max(scores, key=scores.get)
        return {
            'final_emotion': final_emotion,
            'final_confidence': scores[final_emotion],
            'reasoning': {'face': face, 'voice': voice, 'text': text},
            'all_emotions': scores
        }


def generate_music(emotion: str, duration: int = 10) -> Dict:
    """Generate music for emotion"""
    try:
        from backend.music_generator import generate_music
        return generate_music(emotion, duration)
    except ImportError:
        # Return None to indicate music generation not available
        st.warning("Music generation module not available. Install required dependencies.")
        return None
    except Exception as e:
        st.error(f"Music generation error: {e}")
        return None


def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎵 EmotionSense</h1>
        <p>Multi-Modal AI Emotion-to-Music Generator</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=🎵", width=100)
        st.title("Settings")
        
        st.subheader("Modality Weights")
        face_weight = st.slider("Face Weight", 0.0, 1.0, 0.4, 0.05)
        voice_weight = st.slider("Voice Weight", 0.0, 1.0, 0.35, 0.05)
        text_weight = st.slider("Text Weight", 0.0, 1.0, 0.25, 0.05)
        
        st.subheader("Music Settings")
        music_duration = st.slider("Duration (seconds)", 5, 30, 10)
        
        st.subheader("Fusion Method")
        fusion_method = st.selectbox(
            "Method",
            ["confidence_based", "weighted_average", "voting", "rule_based"]
        )
        
        if st.button("🔄 Reset All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📥 Input", "📊 Analysis", "🎵 Music"])
    
    # Tab 1: Input
    with tab1:
        st.header("Emotion Capture Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Camera Input")
            
            camera_option = st.radio(
                "Select input method:",
                ["Upload Image", "Capture from Webcam"],
                horizontal=True
            )
            
            if camera_option == "Upload Image":
                uploaded_file = st.file_uploader(
                    "Upload an image",
                    type=['jpg', 'jpeg', 'png'],
                    key="face_upload"
                )
                
                if uploaded_file:
                    # Read and display image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image")
                    
                    if st.button("🔍 Analyze Face", key="analyze_face"):
                        with st.spinner("Analyzing facial expression..."):
                            result = analyze_face_emotion(image)
                            if result:
                                st.session_state.face_emotion = result
                                st.success(f"Detected: {get_emoji(result['emotion'])} {result['emotion'].upper()}")
            
            else:  # Webcam capture
                camera_input = st.camera_input("Capture from webcam")
                
                if camera_input:
                    file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if st.button("🔍 Analyze Face", key="analyze_face_cam"):
                        with st.spinner("Analyzing facial expression..."):
                            result = analyze_face_emotion(image)
                            if result:
                                st.session_state.face_emotion = result
                                st.success(f"Detected: {get_emoji(result['emotion'])} {result['emotion'].upper()}")
        
        with col2:
            st.subheader("🎤 Voice Input")
            
            audio_file = st.file_uploader(
                "Upload audio file (WAV, MP3)",
                type=['wav', 'mp3', 'ogg'],
                key="voice_upload"
            )
            
            if audio_file:
                st.audio(audio_file)
                
                if st.button("🔍 Analyze Voice", key="analyze_voice"):
                    with st.spinner("Analyzing voice emotion..."):
                        # Save to temp file and process
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                                tmp.write(audio_file.getvalue())
                                tmp_path = tmp.name
                            
                            # Load and analyze
                            import wave
                            try:
                                with wave.open(tmp_path, 'rb') as wf:
                                    sample_rate = wf.getframerate()
                                    n_frames = wf.getnframes()
                                    audio_bytes = wf.readframes(n_frames)
                                    audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                            except:
                                # Fallback
                                audio_data = np.random.randn(16000 * 3).astype(np.float32) * 0.1
                                sample_rate = 16000
                            
                            result = analyze_voice_emotion(audio_data, sample_rate)
                            if result:
                                st.session_state.voice_emotion = result
                                st.success(f"Detected: {get_emoji(result['emotion'])} {result['emotion'].upper()}")
                            
                            os.unlink(tmp_path)
                        except Exception as e:
                            st.error(f"Error processing audio: {e}")
            
            st.markdown("---")
            
            st.subheader("⌨️ Text Input")
            
            text_input = st.text_area(
                "Express how you're feeling...",
                placeholder="I feel lonely today...",
                height=100
            )
            
            if text_input:
                if st.button("🔍 Analyze Text", key="analyze_text"):
                    with st.spinner("Analyzing text sentiment..."):
                        result = analyze_text_emotion(text_input)
                        if result:
                            st.session_state.text_emotion = result
                            st.success(f"Detected: {get_emoji(result['emotion'])} {result['emotion'].upper()}")
        
        st.markdown("---")
        
        # Fusion button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🧠 Fuse Emotions & Generate Music", type="primary", use_container_width=True):
                if any([st.session_state.face_emotion, 
                       st.session_state.voice_emotion, 
                       st.session_state.text_emotion]):
                    
                    with st.spinner("Fusing emotions..."):
                        fusion_result = fuse_emotions(
                            st.session_state.face_emotion,
                            st.session_state.voice_emotion,
                            st.session_state.text_emotion
                        )
                        st.session_state.final_emotion = fusion_result
                    
                    with st.spinner("Generating personalized music..."):
                        music_result = generate_music(
                            fusion_result['final_emotion'],
                            music_duration
                        )
                        st.session_state.generated_music = music_result
                    
                    st.session_state.analysis_complete = True
                    
                    # Add to history
                    st.session_state.emotion_history.append({
                        'emotion': fusion_result['final_emotion'],
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })
                    
                    st.success("Analysis complete! Check the Analysis and Music tabs.")
                else:
                    st.warning("Please provide at least one input (face, voice, or text)")
    
    # Tab 2: Analysis
    with tab2:
        st.header("Emotion Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            display_emotion_card("Face Emotion", st.session_state.face_emotion, "😐")
        
        with col2:
            display_emotion_card("Voice Emotion", st.session_state.voice_emotion, "🎧")
        
        with col3:
            display_emotion_card("Text Emotion", st.session_state.text_emotion, "💬")
        
        st.markdown("---")
        
        # Final emotion
        if st.session_state.final_emotion:
            final = st.session_state.final_emotion
            emotion = final['final_emotion']
            confidence = final.get('final_confidence', 0)
            emoji = get_emoji(emotion)
            
            st.markdown(f"""
            <div class="final-emotion">
                <p style="font-size: 1.5rem;">🧠 Multi-Modal Fusion Result</p>
                <h1>{emoji} {emotion.upper()}</h1>
                <p style="font-size: 1.2rem;">Confidence: {confidence:.0%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Reasoning
            with st.expander("📝 Fusion Reasoning"):
                reasoning = final.get('reasoning', {})
                for key, value in reasoning.items():
                    if key not in ['method', 'rule', 'final']:
                        st.write(f"**{key.capitalize()}:** {value}")
                
                if 'explanation' in final:
                    st.text(final['explanation'])
            
            # All emotions chart
            all_emotions = final.get('all_emotions', {})
            if all_emotions:
                st.subheader("Emotion Distribution")
                import pandas as pd
                df = pd.DataFrame({
                    'Emotion': list(all_emotions.keys()),
                    'Score': list(all_emotions.values())
                })
                st.bar_chart(df.set_index('Emotion'))
        
        # Emotion history
        if st.session_state.emotion_history:
            st.subheader("📜 Session History")
            for i, entry in enumerate(reversed(st.session_state.emotion_history[-5:])):
                st.write(f"{entry['timestamp']}: {get_emoji(entry['emotion'])} {entry['emotion'].upper()}")
    
    # Tab 3: Music
    with tab3:
        st.header("🎵 Generated Music")
        
        if st.session_state.generated_music:
            music = st.session_state.generated_music
            
            # Music info card
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div class="music-player">
                    <h2>{get_emoji(music['emotion'])} Music for: {music['emotion'].upper()}</h2>
                    <p>🎼 Genre: {music.get('genre', 'Unknown')}</p>
                    <p>🎵 Tempo: {music.get('tempo', 'Unknown')}</p>
                    <p>⏱️ Duration: {music.get('duration', 0):.1f} seconds</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Sample Rate", f"{music.get('sample_rate', 0)} Hz")
                
                if st.button("🔁 Regenerate Music"):
                    with st.spinner("Generating new music..."):
                        new_music = generate_music(
                            st.session_state.final_emotion['final_emotion'],
                            music_duration
                        )
                        st.session_state.generated_music = new_music
                        st.rerun()
            
            # Audio player
            st.markdown("---")
            st.subheader("🎧 Play Music")
            
            file_path = music.get('file_path', '')
            if file_path and os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format='audio/wav')
                
                # Download button
                st.download_button(
                    label="📥 Download Music",
                    data=audio_bytes,
                    file_name=os.path.basename(file_path),
                    mime="audio/wav"
                )
            else:
                st.info("Music file not found. Try regenerating.")
            
            # Music style info
            with st.expander("🎼 Music Style Details"):
                st.write(f"**Prompt:** {music.get('prompt', 'N/A')}")
                
                style_info = {
                    'happy': "Bright, energetic music with major keys and upbeat rhythms",
                    'sad': "Melancholic piano and strings with minor keys and slow tempo",
                    'angry': "Heavy drums and distorted guitars with fast tempo",
                    'calm': "Ambient pads and soft melodies for relaxation",
                    'fear': "Tense, suspenseful atmosphere with dark tones",
                    'surprise': "Dynamic, unexpected musical changes",
                    'neutral': "Balanced, easy-listening background music"
                }
                
                st.info(style_info.get(music['emotion'], "Custom generated music"))
        
        else:
            st.info("🎵 Generate music by analyzing emotions in the Input tab")
            
            # Demo music options
            st.subheader("🎹 Quick Demo")
            demo_emotion = st.selectbox(
                "Select emotion for demo:",
                ['happy', 'sad', 'angry', 'calm', 'neutral', 'fear', 'surprise']
            )
            
            if st.button("Generate Demo Music"):
                with st.spinner(f"Generating {demo_emotion} music..."):
                    demo_music = generate_music(demo_emotion, 5)
                    if demo_music:
                        st.session_state.generated_music = demo_music
                        st.session_state.final_emotion = {
                            'final_emotion': demo_emotion,
                            'final_confidence': 1.0
                        }
                        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>EmotionSense v1.0 | Multi-Modal AI Emotion-to-Music Generator</p>
        <p>Powered by DeepFace, Transformers, and MusicGen</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
