# 🎵 EmotionSense - Multi-Modal AI Emotion-to-Music Generator

<div align="center">

![EmotionSense Banner](https://via.placeholder.com/800x200.png?text=EmotionSense+-+AI+Emotion+to+Music)

**Transform your emotions into personalized music using AI**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## 🎯 Overview

EmotionSense is an end-to-end AI system that:

1. **Collects multi-modal inputs** (Face, Voice, Text)
2. **Detects human emotions** from each modality
3. **Performs intelligent emotion fusion** using reasoning
4. **Generates personalized music** using generative AI
5. **Provides an interactive UI** for real-time interaction

### 🖥️ Two Frontend Options

| Frontend | Description | Best For |
|----------|-------------|----------|
| **React App** | Modern, interactive SPA with in-browser music playback | Full interactive experience |
| **Streamlit** | Quick-start dashboard interface | Rapid prototyping & demos |

## 🏗️ System Architecture

```
User Input
    ↓
[UI Layer - React / Streamlit]
    ↓
[Flask REST API]
    ↓
[Emotion Detection Models]
    ├── Face Emotion (DeepFace)
    ├── Voice Emotion (Transformers/Librosa)
    └── Text Emotion (BERT/RoBERTa)
    ↓
[Multi-Modal Emotion Fusion Engine]
    ↓
[Music Generation (MusicGen/Procedural)]
    ↓
[Music Playback in Browser]
```

## ✨ Features

### 🔹 Multi-Modal Emotion Detection

| Modality | Technology | Emotions Detected |
|----------|------------|-------------------|
| 📷 Face | DeepFace + OpenCV | Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral |
| 🎤 Voice | Transformers + Librosa | Happy, Sad, Angry, Calm, Fear, Neutral |
| ⌨️ Text | BERT/RoBERTa | Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral |

### 🔹 Intelligent Emotion Fusion

- **Weighted Average**: Combines emotions with configurable weights
- **Voting**: Majority voting with confidence tiebreaker
- **Confidence-Based**: Dynamic weighting based on detection confidence
- **Rule-Based**: Expert knowledge with emotion compatibility

### 🔹 AI Music Generation

| Emotion | Music Style |
|---------|-------------|
| 😄 Happy | Upbeat, energetic, major key |
| 😢 Sad | Slow piano, minor chords |
| 😡 Angry | Heavy drums, fast tempo |
| 😌 Calm | Ambient, soft pads |
| 😨 Fear | Tense, suspenseful |
| 😲 Surprise | Dynamic, unexpected changes |

## 📂 Project Structure

```
EmotionSense/
│
├── frontend/
│   └── app.py              # Streamlit UI
│
├── frontend-react/         # React Interactive UI
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── services/       # API services
│   │   ├── App.js          # Main app
│   │   └── index.js        # Entry point
│   └── package.json
│
├── backend/
│   ├── __init__.py
│   ├── face_emotion.py     # Face emotion detection
│   ├── voice_emotion.py    # Voice emotion detection
│   ├── text_emotion.py     # Text sentiment analysis
│   ├── fusion_engine.py    # Multi-modal fusion
│   ├── music_generator.py  # Music generation
│   └── api.py              # Flask REST API
│
├── utils/
│   ├── __init__.py
│   ├── helpers.py          # Utility functions
│   └── config.py           # Configuration
│
├── models/                 # Trained models (auto-downloaded)
├── data/                   # Data files
├── output/                 # Generated music
│
├── setup.bat               # Windows setup script
├── start.bat               # Windows start script
├── requirements.txt        # Full dependencies
├── requirements-minimal.txt # Minimal dependencies
└── README.md
```

## 🚀 Quick Start

### Option A: Interactive React Frontend (Recommended)

#### 1. Setup (One-time)

**Windows:**
```bash
# Run the setup script
setup.bat
```

**Manual:**
```bash
# Install Python dependencies
pip install -r requirements-minimal.txt

# Install React dependencies
cd frontend-react
npm install
cd ..
```

#### 2. Run the Application

**Windows:**
```bash
# Run the start script
start.bat
```

**Manual (2 terminals):**
```bash
# Terminal 1 - Start Flask API
python -m backend.api

# Terminal 2 - Start React frontend
cd frontend-react
npm start
```

#### 3. Open in Browser

Navigate to **http://localhost:3000**

### Option B: Streamlit Frontend (Simple)

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/EmotionSense.git
cd EmotionSense
```

#### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 3. Install Dependencies

**Full installation (recommended):**
```bash
pip install -r requirements.txt
```

**Minimal installation (for testing):**
```bash
pip install -r requirements-minimal.txt
```

### 4. Run the Application

```bash
streamlit run frontend/app.py
```

The app will open in your browser at `http://localhost:8501`

## 🖥️ Usage

### Step 1: Capture Emotions

1. **Face**: Upload an image or use webcam
2. **Voice**: Upload an audio file (WAV/MP3)
3. **Text**: Type how you're feeling

### Step 2: Analyze

Click **"Fuse Emotions & Generate Music"** to:
- Detect emotions from each modality
- Combine using multi-modal fusion
- Generate personalized music

### Step 3: Enjoy

- View emotion analysis results
- Listen to generated music
- Download or regenerate

## ⚙️ Configuration

### Modality Weights

Adjust in the sidebar:
- Face Weight: 0.0 - 1.0
- Voice Weight: 0.0 - 1.0
- Text Weight: 0.0 - 1.0

### Fusion Methods

- `confidence_based`: Dynamic weighting (recommended)
- `weighted_average`: Static weights
- `voting`: Majority voting
- `rule_based`: Expert rules

### Music Settings

- Duration: 5 - 30 seconds
- Customize per emotion

## 🔧 API Usage

### Face Emotion Detection

```python
from backend.face_emotion import analyze_face_emotion
import cv2

image = cv2.imread("face.jpg")
result = analyze_face_emotion(image)
print(result)
# {'modality': 'face', 'emotion': 'happy', 'confidence': 0.85, ...}
```

### Voice Emotion Detection

```python
from backend.voice_emotion import analyze_voice_emotion
import numpy as np

audio = np.random.randn(16000 * 3)  # 3 seconds
result = analyze_voice_emotion(audio, sample_rate=16000)
print(result)
# {'modality': 'voice', 'emotion': 'calm', 'confidence': 0.72, ...}
```

### Text Emotion Detection

```python
from backend.text_emotion import analyze_text_emotion

text = "I feel so happy today!"
result = analyze_text_emotion(text)
print(result)
# {'modality': 'text', 'emotion': 'happy', 'confidence': 0.89, ...}
```

### Emotion Fusion

```python
from backend.fusion_engine import fuse_emotions

result = fuse_emotions(
    face_result={'emotion': 'sad', 'confidence': 0.8},
    voice_result={'emotion': 'calm', 'confidence': 0.7},
    text_result={'emotion': 'sad', 'confidence': 0.6}
)
print(result['final_emotion'])  # 'sad'
```

### Music Generation

```python
from backend.music_generator import generate_music

result = generate_music(emotion='happy', duration=10)
print(result['file_path'])  # 'output/music_happy_20241225_120000.wav'
```

## 📊 Example Output

### Emotion Analysis

```json
{
  "face_emotion": {
    "emotion": "sad",
    "confidence": 0.82
  },
  "voice_emotion": {
    "emotion": "calm",
    "confidence": 0.74
  },
  "text_emotion": {
    "emotion": "sad",
    "confidence": 0.68
  },
  "final_emotion": "sad",
  "music": {
    "genre": "ambient piano",
    "tempo": "slow",
    "duration_sec": 10,
    "file": "generated_music_sad.wav"
  }
}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=backend tests/
```

## 🐳 Docker

```bash
# Build image
docker build -t emotionsense .

# Run container
docker run -p 8501:8501 emotionsense
```

## 📝 Environment Variables

Create a `.env` file:

```env
# Model settings
FACE_MODEL=deepface
VOICE_MODEL=transformer
TEXT_MODEL=auto
MUSIC_MODEL=facebook/musicgen-small

# GPU settings
USE_GPU=false

# Logging
LOG_LEVEL=INFO
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [DeepFace](https://github.com/serengil/deepface) - Face emotion detection
- [Hugging Face Transformers](https://huggingface.co/transformers/) - NLP models
- [Meta MusicGen](https://github.com/facebookresearch/audiocraft) - Music generation
- [Streamlit](https://streamlit.io/) - Web UI framework
- [Librosa](https://librosa.org/) - Audio processing

## 📧 Contact

- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)

---

<div align="center">

**Made with ❤️ and 🎵**

*EmotionSense v1.0*

</div>
