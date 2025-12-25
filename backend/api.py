"""
EmotionSense Flask API
RESTful API for emotion detection and music generation
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import base64
import os
import sys
import tempfile
import wave
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Output directory for generated music
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def decode_base64_image(base64_string):
    """Decode base64 image to numpy array"""
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    img_bytes = base64.b64decode(base64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image


def decode_base64_audio(base64_string, sample_rate=16000):
    """Decode base64 audio to numpy array"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    audio_bytes = base64.b64decode(base64_string)
    
    # Save to temp file and read
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        with wave.open(tmp_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_bytes = wf.readframes(n_frames)
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    except:
        # Fallback: treat as raw PCM
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    finally:
        os.unlink(tmp_path)
    
    return audio_data, sample_rate


@app.route('/', methods=['GET'])
def root():
    """Root endpoint - shows API is running"""
    return jsonify({
        'message': 'EmotionSense API is running!',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'analyze_face': '/api/analyze/face',
            'analyze_voice': '/api/analyze/voice', 
            'analyze_text': '/api/analyze/text',
            'analyze_all': '/api/analyze/all',
            'analyze_all_youtube': '/api/analyze/all/youtube',
            'generate_music': '/api/generate/music',
            'youtube_song': '/api/youtube/song',
            'youtube_playlist': '/api/youtube/playlist'
        }
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Check if music generator is initialized
    music_ready = hasattr(app, '_music_generator') and app._music_generator._initialized
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'music_generator_ready': music_ready,
        'output_dir': OUTPUT_DIR,
        'output_dir_exists': os.path.exists(OUTPUT_DIR)
    })


@app.route('/api/warmup', methods=['POST'])
def warmup_models():
    """Pre-warm the music generator model to avoid first-request timeout"""
    try:
        logger.info("[WARMUP] Starting model warm-up...")
        
        from backend.music_generator import MusicGenerator
        
        if not hasattr(app, '_music_generator'):
            logger.info("[WARMUP] Initializing MusicGenerator...")
            app._music_generator = MusicGenerator(use_gpu=False)
            
        # Force initialization
        app._music_generator._lazy_init()
        
        logger.info("[WARMUP] Model warm-up complete")
        
        return jsonify({
            'success': True,
            'message': 'Models warmed up successfully',
            'model_loaded': app._music_generator._model is not None,
            'using_procedural_fallback': app._music_generator._model is None
        })
        
    except Exception as e:
        logger.error(f"[WARMUP] Error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/analyze/face', methods=['POST'])
def analyze_face():
    """Analyze face emotion from image"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode image
        image = decode_base64_image(data['image'])
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Analyze emotion
        try:
            from backend.face_emotion import analyze_face_emotion
            result = analyze_face_emotion(image)
        except ImportError:
            # Fallback
            result = {
                'modality': 'face',
                'emotion': 'neutral',
                'confidence': 0.5,
                'all_emotions': {'neutral': 0.5, 'happy': 0.2, 'sad': 0.3}
            }
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Face analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/voice', methods=['POST'])
def analyze_voice():
    """Analyze voice emotion from audio"""
    try:
        data = request.get_json()
        
        if 'audio' not in data:
            return jsonify({'error': 'No audio provided'}), 400
        
        # Decode audio
        sample_rate = data.get('sampleRate', 16000)
        audio_data, actual_sr = decode_base64_audio(data['audio'], sample_rate)
        
        # Analyze emotion
        try:
            from backend.voice_emotion import analyze_voice_emotion
            result = analyze_voice_emotion(audio_data, actual_sr)
        except ImportError:
            # Fallback
            result = {
                'modality': 'voice',
                'emotion': 'calm',
                'confidence': 0.6,
                'all_emotions': {'calm': 0.6, 'neutral': 0.3, 'sad': 0.1}
            }
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Voice analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """Analyze text emotion"""
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Analyze emotion
        try:
            from backend.text_emotion import analyze_text_emotion
            result = analyze_text_emotion(text)
        except ImportError:
            # Simple fallback
            text_lower = text.lower()
            if any(w in text_lower for w in ['happy', 'joy', 'great', 'love', 'excited']):
                emotion, conf = 'happy', 0.7
            elif any(w in text_lower for w in ['sad', 'lonely', 'depressed', 'unhappy']):
                emotion, conf = 'sad', 0.7
            elif any(w in text_lower for w in ['angry', 'mad', 'furious', 'hate']):
                emotion, conf = 'angry', 0.7
            elif any(w in text_lower for w in ['scared', 'afraid', 'fear', 'worried']):
                emotion, conf = 'fear', 0.7
            else:
                emotion, conf = 'neutral', 0.5
            
            result = {
                'modality': 'text',
                'emotion': emotion,
                'confidence': conf,
                'all_emotions': {emotion: conf}
            }
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fuse', methods=['POST'])
def fuse_emotions():
    """Fuse emotions from multiple modalities"""
    try:
        data = request.get_json()
        
        face_result = data.get('face')
        voice_result = data.get('voice')
        text_result = data.get('text')
        method = data.get('method', 'confidence_based')
        
        if not any([face_result, voice_result, text_result]):
            return jsonify({'error': 'At least one emotion result required'}), 400
        
        # Fuse emotions
        try:
            from backend.fusion_engine import fuse_emotions
            result = fuse_emotions(face_result, voice_result, text_result, method=method)
        except ImportError:
            # Simple fallback fusion
            emotions = []
            weights = {'face': 0.4, 'voice': 0.35, 'text': 0.25}
            
            for modality, res in [('face', face_result), ('voice', voice_result), ('text', text_result)]:
                if res:
                    emotions.append((res['emotion'], res['confidence'], weights[modality]))
            
            scores = {}
            for emotion, confidence, weight in emotions:
                if emotion not in scores:
                    scores[emotion] = 0
                scores[emotion] += confidence * weight
            
            final_emotion = max(scores, key=scores.get)
            
            result = {
                'final_emotion': final_emotion,
                'final_confidence': scores[final_emotion],
                'reasoning': {
                    'face': face_result,
                    'voice': voice_result,
                    'text': text_result
                },
                'all_emotions': scores,
                'fusion_method': method
            }
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Fusion error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/music', methods=['POST'])
def generate_music():
    """Generate music based on emotion"""
    try:
        data = request.get_json()
        
        emotion = data.get('emotion', 'neutral')
        duration = data.get('duration', 10)
        custom_prompt = data.get('prompt')
        
        # Validate duration - limit to avoid memory issues
        duration = max(5, min(20, duration))  # Max 20 seconds to prevent memory issues
        
        logger.info(f"[MUSIC] Starting generation: emotion={emotion}, duration={duration}s")
        
        # Generate music with detailed logging
        try:
            logger.info("[MUSIC] Loading music generator module...")
            from backend.music_generator import MusicGenerator
            
            # Use singleton pattern to avoid reloading model
            if not hasattr(app, '_music_generator'):
                logger.info("[MUSIC] Initializing MusicGenerator (first time)...")
                app._music_generator = MusicGenerator(use_gpu=False)  # Force CPU for stability
            
            generator = app._music_generator
            logger.info("[MUSIC] Generator ready, starting music generation...")
            
            gen_result = generator.generate(emotion, duration, custom_prompt)
            
            logger.info(f"[MUSIC] Generation complete: {gen_result.file_path}")
            
            result = {
                'file_path': gen_result.file_path,
                'emotion': gen_result.emotion,
                'prompt': gen_result.prompt,
                'duration': gen_result.duration,
                'sample_rate': gen_result.sample_rate,
                'genre': gen_result.genre,
                'tempo': gen_result.tempo
            }
            
        except Exception as gen_error:
            logger.error(f"[MUSIC] Generation failed: {gen_error}")
            logger.info("[MUSIC] Attempting fallback procedural generation...")
            
            # Fallback to simple procedural generation
            from backend.music_generator import MusicGenerator
            generator = MusicGenerator(use_gpu=False)
            generator._model = None  # Force procedural mode
            generator._initialized = True
            gen_result = generator.generate(emotion, duration)
            
            result = {
                'file_path': gen_result.file_path,
                'emotion': gen_result.emotion,
                'prompt': gen_result.prompt,
                'duration': gen_result.duration,
                'sample_rate': gen_result.sample_rate,
                'genre': gen_result.genre,
                'tempo': gen_result.tempo
            }
        
        # Verify file was created
        if not os.path.exists(result['file_path']):
            raise Exception(f"Music file was not created: {result['file_path']}")
        
        # Convert file path to URL-friendly format
        filename = os.path.basename(result['file_path'])
        logger.info(f"[MUSIC] Returning music file: {filename}")
        
        return jsonify({
            'success': True,
            'result': {
                'filename': filename,
                'url': f'/api/music/{filename}',
                'emotion': result['emotion'],
                'duration': result['duration'],
                'genre': result.get('genre', 'unknown'),
                'tempo': result.get('tempo', 'unknown'),
                'prompt': result.get('prompt', '')
            }
        })
        
    except Exception as e:
        logger.error(f"[MUSIC] Critical error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'details': 'Music generation failed. This may be due to model loading issues or memory constraints.'
        }), 500


@app.route('/api/music/<filename>', methods=['GET'])
def serve_music(filename):
    """Serve generated music file with proper headers"""
    try:
        # Sanitize filename to prevent directory traversal
        safe_filename = os.path.basename(filename)
        file_path = os.path.join(OUTPUT_DIR, safe_filename)
        
        logger.info(f"[SERVE] Requesting file: {safe_filename}")
        
        if not os.path.exists(file_path):
            logger.error(f"[SERVE] File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        # Get file size for content-length header
        file_size = os.path.getsize(file_path)
        logger.info(f"[SERVE] Serving file: {safe_filename} ({file_size} bytes)")
        
        response = send_file(
            file_path,
            mimetype='audio/wav',
            as_attachment=False,
            download_name=safe_filename
        )
        
        # Add headers to prevent caching issues and enable range requests
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Expose-Headers'] = 'Content-Length, Accept-Ranges'
        
        return response
        
    except Exception as e:
        logger.error(f"[SERVE] Error serving music: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/all', methods=['POST'])
def analyze_all():
    """Analyze all modalities and generate music in one request"""
    try:
        data = request.get_json()
        
        results = {}
        
        # Analyze face if provided
        if 'image' in data and data['image']:
            try:
                image = decode_base64_image(data['image'])
                from backend.face_emotion import analyze_face_emotion
                results['face'] = analyze_face_emotion(image)
            except Exception as e:
                logger.warning(f"Face analysis failed: {e}")
                results['face'] = None
        
        # Analyze voice if provided
        if 'audio' in data and data['audio']:
            try:
                sample_rate = data.get('sampleRate', 16000)
                audio_data, actual_sr = decode_base64_audio(data['audio'], sample_rate)
                from backend.voice_emotion import analyze_voice_emotion
                results['voice'] = analyze_voice_emotion(audio_data, actual_sr)
            except Exception as e:
                logger.warning(f"Voice analysis failed: {e}")
                results['voice'] = None
        
        # Analyze text if provided
        if 'text' in data and data['text']:
            try:
                from backend.text_emotion import analyze_text_emotion
                results['text'] = analyze_text_emotion(data['text'])
            except Exception as e:
                logger.warning(f"Text analysis failed: {e}")
                results['text'] = None
        
        # Check if we have any results
        if not any([results.get('face'), results.get('voice'), results.get('text')]):
            return jsonify({'error': 'No valid input provided'}), 400
        
        # Fuse emotions
        try:
            from backend.fusion_engine import fuse_emotions
            fusion_result = fuse_emotions(
                results.get('face'),
                results.get('voice'),
                results.get('text'),
                method=data.get('fusionMethod', 'confidence_based')
            )
        except ImportError:
            # Fallback
            emotions = []
            for key in ['face', 'voice', 'text']:
                if results.get(key):
                    emotions.append(results[key])
            
            if emotions:
                best = max(emotions, key=lambda x: x['confidence'])
                fusion_result = {
                    'final_emotion': best['emotion'],
                    'final_confidence': best['confidence']
                }
            else:
                fusion_result = {'final_emotion': 'neutral', 'final_confidence': 0.5}
        
        # Generate music
        duration = data.get('duration', 10)
        try:
            from backend.music_generator import generate_music as gen_music
            music_result = gen_music(fusion_result['final_emotion'], duration)
        except:
            from backend.music_generator import MusicGenerator
            generator = MusicGenerator()
            gen_result = generator.generate(fusion_result['final_emotion'], duration)
            music_result = {
                'file_path': gen_result.file_path,
                'emotion': gen_result.emotion,
                'duration': gen_result.duration,
                'genre': gen_result.genre,
                'tempo': gen_result.tempo
            }
        
        filename = os.path.basename(music_result['file_path'])
        
        return jsonify({
            'success': True,
            'results': {
                'face': results.get('face'),
                'voice': results.get('voice'),
                'text': results.get('text'),
                'fusion': fusion_result,
                'music': {
                    'filename': filename,
                    'url': f'/api/music/{filename}',
                    'emotion': music_result['emotion'],
                    'duration': music_result['duration'],
                    'genre': music_result.get('genre', 'unknown'),
                    'tempo': music_result.get('tempo', 'unknown')
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Analyze all error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/emotions', methods=['GET'])
def get_emotions():
    """Get list of supported emotions"""
    emotions = [
        {'id': 'happy', 'label': 'Happy', 'emoji': '😄', 'color': '#FFD700'},
        {'id': 'sad', 'label': 'Sad', 'emoji': '😢', 'color': '#4169E1'},
        {'id': 'angry', 'label': 'Angry', 'emoji': '😡', 'color': '#DC143C'},
        {'id': 'fear', 'label': 'Fear', 'emoji': '😨', 'color': '#800080'},
        {'id': 'surprise', 'label': 'Surprise', 'emoji': '😲', 'color': '#FF8C00'},
        {'id': 'disgust', 'label': 'Disgust', 'emoji': '🤢', 'color': '#228B22'},
        {'id': 'neutral', 'label': 'Neutral', 'emoji': '😐', 'color': '#808080'},
        {'id': 'calm', 'label': 'Calm', 'emoji': '😌', 'color': '#87CEEB'}
    ]
    
    return jsonify({
        'success': True,
        'emotions': emotions
    })


# ==========================================
# YouTube Music Endpoints (Full Songs)
# ==========================================

@app.route('/api/youtube/song', methods=['POST'])
def get_youtube_song():
    """Get a YouTube song recommendation based on emotion"""
    try:
        data = request.get_json()
        emotion = data.get('emotion', 'neutral')
        
        logger.info(f"[YouTube] Getting song for emotion: {emotion}")
        
        from backend.youtube_music import get_song_for_emotion
        song = get_song_for_emotion(emotion)
        
        return jsonify({
            'success': True,
            'song': song
        })
        
    except Exception as e:
        logger.error(f"[YouTube] Error getting song: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/youtube/playlist', methods=['POST'])
def get_youtube_playlist():
    """Get a playlist of YouTube songs based on emotion"""
    try:
        data = request.get_json()
        emotion = data.get('emotion', 'neutral')
        count = data.get('count', 5)
        
        logger.info(f"[YouTube] Getting playlist for emotion: {emotion}, count: {count}")
        
        from backend.youtube_music import get_playlist_for_emotion
        playlist = get_playlist_for_emotion(emotion, count)
        
        return jsonify({
            'success': True,
            'playlist': playlist,
            'emotion': emotion
        })
        
    except Exception as e:
        logger.error(f"[YouTube] Error getting playlist: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/all/youtube', methods=['POST'])
def analyze_all_youtube():
    """Analyze all modalities and get YouTube song recommendation"""
    try:
        data = request.get_json()
        
        results = {}
        
        # Analyze face if provided
        if 'image' in data and data['image']:
            try:
                image = decode_base64_image(data['image'])
                from backend.face_emotion import analyze_face_emotion
                results['face'] = analyze_face_emotion(image)
            except Exception as e:
                logger.warning(f"Face analysis failed: {e}")
                results['face'] = None
        
        # Analyze voice if provided
        if 'audio' in data and data['audio']:
            try:
                sample_rate = data.get('sampleRate', 16000)
                audio_data, actual_sr = decode_base64_audio(data['audio'], sample_rate)
                from backend.voice_emotion import analyze_voice_emotion
                results['voice'] = analyze_voice_emotion(audio_data, actual_sr)
            except Exception as e:
                logger.warning(f"Voice analysis failed: {e}")
                results['voice'] = None
        
        # Analyze text if provided
        if 'text' in data and data['text']:
            try:
                from backend.text_emotion import analyze_text_emotion
                results['text'] = analyze_text_emotion(data['text'])
            except Exception as e:
                logger.warning(f"Text analysis failed: {e}")
                results['text'] = None
        
        # Check if we have any results
        if not any([results.get('face'), results.get('voice'), results.get('text')]):
            return jsonify({'error': 'No valid input provided'}), 400
        
        # Fuse emotions with improved handling of neutral
        try:
            from backend.fusion_engine import fuse_emotions
            fusion_result = fuse_emotions(
                results.get('face'),
                results.get('voice'),
                results.get('text'),
                method=data.get('fusionMethod', 'confidence_based')
            )
        except ImportError:
            # Fallback with neutral handling
            emotions = []
            for key in ['face', 'voice', 'text']:
                if results.get(key):
                    emotions.append(results[key])
            
            if emotions:
                # Filter out neutral if other emotions exist
                non_neutral = [e for e in emotions if e['emotion'] != 'neutral']
                if non_neutral:
                    best = max(non_neutral, key=lambda x: x['confidence'])
                else:
                    best = max(emotions, key=lambda x: x['confidence'])
                fusion_result = {
                    'final_emotion': best['emotion'],
                    'final_confidence': best['confidence']
                }
            else:
                fusion_result = {'final_emotion': 'neutral', 'final_confidence': 0.5}
        
        # Get YouTube song
        from backend.youtube_music import get_song_for_emotion
        song = get_song_for_emotion(fusion_result['final_emotion'])
        
        return jsonify({
            'success': True,
            'results': {
                'face': results.get('face'),
                'voice': results.get('voice'),
                'text': results.get('text'),
                'fusion': fusion_result,
                'song': song
            }
        })
        
    except Exception as e:
        logger.error(f"Analyze all (YouTube) error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("EmotionSense API Server")
    print("=" * 50)
    print(f"Output directory: {OUTPUT_DIR}")
    print("Starting server on http://localhost:5000")
    print("=" * 50)
    
    # Disable reloader to prevent constant restarts from transformers library changes
    # This fixes the "Cannot connect to server" error caused by server restarts
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
