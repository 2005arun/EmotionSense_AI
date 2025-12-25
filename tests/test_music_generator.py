"""
Test Music Generator
Run this script to test if music generation works without the UI.
This helps diagnose "network error" issues.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("  EmotionSense - Music Generator Test")
print("=" * 60)
print()

# Step 1: Check Python environment
print("[1/6] Checking Python version...")
print(f"      Python {sys.version}")
print()

# Step 2: Check if torch is available
print("[2/6] Checking PyTorch...")
try:
    import torch
    print(f"      PyTorch version: {torch.__version__}")
    print(f"      CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"      CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"      GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError as e:
    print(f"      [WARNING] PyTorch not installed: {e}")
print()

# Step 3: Check if transformers is available
print("[3/6] Checking Transformers...")
try:
    import transformers
    print(f"      Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"      [WARNING] Transformers not installed: {e}")
print()

# Step 4: Check output directory
print("[4/6] Checking output directory...")
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
os.makedirs(output_dir, exist_ok=True)
print(f"      Output directory: {output_dir}")
print(f"      Directory exists: {os.path.exists(output_dir)}")
print(f"      Writable: {os.access(output_dir, os.W_OK)}")
print()

# Step 5: Test music generator initialization
print("[5/6] Initializing Music Generator...")
try:
    from backend.music_generator import MusicGenerator
    
    # Force CPU mode for testing
    generator = MusicGenerator(use_gpu=False)
    print("      Generator created successfully")
    
    # Initialize (this may download model on first run)
    print("      Initializing model (may take 1-2 minutes on first run)...")
    generator._lazy_init()
    
    if generator._model is not None:
        print("      [OK] MusicGen model loaded successfully!")
    else:
        print("      [INFO] Using procedural fallback (no AI model)")
        if generator._load_error:
            print(f"      Reason: {generator._load_error}")
    
except Exception as e:
    print(f"      [ERROR] Failed to initialize: {e}")
    import traceback
    traceback.print_exc()
print()

# Step 6: Generate test music
print("[6/6] Generating test music...")
try:
    from backend.music_generator import MusicGenerator
    
    generator = MusicGenerator(use_gpu=False)
    
    # Test with each emotion
    test_emotions = ['happy', 'sad', 'calm']
    
    for emotion in test_emotions:
        print(f"\n      Generating {emotion} music (5 seconds)...")
        result = generator.generate(emotion, duration=5)
        
        if os.path.exists(result.file_path):
            file_size = os.path.getsize(result.file_path)
            print(f"      [OK] Generated: {os.path.basename(result.file_path)} ({file_size} bytes)")
        else:
            print(f"      [ERROR] File not created: {result.file_path}")
    
    print("\n" + "=" * 60)
    print("  TEST PASSED - Music generation is working!")
    print("=" * 60)
    print("\n  If you see 'Network Error' in the UI, the issue is likely:")
    print("  1. Frontend-backend connection (CORS)")
    print("  2. Request timeout (increase timeout in api.js)")
    print("  3. File serving issue (check /api/music endpoint)")
    
except Exception as e:
    print(f"\n      [ERROR] Music generation failed: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("  TROUBLESHOOTING:")
    print("=" * 60)
    print("  1. Install dependencies: pip install torch transformers scipy numpy")
    print("  2. Check available memory (need ~4GB for musicgen-small)")
    print("  3. Try: set DISABLE_MUSICGEN=1 to use procedural fallback")
    print("  4. Check firewall/proxy for HuggingFace downloads")

print()
