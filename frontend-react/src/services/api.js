import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

// Create axios instance with generous timeout for music generation
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5 minutes for music generation (model loading can take time)
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('[API] Request error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`[API] Response: ${response.status}`);
    return response;
  },
  (error) => {
    if (error.code === 'ECONNABORTED') {
      console.error('[API] Request timeout - music generation may still be in progress');
      error.message = 'Request timeout. Music generation is taking longer than expected. The server may still be processing - please wait and try again.';
    } else if (error.code === 'ERR_NETWORK') {
      console.error('[API] Network error - check if backend server is running');
      error.message = 'Network error. Please ensure the backend server is running on http://localhost:5000';
    } else if (error.response) {
      console.error(`[API] Error ${error.response.status}:`, error.response.data);
    }
    return Promise.reject(error);
  }
);

// Health check with retry logic
export const healthCheck = async (retries = 3, delay = 1000) => {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await api.get('/health', { timeout: 5000 });
      return response.data;
    } catch (error) {
      console.warn(`Health check attempt ${i + 1}/${retries} failed:`, error.message);
      if (i < retries - 1) {
        await new Promise(resolve => setTimeout(resolve, delay));
      } else {
        console.error('Health check failed after all retries:', error);
        throw error;
      }
    }
  }
};

// Warmup the music generator model (call this on app load)
export const warmupModels = async () => {
  try {
    console.log('[API] Warming up music generator model...');
    const response = await api.post('/warmup', {}, { timeout: 180000 }); // 3 min timeout for warmup
    console.log('[API] Warmup complete:', response.data);
    return response.data;
  } catch (error) {
    console.warn('[API] Warmup failed (this is okay, will use fallback):', error.message);
    return { success: false, error: error.message };
  }
};

// Analyze face emotion
export const analyzeFace = async (imageBase64) => {
  try {
    const response = await api.post('/analyze/face', {
      image: imageBase64,
    });
    return response.data;
  } catch (error) {
    console.error('Face analysis failed:', error);
    throw error;
  }
};

// Analyze voice emotion
export const analyzeVoice = async (audioBase64, sampleRate = 16000) => {
  try {
    const response = await api.post('/analyze/voice', {
      audio: audioBase64,
      sampleRate,
    });
    return response.data;
  } catch (error) {
    console.error('Voice analysis failed:', error);
    throw error;
  }
};

// Analyze text emotion
export const analyzeText = async (text) => {
  try {
    const response = await api.post('/analyze/text', {
      text,
    });
    return response.data;
  } catch (error) {
    console.error('Text analysis failed:', error);
    throw error;
  }
};

// Fuse emotions
export const fuseEmotions = async (faceResult, voiceResult, textResult, method = 'confidence_based') => {
  try {
    const response = await api.post('/fuse', {
      face: faceResult,
      voice: voiceResult,
      text: textResult,
      method,
    });
    return response.data;
  } catch (error) {
    console.error('Fusion failed:', error);
    throw error;
  }
};

// Generate music
export const generateMusic = async (emotion, duration = 10, customPrompt = null) => {
  try {
    const response = await api.post('/generate/music', {
      emotion,
      duration,
      prompt: customPrompt,
    });
    return response.data;
  } catch (error) {
    console.error('Music generation failed:', error);
    throw error;
  }
};

// Analyze all modalities at once
export const analyzeAll = async ({ image, audio, text, duration = 10, fusionMethod = 'confidence_based' }) => {
  try {
    const response = await api.post('/analyze/all', {
      image,
      audio,
      text,
      duration,
      fusionMethod,
    });
    return response.data;
  } catch (error) {
    console.error('Full analysis failed:', error);
    throw error;
  }
};

// Analyze all modalities and get YouTube song (FULL SONGS)
export const analyzeAllYouTube = async ({ image, audio, text, fusionMethod = 'confidence_based' }) => {
  try {
    const response = await api.post('/analyze/all/youtube', {
      image,
      audio,
      text,
      fusionMethod,
    });
    return response.data;
  } catch (error) {
    console.error('Full analysis (YouTube) failed:', error);
    throw error;
  }
};

// Get YouTube song for emotion
export const getYouTubeSong = async (emotion) => {
  try {
    const response = await api.post('/youtube/song', { emotion });
    return response.data;
  } catch (error) {
    console.error('YouTube song fetch failed:', error);
    throw error;
  }
};

// Get YouTube playlist for emotion
export const getYouTubePlaylist = async (emotion, count = 5) => {
  try {
    const response = await api.post('/youtube/playlist', { emotion, count });
    return response.data;
  } catch (error) {
    console.error('YouTube playlist fetch failed:', error);
    throw error;
  }
};

// Get music file URL
export const getMusicUrl = (filename) => {
  return `${API_BASE_URL}/music/${filename}`;
};

// Get supported emotions
export const getEmotions = async () => {
  try {
    const response = await api.get('/emotions');
    return response.data;
  } catch (error) {
    console.error('Failed to get emotions:', error);
    throw error;
  }
};

export default api;
