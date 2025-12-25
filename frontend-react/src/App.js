import React, { useState, useCallback, useEffect } from 'react';
import './App.css';
import Header from './components/Header';
import InputSection from './components/InputSection';
import EmotionDashboard from './components/EmotionDashboard';
import MusicPlayer from './components/MusicPlayer';
import YouTubePlayer from './components/YouTubePlayer';
import LoadingOverlay from './components/LoadingOverlay';
import { 
  analyzeAll, 
  analyzeAllYouTube,
  generateMusic, 
  warmupModels, 
  healthCheck,
  getYouTubeSong,
  getYouTubePlaylist 
} from './services/api';

function App() {
  // State management
  const [faceData, setFaceData] = useState(null);
  const [voiceData, setVoiceData] = useState(null);
  const [textData, setTextData] = useState(null);
  const [fusionResult, setFusionResult] = useState(null);
  const [musicData, setMusicData] = useState(null);
  const [youtubeData, setYoutubeData] = useState(null);
  const [youtubePlaylist, setYoutubePlaylist] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('input');
  const [serverStatus, setServerStatus] = useState('checking');
  const [modelReady, setModelReady] = useState(false);
  
  // Music mode: 'youtube' (full songs) or 'ai' (short generated)
  const [musicMode, setMusicMode] = useState('youtube');

  // Input states
  const [capturedImage, setCapturedImage] = useState(null);
  const [recordedAudio, setRecordedAudio] = useState(null);
  const [textInput, setTextInput] = useState('');

  // Settings
  const [settings, setSettings] = useState({
    duration: 10,
    fusionMethod: 'confidence_based'
  });

  // Check server status and warmup on mount
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Check if server is running
        console.log('[App] Checking server status...');
        const health = await healthCheck();
        setServerStatus('online');
        console.log('[App] Server status:', health);
        
        // Only warmup if using AI mode
        if (musicMode === 'ai' && !health.music_generator_ready) {
          console.log('[App] Warming up music generator...');
          setLoadingMessage('Preparing music generator...');
          const warmupResult = await warmupModels();
          setModelReady(warmupResult.success);
          setLoadingMessage('');
          console.log('[App] Warmup result:', warmupResult);
        } else {
          setModelReady(true);
        }
      } catch (err) {
        console.error('[App] Server not available:', err);
        setServerStatus('offline');
        setError('Backend server is not running. Please start the server with: python -m backend.api');
      }
    };

    initializeApp();
  }, [musicMode]);

  // Handle full analysis
  const handleAnalyze = useCallback(async () => {
    if (!capturedImage && !recordedAudio && !textInput) {
      setError('Please provide at least one input (face, voice, or text)');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      if (musicMode === 'youtube') {
        // YouTube mode - get full song recommendation
        setLoadingMessage('Analyzing emotions & finding perfect song...');
        
        const response = await analyzeAllYouTube({
          image: capturedImage,
          audio: recordedAudio,
          text: textInput,
          fusionMethod: settings.fusionMethod
        });

        if (response.success) {
          const { results } = response;
          
          setFaceData(results.face);
          setVoiceData(results.voice);
          setTextData(results.text);
          setFusionResult(results.fusion);
          setYoutubeData(results.song);
          setYoutubePlaylist(null);
          setActiveTab('results');
        } else {
          throw new Error(response.error || 'Analysis failed');
        }
      } else {
        // AI mode - generate short music
        setLoadingMessage('Analyzing emotions & generating music...');
        
        const response = await analyzeAll({
          image: capturedImage,
          audio: recordedAudio,
          text: textInput,
          duration: settings.duration,
          fusionMethod: settings.fusionMethod
        });

        if (response.success) {
          const { results } = response;
          
          setFaceData(results.face);
          setVoiceData(results.voice);
          setTextData(results.text);
          setFusionResult(results.fusion);
          setMusicData(results.music);
          setActiveTab('results');
        } else {
          throw new Error(response.error || 'Analysis failed');
        }
      }
    } catch (err) {
      console.error('[App] Analysis error:', err);
      
      // Provide user-friendly error messages
      let errorMessage = 'Failed to analyze emotions';
      if (err.code === 'ERR_NETWORK') {
        errorMessage = 'Cannot connect to server. Please ensure the backend is running (python -m backend.api)';
      } else if (err.code === 'ECONNABORTED') {
        errorMessage = 'Request timed out. Please try again.';
      } else if (err.response?.data?.details) {
        errorMessage = err.response.data.details;
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [capturedImage, recordedAudio, textInput, settings, musicMode]);

  // Get new YouTube song
  const handleGetNewSong = useCallback(async () => {
    if (!fusionResult) return;

    setIsLoading(true);
    setLoadingMessage('Finding another song...');
    setError(null);

    try {
      const response = await getYouTubeSong(fusionResult.final_emotion);

      if (response.success) {
        setYoutubeData(response.song);
      } else {
        throw new Error(response.error || 'Failed to get song');
      }
    } catch (err) {
      console.error('[App] YouTube song error:', err);
      setError(err.message || 'Failed to get new song');
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [fusionResult]);

  // Get YouTube playlist
  const handleGetPlaylist = useCallback(async () => {
    if (!fusionResult) return;

    setIsLoading(true);
    setLoadingMessage('Creating playlist...');
    setError(null);

    try {
      const response = await getYouTubePlaylist(fusionResult.final_emotion, 5);

      if (response.success) {
        setYoutubePlaylist(response.playlist);
      } else {
        throw new Error(response.error || 'Failed to get playlist');
      }
    } catch (err) {
      console.error('[App] YouTube playlist error:', err);
      setError(err.message || 'Failed to get playlist');
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [fusionResult]);

  // Regenerate AI music
  const handleRegenerateMusic = useCallback(async () => {
    if (!fusionResult) return;

    setIsLoading(true);
    setLoadingMessage('Generating new music... (this may take 30-60 seconds)');
    setError(null);

    try {
      const response = await generateMusic(
        fusionResult.final_emotion,
        settings.duration
      );

      if (response.success) {
        setMusicData(response.result);
      } else {
        throw new Error(response.error || 'Music generation failed');
      }
    } catch (err) {
      console.error('[App] Music generation error:', err);
      
      let errorMessage = 'Failed to generate music';
      if (err.code === 'ERR_NETWORK') {
        errorMessage = 'Network error. Please check server connection.';
      } else if (err.code === 'ECONNABORTED') {
        errorMessage = 'Generation timed out. The server may still be processing - wait a moment and check the output folder.';
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    } finally {
      setIsLoading(false);
      setLoadingMessage('');
    }
  }, [fusionResult, settings.duration]);

  // Reset all data
  const handleReset = useCallback(() => {
    setFaceData(null);
    setVoiceData(null);
    setTextData(null);
    setFusionResult(null);
    setMusicData(null);
    setYoutubeData(null);
    setYoutubePlaylist(null);
    setCapturedImage(null);
    setRecordedAudio(null);
    setTextInput('');
    setError(null);
    setActiveTab('input');
  }, []);

  return (
    <div className="app">
      {isLoading && <LoadingOverlay message={loadingMessage} />}
      
      <Header />
      
      {/* Music Mode Toggle */}
      <div className="music-mode-toggle">
        <span className="mode-label">Music Mode:</span>
        <div className="mode-buttons">
          <button 
            className={`mode-btn ${musicMode === 'youtube' ? 'active' : ''}`}
            onClick={() => setMusicMode('youtube')}
          >
            🎬 YouTube (Full Songs)
          </button>
          <button 
            className={`mode-btn ${musicMode === 'ai' ? 'active' : ''}`}
            onClick={() => setMusicMode('ai')}
          >
            🤖 AI Generated (Short)
          </button>
        </div>
      </div>
      
      {/* Server Status Banner */}
      {serverStatus === 'offline' && (
        <div className="server-offline-banner">
          <span>🔴 Server Offline</span>
          <p>Start the backend: <code>python -m backend.api</code></p>
        </div>
      )}
      
      {serverStatus === 'online' && !modelReady && (
        <div className="model-loading-banner">
          <span>⏳ Music generator is warming up...</span>
          <p>First generation may take longer. Procedural fallback is available.</p>
        </div>
      )}
      
      <main className="main-content">
        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button
            className={`tab-btn ${activeTab === 'input' ? 'active' : ''}`}
            onClick={() => setActiveTab('input')}
          >
            📥 Input
          </button>
          <button
            className={`tab-btn ${activeTab === 'results' ? 'active' : ''}`}
            onClick={() => setActiveTab('results')}
            disabled={!fusionResult}
          >
            📊 Results
          </button>
          <button
            className={`tab-btn ${activeTab === 'music' ? 'active' : ''}`}
            onClick={() => setActiveTab('music')}
            disabled={musicMode === 'youtube' ? !youtubeData : !musicData}
          >
            🎵 Music
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="error-banner">
            <span>⚠️ {error}</span>
            <button onClick={() => setError(null)}>×</button>
          </div>
        )}

        {/* Tab Content */}
        <div className="tab-content">
          {activeTab === 'input' && (
            <InputSection
              capturedImage={capturedImage}
              setCapturedImage={setCapturedImage}
              recordedAudio={recordedAudio}
              setRecordedAudio={setRecordedAudio}
              textInput={textInput}
              setTextInput={setTextInput}
              settings={settings}
              setSettings={setSettings}
              onAnalyze={handleAnalyze}
              onReset={handleReset}
              disabled={serverStatus === 'offline'}
              musicMode={musicMode}
            />
          )}

          {activeTab === 'results' && (
            <EmotionDashboard
              faceData={faceData}
              voiceData={voiceData}
              textData={textData}
              fusionResult={fusionResult}
            />
          )}

          {activeTab === 'music' && musicMode === 'youtube' && (
            <YouTubePlayer
              song={youtubeData}
              playlist={youtubePlaylist}
              emotion={fusionResult?.final_emotion}
              onGetNewSong={handleGetNewSong}
              onGetPlaylist={handleGetPlaylist}
              isLoading={isLoading}
            />
          )}

          {activeTab === 'music' && musicMode === 'ai' && (
            <MusicPlayer
              musicData={musicData}
              emotion={fusionResult?.final_emotion}
              onRegenerate={handleRegenerateMusic}
            />
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>EmotionSense v1.0 • Multi-Modal AI Emotion-to-Music Generator</p>
        <p>
          Powered by DeepFace, Transformers, and MusicGen
          <span className={`status-dot ${serverStatus}`}></span>
        </p>
      </footer>
    </div>
  );
}

export default App;
