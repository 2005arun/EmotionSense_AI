import React, { useState, useRef, useEffect } from 'react';
import './MusicPlayer.css';

const EMOJI_MAP = {
  happy: '😄',
  sad: '😢',
  angry: '😡',
  fear: '😨',
  surprise: '😲',
  disgust: '🤢',
  neutral: '😐',
  calm: '😌',
};

const MUSIC_INFO = {
  happy: {
    description: 'Bright, energetic music with major keys and upbeat rhythms',
    style: 'Upbeat, cheerful, major scale',
    instruments: 'Piano, Guitar, Drums',
  },
  sad: {
    description: 'Melancholic piano and strings with minor keys and slow tempo',
    style: 'Slow, emotional, minor chords',
    instruments: 'Piano, Strings, Soft pads',
  },
  angry: {
    description: 'Heavy drums and powerful rhythms with fast tempo',
    style: 'Intense, heavy, distorted',
    instruments: 'Electric Guitar, Heavy Drums',
  },
  fear: {
    description: 'Tense, suspenseful atmosphere with dark tones',
    style: 'Dark, mysterious, ambient',
    instruments: 'Synths, Strings, Effects',
  },
  surprise: {
    description: 'Dynamic, unexpected musical changes',
    style: 'Dramatic, orchestral hits',
    instruments: 'Orchestra, Percussion',
  },
  disgust: {
    description: 'Dissonant experimental sounds',
    style: 'Industrial, experimental',
    instruments: 'Synths, Industrial sounds',
  },
  neutral: {
    description: 'Balanced, easy-listening background music',
    style: 'Calm, balanced, ambient',
    instruments: 'Soft synths, Piano',
  },
  calm: {
    description: 'Peaceful relaxing music for meditation',
    style: 'Ambient, peaceful, flowing',
    instruments: 'Piano, Soft pads, Nature sounds',
  },
};

const MusicPlayer = ({ musicData, emotion, onRegenerate }) => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.8);
  const audioRef = useRef(null);

  const emotionLower = emotion?.toLowerCase() || 'neutral';
  const emoji = EMOJI_MAP[emotionLower] || '😐';
  const info = MUSIC_INFO[emotionLower] || MUSIC_INFO.neutral;

  useEffect(() => {
    // Reset player when music changes
    setIsPlaying(false);
    setCurrentTime(0);
    if (audioRef.current) {
      audioRef.current.load();
    }
  }, [musicData]);

  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = volume;
    }
  }, [volume]);

  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  };

  const handleSeek = (e) => {
    const time = parseFloat(e.target.value);
    setCurrentTime(time);
    if (audioRef.current) {
      audioRef.current.currentTime = time;
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const formatTime = (time) => {
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleDownload = () => {
    if (musicData?.url) {
      const link = document.createElement('a');
      link.href = `http://localhost:5000${musicData.url}`;
      link.download = musicData.filename || 'emotion_music.wav';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  if (!musicData) {
    return (
      <div className="music-player empty">
        <div className="empty-state">
          <span className="empty-icon">🎵</span>
          <h3>No Music Generated Yet</h3>
          <p>Analyze your emotions to generate personalized music</p>
        </div>
      </div>
    );
  }

  return (
    <div className="music-player">
      {/* Visualizer Background */}
      <div className="visualizer-bg">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className={`bar ${isPlaying ? 'playing' : ''}`}
            style={{
              animationDelay: `${i * 0.1}s`,
              height: `${Math.random() * 60 + 20}%`,
            }}
          />
        ))}
      </div>

      <div className="player-content">
        {/* Header */}
        <div className="player-header">
          <h2>🎵 Generated Music</h2>
          <span className="emotion-badge">
            {emoji} {emotionLower.toUpperCase()}
          </span>
        </div>

        {/* Album Art */}
        <div className="album-art">
          <div className={`album-disc ${isPlaying ? 'spinning' : ''}`}>
            <span className="album-emoji">{emoji}</span>
          </div>
        </div>

        {/* Music Info */}
        <div className="music-info">
          <h3 className="music-title">
            {emotionLower.charAt(0).toUpperCase() + emotionLower.slice(1)} Mood Music
          </h3>
          <p className="music-artist">Generated by EmotionSense AI</p>
          <div className="music-tags">
            <span className="tag">{musicData.genre || 'Ambient'}</span>
            <span className="tag">{musicData.tempo || 'Moderate'} tempo</span>
          </div>
        </div>

        {/* Audio Element */}
        <audio
          ref={audioRef}
          src={`http://localhost:5000${musicData.url}`}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={handleEnded}
        />

        {/* Progress Bar */}
        <div className="progress-section">
          <span className="time current">{formatTime(currentTime)}</span>
          <div className="progress-bar-container">
            <input
              type="range"
              className="progress-bar"
              min="0"
              max={duration || 0}
              value={currentTime}
              onChange={handleSeek}
            />
            <div
              className="progress-fill"
              style={{ width: `${(currentTime / duration) * 100 || 0}%` }}
            />
          </div>
          <span className="time total">{formatTime(duration || musicData.duration || 0)}</span>
        </div>

        {/* Controls */}
        <div className="controls">
          <button className="control-btn" onClick={() => {
            if (audioRef.current) {
              audioRef.current.currentTime = Math.max(0, audioRef.current.currentTime - 10);
            }
          }}>
            ⏪
          </button>
          <button className="control-btn play-btn" onClick={togglePlay}>
            {isPlaying ? '⏸️' : '▶️'}
          </button>
          <button className="control-btn" onClick={() => {
            if (audioRef.current) {
              audioRef.current.currentTime = Math.min(duration, audioRef.current.currentTime + 10);
            }
          }}>
            ⏩
          </button>
        </div>

        {/* Volume Control */}
        <div className="volume-control">
          <span className="volume-icon">{volume === 0 ? '🔇' : volume < 0.5 ? '🔉' : '🔊'}</span>
          <input
            type="range"
            className="volume-slider"
            min="0"
            max="1"
            step="0.01"
            value={volume}
            onChange={(e) => setVolume(parseFloat(e.target.value))}
          />
        </div>

        {/* Action Buttons */}
        <div className="action-buttons">
          <button className="btn btn-primary" onClick={onRegenerate}>
            🔁 Regenerate Music
          </button>
          <button className="btn btn-secondary" onClick={handleDownload}>
            📥 Download
          </button>
        </div>

        {/* Music Description */}
        <div className="music-description">
          <h4>🎼 Music Style</h4>
          <p>{info.description}</p>
          <div className="description-details">
            <div className="detail-item">
              <span className="detail-label">Style:</span>
              <span className="detail-value">{info.style}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Instruments:</span>
              <span className="detail-value">{info.instruments}</span>
            </div>
            <div className="detail-item">
              <span className="detail-label">Duration:</span>
              <span className="detail-value">{musicData.duration?.toFixed(1) || '10'}s</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MusicPlayer;
