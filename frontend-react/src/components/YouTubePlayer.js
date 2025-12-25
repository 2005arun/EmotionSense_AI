import React, { useState } from 'react';
import { FaPlay, FaRedo, FaList, FaExternalLinkAlt, FaMusic } from 'react-icons/fa';
import './YouTubePlayer.css';

const EMOJI_MAP = {
  happy: '😄',
  sad: '😢',
  angry: '😡',
  fear: '😨',
  surprise: '😲',
  disgust: '🤢',
  neutral: '😐',
  calm: '😌'
};

const YouTubePlayer = ({ song, playlist, emotion, onGetNewSong, onGetPlaylist, isLoading }) => {
  const [showPlaylist, setShowPlaylist] = useState(false);
  const [currentSongIndex, setCurrentSongIndex] = useState(0);

  // Get current song (from single song or playlist)
  const currentSong = playlist && playlist.length > 0 
    ? playlist[currentSongIndex] 
    : song;

  if (!currentSong && !isLoading) {
    return (
      <div className="youtube-player empty">
        <div className="empty-state">
          <span className="empty-icon">🎵</span>
          <h3>No Song Yet</h3>
          <p>Analyze your emotions to get a full song recommendation</p>
        </div>
      </div>
    );
  }

  const handleNextSong = () => {
    if (playlist && currentSongIndex < playlist.length - 1) {
      setCurrentSongIndex(prev => prev + 1);
    }
  };

  const handlePrevSong = () => {
    if (currentSongIndex > 0) {
      setCurrentSongIndex(prev => prev - 1);
    }
  };

  const handlePlayFromPlaylist = (index) => {
    setCurrentSongIndex(index);
    setShowPlaylist(false);
  };

  return (
    <div className="youtube-player">
      {/* Header */}
      <div className="player-header">
        <h2>🎵 Now Playing</h2>
        {emotion && (
          <span className="emotion-badge">
            {EMOJI_MAP[emotion] || '🎵'} {emotion}
          </span>
        )}
      </div>

      {isLoading ? (
        <div className="loading-state">
          <div className="spinner"></div>
          <p>Finding the perfect song for you...</p>
        </div>
      ) : currentSong ? (
        <>
          {/* YouTube Embed */}
          <div className="video-container">
            <iframe
              src={`https://www.youtube.com/embed/${currentSong.video_id}?autoplay=1&rel=0`}
              title={currentSong.title}
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            ></iframe>
          </div>

          {/* Song Info */}
          <div className="song-info">
            <h3 className="song-title">{currentSong.title}</h3>
            <p className="song-artist">{currentSong.artist}</p>
            <div className="song-meta">
              <span className="duration">⏱️ {currentSong.duration}</span>
              <span className="genre">🎸 {currentSong.genre}</span>
            </div>
          </div>

          {/* Controls */}
          <div className="player-controls">
            <button 
              className="control-btn"
              onClick={onGetNewSong}
              title="Get new song"
            >
              <FaRedo /> New Song
            </button>
            
            <button 
              className="control-btn"
              onClick={() => onGetPlaylist && onGetPlaylist()}
              title="Get playlist"
            >
              <FaList /> Get Playlist
            </button>
            
            <a 
              href={currentSong.watch_url}
              target="_blank"
              rel="noopener noreferrer"
              className="control-btn external"
            >
              <FaExternalLinkAlt /> Open in YouTube
            </a>
          </div>

          {/* Playlist Navigation */}
          {playlist && playlist.length > 1 && (
            <div className="playlist-nav">
              <button 
                onClick={handlePrevSong}
                disabled={currentSongIndex === 0}
                className="nav-btn"
              >
                ← Previous
              </button>
              <span className="playlist-position">
                {currentSongIndex + 1} / {playlist.length}
              </span>
              <button 
                onClick={handleNextSong}
                disabled={currentSongIndex >= playlist.length - 1}
                className="nav-btn"
              >
                Next →
              </button>
            </div>
          )}

          {/* Playlist Toggle */}
          {playlist && playlist.length > 1 && (
            <button 
              className="playlist-toggle"
              onClick={() => setShowPlaylist(!showPlaylist)}
            >
              {showPlaylist ? '▲ Hide Playlist' : '▼ Show Playlist'}
            </button>
          )}

          {/* Playlist View */}
          {showPlaylist && playlist && (
            <div className="playlist-view">
              <h4>📋 Playlist for {emotion}</h4>
              <div className="playlist-items">
                {playlist.map((item, index) => (
                  <div 
                    key={item.video_id}
                    className={`playlist-item ${index === currentSongIndex ? 'active' : ''}`}
                    onClick={() => handlePlayFromPlaylist(index)}
                  >
                    <img 
                      src={item.thumbnail} 
                      alt={item.title}
                      className="playlist-thumbnail"
                    />
                    <div className="playlist-item-info">
                      <span className="playlist-item-title">{item.title}</span>
                      <span className="playlist-item-artist">{item.artist}</span>
                    </div>
                    <span className="playlist-item-duration">{item.duration}</span>
                    {index === currentSongIndex && (
                      <FaPlay className="now-playing-icon" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Full Song Notice */}
          <div className="full-song-notice">
            <FaMusic />
            <span>Full 3-4 minute songs from YouTube</span>
          </div>
        </>
      ) : null}
    </div>
  );
};

export default YouTubePlayer;
