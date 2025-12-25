import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <span className="logo-icon">🎵</span>
          <div className="logo-text">
            <h1>EmotionSense</h1>
            <p>Multi-Modal AI Emotion-to-Music Generator</p>
          </div>
        </div>
        
        <nav className="header-nav">
          <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="nav-link">
            GitHub
          </a>
          <a href="#about" className="nav-link">About</a>
        </nav>
      </div>
    </header>
  );
};

export default Header;
