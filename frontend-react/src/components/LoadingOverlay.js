import React from 'react';
import { motion } from 'framer-motion';
import './LoadingOverlay.css';

const LoadingOverlay = ({ message = 'Analyzing emotions...' }) => {
  return (
    <motion.div 
      className="loading-overlay"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
    >
      <div className="loading-content">
        <div className="loading-animation">
          {/* Brain Animation */}
          <div className="brain-container">
            <span className="brain-emoji">🧠</span>
            <div className="pulse-ring"></div>
            <div className="pulse-ring delay-1"></div>
            <div className="pulse-ring delay-2"></div>
          </div>
          
          {/* Neural Network Lines */}
          <svg className="neural-lines" viewBox="0 0 200 100">
            <defs>
              <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#667eea" stopOpacity="0" />
                <stop offset="50%" stopColor="#764ba2" stopOpacity="1" />
                <stop offset="100%" stopColor="#667eea" stopOpacity="0" />
              </linearGradient>
            </defs>
            <path d="M0,50 Q50,20 100,50 T200,50" className="neural-path" />
            <path d="M0,30 Q50,60 100,30 T200,30" className="neural-path delay-1" />
            <path d="M0,70 Q50,40 100,70 T200,70" className="neural-path delay-2" />
          </svg>
        </div>

        {/* Loading Text */}
        <h3 className="loading-message">{message}</h3>
        
        {/* Progress Steps */}
        <div className="loading-steps">
          <div className="step active">
            <div className="step-icon">👁️</div>
            <span>Face</span>
          </div>
          <div className="step-connector">
            <div className="connector-fill"></div>
          </div>
          <div className="step active">
            <div className="step-icon">🎙️</div>
            <span>Voice</span>
          </div>
          <div className="step-connector">
            <div className="connector-fill"></div>
          </div>
          <div className="step active">
            <div className="step-icon">📝</div>
            <span>Text</span>
          </div>
          <div className="step-connector">
            <div className="connector-fill"></div>
          </div>
          <div className="step">
            <div className="step-icon">🎵</div>
            <span>Music</span>
          </div>
        </div>

        {/* Loading Bar */}
        <div className="loading-bar">
          <div className="loading-bar-fill"></div>
        </div>
        
        <p className="loading-tip">
          💡 Tip: For best results, ensure good lighting for face detection
        </p>
      </div>
    </motion.div>
  );
};

export default LoadingOverlay;
