import React from 'react';
import './EmotionDashboard.css';

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

const COLOR_MAP = {
  happy: '#FFD700',
  sad: '#4169E1',
  angry: '#DC143C',
  fear: '#800080',
  surprise: '#FF8C00',
  disgust: '#228B22',
  neutral: '#808080',
  calm: '#87CEEB',
};

const EmotionCard = ({ title, icon, data }) => {
  if (!data) {
    return (
      <div className="emotion-card empty">
        <div className="emotion-card-header">
          <span className="emotion-card-icon">{icon}</span>
          <h3>{title}</h3>
        </div>
        <div className="emotion-card-body">
          <p className="no-data">No data available</p>
        </div>
      </div>
    );
  }

  const emotion = data.emotion?.toLowerCase() || 'neutral';
  const confidence = data.confidence || 0;
  const emoji = EMOJI_MAP[emotion] || '😐';
  const color = COLOR_MAP[emotion] || '#808080';

  return (
    <div className="emotion-card" style={{ '--emotion-color': color }}>
      <div className="emotion-card-header">
        <span className="emotion-card-icon">{icon}</span>
        <h3>{title}</h3>
      </div>
      <div className="emotion-card-body">
        <div className="emotion-result">
          <span className="emotion-emoji">{emoji}</span>
          <span className="emotion-label">{emotion.toUpperCase()}</span>
        </div>
        <div className="confidence-bar-container">
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{ width: `${confidence * 100}%`, background: color }}
            />
          </div>
          <span className="confidence-value">{(confidence * 100).toFixed(0)}%</span>
        </div>
        
        {data.all_emotions && Object.keys(data.all_emotions).length > 0 && (
          <div className="all-emotions">
            <h4>All Emotions</h4>
            <div className="emotions-list">
              {Object.entries(data.all_emotions)
                .sort(([, a], [, b]) => b - a)
                .map(([em, score]) => (
                  <div key={em} className="emotion-item">
                    <span className="emotion-item-emoji">{EMOJI_MAP[em] || '😐'}</span>
                    <span className="emotion-item-label">{em}</span>
                    <div className="emotion-item-bar">
                      <div
                        className="emotion-item-fill"
                        style={{
                          width: `${score * 100}%`,
                          background: COLOR_MAP[em] || '#808080',
                        }}
                      />
                    </div>
                    <span className="emotion-item-value">{(score * 100).toFixed(0)}%</span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const FinalEmotionCard = ({ fusionResult }) => {
  if (!fusionResult) return null;

  const emotion = fusionResult.final_emotion?.toLowerCase() || 'neutral';
  const confidence = fusionResult.final_confidence || 0;
  const emoji = EMOJI_MAP[emotion] || '😐';
  const color = COLOR_MAP[emotion] || '#808080';

  return (
    <div className="final-emotion-card" style={{ '--emotion-color': color }}>
      <div className="final-emotion-glow" style={{ background: color }} />
      <div className="final-emotion-content">
        <h2>🧠 Multi-Modal Fusion Result</h2>
        <div className="final-emotion-result">
          <span className="final-emoji">{emoji}</span>
          <span className="final-label">{emotion.toUpperCase()}</span>
          <span className="final-confidence">{(confidence * 100).toFixed(0)}% Confidence</span>
        </div>
        
        {fusionResult.reasoning && (
          <div className="fusion-reasoning">
            <h4>Reasoning</h4>
            <ul>
              {Object.entries(fusionResult.reasoning).map(([key, value]) => {
                if (!value || key === 'method' || key === 'rule' || key === 'final') return null;
                const em = value.emotion || value;
                const conf = value.confidence;
                return (
                  <li key={key}>
                    <strong>{key.charAt(0).toUpperCase() + key.slice(1)}:</strong>{' '}
                    {typeof value === 'object' 
                      ? `${EMOJI_MAP[em] || '😐'} ${em} (${(conf * 100).toFixed(0)}%)`
                      : value
                    }
                  </li>
                );
              })}
            </ul>
          </div>
        )}

        {fusionResult.all_emotions && (
          <div className="fusion-distribution">
            <h4>Emotion Distribution</h4>
            <div className="distribution-bars">
              {Object.entries(fusionResult.all_emotions)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 5)
                .map(([em, score]) => (
                  <div key={em} className="distribution-item">
                    <span className="distribution-label">
                      {EMOJI_MAP[em] || '😐'} {em}
                    </span>
                    <div className="distribution-bar">
                      <div
                        className="distribution-fill"
                        style={{
                          width: `${Math.min(score * 100, 100)}%`,
                          background: COLOR_MAP[em] || '#808080',
                        }}
                      />
                    </div>
                    <span className="distribution-value">{(score * 100).toFixed(0)}%</span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const EmotionDashboard = ({ faceData, voiceData, textData, fusionResult }) => {
  return (
    <div className="emotion-dashboard">
      <h2 className="dashboard-title">📊 Emotion Analysis Results</h2>
      
      {/* Individual Modality Results */}
      <div className="modality-grid">
        <EmotionCard title="Face Emotion" icon="😐" data={faceData} />
        <EmotionCard title="Voice Emotion" icon="🎧" data={voiceData} />
        <EmotionCard title="Text Emotion" icon="💬" data={textData} />
      </div>

      {/* Final Fusion Result */}
      <FinalEmotionCard fusionResult={fusionResult} />
    </div>
  );
};

export default EmotionDashboard;
