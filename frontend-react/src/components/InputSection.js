import React, { useRef, useCallback, useState } from 'react';
import Webcam from 'react-webcam';
import './InputSection.css';

const InputSection = ({
  capturedImage,
  setCapturedImage,
  recordedAudio,
  setRecordedAudio,
  textInput,
  setTextInput,
  settings,
  setSettings,
  onAnalyze,
  onReset,
}) => {
  const webcamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [cameraActive, setCameraActive] = useState(false);
  const [audioChunks, setAudioChunks] = useState([]);
  const recordingTimerRef = useRef(null);

  // Capture image from webcam
  const captureImage = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      setCapturedImage(imageSrc);
    }
  }, [setCapturedImage]);

  // Handle image upload
  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setCapturedImage(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Start audio recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      const chunks = [];
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        const reader = new FileReader();
        reader.onloadend = () => {
          setRecordedAudio(reader.result);
        };
        reader.readAsDataURL(blob);
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setRecordingTime(0);
      
      // Start timer
      recordingTimerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } catch (err) {
      console.error('Error accessing microphone:', err);
      alert('Could not access microphone. Please ensure you have given permission.');
    }
  };

  // Stop audio recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      clearInterval(recordingTimerRef.current);
    }
  };

  // Handle audio file upload
  const handleAudioUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setRecordedAudio(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  // Format recording time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Check if any input is provided
  const hasInput = capturedImage || recordedAudio || textInput.trim();

  return (
    <div className="input-section">
      <div className="input-grid">
        {/* Face Input */}
        <div className="input-card">
          <div className="card-header">
            <span className="card-icon">📷</span>
            <h3>Face Input</h3>
          </div>
          
          <div className="card-content">
            {cameraActive && !capturedImage ? (
              <div className="webcam-container">
                <Webcam
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  className="webcam"
                  mirrored={true}
                />
                <div className="webcam-controls">
                  <button className="btn btn-capture" onClick={captureImage}>
                    📸 Capture
                  </button>
                  <button className="btn btn-secondary" onClick={() => setCameraActive(false)}>
                    ✕ Close
                  </button>
                </div>
              </div>
            ) : capturedImage ? (
              <div className="preview-container">
                <img src={capturedImage} alt="Captured" className="preview-image" />
                <button className="btn btn-secondary" onClick={() => setCapturedImage(null)}>
                  🗑️ Remove
                </button>
              </div>
            ) : (
              <div className="input-options">
                <button className="btn btn-primary" onClick={() => setCameraActive(true)}>
                  📹 Open Camera
                </button>
                <span className="or-divider">or</span>
                <label className="btn btn-secondary file-upload">
                  📁 Upload Image
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    hidden
                  />
                </label>
              </div>
            )}
          </div>
          
          {capturedImage && (
            <div className="card-status success">
              ✓ Face image captured
            </div>
          )}
        </div>

        {/* Voice Input */}
        <div className="input-card">
          <div className="card-header">
            <span className="card-icon">🎤</span>
            <h3>Voice Input</h3>
          </div>
          
          <div className="card-content">
            {recordedAudio ? (
              <div className="audio-preview">
                <audio controls src={recordedAudio} className="audio-player" />
                <button className="btn btn-secondary" onClick={() => setRecordedAudio(null)}>
                  🗑️ Remove
                </button>
              </div>
            ) : (
              <div className="input-options">
                {isRecording ? (
                  <div className="recording-status">
                    <div className="recording-indicator">
                      <span className="recording-dot"></span>
                      Recording... {formatTime(recordingTime)}
                    </div>
                    <button className="btn btn-danger" onClick={stopRecording}>
                      ⏹️ Stop Recording
                    </button>
                  </div>
                ) : (
                  <>
                    <button className="btn btn-primary" onClick={startRecording}>
                      🎙️ Start Recording
                    </button>
                    <span className="or-divider">or</span>
                    <label className="btn btn-secondary file-upload">
                      📁 Upload Audio
                      <input
                        type="file"
                        accept="audio/*"
                        onChange={handleAudioUpload}
                        hidden
                      />
                    </label>
                  </>
                )}
              </div>
            )}
          </div>
          
          {recordedAudio && (
            <div className="card-status success">
              ✓ Audio recorded
            </div>
          )}
        </div>

        {/* Text Input */}
        <div className="input-card">
          <div className="card-header">
            <span className="card-icon">⌨️</span>
            <h3>Text Input</h3>
          </div>
          
          <div className="card-content">
            <textarea
              className="text-input"
              placeholder="Express how you're feeling today...&#10;&#10;Example: 'I feel lonely and sad today'"
              value={textInput}
              onChange={(e) => setTextInput(e.target.value)}
              rows={5}
            />
            <div className="text-char-count">
              {textInput.length} characters
            </div>
          </div>
          
          {textInput.trim() && (
            <div className="card-status success">
              ✓ Text entered
            </div>
          )}
        </div>
      </div>

      {/* Settings */}
      <div className="settings-section">
        <h4>⚙️ Settings</h4>
        <div className="settings-grid">
          <div className="setting-item">
            <label>Music Duration</label>
            <div className="slider-container">
              <input
                type="range"
                min="5"
                max="30"
                value={settings.duration}
                onChange={(e) => setSettings({ ...settings, duration: parseInt(e.target.value) })}
              />
              <span>{settings.duration}s</span>
            </div>
          </div>
          
          <div className="setting-item">
            <label>Fusion Method</label>
            <select
              value={settings.fusionMethod}
              onChange={(e) => setSettings({ ...settings, fusionMethod: e.target.value })}
            >
              <option value="confidence_based">Confidence Based</option>
              <option value="weighted_average">Weighted Average</option>
              <option value="voting">Voting</option>
              <option value="rule_based">Rule Based</option>
            </select>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="action-buttons">
        <button
          className="btn btn-large btn-primary"
          onClick={onAnalyze}
          disabled={!hasInput}
        >
          🧠 Analyze Emotions & Generate Music
        </button>
        <button className="btn btn-large btn-secondary" onClick={onReset}>
          🔄 Reset All
        </button>
      </div>
    </div>
  );
};

export default InputSection;
