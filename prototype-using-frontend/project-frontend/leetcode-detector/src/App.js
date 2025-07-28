import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [processedCount, setProcessedCount] = useState(0);
  const [stream, setStream] = useState(null);
  const [logs, setLogs] = useState([]);
  const logsEndRef = useRef(null);
  
  // Add streaming states
  const [isStreamingResponse, setIsStreamingResponse] = useState(false);
  const [streamedResponse, setStreamedResponse] = useState('');

  // Add detected text state
  const [detectedText, setDetectedText] = useState('');

  // Add camera selection states
  const [availableDevices, setAvailableDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [hasPermission, setHasPermission] = useState(false);

  // Add multi-page capture states
  const [capturedFrames, setCapturedFrames] = useState([]);
  const [processingTimeLeft, setProcessingTimeLeft] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Add countdown states
  const [isCountingDown, setIsCountingDown] = useState(false);
  const [countdownValue, setCountdownValue] = useState(0);

  
  // Processing duration (7 seconds)
  const PROCESSING_DURATION = 7000; // 7 seconds
  const CAPTURE_INTERVAL = 500; // Capture every 0.5 seconds
  const COUNTDOWN_DURATION = 3000; // 3 seconds countdown

  // Refs to store interval IDs for cleanup
  const countdownIntervalRef = useRef(null);
  const captureIntervalRef = useRef(null);
  const processingTimeoutRef = useRef(null);
  const readyCountdownRef = useRef(null);

  // Add log entry with duplicate prevention
  const addLog = useCallback((message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = {
      id: Date.now() + Math.random(), // More unique ID
      timestamp,
      message,
      type
    };
    
    setLogs(prev => {
      // Prevent duplicate logs within 100ms
      const lastLog = prev[prev.length - 1];
      if (lastLog && lastLog.message === message && 
          (Date.now() - lastLog.id) < 100) {
        return prev;
      }
      return [...prev, logEntry];
    });
  }, []);

  // Auto-scroll logs to bottom
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  // Request camera permission and get devices with proper labels
  const requestPermissionAndGetDevices = useCallback(async () => {
    try {
      addLog('🔐 Requesting camera permission...', 'info');
      
      // Request camera permission first
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setHasPermission(true);
      addLog('✅ Camera permission granted', 'success');
      
      // Stop the temporary stream
      stream.getTracks().forEach(track => track.stop());
      
      // Now get devices with proper labels
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
      setAvailableDevices(videoDevices);
      
      // Set default device if none selected
      if (!selectedDeviceId && videoDevices.length > 0) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
      
      addLog(`📹 Found ${videoDevices.length} camera(s)`, 'info');
      
      // Log each camera for debugging
      videoDevices.forEach((device, index) => {
        addLog(`  📷 ${index + 1}: ${device.label || 'Unknown Camera'}`, 'info');
      });
      
    } catch (error) {
      console.error('Permission error:', error);
      addLog('❌ Camera permission denied: ' + error.message, 'error');
      setHasPermission(false);
    }
  }, [addLog, selectedDeviceId]);

  // Get available video devices (without permission)
  const getVideoDevices = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices.filter(device => device.kind === 'videoinput');
      
      // Add generic labels if no permission
      const devicesWithLabels = videoDevices.map((device, index) => ({
        ...device,
        label: device.label || `Camera ${index + 1}`
      }));
      
      setAvailableDevices(devicesWithLabels);
      
      // Only set default device if none is currently selected
      if (!selectedDeviceId && devicesWithLabels.length > 0) {
        setSelectedDeviceId(devicesWithLabels[0].deviceId);
      }
      
      addLog(`📹 Found ${devicesWithLabels.length} camera(s)`, 'info');
      
      // Check if we have proper labels
      const hasProperLabels = videoDevices.some(device => device.label && device.label.trim() !== '');
      if (!hasProperLabels && videoDevices.length > 0) {
        addLog('⚠️ Camera names not available - permission needed', 'warning');
      }
      
    } catch (error) {
      console.error('Error getting video devices:', error);
      addLog('❌ Failed to get video devices: ' + error.message, 'error');
    }
  }, [addLog, selectedDeviceId]);

  // Load devices on component mount
  useEffect(() => {
    let mounted = true;
    
    const loadDevices = async () => {
      if (mounted) {
        await getVideoDevices();
      }
    };
    
    loadDevices();
    
    return () => {
      mounted = false;
    };
  }, [getVideoDevices]);

  // Start camera with selected device - FIXED CONSTRAINTS
  const startCamera = async () => {
    try {
      addLog('🔄 Starting camera...', 'info');
      
      // Use more flexible constraints
      const constraints = {
        video: {
          width: { min: 640, ideal: 1280, max: 1920 },
          height: { min: 480, ideal: 720, max: 1080 },
          ...(selectedDeviceId && { deviceId: { exact: selectedDeviceId } })
        }
      };
      
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      
      videoRef.current.srcObject = mediaStream;
      setStream(mediaStream);
      setIsStreaming(true);
      setHasPermission(true);
      
      // Get the selected device name for logging
      const selectedDevice = availableDevices.find(device => device.deviceId === selectedDeviceId);
      const deviceName = selectedDevice?.label || 'Camera';
      
      addLog(`✅ Camera started: ${deviceName}`, 'success');
      
      // Refresh device list after permission is granted
      setTimeout(() => {
        requestPermissionAndGetDevices();
      }, 500);
      
    } catch (error) {
      console.error('Error accessing camera:', error);
      addLog('❌ Failed to start camera: ' + error.message, 'error');
      
      // Try fallback with basic constraints if over-constrained
      if (error.name === 'OverconstrainedError') {
        try {
          addLog('🔄 Trying with basic constraints...', 'info');
          
          const fallbackConstraints = {
            video: selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true
          };
          
          const fallbackStream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
          
          videoRef.current.srcObject = fallbackStream;
          setStream(fallbackStream);
          setIsStreaming(true);
          setHasPermission(true);
          
          addLog('✅ Camera started with basic settings', 'success');
          
        } catch (fallbackError) {
          console.error('Fallback also failed:', fallbackError);
          addLog('❌ Camera failed even with basic settings: ' + fallbackError.message, 'error');
          
          if (fallbackError.name === 'NotAllowedError') {
            alert('Camera permission denied. Please allow camera access and try again.');
          } else if (fallbackError.name === 'NotFoundError') {
            alert('No camera found. Please connect a camera and refresh.');
          } else {
            alert('Camera error: ' + fallbackError.message);
          }
        }
      } else {
        // Handle other errors
        if (error.name === 'NotAllowedError') {
          alert('Camera permission denied. Please allow camera access and try again.');
        } else if (error.name === 'NotFoundError') {
          alert('No camera found. Please connect a camera and refresh.');
        } else if (error.name === 'NotReadableError') {
          alert('Camera is busy. Please close other apps using the camera.');
        } else {
          alert('Camera error: ' + error.message);
        }
      }
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
      setIsStreaming(false);
      setIsProcessing(false);
      
      addLog('⏹️ Camera stopped', 'warning');
      console.log('Camera stopped');
    }
  };

  // Capture single frame
  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !isStreaming) return null;

    console.log('📸 Capturing frame...');
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        resolve(blob);
      }, 'image/png');
    });
  }, [isStreaming]);

  // Process captured frames with backend - FIXED DUPLICATES
  const processFrames = useCallback(async (frames) => {
    if (frames.length === 0) return;

    setIsAnalyzing(true);
    addLog(`🔄 Analyzing ${frames.length} captured frames...`, 'info');
    
    try {
      // Create FormData with multiple frames
      const formData = new FormData();
      frames.forEach((frame, index) => {
        formData.append(`frame_${index}`, frame, `frame_${index}.png`);
      });
      formData.append('frame_count', frames.length.toString());

      setIsStreamingResponse(true);
      setStreamedResponse('');
      setDetectedText(''); // Reset detected text
      
      // Use fetch with SSE for streaming GPT response
      const streamResponse = await fetch('http://localhost:8000/process-multiple-frames-stream', {
        method: 'POST',
        body: formData
      });
      
      if (!streamResponse.ok) {
        throw new Error(`HTTP error! status: ${streamResponse.status}`);
      }
      
      // Handle streaming response
      const reader = streamResponse.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedResponse = '';
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6); // Remove 'data: ' prefix
            
            if (data === '[DONE]') {
              // Stream is complete
              setIsStreamingResponse(false);
              setIsAnalyzing(false);
              setProcessedCount(prev => prev + 1);
              addLog('✅ Analysis completed successfully', 'success');
              
              // Set final result
              setResult({
                success: true,
                response: accumulatedResponse,
                frame_count: frames.length,
                detected_text: detectedText
              });
              return;
            }
            
            try {
              const parsedData = JSON.parse(data);
              if (parsedData.content) {
                accumulatedResponse += parsedData.content;
                setStreamedResponse(accumulatedResponse);
              }
              // Handle detected_text from backend
              if (parsedData.detected_text) {
                setDetectedText(parsedData.detected_text);
              }
            } catch (e) {
              // If it's not JSON, treat as plain text
              if (data.trim()) {
                accumulatedResponse += data;
                setStreamedResponse(accumulatedResponse);
              }
            }
          }
        }
      }
      
    } catch (error) {
      console.error('Error processing frames:', error);
      addLog('❌ Failed to process frames: ' + error.message, 'error');
      setIsStreamingResponse(false);
      setIsAnalyzing(false);
      
      setResult({
        success: false,
        error: error.message
      });
    }
  }, [addLog, setProcessedCount, detectedText]);

  // Cleanup function for intervals and timeouts
  const cleanupTimers = useCallback(() => {
    if (countdownIntervalRef.current) {
      clearInterval(countdownIntervalRef.current);
      countdownIntervalRef.current = null;
    }
    if (captureIntervalRef.current) {
      clearInterval(captureIntervalRef.current);
      captureIntervalRef.current = null;
    }
    if (processingTimeoutRef.current) {
      clearTimeout(processingTimeoutRef.current);
      processingTimeoutRef.current = null;
    }
    if (readyCountdownRef.current) {
      clearInterval(readyCountdownRef.current);
      readyCountdownRef.current = null;
    }
  }, []);

  // Start the actual frame capture process
  const startActualProcessing = useCallback(() => {
    setIsProcessing(true);
    setCapturedFrames([]);
    setProcessingTimeLeft(PROCESSING_DURATION / 1000);
    addLog('🔄 Started multi-page capture (7 seconds)', 'info');
    
    const startTime = Date.now();
    
    // Countdown timer
    countdownIntervalRef.current = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const remaining = Math.max(0, (PROCESSING_DURATION - elapsed) / 1000);
      setProcessingTimeLeft(Math.ceil(remaining));
      
      if (remaining <= 0) {
        clearInterval(countdownIntervalRef.current);
        countdownIntervalRef.current = null;
      }
    }, 100);
    
    // Frame capture interval
    captureIntervalRef.current = setInterval(async () => {
      const frame = await captureFrame();
      if (frame) {
        setCapturedFrames(prev => {
          const newFrames = [...prev, frame];
          addLog(`📸 Frame ${newFrames.length} captured`, 'info');
          return newFrames;
        });
      }
    }, CAPTURE_INTERVAL);
    
    // Auto-stop after duration
    processingTimeoutRef.current = setTimeout(() => {
      cleanupTimers();
      setIsProcessing(false);
      setProcessingTimeLeft(0);
      addLog('⏸️ Multi-page capture completed', 'warning');
      
      // Process captured frames
      setCapturedFrames(frames => {
        if (frames.length > 0) {
          processFrames(frames);
        } else {
          addLog('⚠️ No frames captured', 'warning');
        }
        return frames;
      });
    }, PROCESSING_DURATION);
    
  }, [captureFrame, processFrames, addLog, cleanupTimers]);

  // Start processing (with 3-second countdown first)
  const startProcessing = useCallback(() => {
    // Cleanup any existing timers first
    cleanupTimers();
    
    setIsCountingDown(true);
    setCountdownValue(3);
    addLog('⏰ Get ready! Starting countdown...', 'info');
    
    const countdownStart = Date.now();
    
    // Ready countdown timer (3, 2, 1)
    readyCountdownRef.current = setInterval(() => {
      const elapsed = Date.now() - countdownStart;
      const remaining = Math.max(0, (COUNTDOWN_DURATION - elapsed) / 1000);
      const countValue = Math.ceil(remaining);
      
      setCountdownValue(countValue);
      
      if (countValue <= 0) {
        clearInterval(readyCountdownRef.current);
        readyCountdownRef.current = null;
        setIsCountingDown(false);
        
        // Start actual processing
        startActualProcessing();
      }
    }, 100);
  }, [cleanupTimers, addLog, startActualProcessing]);

  // Stop processing manually
  const stopProcessing = useCallback(() => {
    cleanupTimers();
    setIsProcessing(false);
    setIsCountingDown(false);
    setProcessingTimeLeft(0);
    addLog('⏸️ Processing stopped manually', 'warning');
    
    // Process whatever frames we have
    if (capturedFrames.length > 0) {
      processFrames(capturedFrames);
    } else {
      addLog('⚠️ No frames to process', 'warning');
    }
  }, [capturedFrames, processFrames, addLog, cleanupTimers]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cleanupTimers();
    };
  }, [cleanupTimers]);

  // Clear logs
  const clearLogs = () => {
    setLogs([]);
  };

  // Reset function to start scanning again
  const resetScanning = () => {
    setResult(null);
    setStreamedResponse('');
    setIsStreamingResponse(false);
    setCapturedFrames([]);
    setIsAnalyzing(false);
    setDetectedText(''); // Reset detected text
    setIsCountingDown(false);
    setCountdownValue(0);
    addLog('🔄 Ready to scan again', 'info');
  };

  return (
    <div className="app-container">
      <div className="main-content">
        <h1>🎯 LeetCode Multi-Page Detector</h1>
        
        {/* Camera Selection */}
        <div className="camera-selection">
          <label htmlFor="camera-select">📹 Select Camera:</label>
          <select 
            id="camera-select"
            value={selectedDeviceId} 
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            disabled={isStreaming}
            className="camera-dropdown"
          >
            {availableDevices.length === 0 ? (
              <option value="">No cameras found</option>
            ) : (
              availableDevices.map((device, index) => (
                <option key={`${device.deviceId}-${index}`} value={device.deviceId}>
                  {device.label || `Camera ${device.deviceId.slice(0, 8)}...`}
                </option>
              ))
            )}
          </select>
          <button 
            onClick={getVideoDevices} 
            disabled={isStreaming}
            className="btn btn-refresh"
          >
            🔄 Refresh
          </button>
          <button 
            onClick={requestPermissionAndGetDevices} 
            disabled={isStreaming}
            className="btn btn-permission"
            style={{ 
              backgroundColor: hasPermission ? '#28a745' : '#007bff',
              color: 'white' 
            }}
          >
            {hasPermission ? '✅ Permission OK' : '🔐 Grant Permission'}
          </button>
        </div>

        {/* Browser compatibility notice */}
        {availableDevices.length > 0 && !hasPermission && (
          <div style={{ 
            padding: '10px', 
            backgroundColor: '#fff3cd', 
            border: '1px solid #ffeaa7', 
            borderRadius: '4px', 
            margin: '10px 0' 
          }}>
            <p>💡 <strong>Users:</strong> Click "Grant Permission" first to see camera names and enable the Start Camera button.</p>
          </div>
        )}
        
        {/* Controls */}
        <div className="controls">
          <button 
            onClick={startCamera} 
            disabled={isStreaming || !selectedDeviceId || availableDevices.length === 0}
            className="btn btn-primary"
          >
            📹 Start Camera
          </button>
          
          <button 
            onClick={stopCamera} 
            disabled={!isStreaming}
            className="btn btn-secondary"
          >
            ⏹️ Stop Camera
          </button>
          
          {/* Processing Controls */}
          {!isProcessing && !isAnalyzing && !isCountingDown ? (
            <button 
              onClick={startProcessing} 
              disabled={!isStreaming}
              className="btn btn-success"
            >
              🔍 Start Multi-Page Capture (7s)
            </button>
          ) : (
            <button 
              onClick={stopProcessing} 
              disabled={isAnalyzing}
              className="btn btn-warning"
            >
              ⏸️ Stop Capture {processingTimeLeft > 0 && `(${processingTimeLeft}s)`}
            </button>
          )}
          
          {/* Reset button */}
          {(result || capturedFrames.length > 0) && (
            <button 
              onClick={resetScanning}
              className="btn btn-warning"
              disabled={isAnalyzing || isCountingDown}
            >
              🔄 Reset
            </button>
          )}
        </div>

        {/* Status */}
        <div className="status">
          <span className={isStreaming ? 'status-on' : 'status-off'}>
            Camera: {isStreaming ? '🟢 ON' : '🔴 OFF'}
          </span>
          <span className={hasPermission ? 'status-on' : 'status-off'}>
            Permission: {hasPermission ? '🟢 GRANTED' : '🔴 NEEDED'}
          </span>
          <span className={isProcessing ? 'status-on' : 'status-off'}>
            Capturing: {isProcessing ? `🟢 ACTIVE (${processingTimeLeft}s)` : '🔴 STOPPED'}
          </span>
          <span>Frames: {capturedFrames.length}</span>
          <span>Analyzed: {processedCount}</span>
          {isCountingDown && (
            <span className="status-streaming">⏰ Get Ready: {countdownValue}</span>
          )}
          {isAnalyzing && (
            <span className="status-streaming">🔴 Analyzing Frames...</span>
          )}
          {isStreamingResponse && (
            <span className="status-streaming">🔴 Streaming GPT Response...</span>
          )}
        </div>

        {/* Video and Chat Layout */}
        <div className="video-chat-layout">
          {/* Video Feed - Left Side */}
          <div className="video-section">
            <div className="video-container">
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                muted
                className="video-feed"
              />
              <canvas ref={canvasRef} style={{ display: 'none' }} />
              
              {isCountingDown && (
                <div className="overlay">
                  <div className="processing-indicator">
                    ⏰ GET READY...
                    <div style={{ fontSize: '48px', fontWeight: 'bold' }}>{countdownValue}</div>
                  </div>
                </div>
              )}

              {isProcessing && !isCountingDown && (
                <div className="overlay">
                  <div className="processing-indicator">
                    🔍 CAPTURING FRAMES... ({processingTimeLeft}s)
                    <div>Frame {capturedFrames.length}</div>
                  </div>
                </div>
              )}

              {isAnalyzing && (
                <div className="overlay">
                  <div className="processing-indicator">
                    {isStreamingResponse ? '🤖 GPT ANALYZING...' : '🔄 PROCESSING FRAMES...'}
                  </div>
                </div>
              )}
              
              {!isStreaming && (
                <div className="placeholder">
                  {availableDevices.length === 0 ? (
                    <div>
                      <p>📷 No cameras detected</p>
                      <p>Please connect a camera and click "Refresh"</p>
                    </div>
                  ) : !hasPermission ? (
                    <div>
                      <p>📷 Camera permission needed</p>
                      <p>Click "Grant Permission" first, then "Start Camera"</p>
                    </div>
                  ) : (
                    <div>
                      <p>📷 Ready to start camera</p>
                      <p>Click "Start Camera" to begin</p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Chat/Logs Section - Right Side */}
          <div className="chat-section">
            <div className="chat-header">
              <h3>📋 Activity Log</h3>
              <button onClick={clearLogs} className="btn btn-clear">
                🗑️ Clear
              </button>
            </div>
            
            <div className="chat-messages">
              {logs.length === 0 ? (
                <div className="no-messages">
                  <p>💬 No activity yet...</p>
                  <p>Start the camera and processing to see logs</p>
                </div>
              ) : (
                <div className="messages">
                  {logs.map((log) => (
                    <div key={log.id} className={`message message-${log.type}`}>
                      <div className="message-content">
                        <span className="message-text">{log.message}</span>
                        <span className="message-time">{log.timestamp}</span>
                      </div>
                    </div>
                  ))}
                  <div ref={logsEndRef} />
                </div>
              )}
            </div>
            
            {/* Session Stats */}
            <div className="chat-stats">
              <div className="stat">
                <span>📊 Total Logs:</span>
                <span>{logs.length}</span>
              </div>
              <div className="stat">
                <span>🎯 Sessions Analyzed:</span>
                <span>{processedCount}</span>
              </div>
              <div className="stat">
                <span>📹 Status:</span>
                <span className={isProcessing || isCountingDown ? 'status-active' : 'status-inactive'}>
                  {isCountingDown ? 'Getting Ready' : isProcessing ? 'Capturing' : 'Idle'}
                </span>
              </div>
              <div className="stat">
                <span>📸 Frames Captured:</span>
                <span>{capturedFrames.length}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Results Section - Below Video and Chat */}
        {((result && result.success) || isStreamingResponse) && (
          <div className="results">
            <h2>🎯 Multi-Page LeetCode Analysis Complete!</h2>
            <div className="success">
              {result?.frame_count && (
                <p>✅ Analyzed {result.frame_count} frames</p>
              )}
              <h3>💡 GPT Analysis:</h3>
              <div className="response">
                {isStreamingResponse ? (
                  <div className="streaming-response">
                    {streamedResponse}
                    <span className="cursor">▋</span>
                  </div>
                ) : (
                  <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
                    {streamedResponse || result?.response}
                  </pre>
                )}
              </div>
              {(result?.detected_text || detectedText) && (
                <>
                  <h3>🔍 Detected Text:</h3>
                  <textarea 
                    value={result?.detected_text || detectedText} 
                    readOnly 
                    rows={8}
                    className="detected-text"
                    style={{ 
                      width: '100%', 
                      fontFamily: 'monospace', 
                      fontSize: '14px',
                      background: '#f5f5f5',
                      border: '1px solid #ddd',
                      borderRadius: '4px',
                      padding: '10px'
                    }}
                  />
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;