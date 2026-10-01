/**
 * frontend/audio.js
 * Microphone audio capture and dispatch streaming for Crisis Command.
 *
 * NOTE ON BROWSER SECURITY:
 * navigator.mediaDevices.getUserMedia() requires a secure context (HTTPS)
 * or a local development origin (http://localhost or http://127.0.0.1).
 * In insecure HTTP network origins, browser security models will reject the call.
 */

(() => {
  let mediaRecorder = null;
  let audioChunks = [];
  let recordingTimer = null;
  let maxDurationTimer = null;
  let secondsElapsed = 0;
  let activeStream = null;

  const MIME_CANDIDATES = [
    'audio/webm;codecs=opus',
    'audio/ogg;codecs=opus',
    'audio/mp4'
  ];

  function getSupportedMimeType() {
    if (typeof MediaRecorder === 'undefined') return '';
    return MIME_CANDIDATES.find(type => MediaRecorder.isTypeSupported(type)) || '';
  }

  document.addEventListener('DOMContentLoaded', () => {
    const micBtn = document.getElementById('mic-btn');
    if (!micBtn) return;

    micBtn.addEventListener('click', toggleRecording);
  });

  async function toggleRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      stopRecording();
    } else {
      await startRecording();
    }
  }

  async function startRecording() {
    const micBtn = document.getElementById('mic-btn');

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (typeof showToast === 'function') {
        showToast('Microphone access unsupported or blocked. Ensure HTTPS or localhost.', 'error');
      }
      return;
    }

    const mimeType = getSupportedMimeType();

    try {
      activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      const isDenied = err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError';
      const msg = isDenied
        ? 'Microphone permission denied by user or browser.'
        : `Microphone access error: ${err.message}`;
      if (typeof showToast === 'function') {
        showToast(msg, 'error');
      }
      return;
    }

    audioChunks = [];
    secondsElapsed = 0;

    const options = mimeType ? { mimeType } : {};
    try {
      mediaRecorder = new MediaRecorder(activeStream, options);
    } catch (err) {
      mediaRecorder = new MediaRecorder(activeStream);
    }

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) {
        audioChunks.push(e.data);
      }
    };

    mediaRecorder.onstop = async () => {
      cleanupStream();
      resetRecordingUI();

      const blobType = mediaRecorder.mimeType || mimeType || 'audio/webm';
      const audioBlob = new Blob(audioChunks, { type: blobType });

      if (audioBlob.size === 0) {
        if (typeof showToast === 'function') {
          showToast('Recording was empty. Please try again.', 'error');
        }
        return;
      }

      await uploadAudioDispatch(audioBlob, blobType);
    };

    mediaRecorder.start();
    updateRecordingUI(true);

    recordingTimer = setInterval(() => {
      secondsElapsed += 1;
      updateTimerDisplay(secondsElapsed);
    }, 1000);

    maxDurationTimer = setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
        if (typeof showToast === 'function') {
          showToast('Reached maximum 30-second recording limit. Uploading...', 'info');
        }
      }
    }, 30000);
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
    }
    clearInterval(recordingTimer);
    clearTimeout(maxDurationTimer);
  }

  function cleanupStream() {
    if (activeStream) {
      activeStream.getTracks().forEach(track => track.stop());
      activeStream = null;
    }
  }

  function updateRecordingUI(isRecording) {
    const micBtn = document.getElementById('mic-btn');
    if (!micBtn) return;

    if (isRecording) {
      micBtn.classList.add('recording');
      micBtn.setAttribute('title', 'Stop Recording (00:00)');
      updateTimerDisplay(0);
    } else {
      micBtn.classList.remove('recording');
      micBtn.setAttribute('title', 'Record Audio Dispatch');
      const timerSpan = micBtn.querySelector('.mic-timer');
      if (timerSpan) timerSpan.remove();
    }
  }

  function resetRecordingUI() {
    updateRecordingUI(false);
  }

  function updateTimerDisplay(sec) {
    const micBtn = document.getElementById('mic-btn');
    if (!micBtn) return;

    const formatted = `00:${sec < 10 ? '0' : ''}${sec}`;
    micBtn.setAttribute('title', `Recording... (${formatted}) Click to stop`);

    let timerSpan = micBtn.querySelector('.mic-timer');
    if (!timerSpan) {
      timerSpan = document.createElement('span');
      timerSpan.className = 'mic-timer';
      timerSpan.style.cssText = 'position:absolute; bottom:-16px; font-size:10px; font-weight:bold; color:#ff1744;';
      micBtn.style.position = 'relative';
      micBtn.appendChild(timerSpan);
    }
    timerSpan.innerText = formatted;
  }

  async function uploadAudioDispatch(blob, mimeType) {
    if (typeof showToast === 'function') {
      showToast('Processing audio dispatch via Gemini...', 'info');
    }

    const formData = new FormData();
    const ext = mimeType.includes('ogg') ? 'ogg' : mimeType.includes('mp4') ? 'mp4' : 'webm';
    formData.append('file', blob, `dispatch_recording.${ext}`);
    formData.append('mime_type', mimeType);

    try {
      const resp = await fetch('/api/incidents/audio', {
        method: 'POST',
        body: formData
      });

      if (!resp.ok) {
        const errData = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(errData.detail || 'Audio transcription failed');
      }

      const result = await resp.json();

      if (result.status === 'needs_confirmation') {
        const extData = result.extraction;
        const intakeInput = document.getElementById('intake-text');
        if (intakeInput && extData.transcript) {
          intakeInput.value = extData.transcript;
        }

        if (typeof displayConfirmationCard === 'function') {
          displayConfirmationCard(extData, 'voice');
        }
        if (typeof showToast === 'function') {
          showToast('Voice dispatch transcribed. Please review and confirm triage details.', 'info');
        }
      } else {
        if (typeof showToast === 'function') {
          showToast(`Voice incident dispatched: ${result.incident?.id || 'OK'}`, 'success');
        }
        if (result.plan_result && typeof checkAndDisplayApproval === 'function') {
          checkAndDisplayApproval(result.plan_result);
        }
        if (typeof fetchInitialState === 'function') {
          fetchInitialState();
        }
      }
    } catch (err) {
      if (typeof showToast === 'function') {
        showToast(`Audio ingestion error: ${err.message}`, 'error');
      }
    }
  }
})();