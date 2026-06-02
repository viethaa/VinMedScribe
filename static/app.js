'use strict';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const micBtn         = document.getElementById('micBtn');
const micIcon        = document.getElementById('micIcon');
const stopIcon       = document.getElementById('stopIcon');
const recordCard     = document.getElementById('recordCard');
const recordLabel    = document.getElementById('recordLabel');
const recordTimer    = document.getElementById('recordTimer');
const waveform       = document.getElementById('waveform');
const waveBars       = Array.from(waveform.querySelectorAll('.wave-bar'));
const uploadCard     = document.getElementById('uploadCard');
const fileInput      = document.getElementById('fileInput');
const uploadFilename = document.getElementById('uploadFilename');
const submitBtn      = document.getElementById('submitBtn');
const inputSection   = document.getElementById('inputSection');
const processingCard = document.getElementById('processingCard');
const processingStep = document.getElementById('processingStep');
const resultsSection = document.getElementById('resultsSection');
const transcriptText = document.getElementById('transcriptText');
const elapsedBadge   = document.getElementById('elapsedBadge');
const soapCard       = document.getElementById('soapCard');
const soapText       = document.getElementById('soapText');
const newBtn         = document.getElementById('newBtn');
const modelBadge     = document.getElementById('modelBadge');
const errorToast     = document.getElementById('errorToast');
const errorMsg       = document.getElementById('errorMsg');

// ── State ─────────────────────────────────────────────────────────────────────
let state = 'idle'; // idle | recording | processing | results
let mediaRecorder = null;
let audioChunks = [];
let pendingFile = null;     // File from upload or recording
let timerInterval = null;
let timerSeconds = 0;
let audioCtx = null;
let analyser = null;
let animationId = null;
let micStream = null;

// ── Health polling ─────────────────────────────────────────────────────────────
async function pollHealth() {
  try {
    const r = await fetch('/api/health');
    if (!r.ok) return;
    const data = await r.json();
    if (data.model_loaded) {
      modelBadge.textContent = '';
      const dot = document.createElement('span');
      dot.className = 'badge-dot';
      modelBadge.appendChild(dot);
      modelBadge.appendChild(document.createTextNode(' PhoWhisper · Ready'));
      modelBadge.classList.add('ready');
    } else {
      setTimeout(pollHealth, 3000);
    }
  } catch {
    setTimeout(pollHealth, 5000);
  }
}
pollHealth();

// ── Timer ─────────────────────────────────────────────────────────────────────
function startTimer() {
  timerSeconds = 0;
  recordTimer.textContent = '00:00';
  timerInterval = setInterval(() => {
    timerSeconds++;
    const m = String(Math.floor(timerSeconds / 60)).padStart(2, '0');
    const s = String(timerSeconds % 60).padStart(2, '0');
    recordTimer.textContent = `${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
}

// ── Waveform visualizer ───────────────────────────────────────────────────────
function startVisualizer(stream) {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 64;
  const src = audioCtx.createMediaStreamSource(stream);
  src.connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);

  function draw() {
    animationId = requestAnimationFrame(draw);
    analyser.getByteFrequencyData(data);
    waveBars.forEach((bar, i) => {
      const idx = Math.floor(i * data.length / waveBars.length);
      const pct = data[idx] / 255;
      const h = 4 + pct * 28;
      bar.style.height = h + 'px';
    });
  }
  draw();
}

function stopVisualizer() {
  if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
  if (audioCtx)    { audioCtx.close(); audioCtx = null; }
  waveBars.forEach(b => { b.style.height = '4px'; });
}

// ── Recording ─────────────────────────────────────────────────────────────────
micBtn.addEventListener('click', () => {
  if (state === 'idle') {
    startRecording();
  } else if (state === 'recording') {
    stopRecording();
  }
});

async function startRecording() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    showError('Microphone access denied. Please allow microphone access and try again.');
    return;
  }

  audioChunks = [];
  const mimeType = getSupportedMimeType();
  mediaRecorder = new MediaRecorder(micStream, mimeType ? { mimeType } : {});

  mediaRecorder.ondataavailable = e => {
    if (e.data.size > 0) audioChunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    const ext = mimeType ? mimeTypeToExt(mimeType) : 'webm';
    const blob = new Blob(audioChunks, { type: mimeType || 'audio/webm' });
    pendingFile = new File([blob], `recording.${ext}`, { type: mimeType || 'audio/webm' });
    micStream.getTracks().forEach(t => t.stop());
    micStream = null;
    setSubmitReady(true);
    // Auto-submit after recording
    handleSubmit();
  };

  mediaRecorder.start();
  setState('recording');
  startTimer();
  startVisualizer(micStream);
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
  stopTimer();
  stopVisualizer();
  setState('idle');
  recordLabel.textContent = 'Processing…';
}

function getSupportedMimeType() {
  const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4'];
  return types.find(t => MediaRecorder.isTypeSupported(t)) || '';
}

function mimeTypeToExt(mime) {
  if (mime.includes('mp4'))  return 'mp4';
  if (mime.includes('ogg'))  return 'ogg';
  return 'webm';
}

// ── Upload ────────────────────────────────────────────────────────────────────
uploadCard.addEventListener('click', () => fileInput.click());
uploadCard.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

uploadCard.addEventListener('dragover', e => {
  e.preventDefault();
  uploadCard.classList.add('drag-over');
});
uploadCard.addEventListener('dragleave', () => uploadCard.classList.remove('drag-over'));
uploadCard.addEventListener('drop', e => {
  e.preventDefault();
  uploadCard.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

function setFile(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['wav', 'mp3', 'm4a', 'webm', 'ogg', 'mp4'].includes(ext)) {
    showError(`Unsupported file type ".${ext}". Use .wav, .mp3, or .m4a`);
    return;
  }
  pendingFile = file;
  uploadFilename.textContent = file.name;
  uploadCard.classList.add('has-file');
  setSubmitReady(true);
}

function setSubmitReady(ready) {
  submitBtn.disabled = !ready;
}

// ── Submit ────────────────────────────────────────────────────────────────────
submitBtn.addEventListener('click', handleSubmit);

async function handleSubmit() {
  if (!pendingFile) return;
  setState('processing');

  const steps = [
    'Loading model & preparing audio',
    'Running speech recognition…',
    'Generating SOAP note…',
  ];
  let stepIdx = 0;
  processingStep.textContent = steps[stepIdx];
  const stepInterval = setInterval(() => {
    stepIdx = (stepIdx + 1) % steps.length;
    processingStep.textContent = steps[stepIdx];
  }, 4000);

  try {
    const form = new FormData();
    form.append('file', pendingFile, pendingFile.name);

    const res = await fetch('/api/transcribe', { method: 'POST', body: form });
    clearInterval(stepInterval);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }

    const data = await res.json();
    showResults(data);
  } catch (err) {
    clearInterval(stepInterval);
    setState('idle');
    showError(err.message || 'Transcription failed. Please try again.');
  }
}

// ── Results ───────────────────────────────────────────────────────────────────
function showResults(data) {
  transcriptText.textContent = data.transcript || '(no transcript returned)';
  elapsedBadge.textContent = data.elapsed_seconds ? `Completed in ${data.elapsed_seconds}s` : '';

  if (data.soap_note) {
    soapText.innerHTML = formatSoap(data.soap_note);
    soapCard.hidden = false;
  } else {
    soapCard.hidden = true;
  }

  setState('results');
}

function formatSoap(raw) {
  // Bold SOAP section headers (lines starting with **...**)
  return raw
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

// ── New recording ─────────────────────────────────────────────────────────────
newBtn.addEventListener('click', resetToIdle);

function resetToIdle() {
  pendingFile = null;
  audioChunks = [];
  uploadFilename.textContent = '';
  uploadCard.classList.remove('has-file');
  fileInput.value = '';
  recordLabel.textContent = 'Click to record';
  recordTimer.textContent = '';
  setSubmitReady(false);
  setState('idle');
}

// ── State machine ─────────────────────────────────────────────────────────────
function setState(s) {
  state = s;
  inputSection.hidden   = (s === 'processing' || s === 'results');
  processingCard.hidden = (s !== 'processing');
  resultsSection.hidden = (s !== 'results');

  recordCard.classList.toggle('recording', s === 'recording');
  micIcon.style.display  = s === 'recording' ? 'none' : '';
  stopIcon.style.display = s === 'recording' ? '' : 'none';
  micBtn.setAttribute('aria-label', s === 'recording' ? 'Stop recording' : 'Start recording');
}

// ── Copy buttons ──────────────────────────────────────────────────────────────
document.querySelectorAll('.copy-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const target = document.getElementById(btn.dataset.target);
    if (!target) return;
    const text = target.innerText || target.textContent;
    try {
      await navigator.clipboard.writeText(text);
      btn.classList.add('copied');
      btn.querySelector('svg + span') && (btn.querySelector('svg + span').textContent = 'Copied!');
      const origHtml = btn.innerHTML;
      btn.innerHTML = btn.innerHTML.replace(/>Copy</, '>Copied!<');
      setTimeout(() => {
        btn.classList.remove('copied');
        btn.innerHTML = origHtml;
      }, 2000);
    } catch { /* clipboard not available */ }
  });
});

// ── Error toast ───────────────────────────────────────────────────────────────
let toastTimeout;
function showError(msg) {
  errorMsg.textContent = msg;
  errorToast.hidden = false;
  requestAnimationFrame(() => errorToast.classList.add('show'));
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => {
    errorToast.classList.remove('show');
    setTimeout(() => { errorToast.hidden = true; }, 300);
  }, 5000);
}
