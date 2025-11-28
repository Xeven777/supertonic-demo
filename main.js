import {
    loadTextToSpeech,
    loadVoiceStyle,
    writeWavFile
} from './helper.js';

// Configuration
const DEFAULT_VOICE_STYLE_PATH = 'assets/voice_styles/M1.json';
const USE_WORKER = true; // Enable web worker for better performance

// Track previous audio URL for cleanup
let previousAudioUrl = null;

// Worker instance
let ttsWorker = null;
let workerMessageId = 0;
let workerCallbacks = new Map();

// Debounce utility
function debounce(fn, ms) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), ms);
    };
}

// Helper function to extract filename from path
function getFilenameFromPath(path) {
    return path.split('/').pop();
}

// Global state (for non-worker mode)
let textToSpeech = null;
let cfgs = null;

// Pre-computed style
let currentStyle = null;
let currentStylePath = DEFAULT_VOICE_STYLE_PATH;

// UI Elements
const textInput = document.getElementById('text');
const voiceStyleSelect = document.getElementById('voiceStyleSelect');
const voiceStyleInfo = document.getElementById('voiceStyleInfo');
const totalStepInput = document.getElementById('totalStep');
const speedInput = document.getElementById('speed');
const generateBtn = document.getElementById('generateBtn');
const statusBox = document.getElementById('statusBox');
const statusText = document.getElementById('statusText');
const backendBadge = document.getElementById('backendBadge');
const resultsContainer = document.getElementById('results');
const errorBox = document.getElementById('error');

// Progress UI Elements
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const progressStep = document.getElementById('progressStep');
const progressPercent = document.getElementById('progressPercent');

function showStatus(message, type = 'info') {
    statusText.innerHTML = message;
    statusBox.className = 'status-card';
    if (type === 'success') {
        statusBox.classList.add('success');
    } else if (type === 'error') {
        statusBox.classList.add('error');
    }
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.add('active');
}

function hideError() {
    errorBox.classList.remove('active');
}

function updateBackendBadge(provider) {
    backendBadge.innerHTML = `<span class="badge-dot"></span>${provider}`;
    backendBadge.className = 'badge badge-success';
}

// Progress tracking functions
function showProgress() {
    progressContainer.classList.remove('hidden');
}

function hideProgress() {
    progressContainer.classList.add('hidden');
    updateProgress(0, 0, 0);
}

function updateProgress(step, total, overallPercent) {
    const percent = Math.round(overallPercent * 100);
    progressFill.style.width = `${percent}%`;
    progressStep.textContent = `Step ${step}/${total}`;
    progressPercent.textContent = `${percent}%`;
}

// Worker communication
function initWorker() {
    return new Promise((resolve, reject) => {
        ttsWorker = new Worker(new URL('./tts.worker.js', import.meta.url), { type: 'module' });

        ttsWorker.onmessage = (e) => {
            const { type, id, payload } = e.data;

            if (type === 'progress') {
                handleWorkerProgress(payload);
                return;
            }

            const callback = workerCallbacks.get(id);
            if (callback) {
                if (type === 'error') {
                    callback.reject(new Error(payload.message));
                } else {
                    callback.resolve({ type, payload });
                }
                workerCallbacks.delete(id);
            }
        };

        ttsWorker.onerror = (error) => {
            console.error('Worker error:', error);
            reject(error);
        };

        resolve();
    });
}

function handleWorkerProgress(payload) {
    if (payload.stage === 'loading') {
        showStatus(`ℹ️ <strong>Loading ONNX models (${payload.current}/${payload.total}):</strong> ${payload.modelName}...`);
    } else if (payload.stage === 'denoising') {
        const { step, total, chunkIndex, totalChunks, overallProgress } = payload;
        if (totalChunks > 1) {
            showStatus(`ℹ️ <strong>Processing chunk ${chunkIndex}/${totalChunks} - Denoising (${step}/${total})...</strong>`);
        } else {
            showStatus(`ℹ️ <strong>Denoising (${step}/${total})...</strong>`);
        }
        updateProgress(step, total, overallProgress);
    }
}

function sendWorkerMessage(type, payload) {
    return new Promise((resolve, reject) => {
        const id = ++workerMessageId;
        workerCallbacks.set(id, { resolve, reject });
        ttsWorker.postMessage({ type, payload, id });
    });
}

// Load voice style from JSON
async function loadStyleFromJSON(stylePath) {
    try {
        const style = await loadVoiceStyle([stylePath], true);
        return style;
    } catch (error) {
        console.error('Error loading voice style:', error);
        throw error;
    }
}

// Load models on page load
async function initializeModels() {
    try {
        showStatus('ℹ️ <strong>Loading configuration...</strong>');

        const basePath = 'assets/onnx';

        if (USE_WORKER) {
            // Initialize with Web Worker
            await initWorker();

            // Try WebGPU first, fallback to WASM
            let executionProvider = 'wasm';
            try {
                const result = await sendWorkerMessage('init', {
                    basePath,
                    sessionOptions: {
                        executionProviders: ['webgpu'],
                        graphOptimizationLevel: 'all'
                    }
                });

                executionProvider = result.payload.executionProvider;
                updateBackendBadge('WebGPU + Worker');
            } catch (webgpuError) {
                console.log('WebGPU not available, falling back to WebAssembly');

                const result = await sendWorkerMessage('init', {
                    basePath,
                    sessionOptions: {
                        executionProviders: ['wasm'],
                        graphOptimizationLevel: 'all'
                    }
                });

                executionProvider = result.payload.executionProvider;
                updateBackendBadge('WASM + Worker');
            }

            showStatus('ℹ️ <strong>Loading default voice style...</strong>');

            // Load default voice style in worker
            await sendWorkerMessage('loadStyle', { stylePath: currentStylePath });
            voiceStyleInfo.textContent = `${getFilenameFromPath(currentStylePath)} (default)`;

            showStatus(`✅ <strong>Models loaded!</strong> Using ${executionProvider.toUpperCase()} with Web Worker. You can now generate speech.`, 'success');
        } else {
            // Non-worker mode (fallback)
            let executionProvider = 'wasm';
            try {
                const result = await loadTextToSpeech(basePath, {
                    executionProviders: ['webgpu'],
                    graphOptimizationLevel: 'all'
                }, (modelName, current, total) => {
                    showStatus(`ℹ️ <strong>Loading ONNX models (${current}/${total}):</strong> ${modelName}...`);
                });

                textToSpeech = result.textToSpeech;
                cfgs = result.cfgs;

                executionProvider = 'webgpu';
                updateBackendBadge('WebGPU');
            } catch (webgpuError) {
                console.log('WebGPU not available, falling back to WebAssembly');

                const result = await loadTextToSpeech(basePath, {
                    executionProviders: ['wasm'],
                    graphOptimizationLevel: 'all'
                }, (modelName, current, total) => {
                    showStatus(`ℹ️ <strong>Loading ONNX models (${current}/${total}):</strong> ${modelName}...`);
                });

                textToSpeech = result.textToSpeech;
                cfgs = result.cfgs;
                updateBackendBadge('WASM');
            }

            showStatus('ℹ️ <strong>Loading default voice style...</strong>');

            // Load default voice style
            currentStyle = await loadStyleFromJSON(currentStylePath);
            voiceStyleInfo.textContent = `${getFilenameFromPath(currentStylePath)} (default)`;

            showStatus(`✅ <strong>Models loaded!</strong> Using ${executionProvider.toUpperCase()}. You can now generate speech.`, 'success');
        }

        generateBtn.disabled = false;

    } catch (error) {
        console.error('Error loading models:', error);
        showStatus(`❌ <strong>Error loading models:</strong> ${error.message}`, 'error');
    }
}

// Handle voice style selection
voiceStyleSelect.addEventListener('change', async (e) => {
    const selectedValue = e.target.value;

    if (!selectedValue) return;

    try {
        generateBtn.disabled = true;
        showStatus(`ℹ️ <strong>Loading voice style...</strong>`, 'info');

        currentStylePath = selectedValue;

        if (USE_WORKER && ttsWorker) {
            await sendWorkerMessage('loadStyle', { stylePath: currentStylePath });
        } else {
            currentStyle = await loadStyleFromJSON(currentStylePath);
        }

        voiceStyleInfo.textContent = getFilenameFromPath(currentStylePath);

        showStatus(`✅ <strong>Voice style loaded:</strong> ${getFilenameFromPath(currentStylePath)}`, 'success');
        generateBtn.disabled = false;
    } catch (error) {
        showError(`Error loading voice style: ${error.message}`);

        // Restore default style
        currentStylePath = DEFAULT_VOICE_STYLE_PATH;
        voiceStyleSelect.value = currentStylePath;
        try {
            if (USE_WORKER && ttsWorker) {
                await sendWorkerMessage('loadStyle', { stylePath: currentStylePath });
            } else {
                currentStyle = await loadStyleFromJSON(currentStylePath);
            }
            voiceStyleInfo.textContent = `${getFilenameFromPath(currentStylePath)} (default)`;
        } catch (styleError) {
            console.error('Error restoring default style:', styleError);
        }

        generateBtn.disabled = false;
    }
});

// Main synthesis function
async function generateSpeech() {
    const text = textInput.value.trim();
    if (!text) {
        showError('Please enter some text to synthesize.');
        return;
    }

    if (USE_WORKER) {
        if (!ttsWorker) {
            showError('Worker is still loading. Please wait.');
            return;
        }
    } else {
        if (!textToSpeech || !cfgs) {
            showError('Models are still loading. Please wait.');
            return;
        }

        if (!currentStyle) {
            showError('Voice style is not ready. Please wait.');
            return;
        }
    }

    const startTime = Date.now();

    try {
        generateBtn.disabled = true;
        hideError();
        showProgress();

        // Clear results and show placeholder
        resultsContainer.innerHTML = `
            <div class="results-placeholder generating">
                <div class="results-placeholder-icon">⏳</div>
                <p>Generating speech...</p>
            </div>
        `;

        const totalStep = parseInt(totalStepInput.value);
        const speed = parseFloat(speedInput.value);

        showStatus('ℹ️ <strong>Generating speech from text...</strong>');
        const tic = Date.now();

        let wavBuffer, duration, sampleRate;

        if (USE_WORKER && ttsWorker) {
            // Use worker for generation
            const result = await sendWorkerMessage('generate', {
                text,
                totalStep,
                speed,
                silenceDuration: 0.3
            });

            wavBuffer = result.payload.wavBuffer;
            duration = result.payload.duration;
            sampleRate = result.payload.sampleRate;
        } else {
            // Non-worker mode
            const result = await textToSpeech.call(
                text,
                currentStyle,
                totalStep,
                speed,
                0.3,
                (step, total) => {
                    showStatus(`ℹ️ <strong>Denoising (${step}/${total})...</strong>`);
                    updateProgress(step, total, step / total);
                }
            );

            const wavLen = Math.floor(textToSpeech.sampleRate * result.duration[0]);
            const wavOut = result.wav.slice(0, wavLen);
            wavBuffer = writeWavFile(wavOut, textToSpeech.sampleRate);
            duration = result.duration[0];
            sampleRate = textToSpeech.sampleRate;
        }

        const toc = Date.now();
        console.log(`Text-to-speech synthesis: ${((toc - tic) / 1000).toFixed(2)}s`);

        showStatus('ℹ️ <strong>Creating audio file...</strong>');

        const blob = new Blob([wavBuffer], { type: 'audio/wav' });

        // Revoke previous URL to prevent memory leak
        if (previousAudioUrl) {
            URL.revokeObjectURL(previousAudioUrl);
        }
        const url = URL.createObjectURL(blob);
        previousAudioUrl = url;

        // Calculate total time and audio duration
        const endTime = Date.now();
        const totalTimeSec = ((endTime - startTime) / 1000).toFixed(2);
        const audioDurationSec = duration.toFixed(2);

        // Hide progress bar
        hideProgress();

        // Display result with full text
        resultsContainer.innerHTML = `
            <div class="result-item">
                <div class="result-text-container">
                    <div class="result-text-label">Input Text</div>
                    <div class="result-text">${text}</div>
                </div>
                <div class="result-info">
                    <div class="info-item">
                        <span>📊 Audio Length</span>
                        <strong>${audioDurationSec}s</strong>
                    </div>
                    <div class="info-item">
                        <span>⏱️ Generation Time</span>
                        <strong>${totalTimeSec}s</strong>
                    </div>
                </div>
                <div class="result-player">
                    <audio controls>
                        <source src="${url}" type="audio/wav">
                    </audio>
                </div>
                <div class="result-actions">
                    <button onclick="downloadAudio('${url}', 'synthesized_speech.wav')">
                        <span>⬇️</span>
                        <span>Download WAV</span>
                    </button>
                </div>
            </div>
        `;

        showStatus('✅ <strong>Speech synthesis completed successfully!</strong>', 'success');

    } catch (error) {
        console.error('Error during synthesis:', error);
        showStatus(`❌ <strong>Error during synthesis:</strong> ${error.message}`, 'error');
        showError(`Error during synthesis: ${error.message}`);
        hideProgress();

        // Restore placeholder
        resultsContainer.innerHTML = `
            <div class="results-placeholder">
                <div class="results-placeholder-icon">🎤</div>
                <p>Generated speech will appear here</p>
            </div>
        `;
    } finally {
        generateBtn.disabled = false;
    }
}

// Download handler (make it global so it can be called from onclick)
window.downloadAudio = function (url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
};

// Attach generate function to button
generateBtn.addEventListener('click', generateSpeech);

// Character count update with debounce
const charCountEl = document.getElementById('charCount');
function updateCharCount() {
    charCountEl.textContent = textInput.value.length;
}
const debouncedCharCount = debounce(updateCharCount, 50);
textInput.addEventListener('input', debouncedCharCount);
updateCharCount();

// Range slider value updates with throttle
const totalStepValueEl = document.getElementById('totalStepValue');
const speedValueEl = document.getElementById('speedValue');

totalStepInput.addEventListener('input', debounce(() => {
    totalStepValueEl.textContent = totalStepInput.value;
}, 16));

speedInput.addEventListener('input', debounce(() => {
    speedValueEl.textContent = speedInput.value + 'x';
}, 16));

// Initialize on load
window.addEventListener('load', async () => {
    generateBtn.disabled = true;
    await initializeModels();
});
