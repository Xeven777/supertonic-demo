import * as ort from 'onnxruntime-web';

// Pre-compiled regex patterns for performance
const EMOJI_PATTERN = /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F1E6}-\u{1F1FF}]+/gu;
const DIACRITICS_PATTERN = /[\u0302\u0303\u0304\u0305\u0306\u0307\u0308\u030A\u030B\u030C\u0327\u0328\u0329\u032A\u032B\u032C\u032D\u032E\u032F]/g;
const SPECIAL_SYMBOLS_PATTERN = /[♥☆♡©\\]/g;
const SPACING_PUNCTUATION_PATTERN = / ([,.!?;:'])/g;
const MULTIPLE_SPACES_PATTERN = /\s+/g;
const ENDING_PUNCTUATION_PATTERN = /[.!?;:,'"')\]}…。」』】〉》›»]$/;
const SENTENCE_SPLIT_PATTERN = /(?<!Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Sr\.|Jr\.|Ph\.D\.|etc\.|e\.g\.|i\.e\.|vs\.|Inc\.|Ltd\.|Co\.|Corp\.|St\.|Ave\.|Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+/;

// Character replacement maps (pre-defined for performance)
const CHAR_REPLACEMENTS = {
    '–': '-', '‑': '-', '—': '-', '¯': ' ', '_': ' ',
    '"': '"', '"': '"', '\u2018': "'", '\u2019': "'",
    '´': "'", '`': "'", '[': ' ', ']': ' ', '|': ' ',
    '/': ' ', '#': ' ', '→': ' ', '←': ' ',
};
const EXPR_REPLACEMENTS = {
    '@': ' at ',
    'e.g.,': 'for example, ',
    'i.e.,': 'that is, ',
};
const DUPLICATE_QUOTES = [['""', '"'], ["''", "'"], ['``', '`']];

/**
 * Unicode Text Processor
 */
export class UnicodeProcessor {
    constructor(indexer) {
        this.indexer = indexer;
    }

    call(textList) {
        const processedTexts = textList.map(text => this.preprocessText(text));

        const textIdsLengths = processedTexts.map(text => text.length);
        const maxLen = Math.max(...textIdsLengths);

        const textIds = processedTexts.map(text => {
            const row = new Array(maxLen).fill(0);
            for (let j = 0; j < text.length; j++) {
                const codePoint = text.codePointAt(j);
                row[j] = (codePoint < this.indexer.length) ? this.indexer[codePoint] : -1;
            }
            return row;
        });

        const textMask = this.getTextMask(textIdsLengths);
        return { textIds, textMask };
    }

    preprocessText(text) {
        // TODO: Need advanced normalizer for better performance
        text = text.normalize('NFKD');

        // FIXME: this should be fixed for non-English languages

        // Remove emojis (wide Unicode range) - using pre-compiled pattern
        text = text.replace(EMOJI_PATTERN, '');

        // Replace various dashes and symbols using pre-defined map
        for (const [k, v] of Object.entries(CHAR_REPLACEMENTS)) {
            text = text.replaceAll(k, v);
        }

        // Remove combining diacritics - using pre-compiled pattern
        text = text.replace(DIACRITICS_PATTERN, '');

        // Remove special symbols - using pre-compiled pattern
        text = text.replace(SPECIAL_SYMBOLS_PATTERN, '');

        // Replace known expressions using pre-defined map
        for (const [k, v] of Object.entries(EXPR_REPLACEMENTS)) {
            text = text.replaceAll(k, v);
        }

        // Fix spacing around punctuation - single pass
        text = text.replace(SPACING_PUNCTUATION_PATTERN, '$1');

        // Remove duplicate quotes using pre-defined patterns
        for (const [pattern, replacement] of DUPLICATE_QUOTES) {
            while (text.includes(pattern)) {
                text = text.replace(pattern, replacement);
            }
        }

        // Remove extra spaces - using pre-compiled pattern
        text = text.replace(MULTIPLE_SPACES_PATTERN, ' ').trim();

        // If text doesn't end with punctuation, quotes, or closing brackets, add a period
        if (!ENDING_PUNCTUATION_PATTERN.test(text)) {
            text += '.';
        }

        return text;
    }

    getTextMask(textIdsLengths) {
        const maxLen = Math.max(...textIdsLengths);
        return this.lengthToMask(textIdsLengths, maxLen);
    }

    lengthToMask(lengths, maxLen = null) {
        const actualMaxLen = maxLen || Math.max(...lengths);
        return lengths.map(len => {
            // Use regular Array for compatibility with .flat()
            const row = new Array(actualMaxLen).fill(0.0);
            const fillLen = Math.min(len, actualMaxLen);
            for (let j = 0; j < fillLen; j++) {
                row[j] = 1.0;
            }
            return [row];
        });
    }
}

/**
 * Style class to hold TTL and DP tensors
 */
export class Style {
    constructor(ttlTensor, dpTensor) {
        this.ttl = ttlTensor;
        this.dp = dpTensor;
    }
}

/**
 * Text-to-Speech class
 */
export class TextToSpeech {
    constructor(cfgs, textProcessor, dpOrt, textEncOrt, vectorEstOrt, vocoderOrt) {
        this.cfgs = cfgs;
        this.textProcessor = textProcessor;
        this.dpOrt = dpOrt;
        this.textEncOrt = textEncOrt;
        this.vectorEstOrt = vectorEstOrt;
        this.vocoderOrt = vocoderOrt;
        this.sampleRate = cfgs.ae.sample_rate;
    }

    async _infer(textList, style, totalStep, speed = 1.05, progressCallback = null) {
        const bsz = textList.length;

        // Process text
        const { textIds, textMask } = this.textProcessor.call(textList);

        // Optimized: Direct BigInt64Array creation without intermediate array
        const textLen = textIds[0].length;
        const textIdsFlat = new BigInt64Array(bsz * textLen);
        for (let b = 0; b < bsz; b++) {
            for (let j = 0; j < textLen; j++) {
                textIdsFlat[b * textLen + j] = BigInt(textIds[b][j]);
            }
        }
        const textIdsShape = [bsz, textLen];
        const textIdsTensor = new ort.Tensor('int64', textIdsFlat, textIdsShape);

        const textMaskFlat = new Float32Array(textMask.flat(2));
        const textMaskShape = [bsz, 1, textMask[0][0].length];
        const textMaskTensor = new ort.Tensor('float32', textMaskFlat, textMaskShape);

        // Track tensors for disposal
        const tensorsToDispose = [textIdsTensor, textMaskTensor];

        try {
            // Predict duration
            const dpOutputs = await this.dpOrt.run({
                text_ids: textIdsTensor,
                style_dp: style.dp,
                text_mask: textMaskTensor
            });
            const duration = Array.from(dpOutputs.duration.data);

            // Apply speed factor to duration (optimized with single loop)
            for (let i = 0; i < duration.length; i++) {
                duration[i] /= speed;
            }

            // Encode text
            const textEncOutputs = await this.textEncOrt.run({
                text_ids: textIdsTensor,
                style_ttl: style.ttl,
                text_mask: textMaskTensor
            });
            const textEmb = textEncOutputs.text_emb;

            // Sample noisy latent
            let { xt, latentMask } = this.sampleNoisyLatent(
                duration,
                this.sampleRate,
                this.cfgs.ae.base_chunk_size,
                this.cfgs.ttl.chunk_compress_factor,
                this.cfgs.ttl.latent_dim
            );

            const latentMaskFlat = new Float32Array(latentMask.flat(2));
            const latentMaskShape = [bsz, 1, latentMask[0][0].length];
            const latentMaskTensor = new ort.Tensor('float32', latentMaskFlat, latentMaskShape);
            tensorsToDispose.push(latentMaskTensor);

            // Prepare constant tensors (reused across denoising loop)
            const totalStepArray = new Float32Array(bsz).fill(totalStep);
            const totalStepTensor = new ort.Tensor('float32', totalStepArray, [bsz]);
            tensorsToDispose.push(totalStepTensor);

            // Pre-allocate reusable arrays for denoising loop
            const latentDim = xt[0].length;
            const latentLen = xt[0][0].length;
            const xtFlatSize = bsz * latentDim * latentLen;
            let xtFlat = new Float32Array(xtFlatSize);
            const currentStepArray = new Float32Array(bsz);

            // Denoising loop
            for (let step = 0; step < totalStep; step++) {
                if (progressCallback) {
                    progressCallback(step + 1, totalStep);
                }

                currentStepArray.fill(step);
                const currentStepTensor = new ort.Tensor('float32', currentStepArray, [bsz]);

                // Flatten xt directly into pre-allocated array
                let idx = 0;
                for (let b = 0; b < bsz; b++) {
                    for (let d = 0; d < latentDim; d++) {
                        for (let t = 0; t < latentLen; t++) {
                            xtFlat[idx++] = xt[b][d][t];
                        }
                    }
                }

                const xtShape = [bsz, latentDim, latentLen];
                const xtTensor = new ort.Tensor('float32', xtFlat, xtShape);

                const vectorEstOutputs = await this.vectorEstOrt.run({
                    noisy_latent: xtTensor,
                    text_emb: textEmb,
                    style_ttl: style.ttl,
                    latent_mask: latentMaskTensor,
                    text_mask: textMaskTensor,
                    current_step: currentStepTensor,
                    total_step: totalStepTensor
                });

                // Dispose step-specific tensors immediately
                currentStepTensor.dispose();
                xtTensor.dispose();

                const denoised = vectorEstOutputs.denoised_latent.data;

                // Reshape to 3D (reuse xt array structure)
                let reshapeIdx = 0;
                for (let b = 0; b < bsz; b++) {
                    for (let d = 0; d < latentDim; d++) {
                        for (let t = 0; t < latentLen; t++) {
                            xt[b][d][t] = denoised[reshapeIdx++];
                        }
                    }
                }
            }

            // Generate waveform
            let finalIdx = 0;
            for (let b = 0; b < bsz; b++) {
                for (let d = 0; d < latentDim; d++) {
                    for (let t = 0; t < latentLen; t++) {
                        xtFlat[finalIdx++] = xt[b][d][t];
                    }
                }
            }
            const finalXtShape = [bsz, latentDim, latentLen];
            const finalXtTensor = new ort.Tensor('float32', xtFlat, finalXtShape);
            tensorsToDispose.push(finalXtTensor);

            const vocoderOutputs = await this.vocoderOrt.run({
                latent: finalXtTensor
            });

            const wav = Array.from(vocoderOutputs.wav_tts.data);

            return { wav, duration };
        } finally {
            // Dispose all tracked tensors
            for (const tensor of tensorsToDispose) {
                try {
                    tensor.dispose();
                } catch (e) {
                    // Tensor may already be disposed
                }
            }
        }
    }

    async call(text, style, totalStep, speed = 1.05, silenceDuration = 0.3, progressCallback = null) {
        if (style.ttl.dims[0] !== 1) {
            throw new Error('Single speaker text to speech only supports single style');
        }
        const textList = chunkText(text);
        let wavCat = null;
        let durCat = 0;

        for (const chunk of textList) {
            const { wav, duration } = await this._infer([chunk], style, totalStep, speed, progressCallback);

            if (wavCat === null) {
                wavCat = wav;
                durCat = duration[0];
            } else {
                // Optimized: Use TypedArray concatenation instead of spread operator
                const silenceLen = Math.floor(silenceDuration * this.sampleRate);
                const newWav = new Float32Array(wavCat.length + silenceLen + wav.length);
                newWav.set(wavCat instanceof Float32Array ? wavCat : new Float32Array(wavCat), 0);
                // Silence is already zeros in TypedArray
                newWav.set(wav instanceof Float32Array ? wav : new Float32Array(wav), wavCat.length + silenceLen);
                wavCat = newWav;
                durCat += duration[0] + silenceDuration;
            }
        }

        return { wav: wavCat || [], duration: [durCat] };
    }

    async batch(textList, style, totalStep, speed = 1.05, progressCallback = null) {
        return await this._infer(textList, style, totalStep, speed, progressCallback);
    }

    sampleNoisyLatent(duration, sampleRate, baseChunkSize, chunkCompress, latentDim) {
        const bsz = duration.length;
        const maxDur = Math.max(...duration);

        const wavLenMax = Math.floor(maxDur * sampleRate);
        const wavLengths = duration.map(d => Math.floor(d * sampleRate));

        const chunkSize = baseChunkSize * chunkCompress;
        const latentLen = Math.floor((wavLenMax + chunkSize - 1) / chunkSize);
        const latentDimVal = latentDim * chunkCompress;

        // Pre-allocate 3D array structure
        const xt = new Array(bsz);
        for (let b = 0; b < bsz; b++) {
            xt[b] = new Array(latentDimVal);
            for (let d = 0; d < latentDimVal; d++) {
                // Use Float32Array for better memory efficiency
                xt[b][d] = new Float32Array(latentLen);
            }
        }

        // Optimized Box-Muller transform with batch random generation
        const totalValues = bsz * latentDimVal * latentLen;
        const TWO_PI = 2.0 * Math.PI;

        let idx = 0;
        for (let b = 0; b < bsz; b++) {
            for (let d = 0; d < latentDimVal; d++) {
                for (let t = 0; t < latentLen; t++) {
                    const u1 = Math.max(0.0001, Math.random());
                    const u2 = Math.random();
                    xt[b][d][t] = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(TWO_PI * u2);
                }
            }
        }

        const latentLengths = wavLengths.map(len => Math.floor((len + chunkSize - 1) / chunkSize));
        const latentMask = this.lengthToMask(latentLengths, latentLen);

        // Apply mask
        for (let b = 0; b < bsz; b++) {
            for (let d = 0; d < latentDimVal; d++) {
                for (let t = 0; t < latentLen; t++) {
                    xt[b][d][t] *= latentMask[b][0][t];
                }
            }
        }

        return { xt, latentMask };
    }

    lengthToMask(lengths, maxLen = null) {
        const actualMaxLen = maxLen || Math.max(...lengths);
        return lengths.map(len => {
            // Use regular Array for compatibility with .flat()
            const row = new Array(actualMaxLen).fill(0.0);
            const fillLen = Math.min(len, actualMaxLen);
            for (let j = 0; j < fillLen; j++) {
                row[j] = 1.0;
            }
            return [row];
        });
    }
}

/**
 * Load voice style from JSON files
 */
export async function loadVoiceStyle(voiceStylePaths, verbose = false) {
    const bsz = voiceStylePaths.length;

    // Read first file to get dimensions
    const firstResponse = await fetch(voiceStylePaths[0]);
    const firstStyle = await firstResponse.json();

    const ttlDims = firstStyle.style_ttl.dims;
    const dpDims = firstStyle.style_dp.dims;

    const ttlDim1 = ttlDims[1];
    const ttlDim2 = ttlDims[2];
    const dpDim1 = dpDims[1];
    const dpDim2 = dpDims[2];

    // Pre-allocate arrays with full batch size
    const ttlSize = bsz * ttlDim1 * ttlDim2;
    const dpSize = bsz * dpDim1 * dpDim2;
    const ttlFlat = new Float32Array(ttlSize);
    const dpFlat = new Float32Array(dpSize);

    // Fill in the data
    for (let i = 0; i < bsz; i++) {
        const response = await fetch(voiceStylePaths[i]);
        const voiceStyle = await response.json();

        // Flatten TTL data
        const ttlData = voiceStyle.style_ttl.data.flat(Infinity);
        const ttlOffset = i * ttlDim1 * ttlDim2;
        ttlFlat.set(ttlData, ttlOffset);

        // Flatten DP data
        const dpData = voiceStyle.style_dp.data.flat(Infinity);
        const dpOffset = i * dpDim1 * dpDim2;
        dpFlat.set(dpData, dpOffset);
    }

    const ttlShape = [bsz, ttlDim1, ttlDim2];
    const dpShape = [bsz, dpDim1, dpDim2];

    const ttlTensor = new ort.Tensor('float32', ttlFlat, ttlShape);
    const dpTensor = new ort.Tensor('float32', dpFlat, dpShape);

    if (verbose) {
        console.log(`Loaded ${bsz} voice styles`);
    }

    return new Style(ttlTensor, dpTensor);
}

/**
 * Load configuration from JSON
 */
export async function loadCfgs(onnxDir) {
    const response = await fetch(`${onnxDir}/tts.json`);
    const cfgs = await response.json();
    return cfgs;
}

/**
 * Load text processor
 */
export async function loadTextProcessor(onnxDir) {
    const response = await fetch(`${onnxDir}/unicode_indexer.json`);
    const indexer = await response.json();
    return new UnicodeProcessor(indexer);
}

/**
 * Load ONNX model
 */
export async function loadOnnx(onnxPath, options) {
    const session = await ort.InferenceSession.create(onnxPath, options);
    return session;
}

/**
 * Load all TTS components
 */
export async function loadTextToSpeech(onnxDir, sessionOptions = {}, progressCallback = null) {
    console.log('Using WebAssembly/WebGPU for inference');

    const cfgs = await loadCfgs(onnxDir);

    const dpPath = `${onnxDir}/duration_predictor.onnx`;
    const textEncPath = `${onnxDir}/text_encoder.onnx`;
    const vectorEstPath = `${onnxDir}/vector_estimator.onnx`;
    const vocoderPath = `${onnxDir}/vocoder.onnx`;

    const modelPaths = [
        { name: 'Duration Predictor', path: dpPath },
        { name: 'Text Encoder', path: textEncPath },
        { name: 'Vector Estimator', path: vectorEstPath },
        { name: 'Vocoder', path: vocoderPath }
    ];

    const sessions = [];
    for (let i = 0; i < modelPaths.length; i++) {
        if (progressCallback) {
            progressCallback(modelPaths[i].name, i + 1, modelPaths.length);
        }
        const session = await loadOnnx(modelPaths[i].path, sessionOptions);
        sessions.push(session);
    }

    const [dpOrt, textEncOrt, vectorEstOrt, vocoderOrt] = sessions;

    const textProcessor = await loadTextProcessor(onnxDir);
    const textToSpeech = new TextToSpeech(cfgs, textProcessor, dpOrt, textEncOrt, vectorEstOrt, vocoderOrt);

    return { textToSpeech, cfgs };
}

/**
 * Chunk text into manageable segments
 */
function chunkText(text, maxLen = 300) {
    if (typeof text !== 'string') {
        throw new Error(`chunkText expects a string, got ${typeof text}`);
    }

    // Split by paragraph (two or more newlines)
    const paragraphs = text.trim().split(/\n\s*\n+/).filter(p => p.trim());

    const chunks = [];

    for (let paragraph of paragraphs) {
        paragraph = paragraph.trim();
        if (!paragraph) continue;

        // Split by sentence boundaries using pre-compiled pattern
        const sentences = paragraph.split(SENTENCE_SPLIT_PATTERN);

        let currentChunk = "";

        for (let sentence of sentences) {
            if (currentChunk.length + sentence.length + 1 <= maxLen) {
                currentChunk += (currentChunk ? " " : "") + sentence;
            } else {
                if (currentChunk) {
                    chunks.push(currentChunk.trim());
                }
                currentChunk = sentence;
            }
        }

        if (currentChunk) {
            chunks.push(currentChunk.trim());
        }
    }

    return chunks;
}

/**
 * Write WAV file to ArrayBuffer
 */
export function writeWavFile(audioData, sampleRate) {
    const numChannels = 1;
    const bitsPerSample = 16;
    const byteRate = sampleRate * numChannels * bitsPerSample / 8;
    const blockAlign = numChannels * bitsPerSample / 8;
    const dataSize = audioData.length * 2;

    // Create ArrayBuffer
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    // Write WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    // Write audio data
    const int16Data = new Int16Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        const clamped = Math.max(-1.0, Math.min(1.0, audioData[i]));
        int16Data[i] = Math.floor(clamped * 32767);
    }

    const dataView = new Uint8Array(buffer, 44);
    dataView.set(new Uint8Array(int16Data.buffer));

    return buffer;
}
