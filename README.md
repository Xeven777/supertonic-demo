# Supertonic Web Example

This example demonstrates how to use Supertonic in a web browser using ONNX Runtime Web.

![screenshot](ss.png)

## Features

- 🌐 Runs entirely in the browser (no server required for inference)
- 🚀 WebGPU support with automatic fallback to WebAssembly
- ⚡ Pre-extracted voice styles for instant generation
- 🎨 Modern, responsive UI
- 🎭 Multiple voice style presets (2 Male, 2 Female)
- 💾 Download generated audio as WAV files
- 📊 Detailed generation statistics (audio length, generation time)
- ⏱️ Real-time progress tracking

## Requirements

- Node.js (for development server)
- Modern web browser (Chrome, Edge, Firefox, Safari)

## Installation

1. Install dependencies:

```bash
bun install
```

## Running the Demo

Start the development server:

```bash
bun run dev
```

## Prerequisites

Before running the examples, download the ONNX models and preset voices, and place them in the assets directory:

> Note: The Hugging Face repository uses Git LFS. Please ensure Git LFS is installed and initialized before cloning or pulling large model files.

- macOS: brew install git-lfs && git lfs install
- Generic: see https://git-lfs.com for installers

`git clone https://huggingface.co/Supertone/supertonic assets`

This will start a local development server (usually at http://localhost:5173) and open the demo in your browser.

## Usage

1. **Wait for Models to Load**: The app will automatically load models and the default voice style (M1)
2. **Select Voice Style**: Choose from available voice presets
   - **Male 1 (M1)**: Default male voice
   - **Male 2 (M2)**: Alternative male voice
   - **Female 1 (F1)**: Default female voice
   - **Female 2 (F2)**: Alternative female voice
3. **Enter Text**: Type or paste the text you want to convert to speech
4. **Adjust Settings** (optional):
   - **Total Steps**: More steps = better quality but slower (default: 5)
5. **Generate Speech**: Click the "Generate Speech" button
6. **View Results**:
   - See the full input text
   - View audio length and generation time statistics
   - Play the generated audio in the browser
   - Download as WAV file

## Technical Details

### Browser Compatibility

This demo uses:

- **ONNX Runtime Web**: For running models in the browser
- **Web Audio API**: For playing generated audio
- **Vite**: For development and bundling

## Notes

- The ONNX models must be accessible at `assets/onnx/` relative to the web root
- Voice style JSON files must be accessible at `assets/voice_styles/` relative to the web root
- Pre-extracted voice styles enable instant generation without audio processing
- Four voice style presets are provided (M1, M2, F1, F2)

## Troubleshooting

### Models not loading

- Check browser console for errors
- Ensure `assets/onnx/` path is correct and models are accessible
- Check CORS settings if serving from a different domain

### WebGPU not available

- WebGPU is only available in recent Chrome/Edge browsers (version 113+)
- The app will automatically fall back to WebAssembly if WebGPU is not available
- Check the backend badge to see which execution provider is being used

### Out of memory errors

- Try shorter text inputs
- Reduce denoising steps
- Use a browser with more available memory
- Close other tabs to free up memory

### Audio quality issues

- Try different voice style presets
- Increase denoising steps for better quality

### Slow generation

- If using WebAssembly, try a browser that supports WebGPU
- Ensure no other heavy processes are running
- Consider using fewer denoising steps for faster (but lower quality) results
