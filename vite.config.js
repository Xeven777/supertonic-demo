import { defineConfig } from 'vite';

export default defineConfig({
  build: {
    target: 'esnext',
    minify: 'terser',
    rollupOptions: {
      output: {
        manualChunks: {
          onnx: ['onnxruntime-web']
        }
      }
    }
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  },
  worker: {
    format: 'es',
    rollupOptions: {
      output: {
        entryFileNames: '[name].js'
      }
    }
  }
});
