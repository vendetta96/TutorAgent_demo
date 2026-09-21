import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    proxy: {
      '/connect': { target: 'http://127.0.0.1:7860', changeOrigin: true },
      '/report': { target: 'http://127.0.0.1:7860', changeOrigin: true },
    },
  },
});
