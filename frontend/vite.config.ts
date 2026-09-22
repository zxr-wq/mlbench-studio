import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  base: process.env.VITE_BASE ?? '/mlbench-studio/',
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': { target: process.env.MLBENCH_API ?? 'http://127.0.0.1:8765', changeOrigin: true, ws: true },
    },
  },
})
