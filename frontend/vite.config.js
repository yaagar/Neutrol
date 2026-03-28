import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],

  // Tauri requires a fixed port so the webview knows where to connect in dev mode.
  // clearScreen: false keeps Vite output visible alongside Tauri's Rust output.
  server: {
    port: 5173,
    strictPort: true,
    host: '127.0.0.1',
  },
  clearScreen: false,
})
