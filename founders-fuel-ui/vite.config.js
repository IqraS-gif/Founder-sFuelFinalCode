// vite.config.js

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react({
      // This is the critical part that makes styled-components work
      babel: {
        plugins: [
          'babel-plugin-styled-components',
        ],
      },
    }),
  ],
})