import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// 该 SPA 会被 FastAPI 挂载到 /app 路径下服务，静态资源必须带 /app/ 前缀
export default defineConfig({
  base: '/app/',
  plugins: [react(), tailwindcss()],
})
