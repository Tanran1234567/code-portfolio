import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // 让 build 后的资源路径变为相对路径（解决直接打开 dist/index.html 为空白的问题）
  base: './',
  server: {
    fs: {
      // 允许 dev server 读取仓库根目录（含 ../dataset）
      allow: [path.resolve(__dirname, '..')],
    },
    // dev-only：把仓库根目录的 dataset/ 暴露为 /dataset/...
    // 这样即使你只跑 frontend，也能 fetch('/dataset/demo-001.json')
    // （build 产物不包含 dataset，这个仅用于开发调试）
    middlewareMode: false,
    configureServer(server) {
      const repoRoot = path.resolve(__dirname, '..')
      const datasetDir = path.join(repoRoot, 'dataset')

      server.middlewares.use('/dataset', (req, res, next) => {
        try {
          // 仅支持 GET/HEAD
          if (req.method !== 'GET' && req.method !== 'HEAD') return next()
          if (!req.url) return next()

          // /demo-001.json or /demo-001
          const rawUrl = req.url.split('?')[0] || ''
          const rel = decodeURIComponent(rawUrl).replace(/^\/+/, '')
          if (!rel) return next()

          // 防目录穿越
          if (rel.includes('..') || path.isAbsolute(rel)) return next()

          const file = rel.toLowerCase().endsWith('.json') ? rel : `${rel}.json`
          const abs = path.join(datasetDir, file)
          if (!abs.startsWith(datasetDir)) return next()

          if (!fs.existsSync(abs) || !fs.statSync(abs).isFile()) return next()

          res.statusCode = 200
          res.setHeader('Content-Type', 'application/json; charset=utf-8')
          const text = fs.readFileSync(abs, 'utf-8')
          res.end(text)
        } catch (e) {
          next(e)
        }
      })
    },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
