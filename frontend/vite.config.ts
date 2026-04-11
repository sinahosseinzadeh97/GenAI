import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/query": { target: "http://api:8000", changeOrigin: true },
      "/schema": { target: "http://api:8000", changeOrigin: true },
      "/cache": { target: "http://api:8000", changeOrigin: true },
      "/history": { target: "http://api:8000", changeOrigin: true },
      "/health": { target: "http://api:8000", changeOrigin: true }
    }
  }
});
