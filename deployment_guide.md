# Deployment Guide

The Farm-AI application consists of two components that must run simultaneously:
1.  **Streamlit App**: The frontend interface.
2.  **MCP Server**: A background process (DuckDuckGo Search) that the app communicates with.

Because of this multi-process requirement, **Docker** is the supported deployment method.

## 🐳 Deployment with Docker

You can deploy this Docker container to any platform that supports it (Render, Railway, AWS ECS, Google Cloud Run, DigitalOcean App Platform, etc.).

### 1. Build the Image
```bash
docker build -t farm-ai .
```

### 2. Run Locally (for testing)
```bash
docker run -p 8501:8501 --env-file .env farm-ai
```
Visit `http://localhost:8501` to use the app.

### 3. Deploy to Render (Example)
1.  Create a new **Web Service** on [Render](https://render.com).
2.  Connect your GitHub repository.
3.  Select **Docker** as the Runtime environment.
4.  Add your Environment Variables (OpenAI API Key, etc.).
5.  Deploy!

### 4. Deploy to Railway (Example)
1.  Create a new project on [Railway](https://railway.app).
2.  Deploy from GitHub repo.
3.  It will automatically detect the `Dockerfile` and build it.
4.  Add your variables in the "Variables" tab.

### NOTE:
The `Dockerfile` uses `entrypoint.sh` to start both the MCP server (port 8000) and the Streamlit app (port 8501) in the same container.
