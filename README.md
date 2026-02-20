---
title: Qwen3-TTS Demo
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: apache-2.0
suggested_hardware: zero-a10g
---

# Qwen3-TTS

Text-to-Speech model based on Qwen3 architecture with voice cloning and voice design capabilities.

## 🚀 Quick Start

### Option 1: Docker Setup (Recommended)

#### Using Pre-built Docker Image

A pre-built Docker image is available on DockerHub at `starlento/qwen3-tts:latest`.

**Prerequisites:**
- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed

**Steps:**

1. Start the container:
```bash
docker-compose up -d
```

The application will be available at `http://localhost:7860`

**Available Apps:**
- `app_custom_voice.py` - Custom voice synthesis (default)
- `app_voice_clone.py` - Voice cloning
- `app_voice_design.py` - Voice design

To use a different app, modify the `command` line in [docker-compose.yaml](docker-compose.yaml):
```yaml
command: python app_voice_clone.py --server-name 0.0.0.0 --server-port 7860
```

### Option 2: Host Setup with UV

#### Prerequisites
- Python 3.10
- NVIDIA GPU with CUDA 12.9 support
- UV package manager (will be installed automatically if not present)

#### Installation Steps

2. Run the UV setup script:
```bash
chmod +x setup_uv_env.sh
./setup_uv_env.sh
```

The script will:
- Install UV if not already installed
- Create a virtual environment with Python 3.10
- Install PyTorch 2.8.0 with CUDA 12.9 support
- Install all required dependencies

3. Activate the environment:
```bash
source .venv/bin/activate
```

4. Run the application:
```bash
# Default app
python app.py

# Or choose a specific app
python app_custom_voice.py
python app_voice_clone.py
python app_voice_design.py
```

## ⚠️ Important Notes

### Flash Attention
**Note:** Flash Attention is **NOT** installed by default in either setup method. If you need flash-attn for optimized attention mechanisms, you'll need to install it manually.

### Model Storage
Models are downloaded to:
- **Docker**: `/models` directory (mapped to `~/models` on host)
- **Host**: Default HuggingFace cache (`~/.cache/huggingface`)

## 📦 Docker Image

- **DockerHub**: `starlento/qwen3-tts:latest`
- **Base**: Python 3.10-slim
- **Includes**: PyTorch 2.8.0 with CUDA 12.9 support
- **Size**: ~8GB
