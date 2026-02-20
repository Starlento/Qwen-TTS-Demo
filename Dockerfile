# Use Python slim base image (3.10 recommended for PyTorch)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Install PyTorch with CUDA 12.9 support first
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129

# Install other requirements
RUN pip install \
    transformers==4.57.3 \
    accelerate==1.12.0 \
    einops \
    gradio \
    librosa \
    soundfile \
    sox \
    onnxruntime \
    spaces \
    numpy \
    huggingface_hub \
    kernels

# Copy the application code
COPY . .

# Expose Gradio default port
EXPOSE 7860

# Set HF_HOME to the mounted models directory
ENV HF_HOME=/models

# Run the application
CMD ["python", "app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]
