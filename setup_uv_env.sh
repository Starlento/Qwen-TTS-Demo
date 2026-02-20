#!/bin/bash
# Setup script for Qwen3-TTS using UV Python environment manager
# This script prepares the Python environment with PyTorch CUDA 12.9 support

set -e  # Exit on error

echo "=========================================="
echo "Qwen3-TTS Environment Setup with UV"
echo "=========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    echo "UV installed successfully!"
else
    echo "UV is already installed: $(uv --version)"
fi

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Working directory: $SCRIPT_DIR"

# Create UV virtual environment with Python 3.10 (recommended for PyTorch)
echo ""
echo "Creating UV virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv --python 3.10
    echo "Virtual environment created at .venv"
else
    echo "Virtual environment already exists at .venv"
fi

# Activate the virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip and setuptools
echo ""
echo "Upgrading pip and setuptools..."
uv pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.9 support
echo ""
echo "=========================================="
echo "Installing PyTorch 2.8.0 with CUDA 12.9..."
echo "=========================================="
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu129

# Verify PyTorch installation
echo ""
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install other dependencies from requirements.txt
echo ""
echo "=========================================="
echo "Installing other dependencies..."
echo "=========================================="

# Create a temporary requirements file excluding torch packages (already installed)
echo "Creating filtered requirements..."
grep -v "^torch" requirements.txt > requirements_filtered.txt || true
grep -v "^torchaudio" requirements_filtered.txt > temp.txt && mv temp.txt requirements_filtered.txt || true

# Install remaining dependencies
if [ -s requirements_filtered.txt ]; then
    echo "Installing dependencies:"
    cat requirements_filtered.txt
    uv pip install -r requirements_filtered.txt
    rm requirements_filtered.txt
else
    echo "No additional dependencies to install."
fi

# Install the local package if setup.py exists
if [ -f "setup.py" ]; then
    echo ""
    echo "Installing local package..."
    uv pip install -e .
fi

# Display installed packages
echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Installed packages:"
uv pip list

echo ""
echo "=========================================="
echo "Environment is ready!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the app:"
echo "  python app.py"
echo ""
