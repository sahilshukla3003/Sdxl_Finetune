#!/bin/bash
# SDXL Training Environment Setup Script
# Based on working installation 

set -e

echo "================================================================================"
echo "SDXL Training Environment Setup - Step-by-Step"
echo "================================================================================"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}ERROR: This script is designed for Linux${NC}"
    exit 1
fi

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}ERROR: conda not found. Please install Anaconda or Miniconda first.${NC}"
    exit 1
fi

# Check CUDA 11.8
echo ""
echo "Step 1: Checking CUDA installation"
echo "--------------------------------------------------------------------------------"
if [ -d "/usr/local/cuda-11.8" ]; then
    echo -e "${GREEN}✓ CUDA 11.8 found at /usr/local/cuda-11.8${NC}"
else
    echo -e "${RED}ERROR: CUDA 11.8 not found at /usr/local/cuda-11.8${NC}"
    echo "Please install CUDA 11.8 from: https://developer.nvidia.com/cuda-11-8-0-download-archive"
    exit 1
fi

# Remove conflicting CUDA symlinks
echo ""
echo "Step 2: Removing conflicting CUDA versions"
echo "--------------------------------------------------------------------------------"
if [ -L "/usr/local/cuda-12.1" ]; then
    echo "Removing CUDA 12.1 symlink..."
    sudo rm -f /usr/local/cuda-12.1
    echo -e "${GREEN}✓ CUDA 12.1 symlink removed${NC}"
fi

# Create conda environment
echo ""
echo "Step 3: Creating conda environment: sdxl_py310"
echo "--------------------------------------------------------------------------------"
if conda env list | grep -q "^sdxl_py310 "; then
    echo -e "${YELLOW}Environment sdxl_py310 already exists. Removing...${NC}"
    conda env remove -n sdxl_py310 -y
fi
conda create -n sdxl_py310 python=3.10 -y
echo -e "${GREEN}✓ Environment created${NC}"

# Activate environment
echo ""
echo "Step 4: Activating environment"
echo "--------------------------------------------------------------------------------"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sdxl_py310
echo -e "${GREEN}✓ Environment activated${NC}"

# Clean install of PyTorch
echo ""
echo "Step 5: Installing PyTorch 2.1.1 with CUDA 11.8"
echo "--------------------------------------------------------------------------------"
pip uninstall torch torchvision torchaudio xformers bitsandbytes -y 2>/dev/null || true
pip install torch==2.1.1 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
echo -e "${GREEN}✓ PyTorch installed${NC}"

# Install xformers (will downgrade torch to 2.1.1 - this is expected)
echo ""
echo "Step 6: Installing xformers 0.0.23"
echo "--------------------------------------------------------------------------------"
pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118
echo -e "${GREEN}✓ xformers installed${NC}"
echo -e "${YELLOW}Note: torch may be downgraded to 2.1.1 - this is expected and correct${NC}"

# Install bitsandbytes
echo ""
echo "Step 7: Installing bitsandbytes"
echo "--------------------------------------------------------------------------------"
pip install bitsandbytes==0.41.3.post2
echo -e "${GREEN}✓ bitsandbytes installed${NC}"

# Set CUDA library path
echo ""
echo "Step 8: Configuring CUDA library path"
echo "--------------------------------------------------------------------------------"
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Add to .bashrc if not already there
if ! grep -q "LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo -e "${GREEN}✓ Added CUDA path to ~/.bashrc${NC}"
else
    echo -e "${YELLOW}CUDA path already in ~/.bashrc${NC}"
fi

# Install other dependencies
echo ""
echo "Step 9: Installing diffusers and other dependencies"
echo "--------------------------------------------------------------------------------"
pip install diffusers==0.25.0
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install peft==0.7.1
pip install datasets==2.15.0
pip install Pillow==10.1.0
pip install safetensors==0.4.1
pip install tensorboard==2.15.1
echo -e "${GREEN}✓ Core dependencies installed${NC}"

# Fix huggingface_hub compatibility
echo ""
echo "Step 10: Fixing huggingface_hub compatibility"
echo "--------------------------------------------------------------------------------"
pip install huggingface_hub==0.19.4 --force-reinstall
echo -e "${GREEN}✓ huggingface_hub downgraded to 0.19.4${NC}"

# Verify installation
echo ""
echo "Step 11: Verifying installation"
echo "--------------------------------------------------------------------------------"
python verify_setup.py

echo ""
echo "================================================================================"
echo "Setup complete!"
echo "================================================================================"
echo ""
echo "IMPORTANT: To activate the environment in new terminals:"
echo "  1. Run: conda activate sdxl_py310"
echo "  2. Run: export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:\$LD_LIBRARY_PATH"
echo ""
echo "Or just restart your terminal (LD_LIBRARY_PATH is in ~/.bashrc)"
echo ""
echo "To start training:"
echo "  python train_sdxl.py"
echo "================================================================================"
