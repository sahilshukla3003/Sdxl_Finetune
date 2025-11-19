# SDXL LoRA Fine-Tuning: Naruto Style Transfer

**Fast, memory-efficient fine-tuning of Stable Diffusion XL with LoRA for anime-style image generation.**

[![Python](https://img.shields.io/badge/python-https://img.shields.io/badge/pytorch
[![CUDA](https://img.shields.io 🎯 Overview

This project provides **two optimized training scripts** for fine-tuning Stable Diffusion XL (SDXL) using Low-Rank Adaptation (LoRA) to generate images in Naruto anime style. Both scripts are production-ready and support different GPU configurations.

**Key Features:**
- ✅ Two training modes: Standard (14GB) and Memory-Optimized (12GB)
- ✅ Full SDXL 1024×1024 resolution support
- ✅ EMA (Exponential Moving Average) for better quality
- ✅ Min-SNR weighted loss for improved convergence
- ✅ Disk-based caching for minimal memory usage
- ✅ Complete setup automation and verification

***

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone ...
cd sdxl-naruto-training

# Run automated setup
chmod +x setup.sh
./setup.sh

# Verify installation
conda activate sdxl_py310
python verify_setup.py
```

### 2. Choose Training Script

| Script | GPU Requirement | Resolution | Training Time |
|--------|----------------|------------|---------------|
| `train_sdxl.py` | 14GB VRAM | 1024×1024 | 2-3 hours |
| `train_opt.py` | 12GB VRAM | 768×768 | 3-4 hours |

### 3. Train the Model

```bash
# For 14GB+ GPUs (higher quality, faster)
python train_sdxl.py

# For 8-12GB GPUs (memory-optimized)
python train_opt.py
```

### 4. **Convert LoRA (Required Before Inference)**

```bash
# Convert trained LoRA to SD-compatible format
python convert_lora.py 
```

⚠️ **Important:** Always run `convert_lora.py` before using the LoRA model for inference. This ensures compatibility with standard Stable Diffusion pipelines.

### 5. Run Inference

```bash
# Fast inference (7-8 GB VRAM, no speed loss)
python inference_fast.py

# Memory-optimized inference (1-4 GB VRAM)
python opt_inference.py
```

***

## 📋 Requirements

### Hardware
- **GPU:** 12GB+ VRAM (RTX 3060 12GB, RTX 4070 Ti, RTX 4080/4090)
- **RAM:** 16GB+ system memory
- **Storage:** 50GB+ free space (SSD recommended)

### Software
- **OS:** Linux (Ubuntu 20.04+)
- **CUDA:** 11.8
- **Python:** 3.10
- **PyTorch:** 2.1.1+cu118

***

## 📦 Installation

### Automated Setup

```bash
bash setup.sh
```

This will:
1. Create conda environment `sdxl_py310`
2. Install PyTorch with CUDA 11.8
3. Install all dependencies
4. Configure CUDA library paths
5. Verify installation

### Manual Setup

```bash
# Create environment
conda create -n sdxl_py310 python=3.10 -y
conda activate sdxl_py310

# Install PyTorch
pip install torch==2.1.1 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

# Install xformers
pip install xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Set CUDA path
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

***

## 🎓 Training

### Standard Training (train_sdxl.py)

**Best for:** Maximum quality, 14GB VRAM

```bash
python train_sdxl.py
```

**Features:**
- 1024×1024 native SDXL resolution
- LoRA rank 16 (better capacity)
- On-the-fly text encoding (faster)
- 2-3 hour training time

### Memory-Optimized Training (train_opt.py)

**Best for:** 8-12GB VRAM, production deployment

```bash
python train_opt.py
```

**Features:**
- 768×768 resolution (saves 3GB VRAM)
- LoRA rank 8 (memory-efficient)
- Disk-based text caching
- EMA model for better quality
- Min-SNR weighted loss
- 3-4 hour training time

### Configuration

Edit the `TrainingConfig` class in either script:

```python
class TrainingConfig:
    resolution = 1024              # Image resolution
    lora_rank = 16                 # LoRA capacity
    learning_rate = 1e-4           # Training speed
    max_train_steps = 2000         # Total steps
    num_train_epochs = 10          # Training epochs
```

***

## 🔄 LoRA Conversion

**Before using your trained LoRA for inference, convert it to SD-compatible format:**

```bash
python convert_lora.py \
```

**Why convert?**
- Ensures compatibility with all SD pipelines
- Optimizes weight format
- Validates LoRA structure
- Required for ComfyUI, AUTOMATIC1111, and other UIs

***

## 🖼️ Inference

### Fast Inference (Recommended)

**5-6GB VRAM, 20-30% faster than standard**

```bash
python inference.py
```

**Features:**
- torch.compile() optimization
- SDPA attention
- Channels-last memory format
- QKV projection fusion
- Minimal VAE tiling

### Memory-Optimized Inference

**4-6GB VRAM, slower but works on low-end GPUs**

```bash
python opt_inference.py
```

**Features:**
- Sequential CPU offload
- VAE slicing and tiling
- Works on 8GB GPUs (RTX 3060 8GB, RTX 4060)

### Custom Inference

```python
from diffusers import StableDiffusionXLPipeline
import torch

# Load pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# Load converted LoRA
pipe.load_lora_weights("./output/sdxl-naruto-lora-converted")
pipe.fuse_lora(lora_scale=1.0)

# Generate
image = pipe(
    prompt="Naruto Uzumaki eating ramen in anime style",
    height=1024,
    width=1024,
    num_inference_steps=50
).images[0]

image.save("output.png")
```

### Adjust LoRA Strength

```python
# Subtle effect
pipe.fuse_lora(lora_scale=0.7)

# Standard (trained strength)
pipe.fuse_lora(lora_scale=1.0)

# Strong effect
pipe.fuse_lora(lora_scale=1.5)
```

***

## 📊 Results

| Model | Resolution | LoRA Rank | Final Loss | Quality |
|-------|-----------|-----------|------------|---------|
| train_sdxl.py | 1024×1024 | 16 | 0.07 | Excellent |
| train_opt.py | 768×768 | 8 | 0.06 | Very Good |

**Sample Prompts:**
- "Naruto Uzumaki eating ramen in Naruto anime style"
- "A ninja with blue eyes and spiky hair under cherry blossoms"
- "Sakura Haruno in a forest, anime style"

***

## 🔍 Troubleshooting

### Out of Memory (OOM)

```bash
# Use memory-optimized script
python train_opt.py

# Or reduce resolution
resolution = 512  # In TrainingConfig
```

### Slow Training

```bash
# Check GPU utilization
nvidia-smi -l 1

# Ensure fast SSD for cache
df -h ./cache
```

### Import Errors

```bash
# Reinstall huggingface_hub
pip install huggingface_hub==0.19.4 --force-reinstall
```

### LoRA Not Loading

```bash
# Verify conversion
python convert_lora.py --validate ./output/sdxl-naruto-lora-converted

# Check file structure
ls -lh ./output/sdxl-naruto-lora-converted/
```

***

## 📚 Documentation

- **Detailed Methodology:** See `documentation_methodology.md`
- **Training Comparison:** Comparison of both training scripts
- **Setup Guide:** Complete environment setup instructions
- **Inference Options:** All inference configurations

***

## 📄 License

This project uses:
- **SDXL Base Model:** CreativeML Open RAIL++-M License
- **Training Scripts:** MIT License
- **Dataset:** lambdalabs/naruto-blip-captions

***

## 🙏 Acknowledgments

- **Stability AI** - SDXL base model
- **Hugging Face** - Diffusers library
- **Lambda Labs** - Naruto dataset
- **Community** - LoRA training methods

***

## 📞 Support

- **Issues:** Open a GitHub issue
- **Documentation:** See `documentation_methodology.md`
- **Verification:** Run `python verify_setup.py`

***

[1](https://github.com/Avaray/stable-diffusion-templates)
[2](https://github.com/nuwandda/sdxl-lora-training)
[3](https://huggingface.co/dog-god/texture-synthesis-sdxl-lora/blob/main/README.md)
[4](https://www.datacamp.com/tutorial/fine-tuning-stable-diffusion-xl-with-dreambooth-and-lora)
[5](https://www.digitalocean.com/community/tutorials/training-a-lora-model-for-stable-diffusion-xl-with-paperspace)
[6](https://creatixai.com/how-to-train-lora-locally-kohya-tutorial-sdxl/)
[7](https://stable-diffusion-art.com/train-lora/)
[8](https://huggingface.co/AiWise/sdxl-faetastic-details_v24/blob/main/README.md)
[9](https://www.youtube.com/watch?v=d4QJg4YPm1c)