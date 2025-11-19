#!/usr/bin/env python3
"""
SDXL Simple Prompt Inference - Naruto Style Fine-tuned LoRA
"""

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
from pathlib import Path
import sys

# ========================== CONFIG ==========================
MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_PATH = "madebyollin/sdxl-vae-fp16-fix"
LORA_PATH = "./output/sdxl-naruto-lora-fixed"
CACHE_DIR = "./cache"
OUTPUT_DIR = "./output/simple_inference"

HEIGHT = 1024
WIDTH = 1024
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 50
SEED = 42

PROMPT = "A man with blue eyes and spiky hair, standing under cherry blossom trees"  # <-- CHANGE THIS
NEGATIVE_PROMPT = "low quality, blurry, distorted, western cartoon"

# ============================================================

Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

if not torch.cuda.is_available():
    print("⚠️ No GPU detected! Using CPU (slow).")
    device = "cpu"
    dtype = torch.float32
else:
    device = "cuda"
    dtype = torch.float16
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# === LOAD PIPELINE ===
vae = AutoencoderKL.from_pretrained(
    VAE_PATH, torch_dtype=dtype, cache_dir=CACHE_DIR
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_BASE,
    vae=vae,
    torch_dtype=dtype,
    use_safetensors=True,
    cache_dir=CACHE_DIR,
    variant="fp16" if dtype == torch.float16 else None,
)
pipe.to(device)

# Enable attention slicing for memory efficiency
if device == "cuda":
    pipe.enable_attention_slicing(1)

# === LOAD LORA ===
if not Path(LORA_PATH).exists():
    print(f"❌ LoRA weights not found at {LORA_PATH}")
    sys.exit(1)
print(f"✓ Loading LoRA weights from: {LORA_PATH}")
pipe.load_lora_weights(LORA_PATH)

# Fuse for speed (optional)
try:
    pipe.fuse_lora(lora_scale=1.0)
    print("✓ LoRA weights fused (scale=1.0)")
except Exception as e:
    print("✓ LoRA loaded in unfused mode")

# === GENERATE ===
generator = torch.Generator(device=device).manual_seed(SEED)
print(f"\nGenerating: '{PROMPT}'")

output = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    height=HEIGHT,
    width=WIDTH,
    guidance_scale=GUIDANCE_SCALE,
    num_inference_steps=NUM_INFERENCE_STEPS,
    generator=generator,
)
image = output.images[0]
filename = Path(OUTPUT_DIR) / "naruto_lora_output.png"
image.save(filename)
print(f"✓ Saved image: {filename}")

