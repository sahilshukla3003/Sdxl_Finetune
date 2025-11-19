#!/usr/bin/env python3
"""
SDXL Simple Prompt Inference - Memory Optimized (<6GB VRAM)
Naruto Style Fine-tuned LoRA
"""

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
from pathlib import Path
import sys
import gc

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

PROMPT = "A man with blue eyes and spiky hair, standing under cherry blossom trees"
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

print("\n" + "="*60)
print("MEMORY-OPTIMIZED INFERENCE MODE")
print("="*60)

# === LOAD PIPELINE ===
print("\nLoading models...")

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

# ============================================================
# MEMORY OPTIMIZATION TECHNIQUES
# ============================================================

if device == "cuda":
    print("\nApplying memory optimizations...")
    
    # 1. Attention Slicing (saves ~1GB)
    pipe.enable_attention_slicing(1)
    print("✓ Attention slicing enabled")
    
    # 2. VAE Slicing (saves ~1.5GB)
    pipe.enable_vae_slicing()
    print("✓ VAE slicing enabled")
    
    # 3. VAE Tiling (saves additional ~1GB for large images)
    pipe.enable_vae_tiling()
    print("✓ VAE tiling enabled")
    
    # 4. Sequential CPU Offload (saves ~3-4GB)
    # Moves model components to CPU when not in use
    pipe.enable_sequential_cpu_offload()
    print("✓ Sequential CPU offload enabled")
    
    # Note: Don't use pipe.to(device) after enable_sequential_cpu_offload()
    print("\nExpected VRAM usage: 4-6GB (vs 8GB without optimizations)")
else:
    pipe.to(device)

# === LOAD LORA ===
print("\nLoading LoRA...")
if not Path(LORA_PATH).exists():
    print(f"❌ LoRA weights not found at {LORA_PATH}")
    sys.exit(1)

pipe.load_lora_weights(LORA_PATH)
print(f"✓ LoRA weights loaded from: {LORA_PATH}")

# Don't fuse LoRA - unfused is more memory efficient
print("✓ LoRA kept unfused for memory efficiency")

# === GENERATE ===
print("\n" + "="*60)
print(f"Generating: '{PROMPT}'")
print("="*60)

generator = torch.Generator(device="cpu" if device == "cuda" else device).manual_seed(SEED)

# Clear cache before generation
if device == "cuda":
    torch.cuda.empty_cache()
    gc.collect()

# Monitor memory
if device == "cuda":
    torch.cuda.reset_peak_memory_stats()
    start_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"\nMemory before generation: {start_memory:.2f} GB")

output = pipe(
    prompt=PROMPT,
    negative_prompt=NEGATIVE_PROMPT,
    height=HEIGHT,
    width=WIDTH,
    guidance_scale=GUIDANCE_SCALE,
    num_inference_steps=NUM_INFERENCE_STEPS,
    generator=generator,
)

# Show peak memory usage
if device == "cuda":
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak memory during generation: {peak_memory:.2f} GB")

image = output.images[0]
filename = Path(OUTPUT_DIR) / "naruto_lora_output.png"
image.save(filename)

print(f"\n✓ Image saved: {filename}")
print("="*60)

# Cleanup
if device == "cuda":
    torch.cuda.empty_cache()
    gc.collect()
