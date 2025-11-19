#!/usr/bin/env python3
"""
SDXL Inference - Base vs Fine-tuned Model Comparison
Matches the training script configuration
"""

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys

print("="*80)
print("SDXL INFERENCE - BASE vs FINE-TUNED COMPARISON")
print("="*80)

# Configuration (matching train.py)
MODEL_BASE = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_PATH = "madebyollin/sdxl-vae-fp16-fix"
LORA_PATH = "./output/sdxl-naruto-lora-fixed"  # Final model, not checkpoint
CACHE_DIR = "./cache"
OUTPUT_DIR = "./output/inference_results"

# Create output directory
Path(OUTPUT_DIR).mkdir(exist_ok=True, parents=True)

# Test prompts
# prompts = [
#     "Naruto Uzumaki eating ramen in Naruto anime style",
#     "Bill Gates as a character in Naruto anime style",
#     "A boy with blue eyes in Naruto anime style",
#     "Sakura Haruno in a cherry blossom forest, Naruto style",
# ]
prompts = [
    "A man in a blue and white outfit",  # No "in Naruto style"
    "a woman with a sword in her hand",     # Photorealistic
    "A teenage boy with blue eyes",  # Generic
    "a dog in a blue shirt laying on the ground",
]

# Generation parameters
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
HEIGHT = 1024
WIDTH = 1024
SEED = 42

# Check GPU
if not torch.cuda.is_available():
    print("⚠️ WARNING: No GPU detected! Using CPU (very slow)")
    device = "cpu"
    dtype = torch.float32
else:
    device = "cuda"
    dtype = torch.float16
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("="*80)

# Check if LoRA weights exist
if not Path(LORA_PATH).exists():
    print(f"\n❌ Error: LoRA weights not found at {LORA_PATH}")
    print("Please train the model first using: python train_sdxl.py")
    sys.exit(1)

# ====================================================================
# LOAD BASE MODEL
# ====================================================================

print("\nLoading base SDXL model...")

try:
    # Load VAE
    vae = AutoencoderKL.from_pretrained(
        VAE_PATH,
        torch_dtype=dtype,
        cache_dir=CACHE_DIR
    )
    
    # Load base pipeline
    base_pipeline = StableDiffusionXLPipeline.from_pretrained(
        MODEL_BASE,
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
        cache_dir=CACHE_DIR,
        variant="fp16" if dtype == torch.float16 else None,
    )
    base_pipeline.to(device)
    
    # Memory optimizations
    if device == "cuda":
        # Use attention slicing for memory efficiency
        base_pipeline.enable_attention_slicing(1)
        print("✓ Attention slicing enabled")
        
        # Only use CPU offload if low on VRAM
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb < 16:
            try:
                base_pipeline.enable_model_cpu_offload()
                print("✓ Model CPU offload enabled")
            except Exception as e:
                print(f"⚠️ CPU offload not available: {e}")
    
    print("✓ Base model loaded successfully")
    
except Exception as e:
    print(f"❌ Error loading base model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================
# GENERATE WITH BASE MODEL
# ====================================================================

print("\n" + "="*80)
print("GENERATING WITH BASE MODEL")
print("="*80 + "\n")

base_images = []

for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] Generating: {prompt}")
    
    try:
        # Create generator with unique seed per image
        generator = torch.Generator(device=device).manual_seed(SEED + i)
        
        image = base_pipeline(
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted",
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        ).images[0]
        
        base_images.append(image)
        output_path = f"{OUTPUT_DIR}/base_{i+1}.png"
        image.save(output_path)
        print(f"✓ Saved: {output_path}\n")
        
    except Exception as e:
        print(f"❌ Error generating base image: {e}\n")
        import traceback
        traceback.print_exc()
        base_images.append(Image.new('RGB', (WIDTH, HEIGHT), color='gray'))

# ====================================================================
# LOAD FINE-TUNED MODEL (LoRA)
# ====================================================================

print("="*80)
print("LOADING FINE-TUNED MODEL")
print("="*80 + "\n")

try:
    # Load LoRA weights
    print(f"Loading LoRA weights from: {LORA_PATH}")
    base_pipeline.load_lora_weights(LORA_PATH)
    
    # Fuse LoRA weights for faster inference (optional)
    try:
        base_pipeline.fuse_lora(lora_scale=1.0)
        print("✓ LoRA weights fused (scale: 1.0)")
    except:
        print("✓ LoRA weights loaded (unfused mode)")
    
    finetuned_pipeline = base_pipeline
    print("✓ Fine-tuned LoRA model ready")
    
except Exception as e:
    print(f"❌ Error loading LoRA weights: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ====================================================================
# GENERATE WITH FINE-TUNED MODEL
# ====================================================================

print("\n" + "="*80)
print("GENERATING WITH FINE-TUNED MODEL (NARUTO STYLE)")
print("="*80 + "\n")

finetuned_images = []

for i, prompt in enumerate(prompts):
    print(f"[{i+1}/{len(prompts)}] Generating: {prompt}")
    
    try:
        # Use same seed as base model for fair comparison
        generator = torch.Generator(device=device).manual_seed(SEED + i)
        
        image = finetuned_pipeline(
            prompt=prompt,
            negative_prompt="low quality, blurry, distorted, western cartoon",
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            height=HEIGHT,
            width=WIDTH,
            generator=generator,
        ).images[0]
        
        finetuned_images.append(image)
        output_path = f"{OUTPUT_DIR}/finetuned_{i+1}.png"
        image.save(output_path)
        print(f"✓ Saved: {output_path}\n")
        
    except Exception as e:
        print(f"❌ Error generating fine-tuned image: {e}\n")
        import traceback
        traceback.print_exc()
        finetuned_images.append(Image.new('RGB', (WIDTH, HEIGHT), color='gray'))

# ====================================================================
# CREATE COMPARISON GRID
# ====================================================================

print("="*80)
print("CREATING COMPARISON GRID")
print("="*80 + "\n")

try:
    fig, axes = plt.subplots(len(prompts), 2, figsize=(16, 5.5 * len(prompts)))
    
    if len(prompts) == 1:
        axes = axes.reshape(1, -1)
    
    for i, prompt in enumerate(prompts):
        # Base model column
        axes[i, 0].imshow(base_images[i])
        axes[i, 0].set_title(
            f"Base SDXL Model\n\n{prompt}", 
            fontsize=10, 
            pad=10,
            wrap=True
        )
        axes[i, 0].axis('off')
        
        # Fine-tuned model column
        axes[i, 1].imshow(finetuned_images[i])
        axes[i, 1].set_title(
            f"Fine-tuned (Naruto Style)\n\n{prompt}", 
            fontsize=10, 
            pad=10,
            wrap=True
        )
        axes[i, 1].axis('off')
    
    plt.suptitle(
        "SDXL Comparison: Base vs Naruto Style Fine-tuned", 
        fontsize=16, 
        fontweight='bold', 
        y=0.998
    )
    plt.tight_layout()
    
    comparison_path = f"{OUTPUT_DIR}/comparison_grid.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    print(f"✓ Comparison grid saved: {comparison_path}")
    
    # Try to display
    try:
        plt.show()
    except:
        print("(Display not available in headless mode)")
    
    plt.close()

except Exception as e:
    print(f"⚠️ Warning: Could not create comparison grid: {e}")
    import traceback
    traceback.print_exc()

# ====================================================================
# CREATE INDIVIDUAL SIDE-BY-SIDE COMPARISONS
# ====================================================================

print("\nCreating individual side-by-side comparisons...")

for i, prompt in enumerate(prompts):
    try:
        # Create side-by-side image
        total_width = WIDTH * 2 + 40  # 40px gap
        max_height = HEIGHT + 80  # Extra for text
        
        combined = Image.new('RGB', (total_width, max_height), color='white')
        combined.paste(base_images[i], (0, 40))
        combined.paste(finetuned_images[i], (WIDTH + 40, 40))
        
        output_path = f"{OUTPUT_DIR}/comparison_{i+1}.png"
        combined.save(output_path)
        print(f"✓ Saved: {output_path}")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create comparison {i+1}: {e}")

print("\n" + "="*80)
print("INFERENCE COMPLETE!")
print("="*80)
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print(f"  - Base model images: base_*.png")
print(f"  - Fine-tuned images: finetuned_*.png")
print(f"  - Side-by-side comparisons: comparison_*.png")
print(f"  - Full comparison grid: comparison_grid.png")
print("\nTo use a specific checkpoint instead of final model:")
print(f"  Set LORA_PATH = './output/sdxl-naruto-lora/checkpoint-1000'")
print("="*80)

# Cleanup
del base_pipeline, finetuned_pipeline
if device == "cuda":
    torch.cuda.empty_cache()

print("\n✓ Done!")
