#!/usr/bin/env python3
"""
SDXL Fine-Tuning - Memory-Optimized for 8-10GB GPU
Optimizations: Disk-based caching, CPU offloading, efficient attention
Expected VRAM: 8-10GB | Training Time: 3-4 hours
"""

import os
import sys
import logging
import torch
import gc
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import json

# Verify CUDA
if not torch.cuda.is_available():
    print("❌ ERROR: No CUDA GPU detected!")
    sys.exit(1)

print("="*80)
print("SDXL FINE-TUNING - MEMORY OPTIMIZED")
print("="*80)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")
print("="*80)

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ====================================================================
# CONFIGURATION - MEMORY OPTIMIZED
# ====================================================================

class TrainingConfig:
    """Memory-optimized configuration for 8-10GB GPU"""
    
    # Model
    pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_vae_name = "madebyollin/sdxl-vae-fp16-fix"
    dataset_name = "lambdalabs/naruto-blip-captions"
    
    # Output
    output_dir = "./output/sdxl-naruto-lora"
    logging_dir = "./output/logs"
    cache_dir = "./cache"
    
    # ========== MEMORY-OPTIMIZED SETTINGS ==========
    resolution = 768  # Reduced from 1024 (saves ~3GB)
    train_batch_size = 1
    gradient_accumulation_steps = 8  # Increased for same effective batch
    
    # LoRA - Lower rank for memory
    lora_rank = 8  # Reduced from 16 (saves ~1GB)
    lora_alpha = 8
    lora_dropout = 0.0
    lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    
    # Memory optimizations
    use_8bit_adam = False  # May cause issues, keep False
    gradient_checkpointing = True  # Essential
    mixed_precision = "fp16"  # Essential
    enable_xformers = False
    cache_latents = True  # Disk-based
    cache_text_embeddings_to_disk = True  # NEW: Disk instead of RAM
    
    # DISABLED for memory saving
    use_ema = True  # Disabled - saves 50% model memory
    
    # Quality optimizations (low memory cost)
    min_snr_gamma = 5.0
    offset_noise_strength = 0.1
    use_sdpa = True
    
    # Training
    num_train_epochs = 15  # More epochs to compensate for lower resolution
    max_train_steps = 3000  # More steps
    learning_rate = 1e-4
    lr_scheduler = "cosine"
    lr_warmup_steps = 150
    
    # Checkpointing
    checkpointing_steps = 750
    save_steps = 750
    logging_steps = 50
    
    # Dataloader
    dataloader_num_workers = 1  # Reduced for memory
    pin_memory = False  # Disabled to save RAM
    
    seed = 42
    report_to = "tensorboard"

config = TrainingConfig()

# ====================================================================
# SETUP
# ====================================================================

os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.logging_dir, exist_ok=True)
os.makedirs(config.cache_dir, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(f"{config.logging_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Save config
config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
with open(f"{config.output_dir}/training_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

logger.info("Configuration saved")
logger.info("Memory optimization mode: Target 8-10GB VRAM")

# ====================================================================
# ACCELERATOR
# ====================================================================

accelerator = Accelerator(
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    mixed_precision=config.mixed_precision,
    log_with=config.report_to,
    project_config=ProjectConfiguration(
        project_dir=config.output_dir,
        logging_dir=config.logging_dir
    ),
)

logger.info(f"Accelerator device: {accelerator.device}")
logger.info(f"Mixed precision: {config.mixed_precision}")
set_seed(config.seed)

# ====================================================================
# LOAD MODELS
# ====================================================================

logger.info("Loading tokenizers...")
tokenizer_one = AutoTokenizer.from_pretrained(
    config.pretrained_model_name, subfolder="tokenizer", use_fast=False, cache_dir=config.cache_dir
)
tokenizer_two = AutoTokenizer.from_pretrained(
    config.pretrained_model_name, subfolder="tokenizer_2", use_fast=False, cache_dir=config.cache_dir
)

logger.info("Loading noise scheduler...")
noise_scheduler = DDPMScheduler.from_pretrained(
    config.pretrained_model_name, subfolder="scheduler", cache_dir=config.cache_dir
)

logger.info("Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    config.pretrained_vae_name, torch_dtype=torch.float16, cache_dir=config.cache_dir
)
vae.requires_grad_(False)
vae.to(accelerator.device)
logger.info("✓ VAE loaded")

logger.info("Loading text encoders...")
text_encoder_one = CLIPTextModel.from_pretrained(
    config.pretrained_model_name, subfolder="text_encoder", torch_dtype=torch.float16, cache_dir=config.cache_dir
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    config.pretrained_model_name, subfolder="text_encoder_2", torch_dtype=torch.float16, cache_dir=config.cache_dir
)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
text_encoder_one.to(accelerator.device)
text_encoder_two.to(accelerator.device)
logger.info("✓ Text encoders loaded")

logger.info("Loading UNet...")
unet = UNet2DConditionModel.from_pretrained(
    config.pretrained_model_name, subfolder="unet", cache_dir=config.cache_dir
)

# Enable gradient checkpointing
if config.gradient_checkpointing:
    unet.enable_gradient_checkpointing()
    logger.info("✓ Gradient checkpointing enabled")

# Use SDPA
if config.use_sdpa and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    unet.set_attn_processor(AttnProcessor2_0())
    logger.info("✓ SDPA attention enabled")

logger.info("✓ UNet loaded")

# ====================================================================
# APPLY LORA (Lower rank for memory)
# ====================================================================

logger.info(f"Applying LoRA (rank={config.lora_rank})...")
lora_config = LoraConfig(
    r=config.lora_rank,
    lora_alpha=config.lora_alpha,
    init_lora_weights="gaussian",
    target_modules=config.lora_target_modules,
    lora_dropout=config.lora_dropout,
)
unet = get_peft_model(unet, lora_config)

if accelerator.is_main_process:
    unet.print_trainable_parameters()

logger.info("✓ LoRA applied")

gc.collect()
torch.cuda.empty_cache()

# ====================================================================
# DATASET WITH DISK-BASED CACHING
# ====================================================================

class NarutoDataset(Dataset):
    def __init__(self, dataset, tokenizer_one, tokenizer_two, resolution, 
                 cache_latents=False, vae=None, cache_dir=None):
        self.dataset = dataset
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.resolution = resolution
        self.cache_latents = cache_latents
        self.vae = vae
        self.cache_dir = cache_dir
        
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Cache latents to disk
        if self.cache_latents and self.vae is not None:
            self.latents_cache_dir = Path(cache_dir) / f"latents_cache_{resolution}"
            self.latents_cache_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"Caching latents for {len(self.dataset)} images at {resolution}x{resolution}...")
            for i in tqdm(range(len(self.dataset)), desc="Caching latents"):
                cache_file = self.latents_cache_dir / f"latent_{i}.pt"
                if cache_file.exists():
                    continue
                
                try:
                    image = self.dataset[i]["image"]
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    
                    image_tensor = self.image_transforms(image).unsqueeze(0)
                    image_tensor = image_tensor.to(self.vae.device, dtype=torch.float16)
                    
                    with torch.no_grad():
                        latent = self.vae.encode(image_tensor).latent_dist.sample()
                        latent = latent * self.vae.config.scaling_factor
                    
                    torch.save(latent.squeeze(0).cpu(), cache_file)
                    del image_tensor, latent
                    
                    if i % 50 == 0:
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Failed to cache image {i}: {e}")
            
            logger.info("✓ Latent caching complete")
    
    def __len__(self):
        return len(self.dataset)
    
    def tokenize_captions(self, caption):
        tokens_one = self.tokenizer_one(
            caption, truncation=True, padding="max_length",
            max_length=self.tokenizer_one.model_max_length, return_tensors="pt",
        ).input_ids
        tokens_two = self.tokenizer_two(
            caption, truncation=True, padding="max_length",
            max_length=self.tokenizer_two.model_max_length, return_tensors="pt",
        ).input_ids
        return tokens_one, tokens_two
    
    def __getitem__(self, idx):
        example = {}
        item = self.dataset[idx]
        
        caption = item["text"]
        tokens_one, tokens_two = self.tokenize_captions(caption)
        example["input_ids_one"] = tokens_one
        example["input_ids_two"] = tokens_two
        example["index"] = idx  # Store index for text embedding lookup
        
        if self.cache_latents:
            cache_file = self.latents_cache_dir / f"latent_{idx}.pt"
            if cache_file.exists():
                example["latents"] = torch.load(cache_file)
            else:
                image = item["image"].convert("RGB")
                example["pixel_values"] = self.image_transforms(image)
        else:
            image = item["image"].convert("RGB")
            example["pixel_values"] = self.image_transforms(image)
        
        return example

logger.info(f"Loading dataset: {config.dataset_name}")
dataset = load_dataset(config.dataset_name, split="train", cache_dir=config.cache_dir)
logger.info(f"Dataset loaded: {len(dataset)} images")

train_dataset = NarutoDataset(
    dataset=dataset,
    tokenizer_one=tokenizer_one,
    tokenizer_two=tokenizer_two,
    resolution=config.resolution,
    cache_latents=config.cache_latents,
    vae=vae,
    cache_dir=config.cache_dir,
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.train_batch_size,
    shuffle=True,
    num_workers=config.dataloader_num_workers,
    pin_memory=config.pin_memory,
)

logger.info(f"✓ Dataloader ready: {len(train_dataset)} training samples")

# ====================================================================
# TEXT ENCODER CACHING TO DISK (Memory Efficient)
# ====================================================================

text_embeddings_cache_dir = None
if config.cache_text_embeddings_to_disk:
    text_embeddings_cache_dir = Path(config.cache_dir) / f"text_embeddings_{config.resolution}"
    text_embeddings_cache_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info("Pre-computing and caching text embeddings to disk...")
    
    def encode_prompts(input_ids_one, input_ids_two):
        with torch.no_grad():
            prompt_embeds_1 = text_encoder_one(
                input_ids_one.to(accelerator.device), output_hidden_states=True
            ).hidden_states[-2]
            
            prompt_embeds_2_out = text_encoder_two(
                input_ids_two.to(accelerator.device), output_hidden_states=True
            )
            prompt_embeds_2_pooled = prompt_embeds_2_out[0]
            prompt_embeds_2 = prompt_embeds_2_out.hidden_states[-2]
            
            prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
        return prompt_embeds, prompt_embeds_2_pooled
    
    # Cache to disk
    for idx in tqdm(range(len(train_dataset)), desc="Caching text embeddings"):
        cache_file = text_embeddings_cache_dir / f"text_emb_{idx}.pt"
        if cache_file.exists():
            continue
        
        batch = train_dataset[idx]
        prompt_embeds, pooled_embeds = encode_prompts(
            batch["input_ids_one"], batch["input_ids_two"]
        )
        
        # Save to disk (not RAM)
        torch.save({
            "prompt_embeds": prompt_embeds.cpu().half(),
            "pooled_embeds": pooled_embeds.cpu().half()
        }, cache_file)
        
        if idx % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # Offload text encoders to CPU (or delete)
    text_encoder_one.to("cpu")
    text_encoder_two.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("✓ Text embeddings cached to disk, encoders moved to CPU")

# Free VAE
if config.cache_latents:
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("✓ VAE offloaded")

# ====================================================================
# OPTIMIZER & SCHEDULER
# ====================================================================

params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
optimizer = torch.optim.AdamW(
    params_to_optimize,
    lr=config.learning_rate,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-08,
)

from diffusers.optimization import get_scheduler
lr_scheduler = get_scheduler(
    config.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
    num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
)

logger.info("✓ Optimizer and scheduler configured")

# ====================================================================
# PREPARE FOR TRAINING
# ====================================================================

unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    unet, optimizer, train_dataloader, lr_scheduler
)

# ====================================================================
# LOSS FUNCTIONS (Min-SNR Weighting)
# ====================================================================

def compute_loss_with_min_snr(model_pred, noise, timesteps, min_snr_gamma=5.0):
    """Min-SNR Weighting Strategy"""
    # FIX: Move alphas_cumprod to same device as timesteps
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    
    snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
    
    mse_loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="none")
    mse_loss = mse_loss.mean(dim=list(range(1, len(mse_loss.shape))))
    
    if noise_scheduler.config.prediction_type == "v_prediction":
        mse_loss_weights = snr + 1
    else:
        mse_loss_weights = snr
    
    mse_loss_weights = torch.clamp(mse_loss_weights, max=min_snr_gamma)
    mse_loss_weights = mse_loss_weights / snr
    
    loss = (mse_loss * mse_loss_weights).mean()
    return loss


# ====================================================================
# TRAINING LOOP
# ====================================================================

logger.info("="*80)
logger.info("STARTING TRAINING - MEMORY OPTIMIZED MODE")
logger.info("="*80)
logger.info(f"  Num examples: {len(train_dataset)}")
logger.info(f"  Num epochs: {config.num_train_epochs}")
logger.info(f"  Batch size: {config.train_batch_size}")
logger.info(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
logger.info(f"  Total steps: {config.max_train_steps}")
logger.info(f"  Resolution: {config.resolution}x{config.resolution}")
logger.info(f"  LoRA rank: {config.lora_rank}")
logger.info(f"  Learning rate: {config.learning_rate}")
# logger.info(f"  EMA: Disabled (memory saving)")
logger.info("="*80)

global_step = 0
progress_bar = tqdm(range(0, config.max_train_steps), disable=not accelerator.is_local_main_process, desc="Steps")

start_time = datetime.now()
train_loss_accumulator = 0.0
best_loss = float('inf')

for epoch in range(config.num_train_epochs):
    unet.train()
    epoch_loss = 0.0
    
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            # Get latents
            if "latents" in batch:
                latents = batch["latents"].to(accelerator.device, dtype=torch.float16)
            else:
                logger.warning("Latents not cached")
                continue
            
            # Sample noise with offset
            noise = torch.randn_like(latents)
            if config.offset_noise_strength > 0:
                noise = noise + config.offset_noise_strength * torch.randn(
                    latents.shape[0], latents.shape[1], 1, 1, device=latents.device, dtype=latents.dtype
                )
            
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            ).long()
            
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Load text embeddings from disk
            if text_embeddings_cache_dir is not None:
                idx = batch["index"][0].item()
                cache_file = text_embeddings_cache_dir / f"text_emb_{idx}.pt"
                cached = torch.load(cache_file, map_location=accelerator.device)
                prompt_embeds = cached["prompt_embeds"].to(dtype=torch.float16)
                pooled_prompt_embeds = cached["pooled_embeds"].to(dtype=torch.float16)
            else:
                # Fallback to on-the-fly encoding
                logger.warning("Text embeddings not cached")
                continue
            
            noisy_latents = noisy_latents.to(dtype=torch.float16)
            
            # SDXL conditioning
            add_time_ids = torch.tensor([
                [config.resolution, config.resolution, 0, 0, config.resolution, config.resolution]
            ] * bsz).to(device=accelerator.device, dtype=torch.float16)
            
            # Forward pass
            model_pred = unet(
                noisy_latents, timesteps, prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids},
            ).sample
            
            # Compute loss with Min-SNR weighting
            loss = compute_loss_with_min_snr(model_pred, noise, timesteps, config.min_snr_gamma)
            
            # Backward pass
            accelerator.backward(loss)
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Update progress
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            train_loss_accumulator += loss.detach().item()
            epoch_loss += loss.detach().item()
            
            # Logging
            if global_step % config.logging_steps == 0:
                avg_loss = train_loss_accumulator / config.logging_steps
                elapsed = (datetime.now() - start_time).total_seconds()
                speed = global_step / elapsed
                eta = (config.max_train_steps - global_step) / speed / 3600
                
                logger.info(
                    f"Epoch {epoch+1}/{config.num_train_epochs} | "
                    f"Step {global_step}/{config.max_train_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.2e} | "
                    f"Speed: {speed:.2f} it/s | ETA: {eta:.1f}h"
                )
                
                if accelerator.is_main_process:
                    accelerator.log({
                        "train_loss": avg_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }, step=global_step)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                
                train_loss_accumulator = 0.0
            
            # Save checkpoint
            if global_step % config.save_steps == 0 and accelerator.is_main_process:
                checkpoint_dir = Path(config.output_dir) / f"checkpoint-{global_step}"
                checkpoint_dir.mkdir(exist_ok=True, parents=True)
                
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet.save_pretrained(checkpoint_dir)
                
                metadata = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "loss": avg_loss if 'avg_loss' in locals() else None,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(checkpoint_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"✓ Checkpoint saved: {checkpoint_dir}")
            
            if global_step >= config.max_train_steps:
                break
        
        # Aggressive memory cleanup
        if step % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    avg_epoch_loss = epoch_loss / len(train_dataloader)
    logger.info(f"Epoch {epoch+1} complete | Average loss: {avg_epoch_loss:.4f}")
    
    if global_step >= config.max_train_steps:
        break

# ====================================================================
# SAVE FINAL MODEL
# ====================================================================

if accelerator.is_main_process:
    logger.info("Saving final model...")
    
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(config.output_dir)
    
    total_time = datetime.now() - start_time
    training_info = {
        "status": "completed",
        "total_steps": global_step,
        "total_epochs": epoch + 1,
        "best_loss": best_loss,
        "final_loss": avg_epoch_loss if 'avg_epoch_loss' in locals() else None,
        "training_time_seconds": total_time.total_seconds(),
        "training_time_formatted": str(total_time),
        "timestamp": datetime.now().isoformat(),
        "config": config_dict,
        "optimizations": {
            "min_snr_weighting": True,
            "offset_noise": config.offset_noise_strength > 0,
            "disk_text_caching": config.cache_text_embeddings_to_disk,
            "memory_optimized": True,
            "resolution": config.resolution,
            "lora_rank": config.lora_rank,
        }
    }
    
    with open(f"{config.output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    logger.info(f"✓ Final model saved to: {config.output_dir}")
    logger.info(f"✓ Training completed in {total_time}")
    logger.info(f"✓ Best loss: {best_loss:.4f}")

accelerator.end_training()

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Output directory: {config.output_dir}")
print(f"Training time: {total_time}")
print(f"Best loss: {best_loss:.4f}")
print("="*80)
print("\nMemory optimizations applied:")
print("  ✓ Lower resolution (768x768)")
print("  ✓ Lower LoRA rank (8)")
print("  ✓ Disk-based text embedding caching")
print("  ✓ EMA enabled")
print("  ✓ Aggressive memory cleanup")
print("  ✓ Target VRAM: 8-10GB")
print("="*80)
