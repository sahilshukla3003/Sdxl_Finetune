#!/usr/bin/env python3
"""
SDXL Fine-Tuning for Naruto Style - Optimized for 12GB GPU
Hardware: 12GB GPU (RTX 3060, RTX 4070 Ti, etc.)
Expected Training Time: 4-6 hours
Resolution: 1024x1024 (native SDXL)
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
print("SDXL FINE-TUNING - NARUTO STYLE TRANSFER")
print("="*80)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")
print("="*80)

# GPU memory check
gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
if gpu_memory_gb < 10:
    print(f"\n⚠️  WARNING: Your GPU has {gpu_memory_gb:.1f}GB VRAM")
    print("   Recommended: 12GB+ for optimal SDXL training")
    response = input("\nContinue anyway? (yes/no): ")
    if response.lower() != 'yes':
        sys.exit(0)

# Imports
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
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
# CONFIGURATION - OPTIMIZED FOR 12GB GPU
# ====================================================================
class TrainingConfig:
    """Battle-tested configuration for 12GB GPU - No issues, maximum efficiency"""
    
    # Model
    pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    pretrained_vae_name = "madebyollin/sdxl-vae-fp16-fix"
    dataset_name = "lambdalabs/naruto-blip-captions"
    
    # Output
    output_dir = "./output/sdxl-naruto-lora"
    logging_dir = "./output/logs"
    cache_dir = "./cache"
    
    # ========== CRITICAL 12GB SETTINGS ==========
    resolution = 1024        # Native SDXL - best quality
    train_batch_size = 1     # Minimum
    gradient_accumulation_steps = 4  # Effective batch = 4
    
    # LoRA - Balanced quality/memory
    lora_rank = 16           # Sweet spot (8=min, 16=balanced, 32=high quality)
    lora_alpha = 16
    lora_dropout = 0.0
    lora_target_modules = ["to_q", "to_k", "to_v", "to_out.0"]
    
    # Memory optimizations (pick what works)
    use_8bit_adam = False                    # ✓ CRITICAL - saves 15GB
    gradient_checkpointing = True           # ✓ CRITICAL - saves 8GB
    mixed_precision = "fp16"                # ✓ CRITICAL - saves 5GB
    enable_xformers = False                 # ✗ DISABLE - causes dtype issues
    cache_latents = True                    # ✓ CRITICAL - saves 3GB
    cache_text_encoder_outputs = False      # Optional (complicates training)
    
    # Training
    num_train_epochs = 10
    max_train_steps = 2000
    learning_rate = 1e-4
    lr_scheduler = "cosine"
    lr_warmup_steps = 100
    max_grad_norm = 1.0
    
    # Checkpointing
    checkpointing_steps = 500
    save_steps = 500
    logging_steps = 50
    
    # Dataloader (reduce workers if RAM limited)
    dataloader_num_workers = 2  # Lower than 4 for stability
    pin_memory = True
    
    seed = 42
    report_to = "tensorboard"

config = TrainingConfig()

# ====================================================================
# SETUP LOGGING & DIRECTORIES
# ====================================================================

os.makedirs(config.output_dir, exist_ok=True)
os.makedirs(config.logging_dir, exist_ok=True)
os.makedirs(config.cache_dir, exist_ok=True)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(
            f"{config.logging_dir}/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Save config
config_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
with open(f"{config.output_dir}/training_config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

logger.info("Configuration saved")

# ====================================================================
# INITIALIZE ACCELERATOR
# ====================================================================

accelerator_project_config = ProjectConfiguration(
    project_dir=config.output_dir,
    logging_dir=config.logging_dir
)

accelerator = Accelerator(
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    mixed_precision=config.mixed_precision,
    log_with=config.report_to,
    project_config=accelerator_project_config,
)

if accelerator.is_main_process:
    logger.info(f"Accelerator device: {accelerator.device}")
    logger.info(f"Mixed precision: {config.mixed_precision}")
    logger.info(f"Num processes: {accelerator.num_processes}")

set_seed(config.seed)

# ====================================================================
# LOAD MODELS
# ====================================================================

logger.info("Loading tokenizers...")
tokenizer_one = AutoTokenizer.from_pretrained(
    config.pretrained_model_name,
    subfolder="tokenizer",
    use_fast=False,
    cache_dir=config.cache_dir
)
tokenizer_two = AutoTokenizer.from_pretrained(
    config.pretrained_model_name,
    subfolder="tokenizer_2",
    use_fast=False,
    cache_dir=config.cache_dir
)

logger.info("Loading noise scheduler...")
noise_scheduler = DDPMScheduler.from_pretrained(
    config.pretrained_model_name,
    subfolder="scheduler",
    cache_dir=config.cache_dir
)

logger.info("Loading VAE...")
vae = AutoencoderKL.from_pretrained(
    config.pretrained_vae_name,
    torch_dtype=torch.float16,
    cache_dir=config.cache_dir
)
vae.requires_grad_(False)
vae.to(accelerator.device)
logger.info("✓ VAE loaded")

logger.info("Loading text encoders...")
text_encoder_one = CLIPTextModel.from_pretrained(
    config.pretrained_model_name,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
    cache_dir=config.cache_dir
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    config.pretrained_model_name,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
    cache_dir=config.cache_dir
)

text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
text_encoder_one.to(accelerator.device)
text_encoder_two.to(accelerator.device)
logger.info("✓ Text encoders loaded")

logger.info("Loading UNet...")
unet = UNet2DConditionModel.from_pretrained(
    config.pretrained_model_name,
    subfolder="unet",
    # torch_dtype=torch.float16,
    cache_dir=config.cache_dir
)

# Enable optimizations
if config.gradient_checkpointing:
    unet.enable_gradient_checkpointing()
    logger.info("✓ Gradient checkpointing enabled")

if config.enable_xformers:
    try:
        unet.enable_xformers_memory_efficient_attention()
        logger.info("✓ xformers memory-efficient attention enabled")
    except Exception as e:
        logger.warning(f"xformers not available: {e}")

logger.info("✓ UNet loaded")

# ====================================================================
# APPLY LORA
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

# unet.to(accelerator.device)

# Memory cleanup
gc.collect()
torch.cuda.empty_cache()

logger.info("✓ LoRA applied")


class NarutoDataset(Dataset):
    """Dataset for Naruto images with caching"""
    
    def __init__(self, dataset, tokenizer_one, tokenizer_two, resolution, 
                 cache_latents=False, vae=None, cache_dir=None):
        self.dataset = dataset
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.resolution = resolution
        self.cache_latents = cache_latents
        self.vae = vae
        self.cache_dir = cache_dir
        
        # Image preprocessing
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Cache latents to disk
        if self.cache_latents and self.vae is not None:
            self.latents_cache_dir = Path(cache_dir) / "latents_cache"
            self.latents_cache_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"Caching latents for {len(self.dataset)} images...")
            
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
                    
                    # FIX: Save without batch dimension
                    torch.save(latent.squeeze(0).cpu(), cache_file)  # Remove batch dim
                    
                    del image_tensor, latent
                    
                    if i % 50 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.warning(f"Failed to cache image {i}: {e}")
            
            logger.info("✓ Latent caching complete")
    
    def __len__(self):
        return len(self.dataset)
    
    def tokenize_captions(self, caption):
        """Tokenize caption with both CLIP tokenizers"""
        tokens_one = self.tokenizer_one(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_one.model_max_length,
            return_tensors="pt",
        ).input_ids
        
        tokens_two = self.tokenizer_two(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer_two.model_max_length,
            return_tensors="pt",
        ).input_ids
        
        return tokens_one, tokens_two
    
    def __getitem__(self, idx):
        example = {}
        item = self.dataset[idx]
        
        # Get caption
        caption = item["text"]
        tokens_one, tokens_two = self.tokenize_captions(caption)
        example["input_ids_one"] = tokens_one
        example["input_ids_two"] = tokens_two
        
        # Get latent
        if self.cache_latents:
            cache_file = self.latents_cache_dir / f"latent_{idx}.pt"
            if cache_file.exists():
                # Load cached latent (shape should be [4, 128, 128])
                example["latents"] = torch.load(cache_file)
            else:
                # Fallback to on-the-fly encoding
                image = item["image"].convert("RGB")
                example["pixel_values"] = self.image_transforms(image)
        else:
            image = item["image"].convert("RGB")
            example["pixel_values"] = self.image_transforms(image)
        
        return example


# Load dataset
logger.info(f"Loading dataset: {config.dataset_name}")
dataset = load_dataset(config.dataset_name, split="train", cache_dir=config.cache_dir)
logger.info(f"Dataset loaded: {len(dataset)} images")

# Create dataset instance
train_dataset = NarutoDataset(
    dataset=dataset,
    tokenizer_one=tokenizer_one,
    tokenizer_two=tokenizer_two,
    resolution=config.resolution,
    cache_latents=config.cache_latents,
    vae=vae,
    cache_dir=config.cache_dir,
)

# Create dataloader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.train_batch_size,
    shuffle=True,
    num_workers=config.dataloader_num_workers,
    pin_memory=config.pin_memory,
)

logger.info(f"✓ Dataloader ready: {len(train_dataset)} training samples")

# Free VAE memory after caching
if config.cache_latents:
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("✓ VAE offloaded (latents cached)")

# ====================================================================
# OPTIMIZER & SCHEDULER
# ====================================================================

if config.use_8bit_adam:
    try:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
        logger.info("Using 8-bit Adam optimizer")
    except ImportError:
        logger.warning("bitsandbytes not available, using AdamW")
        optimizer_class = torch.optim.AdamW
else:
    optimizer_class = torch.optim.AdamW

params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))

optimizer = optimizer_class(
    params_to_optimize,
    lr=config.learning_rate,
    betas=(0.9, 0.999),
    weight_decay=1e-2,
    eps=1e-08,
)

# Learning rate scheduler
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
# HELPER FUNCTIONS
# ====================================================================

def encode_prompts(input_ids_one, input_ids_two):
    """Encode prompts using both text encoders"""
    with torch.no_grad():
        # Encoder 1
        prompt_embeds_1 = text_encoder_one(
            input_ids_one.to(accelerator.device),
            output_hidden_states=True,
        )
        prompt_embeds_1 = prompt_embeds_1.hidden_states[-2]
        
        # Encoder 2
        prompt_embeds_2 = text_encoder_two(
            input_ids_two.to(accelerator.device),
            output_hidden_states=True,
        )
        prompt_embeds_2_pooled = prompt_embeds_2[0]
        prompt_embeds_2 = prompt_embeds_2.hidden_states[-2]
        
        # Concatenate
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    
    return prompt_embeds, prompt_embeds_2_pooled

# ====================================================================
# TRAINING LOOP
# ====================================================================

logger.info("="*80)
logger.info("STARTING TRAINING")
logger.info("="*80)
logger.info(f"  Num examples: {len(train_dataset)}")
logger.info(f"  Num epochs: {config.num_train_epochs}")
logger.info(f"  Batch size per device: {config.train_batch_size}")
logger.info(f"  Gradient accumulation steps: {config.gradient_accumulation_steps}")
logger.info(f"  Total optimization steps: {config.max_train_steps}")
logger.info(f"  Resolution: {config.resolution}x{config.resolution}")
logger.info(f"  LoRA rank: {config.lora_rank}")
logger.info(f"  Learning rate: {config.learning_rate}")
logger.info("="*80)

global_step = 0
first_epoch = 0

progress_bar = tqdm(
    range(0, config.max_train_steps),
    initial=0,
    desc="Steps",
    disable=not accelerator.is_local_main_process,
)

# Training metrics
start_time = datetime.now()
train_loss_accumulator = 0.0
best_loss = float('inf')


for epoch in range(first_epoch, config.num_train_epochs):
    unet.train()
    epoch_loss = 0.0
    
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            # Get latents
            if "latents" in batch:
                latents = batch["latents"].to(accelerator.device, dtype=torch.float16)
            else:
                logger.warning("Latents not cached, encoding on-the-fly")
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=torch.float16)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,),
                device=latents.device
            )
            timesteps = timesteps.long()
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings
            prompt_embeds, pooled_prompt_embeds = encode_prompts(
                batch["input_ids_one"],
                batch["input_ids_two"]
            )
            
            # Ensure FP16 dtype
            prompt_embeds = prompt_embeds.to(dtype=torch.float16)
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=torch.float16)
            noisy_latents = noisy_latents.to(dtype=torch.float16)
            
            # Prepare SDXL conditioning
            add_time_ids = torch.tensor([
                [config.resolution, config.resolution, 0, 0, config.resolution, config.resolution]
            ] * bsz).to(device=accelerator.device, dtype=torch.float16)
            
            # Forward pass
            model_pred = unet(
                noisy_latents,
                timesteps,
                prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids
                },
            ).sample
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(
                model_pred.float(), noise.float(), reduction="mean"
            )
            
            # Backward pass
            accelerator.backward(loss)
            
            # Update weights
            # if accelerator.sync_gradients:
                # FIX: Gradient clipping with FP16 - use unwrapped model
                # unwrapped_model = accelerator.unwrap_model(unet)
                # accelerator.clip_grad_norm_(unwrapped_model.parameters(), config.max_grad_norm)
            
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
                elapsed_time = (datetime.now() - start_time).total_seconds()
                steps_per_sec = global_step / elapsed_time
                eta_seconds = (config.max_train_steps - global_step) / steps_per_sec
                eta_hours = eta_seconds / 3600
                
                log_message = (
                    f"Epoch {epoch+1}/{config.num_train_epochs} | "
                    f"Step {global_step}/{config.max_train_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.2e} | "
                    f"Speed: {steps_per_sec:.2f} it/s | "
                    f"ETA: {eta_hours:.1f}h"
                )
                
                logger.info(log_message)
                
                if accelerator.is_main_process:
                    accelerator.log({
                        "train_loss": avg_loss,
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                    }, step=global_step)
                
                # Track best loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                
                train_loss_accumulator = 0.0
            
            # Save checkpoint
            if global_step % config.save_steps == 0:
                if accelerator.is_main_process:
                    checkpoint_dir = Path(config.output_dir) / f"checkpoint-{global_step}"
                    checkpoint_dir.mkdir(exist_ok=True, parents=True)
                    
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    unwrapped_unet.save_pretrained(checkpoint_dir)
                    
                    # Save checkpoint metadata
                    metadata = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "loss": avg_loss if 'avg_loss' in locals() else None,
                        "timestamp": datetime.now().isoformat(),
                    }
                    
                    with open(checkpoint_dir / "metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)
                    
                    logger.info(f"✓ Checkpoint saved: {checkpoint_dir}")
            
            # Stop if max steps reached
            if global_step >= config.max_train_steps:
                break
        
        # Periodic memory cleanup
        if step % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()
    
    # End of epoch logging
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
    
    # Save final training info
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
