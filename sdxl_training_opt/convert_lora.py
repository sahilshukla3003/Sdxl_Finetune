#!/usr/bin/env python3
"""
Convert PEFT LoRA to diffusers-compatible format
"""
import torch
from pathlib import Path

LORA_PATH = "./output/sdxl-naruto-lora"
OUTPUT_PATH = "./output/sdxl-naruto-lora-fixed"

print("Converting LoRA weights...")

# Load the adapter_model.safetensors or .bin file
lora_file = Path(LORA_PATH) / "adapter_model.safetensors"
if not lora_file.exists():
    lora_file = Path(LORA_PATH) / "adapter_model.bin"

if lora_file.exists():
    from safetensors.torch import load_file, save_file
    
    # Load weights
    if str(lora_file).endswith('.safetensors'):
        state_dict = load_file(lora_file)
    else:
        state_dict = torch.load(lora_file, map_location='cpu')
    
    # Remove 'base_model.model.' prefix
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('base_model.model.'):
            new_key = key.replace('base_model.model.', '')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Save converted weights
    Path(OUTPUT_PATH).mkdir(exist_ok=True, parents=True)
    
    if str(lora_file).endswith('.safetensors'):
        save_file(new_state_dict, Path(OUTPUT_PATH) / "pytorch_lora_weights.safetensors")
    else:
        torch.save(new_state_dict, Path(OUTPUT_PATH) / "pytorch_lora_weights.bin")
    
    # Copy adapter_config.json
    import shutil
    config_file = Path(LORA_PATH) / "adapter_config.json"
    if config_file.exists():
        shutil.copy(config_file, Path(OUTPUT_PATH) / "adapter_config.json")
    
    print(f"✓ Converted LoRA saved to: {OUTPUT_PATH}")
    print(f"  - Use LORA_PATH = '{OUTPUT_PATH}' in inference.py")
else:
    print(f"❌ Error: {lora_file} not found")
