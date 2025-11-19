#!/usr/bin/env python3
"""Quick environment verification"""

import sys

def check_package(package_name, import_name=None):
    """Check if package is installed and return version"""
    if import_name is None:
        import_name = package_name
    
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {package_name}: {version}")
        return True
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        return False

print("="*60)
print("ENVIRONMENT VERIFICATION")
print("="*60)

print(f"\nPython: {sys.version.split()[0]}")

# Check PyTorch
import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ WARNING: CUDA not available!")

print("\nPackages:")
packages = [
    'transformers',
    'diffusers',
    'accelerate',
    'peft',
    'datasets',
    'bitsandbytes',
    'xformers',
    'safetensors',
    'tensorboard',
    ('Pillow', 'PIL'),
]

all_ok = True
for pkg in packages:
    if isinstance(pkg, tuple):
        name, import_name = pkg
        if not check_package(name, import_name):
            all_ok = False
    else:
        if not check_package(pkg):
            all_ok = False

print("\n" + "="*60)
if all_ok and torch.cuda.is_available():
    print("✓ ALL CHECKS PASSED - READY FOR TRAINING!")
else:
    print("⚠️ Some issues detected - check above")
print("="*60)
