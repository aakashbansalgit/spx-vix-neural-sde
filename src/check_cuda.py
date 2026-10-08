"""Quick sanity check for CUDA availability and device info."""
import torch

print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device         : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("Running on CPU — training will be slow.")
