import torch
import sys

print("--- System Check ---")
print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")

print("\n--- GPU Check ---")
cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {cuda_available}")

if cuda_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test a small calculation on the GPU
    x = torch.rand(5, 3).cuda()
    print("\n--- Performance Test ---")
    print("Success! A tensor was moved to the GPU.")
else:
    print("\n[!] Warning: GPU not detected. PyTorch is running on the CPU.")