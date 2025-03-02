import sys
import torch

def check_system_info():
    print("🔹 Checking system information...\n")

    # Python version
    print(f"🐍 Python Version: {sys.version.split()[0]}")

    try:
        # PyTorch version
        print(f"🔥 PyTorch Version: {torch.__version__}")

        # CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"⚡ CUDA Available: {cuda_available}")

        if cuda_available:
            print(f"🖥️ CUDA Version: {torch.version.cuda}")
            print(f"🔧 cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"📀 Number of GPUs: {torch.cuda.device_count()}")

            # GPU details
            for i in range(torch.cuda.device_count()):
                print(f"  🏎️ GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  🚀 Compute Capability: {torch.cuda.get_device_capability(i)}")
                print(f"  🔋 GPU Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 3:.2f} GB")
                print(f"  📦 GPU Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 3:.2f} GB")
        else:
            print("❌ CUDA is **not** available. Running on CPU.")

    except ImportError:
        print("❌ PyTorch is **not installed**. Install it using:")
        print("   pip install torch")

# Run the check
check_system_info()

