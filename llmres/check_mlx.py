#!/usr/bin/env python3
"""
MLX System Check Script
Verifies MLX installation and Apple Silicon compatibility
"""

import sys
import platform
import subprocess

def check_apple_silicon():
    """Check if running on Apple Silicon."""
    print("=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    
    # Check for Apple Silicon
    if platform.machine() == 'arm64':
        print("✅ Apple Silicon detected (arm64)")
        return True
    else:
        print("❌ Not running on Apple Silicon")
        print("MLX is optimized for Apple Silicon (M1/M2/M3)")
        return False

def check_python_version():
    """Check Python version compatibility."""
    print("\n=== Python Version ===")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version compatible")
        return True
    else:
        print("❌ Python 3.8+ required for MLX")
        return False

def check_mlx_installation():
    """Check MLX installation."""
    print("\n=== MLX Installation ===")
    
    try:
        import mlx.core as mx
        print(f"✅ MLX Core installed: {mx.__version__}")
        
        # Test basic MLX functionality
        test_array = mx.array([1, 2, 3, 4])
        result = mx.sum(test_array)
        print(f"✅ MLX basic operations working: sum([1,2,3,4]) = {result.item()}")
        
        mlx_core_ok = True
    except ImportError as e:
        print(f"❌ MLX Core not installed: {e}")
        mlx_core_ok = False
    except Exception as e:
        print(f"❌ MLX Core error: {e}")
        mlx_core_ok = False
    
    try:
        import mlx.nn as nn
        print("✅ MLX Neural Networks available")
        mlx_nn_ok = True
    except ImportError as e:
        print(f"❌ MLX Neural Networks not available: {e}")
        mlx_nn_ok = False
    
    try:
        import mlx.optimizers as optim
        print("✅ MLX Optimizers available")
        mlx_optim_ok = True
    except ImportError as e:
        print(f"❌ MLX Optimizers not available: {e}")
        mlx_optim_ok = False
    
    return mlx_core_ok and mlx_nn_ok and mlx_optim_ok

def check_mlx_lm():
    """Check MLX-LM installation."""
    print("\n=== MLX-LM Installation ===")
    
    try:
        import mlx_lm
        print("✅ MLX-LM installed")
        return True
    except ImportError as e:
        print(f"❌ MLX-LM not installed: {e}")
        return False

def check_dependencies():
    """Check other required dependencies."""
    print("\n=== Dependencies ===")
    
    dependencies = [
        ('numpy', 'numpy'),
        ('PIL', 'Pillow'),
        ('datasets', 'datasets'),
        ('safetensors', 'safetensors'),
        ('huggingface_hub', 'huggingface-hub')
    ]
    
    all_ok = True
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✅ {package_name} installed")
        except ImportError:
            print(f"❌ {package_name} not installed")
            all_ok = False
    
    return all_ok

def check_memory():
    """Check available memory."""
    print("\n=== Memory Information ===")
    
    try:
        # Get memory info on macOS
        result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
        if result.returncode == 0:
            memory_bytes = int(result.stdout.split(':')[1].strip())
            memory_gb = memory_bytes / (1024**3)
            print(f"Total Memory: {memory_gb:.1f} GB")
            
            if memory_gb >= 8:
                print("✅ Sufficient memory for MLX training")
                return True
            else:
                print("⚠️  Limited memory - consider reducing model size")
                return True
        else:
            print("❓ Could not determine memory size")
            return True
    except Exception as e:
        print(f"❓ Memory check failed: {e}")
        return True

def installation_guide():
    """Print installation guide."""
    print("\n=== Installation Guide ===")
    print("To install MLX and dependencies:")
    print("")
    print("1. Ensure you're on Apple Silicon (M1/M2/M3)")
    print("2. Install MLX:")
    print("   pip install mlx")
    print("   pip install mlx-lm")
    print("")
    print("3. Install other dependencies:")
    print("   pip install -r requirements.txt")
    print("")
    print("4. Verify installation:")
    print("   python check_mlx.py")

def main():
    """Main check function."""
    print("MLX System Compatibility Check")
    print("=" * 40)
    
    checks = [
        check_apple_silicon(),
        check_python_version(),
        check_mlx_installation(),
        check_mlx_lm(),
        check_dependencies(),
        check_memory()
    ]
    
    print("\n=== Summary ===")
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✅ All checks passed ({passed}/{total})")
        print("Your system is ready for MLX training!")
    else:
        print(f"❌ {total - passed} checks failed ({passed}/{total})")
        print("Please address the issues above before training.")
        installation_guide()
    
    print("\n=== Next Steps ===")
    if passed == total:
        print("Run: python train.py")
    else:
        print("Fix installation issues and run this check again.")

if __name__ == '__main__':
    main()