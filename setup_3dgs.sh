#!/bin/bash

# Automated setup script for 3D Gaussian Splatting pipeline without conda
# For RTX 4090 GPU environment with text-based segmentation

set -e  # Exit on any error

echo "=========================================="
echo "3D Gaussian Splatting Pipeline Setup"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root is not recommended. Consider using a regular user."
fi

# Function to check command existence
check_command() {
    if command -v "$1" &> /dev/null; then
        print_status "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# Function to check Python version
check_python_version() {
    python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
    required_version="3.8"
    
    if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
        print_status "Python $python_version is compatible"
        return 0
    else
        print_error "Python $python_version is too old. Minimum required: $required_version"
        return 1
    fi
}

# Step 1: System dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Update system
    sudo apt update -y
    
    # Install system packages
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ffmpeg \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libswscale-dev \
        colmap \
        ninja-build
    
    print_status "System dependencies installed successfully"
}

# Step 2: Check CUDA
check_cuda() {
    print_status "Checking CUDA installation..."
    
    if check_command nvidia-smi && check_command nvcc; then
        # Set CUDA environment variables
        echo "export CUDA_HOME=/usr/local/cuda" >> ~/.bashrc
        echo "export PATH=\$CUDA_HOME/bin:\$PATH" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
        echo "export AM_I_DOCKER=False" >> ~/.bashrc
        echo "export BUILD_WITH_CUDA=True" >> ~/.bashrc
        
        # Apply immediately
        export CUDA_HOME=/usr/local/cuda
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        export AM_I_DOCKER=False
        export BUILD_WITH_CUDA=True
        
        print_status "CUDA environment configured"
    else
        print_error "CUDA not found. Please install CUDA toolkit first."
        return 1
    fi
}

# Step 3: Install Python packages
install_python_packages() {
    print_status "Installing Python packages..."
    
    # Upgrade pip
    python3 -m pip install --upgrade pip
    
    # Install PyTorch for CUDA 11.8 (adjust as needed)
    print_status "Installing PyTorch..."
    python3 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    
    # Basic scientific libraries
    print_status "Installing scientific libraries..."
    python3 -m pip install \
        numpy==1.24.3 \
        scipy==1.10.1 \
        matplotlib==3.7.1 \
        pillow==10.0.0 \
        opencv-python==4.8.0.74 \
        scikit-learn==1.3.0
    
    # 3DGS specific dependencies
    print_status "Installing 3DGS dependencies..."
    python3 -m pip install \
        plyfile==0.8.1 \
        tqdm==4.65.0 \
        wandb==0.15.4 \
        lpips==0.1.4
    
    print_status "Python packages installed successfully"
}

# Step 4: Install CUDA modules
install_cuda_modules() {
    print_status "Installing CUDA modules..."
    
    # Check if submodules exist
    if [ -d "submodules/diff-gaussian-rasterization" ]; then
        print_status "Installing diff-gaussian-rasterization..."
        cd submodules/diff-gaussian-rasterization
        python3 -m pip install .
        cd ../..
    else
        print_warning "diff-gaussian-rasterization submodule not found"
    fi
    
    if [ -d "submodules/simple-knn" ]; then
        print_status "Installing simple-knn..."
        cd submodules/simple-knn
        python3 -m pip install .
        cd ../..
    else
        print_warning "simple-knn submodule not found"
    fi
}

# Step 5: Install DEVA
install_deva() {
    print_status "Installing DEVA for text-based segmentation..."
    
    if [ ! -d "Tracking-Anything-with-DEVA" ]; then
        print_status "Cloning DEVA repository..."
        git clone https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git
    fi
    
    cd Tracking-Anything-with-DEVA
    python3 -m pip install -e .
    
    # Download models
    print_status "Downloading DEVA models..."
    bash scripts/download_models.sh
    
    cd ..
    print_status "DEVA installed successfully"
}

# Step 6: Install Grounded Segment Anything
install_grounded_sam() {
    print_status "Installing Grounded Segment Anything..."
    
    cd Tracking-Anything-with-DEVA
    
    if [ ! -d "Grounded-Segment-Anything" ]; then
        print_status "Cloning Grounded-Segment-Anything..."
        git clone https://github.com/hkchengrex/Grounded-Segment-Anything.git
    fi
    
    cd Grounded-Segment-Anything
    
    # Install Segment Anything
    python3 -m pip install -e segment_anything
    
    # Install GroundingDINO
    python3 -m pip install -e GroundingDINO
    
    # Additional dependencies
    python3 -m pip install \
        transformers==4.31.0 \
        addict==2.4.0 \
        yapf==0.40.1 \
        timm==0.9.2 \
        supervision==0.12.0
    
    cd ../..
    print_status "Grounded Segment Anything installed successfully"
}

# Step 7: Install LaMa (optional)
install_lama() {
    print_status "Installing LaMa for inpainting..."
    
    if [ ! -d "lama" ]; then
        print_status "Cloning LaMa repository..."
        git clone https://github.com/advimman/lama.git
    fi
    
    cd lama
    python3 -m pip install -r requirements.txt
    
    # Additional dependencies
    python3 -m pip install \
        albumentations==1.3.1 \
        hydra-core==1.3.2 \
        pytorch-lightning==2.0.6 \
        tabulate==0.9.0 \
        kornia==0.6.12 \
        webdataset==0.2.77
    
    cd ..
    print_status "LaMa installed successfully"
}

# Step 8: Verification
verify_installation() {
    print_status "Verifying installation..."
    
    # Test basic imports
    python3 -c "
import torch
import torchvision
import cv2
import numpy as np
import scipy
import plyfile
import lpips
print('✓ Basic libraries OK')
" || print_error "Basic libraries verification failed"
    
    # Test CUDA modules
    python3 -c "
try:
    import diff_gaussian_rasterization
    import simple_knn
    print('✓ CUDA modules OK')
except ImportError as e:
    print('✗ CUDA modules failed:', e)
"
    
    # Test GPU
    python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Current GPU: {torch.cuda.get_device_name()}')
"
    
    # Test DEVA
    cd Tracking-Anything-with-DEVA
    python3 -c "
try:
    from deva.inference.inference_core import DEVAInferenceCore
    print('✓ DEVA OK')
except ImportError as e:
    print('✗ DEVA failed:', e)
"
    
    # Test GroundingDINO
    python3 -c "
try:
    from groundingdino.util.inference import Model as GroundingDINOModel
    print('✓ GroundingDINO OK')
except ImportError as e:
    print('✗ GroundingDINO failed:', e)
    print('Make sure CUDA_HOME is set correctly')
"
    cd ..
    
    print_status "Verification completed"
}

# Step 9: Create test script
create_test_script() {
    print_status "Creating test script..."
    
    cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify 3DGS pipeline installation
"""
import sys
import torch

def test_basic_imports():
    """Test basic Python packages"""
    try:
        import numpy as np
        import cv2
        import plyfile
        import scipy
        import matplotlib
        print("✓ Basic packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Basic import failed: {e}")
        return False

def test_cuda_modules():
    """Test CUDA-compiled modules"""
    try:
        import diff_gaussian_rasterization
        import simple_knn
        print("✓ CUDA modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ CUDA modules failed: {e}")
        return False

def test_gpu():
    """Test GPU availability"""
    if torch.cuda.is_available():
        print(f"✓ CUDA available with {torch.cuda.device_count()} GPU(s)")
        print(f"  Current device: {torch.cuda.get_device_name()}")
        return True
    else:
        print("✗ CUDA not available")
        return False

def test_segmentation():
    """Test segmentation modules"""
    try:
        sys.path.append('Tracking-Anything-with-DEVA')
        from deva.inference.inference_core import DEVAInferenceCore
        print("✓ DEVA segmentation module available")
        
        from groundingdino.util.inference import Model as GroundingDINOModel
        print("✓ GroundingDINO module available")
        return True
    except ImportError as e:
        print(f"✗ Segmentation modules failed: {e}")
        return False

def main():
    print("Testing 3DGS Pipeline Installation")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_cuda_modules,
        test_gpu,
        test_segmentation
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    passed = sum(results)
    total = len(results)
    
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        return 0
    else:
        print("❌ Some tests failed. Check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF
    
    chmod +x test_installation.py
    print_status "Test script created: test_installation.py"
}

# Main installation function
main() {
    print_status "Starting 3DGS pipeline installation..."
    
    # Pre-checks
    if ! check_python_version; then
        print_error "Python version check failed"
        exit 1
    fi
    
    # Installation steps
    install_system_deps
    check_cuda
    install_python_packages
    install_cuda_modules
    install_deva
    install_grounded_sam
    
    # Optional components
    read -p "Install LaMa for inpainting? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_lama
    fi
    
    # Verification and testing
    verify_installation
    create_test_script
    
    print_status "Installation completed!"
    print_status "Run 'python3 test_installation.py' to verify everything works"
    print_status "Don't forget to run 'source ~/.bashrc' to load environment variables"
    
    echo ""
    echo "=========================================="
    echo "Next steps:"
    echo "1. Restart your shell or run: source ~/.bashrc"
    echo "2. Test installation: python3 test_installation.py"
    echo "3. Run CPU test pipeline: python3 test_cpu_pipeline.py"
    echo "4. For COLMAP processing, place images in data/scene_name/images/"
    echo "5. For text segmentation, use DEVA demo scripts"
    echo "=========================================="
}

# Run main function
main "$@"