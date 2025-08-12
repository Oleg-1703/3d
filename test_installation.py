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
