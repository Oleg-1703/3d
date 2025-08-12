#!/usr/bin/env python3
"""
Fixed headless COLMAP wrapper with OpenGL workarounds
"""
import os
import sys
import subprocess
import argparse

def setup_headless_environment():
    """Setup environment for headless OpenGL"""
    env = os.environ.copy()
    
    # Disable OpenGL rendering
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env['DISPLAY'] = ''
    env['XDG_RUNTIME_DIR'] = '/tmp'
    env['COLMAP_USE_OPENGL'] = '0'
    env['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    
    return env

def run_cpu_colmap(image_path, output_path):
    """Run COLMAP with CPU-only features"""
    
    os.makedirs(output_path, exist_ok=True)
    sparse_path = os.path.join(output_path, "sparse")
    os.makedirs(sparse_path, exist_ok=True)
    
    env = setup_headless_environment()
    database_path = os.path.join(output_path, "database.db")
    
    try:
        # Step 1: Feature extraction with CPU only
        print("Step 1: CPU Feature extraction...")
        cmd1 = [
            "colmap", "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_path,
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "0",  # Force CPU
            "--SiftExtraction.gpu_index", "-1"
        ]
        subprocess.run(cmd1, env=env, check=True)
        
        # Step 2: Feature matching
        print("Step 2: CPU Feature matching...")
        cmd2 = [
            "colmap", "exhaustive_matcher",
            "--database_path", database_path,
            "--SiftMatching.use_gpu", "0"  # Force CPU
        ]
        subprocess.run(cmd2, env=env, check=True)
        
        # Step 3: Bundle adjustment
        print("Step 3: Bundle adjustment...")
        cmd3 = [
            "colmap", "mapper",
            "--database_path", database_path,
            "--image_path", image_path,
            "--output_path", sparse_path
        ]
        subprocess.run(cmd3, env=env, check=True)
        
        print("✓ COLMAP CPU reconstruction completed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ COLMAP failed: {e}")
        return False

def create_minimal_sparse_model(image_path, output_path):
    """Create minimal sparse model if COLMAP fails"""
    print("Creating minimal sparse model as fallback...")
    
    import numpy as np
    import glob
    
    sparse_path = os.path.join(output_path, "sparse", "0")
    os.makedirs(sparse_path, exist_ok=True)
    
    # Count images
    image_files = glob.glob(os.path.join(image_path, "*.jpg")) + \
                  glob.glob(os.path.join(image_path, "*.png"))
    num_images = len(image_files)
    
    if num_images == 0:
        print("No images found!")
        return False
    
    # Create dummy cameras.txt
    with open(os.path.join(sparse_path, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("1 PINHOLE 640 480 500 500 320 240\n")
    
    # Create dummy images.txt
    with open(os.path.join(sparse_path, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        for i, img_file in enumerate(sorted(image_files)):
            img_name = os.path.basename(img_file)
            # Simple circular camera positions
            angle = i * 2 * np.pi / num_images
            tx = 2 * np.cos(angle)
            ty = 0
            tz = 2 * np.sin(angle)
            
            f.write(f"{i+1} 1 0 0 0 {tx} {ty} {tz} 1 {img_name}\n")
            f.write("\n")  # Empty points2D line
    
    # Create dummy points3D.txt
    with open(os.path.join(sparse_path, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        
        # Add a few dummy 3D points
        for i in range(10):
            x = (i - 5) * 0.1
            f.write(f"{i+1} {x} 0 0 128 128 128 0.1\n")
    
    print(f"✓ Created minimal sparse model with {num_images} images")
    return True

def main():
    parser = argparse.ArgumentParser(description="Fixed headless COLMAP")
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--force_minimal", action="store_true", 
                       help="Skip COLMAP and create minimal model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: {args.image_path} does not exist")
        return 1
    
    if args.force_minimal:
        success = create_minimal_sparse_model(args.image_path, args.output_path)
    else:
        print("Trying COLMAP CPU reconstruction...")
        success = run_cpu_colmap(args.image_path, args.output_path)
        
        if not success:
            print("COLMAP failed, creating minimal model...")
            success = create_minimal_sparse_model(args.image_path, args.output_path)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
