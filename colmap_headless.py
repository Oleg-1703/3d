#!/usr/bin/env python3
"""
Headless COLMAP wrapper for container environments
"""
import os
import sys
import subprocess
import argparse

def run_colmap_headless(image_path, output_path, quality="high"):
    """Run COLMAP reconstruction without GUI"""
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    sparse_path = os.path.join(output_path, "sparse")
    dense_path = os.path.join(output_path, "dense")
    os.makedirs(sparse_path, exist_ok=True)
    os.makedirs(dense_path, exist_ok=True)
    
    # Set environment for headless mode
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env['DISPLAY'] = ''
    
    try:
        # Step 1: Feature extraction
        print("Step 1: Feature extraction...")
        cmd1 = [
            "colmap", "feature_extractor",
            "--database_path", os.path.join(output_path, "database.db"),
            "--image_path", image_path,
            "--ImageReader.single_camera", "1"
        ]
        subprocess.run(cmd1, env=env, check=True)
        
        # Step 2: Feature matching
        print("Step 2: Feature matching...")
        cmd2 = [
            "colmap", "exhaustive_matcher",
            "--database_path", os.path.join(output_path, "database.db")
        ]
        subprocess.run(cmd2, env=env, check=True)
        
        # Step 3: Bundle adjustment (sparse reconstruction)
        print("Step 3: Bundle adjustment...")
        cmd3 = [
            "colmap", "mapper",
            "--database_path", os.path.join(output_path, "database.db"),
            "--image_path", image_path,
            "--output_path", sparse_path
        ]
        subprocess.run(cmd3, env=env, check=True)
        
        # Step 4: Dense reconstruction
        print("Step 4: Dense reconstruction...")
        
        # Image undistortion
        cmd4 = [
            "colmap", "image_undistorter",
            "--image_path", image_path,
            "--input_path", os.path.join(sparse_path, "0"),
            "--output_path", dense_path,
            "--output_type", "COLMAP"
        ]
        subprocess.run(cmd4, env=env, check=True)
        
        # Patch match stereo
        cmd5 = [
            "colmap", "patch_match_stereo",
            "--workspace_path", dense_path,
            "--workspace_format", "COLMAP"
        ]
        subprocess.run(cmd5, env=env, check=True)
        
        # Stereo fusion
        cmd6 = [
            "colmap", "stereo_fusion",
            "--workspace_path", dense_path,
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", os.path.join(dense_path, "fused.ply")
        ]
        subprocess.run(cmd6, env=env, check=True)
        
        print(f"✓ COLMAP reconstruction completed successfully!")
        print(f"✓ Sparse model: {sparse_path}")
        print(f"✓ Dense model: {dense_path}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ COLMAP failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Headless COLMAP reconstruction")
    parser.add_argument("--image_path", required=True, help="Path to input images")
    parser.add_argument("--output_path", required=True, help="Path to output directory")
    parser.add_argument("--quality", default="high", choices=["low", "medium", "high"])
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image path {args.image_path} does not exist")
        return 1
    
    success = run_colmap_headless(args.image_path, args.output_path, args.quality)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
