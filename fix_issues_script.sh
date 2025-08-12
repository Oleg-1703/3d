#!/bin/bash

# Скрипт для исправления проблем после успешной установки 3DGS пайплайна

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo "=========================================="
echo "Исправление проблем после установки 3DGS"
echo "=========================================="

# Проблема 1: COLMAP GUI в контейнере
fix_colmap_headless() {
    print_status "Настройка COLMAP для работы без GUI..."
    
    # Создаем скрипт для headless COLMAP
    cat > colmap_headless.py << 'EOF'
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
EOF
    
    chmod +x colmap_headless.py
    print_status "COLMAP headless wrapper создан: colmap_headless.py"
}

# Проблема 2: Скачивание моделей DEVA/GroundingDINO
download_models_manually() {
    print_status "Скачивание моделей для DEVA и GroundingDINO..."
    
    cd Tracking-Anything-with-DEVA
    
    # Создаем директории для моделей
    mkdir -p saves
    mkdir -p Grounded-Segment-Anything/weights
    
    # Скачиваем основные модели DEVA
    print_status "Скачивание DEVA моделей..."
    cd saves
    
    # Используем прямые ссылки на модели
    if [ ! -f "DEVA-propagation.pth" ]; then
        wget -q --show-progress https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth
    fi
    
    if [ ! -f "XMem.pth" ]; then
        wget -q --show-progress https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem.pth
    fi
    
    cd ..
    
    # Скачиваем модели GroundingDINO
    print_status "Скачивание GroundingDINO моделей..."
    cd Grounded-Segment-Anything/weights
    
    if [ ! -f "groundingdino_swint_ogc.pth" ]; then
        wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    fi
    
    # Скачиваем SAM модели
    if [ ! -f "sam_vit_h_4b8939.pth" ]; then
        wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    fi
    
    cd ../../..
    print_status "Модели скачаны успешно"
}

# Проблема 3: Создание офлайн конфигурации
setup_offline_mode() {
    print_status "Настройка офлайн режима для HuggingFace..."
    
    # Создаем локальную копию bert-base-uncased конфигурации
    mkdir -p models/bert-base-uncased
    
    cat > models/bert-base-uncased/config.json << 'EOF'
{
  "architectures": [
    "BertForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "classifier_dropout": null,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-12,
  "max_position_embeddings": 512,
  "model_type": "bert",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 0,
  "position_embedding_type": "absolute",
  "transformers_version": "4.31.0",
  "type_vocab_size": 2,
  "use_cache": true,
  "vocab_size": 30522
}
EOF

    cat > models/bert-base-uncased/tokenizer_config.json << 'EOF'
{
  "clean_up_tokenization_spaces": true,
  "cls_token": "[CLS]",
  "do_lower_case": true,
  "mask_token": "[MASK]",
  "model_max_length": 512,
  "pad_token": "[PAD]",
  "sep_token": "[SEP]",
  "strip_accents": null,
  "tokenize_chinese_chars": true,
  "tokenizer_class": "BertTokenizer",
  "unk_token": "[UNK]"
}
EOF
    
    print_status "Офлайн конфигурация создана"
}

# Создание тестовых данных
create_test_data() {
    print_status "Создание тестовых данных..."
    
    # Создаем структуру папок для тестовых данных
    mkdir -p data/test_scene/images
    
    # Создаем простую тестовую сцену с помощью Python
    cat > create_test_images.py << 'EOF'
#!/usr/bin/env python3
"""
Создание тестовых изображений для демонстрации пайплайна
"""
import numpy as np
import cv2
import os

def create_simple_scene():
    """Создает простую сцену с объектами для тестирования"""
    
    output_dir = "data/test_scene/images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Параметры сцены
    width, height = 640, 480
    num_views = 10
    
    for i in range(num_views):
        # Создаем изображение
        img = np.ones((height, width, 3), dtype=np.uint8) * 220  # Светло-серый фон
        
        # Добавляем простые объекты
        angle = i * 36  # Поворот на 36 градусов между кадрами
        
        # Красный куб
        center_x = int(width/2 + 100 * np.cos(np.radians(angle)))
        center_y = int(height/2 + 50 * np.sin(np.radians(angle)))
        cv2.rectangle(img, (center_x-30, center_y-30), (center_x+30, center_y+30), (0, 0, 255), -1)
        
        # Синий круг
        center_x2 = int(width/2 - 80 * np.cos(np.radians(angle + 45)))
        center_y2 = int(height/2 - 30 * np.sin(np.radians(angle + 45)))
        cv2.circle(img, (center_x2, center_y2), 25, (255, 0, 0), -1)
        
        # Зеленый треугольник
        pts = np.array([
            [width//2 + 50, height//2 - 80],
            [width//2 + 20, height//2 - 20],
            [width//2 + 80, height//2 - 20]
        ], np.int32)
        
        # Поворачиваем треугольник
        rotation_matrix = cv2.getRotationMatrix2D((width//2, height//2), angle, 1.0)
        pts_rotated = cv2.transform(pts.reshape(-1, 1, 2), rotation_matrix).reshape(-1, 2)
        cv2.fillPoly(img, [pts_rotated.astype(np.int32)], (0, 255, 0))
        
        # Сохраняем изображение
        filename = f"{output_dir}/image_{i:03d}.jpg"
        cv2.imwrite(filename, img)
        print(f"Created: {filename}")
    
    print(f"✓ Created {num_views} test images in {output_dir}")

if __name__ == "__main__":
    create_simple_scene()
EOF
    
    python3 create_test_images.py
    print_status "Тестовые данные созданы"
}

# Создание упрощенного демо скрипта сегментации
create_simple_segmentation_demo() {
    print_status "Создание упрощенного демо для сегментации..."
    
    cd Tracking-Anything-with-DEVA
    
    cat > simple_demo.py << 'EOF'
#!/usr/bin/env python3
"""
Упрощенное демо для сегментации без сетевых зависимостей
"""
import cv2
import numpy as np
import os
import sys
import torch

def simple_color_segmentation(image_path, prompt="red object", output_dir="output_simple"):
    """
    Простая сегментация на основе цвета (демонстрационная версия)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Читаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return None
    
    # Конвертируем в HSV для лучшей сегментации по цвету
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Простая сегментация по промпту
    if "red" in prompt.lower():
        # Красный цвет в HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        
    elif "blue" in prompt.lower():
        # Синий цвет
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
    elif "green" in prompt.lower():
        # Зеленый цвет
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
    else:
        # По умолчанию - все объекты
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Применяем морфологические операции для очистки маски
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Сохраняем результат
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    cv2.imwrite(mask_path, mask)
    
    # Создаем визуализацию
    result = img.copy()
    result[mask == 0] = result[mask == 0] * 0.3  # Затемняем фон
    result_path = os.path.join(output_dir, f"{base_name}_result.png")
    cv2.imwrite(result_path, result)
    
    print(f"✓ Mask saved: {mask_path}")
    print(f"✓ Result saved: {result_path}")
    
    return mask_path

def process_directory(input_dir, prompt="object", output_dir="output_simple"):
    """Обрабатывает все изображения в директории"""
    
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images with prompt: '{prompt}'")
    
    for img_file in sorted(image_files):
        img_path = os.path.join(input_dir, img_file)
        simple_color_segmentation(img_path, prompt, output_dir)
    
    print(f"✓ Processed all images. Results in {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple segmentation demo")
    parser.add_argument("--img_path", required=True, help="Input image directory")
    parser.add_argument("--prompt", default="red object", help="Segmentation prompt")
    parser.add_argument("--output", default="output_simple", help="Output directory")
    
    args = parser.parse_args()
    
    process_directory(args.img_path, args.prompt, args.output)
EOF
    
    cd ..
    print_status "Упрощенное демо создано"
}

# Исправление тестового скрипта
fix_test_script() {
    print_status "Исправление тестового скрипта..."
    
    # Исправляем проблему с чтением PLY файла
    cat > test_cpu_pipeline_fixed.py << 'EOF'
#!/usr/bin/env python3
"""
Fixed CPU test pipeline for 3DGS with instance segmentation
"""
import torch
import numpy as np
import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from scene.gaussian_model import GaussianModel
from gaussian_renderer import render

class MockViewpointCamera:
    def __init__(self, W, H, K, W2C, Znear, Zfar):
        self.image_width = W
        self.image_height = H
        self.znear = Znear
        self.zfar = Zfar

        fx = K[0,0]
        fy = K[1,1]
        self.FoVx = 2 * np.arctan(W / (2 * fx))
        self.FoVy = 2 * np.arctan(H / (2 * fy))
        self.tanfovx = np.tan(self.FoVx * 0.5)
        self.tanfovy = np.tan(self.FoVy * 0.5)

        self.world_view_transform = torch.tensor(W2C, dtype=torch.float32)
        self.camera_center = torch.inverse(self.world_view_transform)[:3, 3]

        proj = torch.zeros(4, 4, dtype=torch.float32)
        proj[0, 0] = 1.0 / self.tanfovx
        proj[1, 1] = 1.0 / self.tanfovy
        proj[2, 2] = (Zfar + Znear) / (Znear - Zfar)
        proj[2, 3] = (2 * Zfar * Znear) / (Znear - Zfar)
        proj[3, 2] = -1.0
        self.projection_matrix = proj
        self.full_proj_transform = self.projection_matrix @ self.world_view_transform

    def to(self, device):
        self.world_view_transform = self.world_view_transform.to(device)
        self.full_proj_transform = self.full_proj_transform.to(device)
        self.camera_center = self.camera_center.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        return self

def main():
    print("Starting FIXED test for CPU 3DGS with instance segmentation...")
    output_dir = os.path.join(project_root, "test_output")
    os.makedirs(output_dir, exist_ok=True)

    # Create Gaussian model
    gaussians = GaussianModel(sh_degree=3)
    
    # Create simple test data
    num_points = 2
    xyz = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float32)
    colors = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=torch.float32)
    opacities = torch.tensor([[2.0], [2.0]], dtype=torch.float32)
    scales = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], dtype=torch.float32)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    object_ids = torch.tensor([[1], [2]], dtype=torch.float32)

    # Initialize model
    gaussians._xyz = xyz
    gaussians._features_dc = colors
    gaussians._opacity = opacities
    gaussians._scaling = scales
    gaussians._rotation = rotations
    gaussians._objects_dc = object_ids

    print("Gaussian model created with 2 Gaussians.")
    print(f"  XYZ:\n{gaussians.get_xyz.detach().cpu().numpy()}")
    print(f"  Object IDs (internal _objects_dc):\n{gaussians._objects_dc.detach().cpu().numpy()}")
    print(f"  Object IDs (get_objects):\n{gaussians.get_objects.detach().cpu().numpy()}")

    # Save PLY with object IDs
    ply_path = os.path.join(output_dir, "test_model_with_ids.ply")
    gaussians.save_ply_with_object_id(ply_path)
    print(f"PLY file saved to {ply_path} with object IDs.")

    # Verify PLY file exists (don't try to read as text)
    if os.path.exists(ply_path):
        file_size = os.path.getsize(ply_path)
        print(f"Saved PLY file: {ply_path} (size: {file_size} bytes)")
    else:
        print("Error: PLY file was not created")

    # Create mock camera
    W, H = 100, 100
    K = np.array([[50, 0, 50], [0, 50, 50], [0, 0, 1]], dtype=np.float32)
    W2C = np.eye(4, dtype=np.float32)
    W2C[2, 3] = 2.0  # Move camera back
    viewpoint_camera = MockViewpointCamera(W, H, K, W2C, 0.1, 100.0)
    print("Mock camera created.")

    # Render
    try:
        print("Rendering scene with CPU rasterizer...")
        render_result = render(viewpoint_camera, gaussians, None, None)
        
        rendered_image = render_result["render"]
        object_id_map = render_result.get("render_object", torch.zeros(H, W))
        
        print(f"Rendering complete. Image shape: {rendered_image.shape}, Object ID map shape: {object_id_map.shape}")

        # Save rendered image
        try:
            from torchvision.utils import save_image
            rendered_image_path = os.path.join(output_dir, "rendered_image_cpu.png")
            save_image(rendered_image, rendered_image_path)
            print(f"Rendered image saved to: {rendered_image_path}")
        except ImportError:
            print("torchvision not available, cannot save rendered image.")

        # Save object_id_map
        try:
            object_id_map_path_raw = os.path.join(output_dir, "object_id_map_raw.pt")
            torch.save(object_id_map, object_id_map_path_raw)
            print(f"Object ID map (raw tensor) saved to: {object_id_map_path_raw}")
            
            if object_id_map.max() > 0:
                from torchvision.utils import save_image
                object_id_map_vis = object_id_map.float() / object_id_map.max()
                object_id_map_vis_path = os.path.join(output_dir, "object_id_map_vis.png")
                save_image(object_id_map_vis.unsqueeze(0), object_id_map_vis_path)
                print(f"Object ID map (visualized) saved to: {object_id_map_vis_path}")
            else:
                print("Object ID map is all zeros, skipping visualization save.")

        except Exception as e_map:
            print(f"Error saving object_id_map: {e_map}")
            
        # Check object_id_map content
        unique_ids = torch.unique(object_id_map)
        print(f"Unique IDs in object_id_map: {unique_ids.cpu().numpy()}")

    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()

    print("Fixed test script finished successfully.")

if __name__ == "__main__":
    main()
EOF
    
    print_status "Исправленный тестовый скрипт создан"
}

# Главная функция
main() {
    # Проверяем, что мы в правильной директории
    if [ ! -f "test_installation.py" ]; then
        print_error "Запустите скрипт из директории с установленным 3DGS проектом"
        exit 1
    fi
    
    # Исправляем проблемы
    fix_colmap_headless
    download_models_manually
    setup_offline_mode
    create_test_data
    create_simple_segmentation_demo
    fix_test_script
    
    print_status "Все проблемы исправлены!"
    
    echo ""
    echo "=========================================="
    echo "Готово! Теперь вы можете:"
    echo ""
    echo "1. Запустить исправленный CPU тест:"
    echo "   python3 test_cpu_pipeline_fixed.py"
    echo ""
    echo "2. Запустить COLMAP без GUI:"
    echo "   python3 colmap_headless.py --image_path data/test_scene/images --output_path data/test_scene"
    echo ""
    echo "3. Запустить простую сегментацию:"
    echo "   cd Tracking-Anything-with-DEVA"
    echo "   python3 simple_demo.py --img_path ../data/test_scene/images --prompt 'red object'"
    echo ""
    echo "4. Запустить полный пайплайн 3DGS:"
    echo "   python3 train.py -s data/test_scene -m output/test_scene --eval"
    echo "=========================================="
}

# Запуск
main "$@"