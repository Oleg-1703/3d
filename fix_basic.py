#!/usr/bin/env python3
"""
НЕМЕДЛЕННОЕ ИСПРАВЛЕНИЕ - запустите этот скрипт и сразу тренируйте
"""

import os
import sys

print("🚀 НЕМЕДЛЕННОЕ ИСПРАВЛЕНИЕ train.py")
print("=" * 50)

# ===== 1. ИСПРАВЛЯЕМ BASICPOINTCLOUD =====
print("1️⃣ Исправляем BasicPointCloud...")

gaussian_model_file = "scene/gaussian_model.py"

if os.path.exists(gaussian_model_file):
    with open(gaussian_model_file, 'r') as f:
        content = f.read()
    
    # Проверяем есть ли BasicPointCloud
    if 'class BasicPointCloud' not in content:
        # Добавляем в начало файла
        basic_point_cloud = '''from typing import NamedTuple
import numpy as np

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

'''
        # Вставляем после первых импортов
        lines = content.split('\\n')
        insert_index = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_index = max(insert_index, i + 1)
        
        lines.insert(insert_index, basic_point_cloud)
        content = '\\n'.join(lines)
        
        with open(gaussian_model_file, 'w') as f:
            f.write(content)
        
        print("✅ BasicPointCloud добавлен в scene/gaussian_model.py")
    else:
        print("✅ BasicPointCloud уже существует")
else:
    print("❌ scene/gaussian_model.py не найден")

# ===== 2. ИСПРАВЛЯЕМ TRAIN.PY =====
print("\\n2️⃣ Исправляем device errors в train.py...")

if os.path.exists('train.py'):
    with open('train.py', 'r') as f:
        train_content = f.read()
    
    # Создаем бэкап
    with open('train.py.backup', 'w') as f:
        f.write(train_content)
    
    print("✅ Создан бэкап train.py.backup")
    
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 1: max_radii2D
    old_problem = 'gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])'
    new_solution = '''# ИСПРАВЛЕНИЕ DEVICE ERROR
                if gaussians.max_radii2D.device != radii.device:
                    radii = radii.to(gaussians.max_radii2D.device)
                if gaussians.max_radii2D.device != visibility_filter.device:
                    visibility_filter = visibility_filter.to(gaussians.max_radii2D.device)
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])'''
    
    if old_problem in train_content:
        train_content = train_content.replace(old_problem, new_solution)
        print("✅ Исправлена ошибка max_radii2D")
    
    # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ 2: cuda() calls
    cuda_fixes = [
        ('gt_obj = viewpoint_cam.objects.cuda().long()', 
         'device = next(gaussians.parameters()).device if gaussians.get_xyz.numel() > 0 else torch.device("cuda")\\n        gt_obj = viewpoint_cam.objects.to(device).long() if hasattr(viewpoint_cam, "objects") and viewpoint_cam.objects is not None else None'),
        
        ('objects.float().cuda().unsqueeze', 'objects.float().to(device).unsqueeze'),
        
        ('background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")', 
         'device = next(gaussians.parameters()).device if gaussians.get_xyz.numel() > 0 else torch.device("cuda")\\n    background = torch.tensor(bg_color, dtype=torch.float32, device=device)'),
    ]
    
    for old, new in cuda_fixes:
        if old in train_content:
            train_content = train_content.replace(old, new)
            print(f"✅ Исправлено: {old[:40]}...")
    
    # Сохраняем исправленный файл
    with open('train.py', 'w') as f:
        f.write(train_content)
    
    print("✅ train.py полностью исправлен!")

else:
    print("❌ train.py не найден")

# ===== 3. ПРОВЕРЯЕМ СТРУКТУРУ =====
print("\\n3️⃣ Проверяем структуру данных...")

required_dirs = [
    "data/dataset/images",
    "data/dataset/sparse/0", 
    "config/gaussian_dataset",
    "output"
]

for dir_path in required_dirs:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Создана директория: {dir_path}")
    else:
        print(f"✅ Существует: {dir_path}")

# ===== 4. СОЗДАЕМ КОНФИГУРАЦИЮ =====
config_path = "config/gaussian_dataset/train.json"
if not os.path.exists(config_path):
    config_content = '''{
    "densify_until_iter": 10000,
    "num_classes": 256,
    "reg3d_interval": 5,
    "reg3d_k": 5,
    "reg3d_lambda_val": 2,
    "reg3d_max_points": 200000,
    "reg3d_sample_size": 1000
}'''
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"✅ Создана конфигурация: {config_path}")

# ===== 5. ПРОВЕРЯЕМ ДАННЫЕ =====
print("\\n4️⃣ Проверяем ваши данные...")

if os.path.exists("data/dataset/images"):
    images = [f for f in os.listdir("data/dataset/images") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📸 Найдено изображений: {len(images)}")
    if len(images) == 0:
        print("⚠️  Поместите ваши изображения в data/dataset/images/")
else:
    print("❌ Директория data/dataset/images не найдена")

if os.path.exists("data/dataset/sparse/0"):
    colmap_files = ['cameras.bin', 'images.bin', 'points3D.bin']
    found_files = [f for f in colmap_files if os.path.exists(f"data/dataset/sparse/0/{f}")]
    print(f"🗂️  COLMAP файлов: {len(found_files)}/{len(colmap_files)}")
    if len(found_files) == 0:
        print("⚠️  Поместите COLMAP данные в data/dataset/sparse/0/")
        print("   Нужны файлы: cameras.bin, images.bin, points3D.bin")
else:
    print("❌ Директория data/dataset/sparse/0 не найдена")

# ===== ФИНАЛ =====
print("\\n" + "=" * 50)
print("🎯 ВСЕ ИСПРАВЛЕНИЯ ПРИМЕНЕНЫ!")
