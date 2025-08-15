#!/usr/bin/env python3
"""
КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ МАСОК
Конвертируем RGB маски с множественными значениями в бинарные маски 0/1
"""

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def convert_masks_to_binary():
    """Конвертируем все маски в бинарный формат"""
    
    mask_dir = 'data/dataset/object_mask'
    backup_dir = 'data/dataset/object_mask_original'
    
    # Создаем бэкап оригинальных масок
    if not os.path.exists(backup_dir):
        print("📦 Создаем бэкап оригинальных масок...")
        os.makedirs(backup_dir)
        
        # Копируем все файлы в бэкап
        import shutil
        for filename in os.listdir(mask_dir):
            if filename.endswith('.png'):
                src = os.path.join(mask_dir, filename)
                dst = os.path.join(backup_dir, filename)
                shutil.copy2(src, dst)
        
        print(f"✅ Бэкап создан в {backup_dir}")
    
    # Получаем список масок
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
    print(f"🔄 Конвертируем {len(mask_files)} масок...")
    
    converted_count = 0
    
    for filename in tqdm(mask_files, desc="Конвертация масок"):
        mask_path = os.path.join(mask_dir, filename)
        
        try:
            # Загружаем маску
            mask = np.array(Image.open(mask_path))
            
            # Если RGB, конвертируем в grayscale
            if len(mask.shape) == 3:
                # Берем первый канал или среднее
                mask = mask[:, :, 0]  # или np.mean(mask, axis=2)
            
            # Бинаризуем: все что > 0 становится 1
            binary_mask = (mask > 0).astype(np.uint8)
            
            # Проверяем что получили правильную бинарную маску
            unique_vals = np.unique(binary_mask)
            if set(unique_vals) <= {0, 1}:
                # Сохраняем бинарную маску
                Image.fromarray(binary_mask).save(mask_path)
                converted_count += 1
            else:
                print(f"⚠️  Проблема с {filename}: {unique_vals}")
                
        except Exception as e:
            print(f"❌ Ошибка с {filename}: {e}")
    
    print(f"✅ Конвертировано {converted_count} масок")
    
    # Проверяем результат
    print("\n🔍 Проверка результата:")
    test_files = mask_files[:3]
    
    for filename in test_files:
        try:
            mask = np.array(Image.open(os.path.join(mask_dir, filename)))
            unique_vals = np.unique(mask)
            object_percent = (mask > 0).sum() / mask.size * 100
            
            print(f"{filename}:")
            print(f"  Размер: {mask.shape}")
            print(f"  Значения: {unique_vals}")
            print(f"  Объект: {object_percent:.1f}%")
            
            if set(unique_vals) == {0, 1}:
                print("  ✅ Правильная бинарная маска")
            else:
                print("  ❌ Все еще не бинарная!")
            print()
            
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")

def check_camera_loading():
    """Проверяем как загружаются маски в камерах"""
    print("\n🔍 Проверяем загрузку масок в dataset...")
    
    # Проверим что dataset правильно загружает маски
    test_code = '''
import sys
sys.path.append('.')
from scene.dataset_readers import readColmapSceneInfo
from arguments import ModelParams

# Создаем тестовые аргументы
class TestArgs:
    source_path = "data/dataset"
    images = "images"
    eval = False
    object_path = "object_mask"
    n_views = 100
    random_init = False
    train_split = False

try:
    scene_info = readColmapSceneInfo(
        TestArgs.source_path, 
        TestArgs.images, 
        TestArgs.eval, 
        TestArgs.object_path,
        n_views=TestArgs.n_views,
        random_init=TestArgs.random_init,
        train_split=TestArgs.train_split
    )
    
    # Проверяем первую камеру
    if scene_info.train_cameras and len(scene_info.train_cameras) > 0:
        cam = scene_info.train_cameras[0]
        if hasattr(cam, 'objects') and cam.objects is not None:
            print(f"✅ Объекты загружены: {cam.objects.shape}")
            if hasattr(cam.objects, 'unique'):
                unique_vals = cam.objects.unique() if hasattr(cam.objects, 'unique') else set(cam.objects.flatten())
                print(f"  Уникальные значения: {unique_vals}")
            else:
                import numpy as np
                unique_vals = np.unique(cam.objects)
                print(f"  Уникальные значения: {unique_vals}")
        else:
            print("❌ Объекты не загружены")
    else:
        print("❌ Нет камер")
        
except Exception as e:
    print(f"❌ Ошибка загрузки: {e}")
    '''
    
    # Сохраняем и выполняем проверку
    with open('test_loading.py', 'w') as f:
        f.write(test_code)
    
    print("Выполняем проверку загрузки...")
    os.system('python3 test_loading.py')
    os.remove('test_loading.py')

if __name__ == "__main__":
    print("🚨 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ МАСОК СЕГМЕНТАЦИИ")
    print("=" * 60)
    
    convert_masks_to_binary()
    check_camera_loading()
    
    print("\n🎯 МАСКИ ИСПРАВЛЕНЫ!")
    print("Теперь запускайте train.py снова:")
    print("python3 train.py -s data/dataset -r 1 -m output/dataset --config_file config/gaussian_dataset/train.json")
    print("\nОжидаем:")
    print("- ✅ Object loss: 0.1-2.0 (вместо 85-98)")
    print("- ✅ Размер масок: (H, W) вместо (H, W, 3)")
    print("- ✅ Значения: [0, 1] вместо [0, 102, 221, 225]")