#!/usr/bin/env python3
"""
Диагностика проблемы с пустым списком train_cam_infos
"""
import os
import sys
import struct
import numpy as np
from PIL import Image

def debug_dataset(dataset_path="data/dataset1024"):
    """Диагностирует проблему с загрузкой камер"""
    
    print("=== ПОЛНАЯ ДИАГНОСТИКА ДАТАСЕТА ===")
    print(f"Папка датасета: {dataset_path}")
    
    # 1. Проверка структуры папок
    print("\n1. СТРУКТУРА ПАПОК:")
    for folder in ["images", "objects", "sparse/0", "input"]:
        path = os.path.join(dataset_path, folder)
        exists = os.path.exists(path)
        print(f"  {folder}: {'✅' if exists else '❌'}")
        if exists and os.path.isdir(path):
            files = [f for f in os.listdir(path) if not f.startswith('.')]
            print(f"    Файлов: {len(files)}")
            if len(files) <= 10:
                print(f"    Содержимое: {files}")
    
    # 2. Проверка COLMAP файлов
    print("\n2. COLMAP ФАЙЛЫ:")
    sparse_path = os.path.join(dataset_path, "sparse/0")
    colmap_files = ["cameras.bin", "images.bin", "points3D.bin"]
    
    for file in colmap_files:
        file_path = os.path.join(sparse_path, file)
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        print(f"  {file}: {'✅' if exists else '❌'} ({size} bytes)")
    
    # 3. Анализ cameras.bin
    print("\n3. АНАЛИЗ CAMERAS.BIN:")
    cameras_path = os.path.join(sparse_path, "cameras.bin")
    if os.path.exists(cameras_path):
        try:
            with open(cameras_path, 'rb') as f:
                num_cameras = struct.unpack('Q', f.read(8))[0]
                print(f"  Количество камер: {num_cameras}")
                
                for i in range(num_cameras):
                    camera_id = struct.unpack('I', f.read(4))[0]
                    model_id = struct.unpack('I', f.read(4))[0]
                    width = struct.unpack('Q', f.read(8))[0]
                    height = struct.unpack('Q', f.read(8))[0]
                    
                    model_names = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 4: "OPENCV"}
                    model_name = model_names.get(model_id, f"UNKNOWN_{model_id}")
                    
                    print(f"  Камера {camera_id}: {model_name} {width}x{height}")
                    
                    # Пропускаем параметры камеры
                    if model_id == 0:  # SIMPLE_PINHOLE
                        f.read(3 * 8)
                    elif model_id == 1:  # PINHOLE
                        f.read(4 * 8)
                    elif model_id == 2:  # SIMPLE_RADIAL
                        f.read(4 * 8)
                    elif model_id == 4:  # OPENCV
                        f.read(8 * 8)
        except Exception as e:
            print(f"  ❌ Ошибка чтения: {e}")
    
    # 4. Анализ images.bin
    print("\n4. АНАЛИЗ IMAGES.BIN:")
    images_path = os.path.join(sparse_path, "images.bin")
    if os.path.exists(images_path):
        try:
            sys.path.append('.')
            from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
            
            cam_extrinsics = read_extrinsics_binary(images_path)
            cam_intrinsics = read_intrinsics_binary(cameras_path)
            
            print(f"  Загружено extrinsics: {len(cam_extrinsics)}")
            print(f"  Загружено intrinsics: {len(cam_intrinsics)}")
            
            print("\n  Соответствие extrinsics -> intrinsics:")
            valid_cameras = 0
            for img_id, extr in cam_extrinsics.items():
                camera_id = extr.camera_id
                image_name = extr.name
                if camera_id in cam_intrinsics:
                    print(f"    ✅ {image_name} -> камера {camera_id}")
                    valid_cameras += 1
                else:
                    print(f"    ❌ {image_name} -> камера {camera_id} НЕ НАЙДЕНА")
            
            print(f"  Валидных связей: {valid_cameras}")
            
        except Exception as e:
            print(f"  ❌ Ошибка загрузки: {e}")
    
    # 5. Проверка изображений и масок
    print("\n5. СООТВЕТСТВИЕ ИЗОБРАЖЕНИЙ И МАСОК:")
    images_folder = os.path.join(dataset_path, "images")
    objects_folder = os.path.join(dataset_path, "objects")
    
    if os.path.exists(images_folder) and os.path.exists(objects_folder):
        image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        object_files = [f for f in os.listdir(objects_folder) if f.lower().endswith('.png')]
        
        print(f"  Изображений: {len(image_files)}")
        print(f"  Масок: {len(object_files)}")
        
        # Проверяем соответствие
        matched = 0
        for img_file in image_files[:5]:  # Проверяем первые 5
            base_name = os.path.splitext(img_file)[0]
            mask_file = base_name + '.png'
            mask_path = os.path.join(objects_folder, mask_file)
            
            if os.path.exists(mask_path):
                print(f"    ✅ {img_file} -> {mask_file}")
                matched += 1
            else:
                print(f"    ❌ {img_file} -> {mask_file} НЕ НАЙДЕНА")
        
        print(f"  Совпадений (первые 5): {matched}")

def test_readColmapCameras():
    """Тестирует загрузку камер напрямую"""
    print("\n=== ТЕСТ READCOLMAPCAMERAS ===")
    
    try:
        sys.path.append('.')
        from scene.dataset_readers import readColmapCameras
        from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
        
        dataset_path = "data/dataset1024"
        sparse_path = os.path.join(dataset_path, "sparse/0")
        
        # Загружаем данные
        cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_path, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_path, "cameras.bin"))
        
        images_folder = os.path.join(dataset_path, "images")
        objects_folder = os.path.join(dataset_path, "objects")
        
        print(f"Параметры:")
        print(f"  cam_extrinsics: {len(cam_extrinsics)} элементов")
        print(f"  cam_intrinsics: {len(cam_intrinsics)} элементов") 
        print(f"  images_folder: {images_folder}")
        print(f"  objects_folder: {objects_folder}")
        
        # Вызываем функцию
        cam_infos = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=images_folder,
            objects_folder=objects_folder
        )
        
        print(f"\nРезультат: {len(cam_infos)} камер загружено")
        
        if len(cam_infos) == 0:
            print("\n❌ СПИСОК КАМЕР ПУСТОЙ! Диагностика:")
            
            # Проверяем каждую камеру отдельно
            for img_id, extr in list(cam_extrinsics.items())[:3]:
                camera_id = extr.camera_id
                image_name = extr.name
                
                print(f"\n  Камера {img_id} ('{image_name}'):")
                
                # 1. Проверяем intrinsics
                if camera_id not in cam_intrinsics:
                    print(f"    ❌ Camera ID {camera_id} не найден в intrinsics")
                    continue
                else:
                    intr = cam_intrinsics[camera_id]
                    print(f"    ✅ Intrinsics: {intr.model} {intr.width}x{intr.height}")
                
                # 2. Проверяем файл изображения
                image_path = os.path.join(images_folder, image_name)
                if not os.path.exists(image_path):
                    print(f"    ❌ Изображение не найдено: {image_path}")
                    continue
                else:
                    print(f"    ✅ Изображение найдено: {image_path}")
                
                # 3. Проверяем маску
                object_path = os.path.join(objects_folder, image_name + '.png')
                if not os.path.exists(object_path):
                    print(f"    ❌ Маска не найдена: {object_path}")
                    continue
                else:
                    print(f"    ✅ Маска найдена: {object_path}")
                
                # 4. Проверяем модель камеры
                if intr.model not in ["SIMPLE_PINHOLE", "PINHOLE"]:
                    print(f"    ❌ Неподдерживаемая модель: {intr.model}")
                    continue
                else:
                    print(f"    ✅ Модель камеры поддерживается: {intr.model}")
                
                print(f"    ✅ Эта камера должна загружаться нормально")
        else:
            for i, cam_info in enumerate(cam_infos[:3]):
                print(f"  Камера {i}: {cam_info.image_name} {cam_info.width}x{cam_info.height}")
                
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dataset()
    test_readColmapCameras()