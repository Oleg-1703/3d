#!/usr/bin/env python3
"""
Скрипт для проверки и исправления модели камеры COLMAP
"""
import struct
import os
import shutil

def read_camera_model(cameras_path):
    """Читает модель камеры из cameras.bin"""
    with open(cameras_path, 'rb') as f:
        num_cameras = struct.unpack('Q', f.read(8))[0]
        print(f"Количество камер: {num_cameras}")
        
        for i in range(num_cameras):
            camera_id = struct.unpack('I', f.read(4))[0]
            model_id = struct.unpack('I', f.read(4))[0]
            width = struct.unpack('Q', f.read(8))[0]
            height = struct.unpack('Q', f.read(8))[0]
            
            # Определяем модель по ID
            model_names = {
                0: "SIMPLE_PINHOLE",
                1: "PINHOLE", 
                2: "SIMPLE_RADIAL",
                3: "RADIAL",
                4: "OPENCV",
                5: "OPENCV_FISHEYE",
                6: "FULL_OPENCV",
                7: "FOV",
                8: "SIMPLE_RADIAL_FISHEYE",
                9: "RADIAL_FISHEYE",
                10: "THIN_PRISM_FISHEYE"
            }
            
            model_name = model_names.get(model_id, f"UNKNOWN_{model_id}")
            print(f"Камера {camera_id}: модель {model_name} (ID: {model_id}), размер {width}x{height}")
            
            # Читаем параметры в зависимости от модели
            if model_id == 0:  # SIMPLE_PINHOLE
                f.read(3 * 8)  # 3 параметра
                print("  ✓ SIMPLE_PINHOLE - поддерживается")
            elif model_id == 1:  # PINHOLE
                f.read(4 * 8)  # 4 параметра
                print("  ✓ PINHOLE - поддерживается")
            elif model_id == 2:  # SIMPLE_RADIAL
                f.read(4 * 8)  # 4 параметра
                print("  ❌ SIMPLE_RADIAL - требует конвертации")
            elif model_id == 4:  # OPENCV
                f.read(8 * 8)  # 8 параметров
                print("  ❌ OPENCV - требует конвертации")
            else:
                print(f"  ❌ Модель {model_name} не поддерживается")

def convert_to_pinhole(input_path, output_path):
    """Конвертирует камеру в PINHOLE модель"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'rb') as f:
        num_cameras = struct.unpack('Q', f.read(8))[0]
        output_data = struct.pack('Q', num_cameras)
        
        for i in range(num_cameras):
            camera_id = struct.unpack('I', f.read(4))[0]
            model_id = struct.unpack('I', f.read(4))[0]
            width = struct.unpack('Q', f.read(8))[0]
            height = struct.unpack('Q', f.read(8))[0]
            
            if model_id == 2:  # SIMPLE_RADIAL -> PINHOLE
                fx = struct.unpack('d', f.read(8))[0]
                cx = struct.unpack('d', f.read(8))[0]
                cy = struct.unpack('d', f.read(8))[0]
                k = struct.unpack('d', f.read(8))[0]  # искажение - игнорируем
                
                print(f"Конвертируем SIMPLE_RADIAL -> PINHOLE")
                print(f"  fx={fx:.2f}, cx={cx:.2f}, cy={cy:.2f} (k={k:.6f} игнорируется)")
                
                # Записываем как PINHOLE: fx, fy, cx, cy
                output_data += struct.pack('I', camera_id)
                output_data += struct.pack('I', 1)  # PINHOLE model_id
                output_data += struct.pack('Q', width)
                output_data += struct.pack('Q', height)
                output_data += struct.pack('d', fx)  # fx
                output_data += struct.pack('d', fx)  # fy = fx (квадратные пиксели)
                output_data += struct.pack('d', cx)  # cx
                output_data += struct.pack('d', cy)  # cy
                
            elif model_id == 4:  # OPENCV -> PINHOLE
                fx = struct.unpack('d', f.read(8))[0]
                fy = struct.unpack('d', f.read(8))[0]
                cx = struct.unpack('d', f.read(8))[0]
                cy = struct.unpack('d', f.read(8))[0]
                # Пропускаем искажения
                for _ in range(4):
                    struct.unpack('d', f.read(8))[0]
                
                print(f"Конвертируем OPENCV -> PINHOLE")
                print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                
                # Записываем как PINHOLE
                output_data += struct.pack('I', camera_id)
                output_data += struct.pack('I', 1)  # PINHOLE model_id
                output_data += struct.pack('Q', width)
                output_data += struct.pack('Q', height)
                output_data += struct.pack('d', fx)
                output_data += struct.pack('d', fy)
                output_data += struct.pack('d', cx)
                output_data += struct.pack('d', cy)
                
            elif model_id in [0, 1]:  # SIMPLE_PINHOLE или PINHOLE
                # Просто копируем данные
                if model_id == 0:
                    params_data = f.read(3 * 8)
                else:
                    params_data = f.read(4 * 8)
                    
                output_data += struct.pack('I', camera_id)
                output_data += struct.pack('I', model_id)
                output_data += struct.pack('Q', width)
                output_data += struct.pack('Q', height)
                output_data += params_data
                print(f"Модель уже поддерживается, копируем без изменений")
                
            else:
                print(f"❌ Неподдерживаемая модель {model_id}")
                return False
    
    with open(output_path, 'wb') as f:
        f.write(output_data)
    
    print(f"✓ Сохранено в {output_path}")
    return True

def main():
    dataset_path = "data/dataset1024"
    sparse_path = os.path.join(dataset_path, "sparse", "0")
    cameras_path = os.path.join(sparse_path, "cameras.bin")
    
    print("=== Проверка модели камеры ===")
    
    if not os.path.exists(cameras_path):
        print(f"❌ Файл {cameras_path} не найден")
        print("Возможные решения:")
        print("1. Запустите COLMAP заново")
        print("2. Проверьте путь к датасету")
        return
    
    print(f"Проверяем {cameras_path}")
    read_camera_model(cameras_path)
    
    print("\n=== Исправление модели камеры ===")
    fixed_sparse_path = os.path.join(dataset_path, "sparse_fixed", "0")
    fixed_cameras_path = os.path.join(fixed_sparse_path, "cameras.bin")
    
    if convert_to_pinhole(cameras_path, fixed_cameras_path):
        # Копируем остальные файлы
        shutil.copy2(os.path.join(sparse_path, "images.bin"), 
                    os.path.join(fixed_sparse_path, "images.bin"))
        shutil.copy2(os.path.join(sparse_path, "points3D.bin"), 
                    os.path.join(fixed_sparse_path, "points3D.bin"))
        
        print("✓ Все файлы скопированы в sparse_fixed/0/")
        print("\nТеперь запустите обучение с исправленными данными:")
        print(f"python3 train.py -s {dataset_path} -r 1 -m output/dataset1024_fixed --config_file config/gaussian_dataset/train.json")
        print("\nИли создайте символическую ссылку:")
        print(f"mv {sparse_path} {sparse_path}_original")
        print(f"mv {fixed_sparse_path} {sparse_path}")
    else:
        print("❌ Ошибка конвертации")

if __name__ == "__main__":
    main()