#!/usr/bin/env python3
import struct
import shutil
import os

def convert_camera_to_pinhole(input_path, output_path):
    """Конвертирует SIMPLE_RADIAL в PINHOLE модель"""
    
    # Создаем выходную папку
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Читаем исходный файл
    with open(input_path, 'rb') as f:
        # Читаем количество камер
        num_cameras = struct.unpack('Q', f.read(8))[0]
        print(f"Обрабатываем {num_cameras} камер(ы)")
        
        # Подготавливаем данные для записи
        output_data = struct.pack('Q', num_cameras)
        
        for i in range(num_cameras):
            # Читаем данные камеры
            camera_id = struct.unpack('I', f.read(4))[0]
            model_id = struct.unpack('I', f.read(4))[0] 
            width = struct.unpack('Q', f.read(8))[0]
            height = struct.unpack('Q', f.read(8))[0]
            
            print(f"Камера {camera_id}: модель {model_id}, размер {width}x{height}")
            
            if model_id == 2:  # SIMPLE_RADIAL
                # Читаем параметры SIMPLE_RADIAL: fx, cx, cy, k
                fx = struct.unpack('d', f.read(8))[0]
                cx = struct.unpack('d', f.read(8))[0] 
                cy = struct.unpack('d', f.read(8))[0]
                k = struct.unpack('d', f.read(8))[0]  # distortion - игнорируем
                
                print(f"  SIMPLE_RADIAL: fx={fx:.2f}, cx={cx:.2f}, cy={cy:.2f}, k={k:.6f}")
                
                # Конвертируем в PINHOLE (модель 1): fx, fy, cx, cy
                # Предполагаем fy = fx (квадратные пиксели)
                fy = fx
                
                # Записываем как PINHOLE
                output_data += struct.pack('I', camera_id)  # camera_id
                output_data += struct.pack('I', 1)          # model_id = 1 (PINHOLE)
                output_data += struct.pack('Q', width)      # width
                output_data += struct.pack('Q', height)     # height
                output_data += struct.pack('d', fx)         # fx
                output_data += struct.pack('d', fy)         # fy 
                output_data += struct.pack('d', cx)         # cx
                output_data += struct.pack('d', cy)         # cy
                
                print(f"  -> PINHOLE: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                
            else:
                print(f"  Неожиданная модель камеры: {model_id}")
                return False
    
    # Записываем новый файл камер
    with open(output_path, 'wb') as f:
        f.write(output_data)
    
    print(f"✓ Модель камеры конвертирована в PINHOLE и сохранена в {output_path}")
    return True

if __name__ == "__main__":
    # Создаем папки
    os.makedirs('data/dataset/sparse_fixed/0', exist_ok=True)
    
    # Конвертируем модель камеры
    success = convert_camera_to_pinhole('data/dataset/sparse/0/cameras.bin', 'data/dataset/sparse_fixed/0/cameras.bin')
    
    if success:
        # Копируем остальные файлы
        shutil.copy2('data/dataset/sparse/0/images.bin', 'data/dataset/sparse_fixed/0/images.bin')
        shutil.copy2('data/dataset/sparse/0/points3D.bin', 'data/dataset/sparse_fixed/0/points3D.bin')
        print("Остальные файлы скопированы")
        print("Конвертация завершена успешно!")
    else:
        print("Ошибка конвертации!")
