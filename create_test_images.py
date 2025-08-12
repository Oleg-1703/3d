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
