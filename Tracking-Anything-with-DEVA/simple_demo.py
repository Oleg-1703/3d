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
