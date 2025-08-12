import os
import cv2
from pathlib import Path
from tqdm import tqdm

def resize_images_for_processing(dataset_path, max_size=1920):
    """Уменьшает изображения для обработки DEVA"""
    
    images_path = Path(dataset_path) / "images"
    resized_path = Path(dataset_path) / "images_resized"
    resized_path.mkdir(exist_ok=True)
    
    images = sorted(list(images_path.glob("*.jpg")) + list(images_path.glob("*.png")) + list(images_path.glob("*.JPG")))
    
    for img_path in tqdm(images, desc="Ресайз изображений"):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Уменьшаем если больше max_size
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img
            
        # Сохраняем
        cv2.imwrite(str(resized_path / img_path.name), img_resized)
    
    print(f"Сохранено {len(images)} изображений в {resized_path}")
    
    print("Папки переименованы: images_original -> оригиналы, images -> уменьшенные")

resize_images_for_processing("data/dataset", max_size=256)
