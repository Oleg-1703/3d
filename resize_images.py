import os
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil

def resize_images_for_processing(dataset_path, max_size=1920):
    """Уменьшает изображения для обработки DEVA, сохраняя оригиналы"""
    
    dataset_path = Path(dataset_path)
    images_path = dataset_path / "images"
    originals_path = dataset_path / "images_original"
    resized_path = dataset_path / "images"

    # Если папка images существует — переименовываем в images_original
    if images_path.exists() and not originals_path.exists():
        images_path.rename(originals_path)
    elif originals_path.exists():
        print("⚠ Папка images_original уже существует. Будут использоваться изображения из неё.")
    else:
        raise FileNotFoundError("Не найдена папка images или images_original")

    # Создаём новую папку images для уменьшенных
    resized_path.mkdir(exist_ok=True)

    # Получаем список изображений из оригиналов
    images = sorted(list(originals_path.glob("*.jpg")) + 
                    list(originals_path.glob("*.png")) + 
                    list(originals_path.glob("*.JPG")))
    
    for img_path in tqdm(images, desc="Ресайз изображений"):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_resized = img
            
        cv2.imwrite(str(resized_path / img_path.name), img_resized)
    
    print(f"Сохранено {len(images)} уменьшенных изображений в {resized_path}")
    print("Оригиналы сохранены в:", originals_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ресайз изображений для обработки DEVA")
    parser.add_argument("dataset_path", type=str, help="Путь к датасету")
    parser.add_argument("--max_size", type=int, default=1920, help="Максимальная сторона изображения (по умолчанию 1920)")
    
    args = parser.parse_args()
    
    resize_images_for_processing(args.dataset_path, max_size=args.max_size)
