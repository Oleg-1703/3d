#!/usr/bin/env python3
"""
КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ МАСОК
Конвертируем RGB маски с множественными значениями в бинарные маски 0/1
"""

import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil


def convert_masks_to_binary(mask_dir: str):
    """Конвертируем все маски в бинарный формат"""
    
    backup_dir = mask_dir.rstrip("/\\") + "_original"

    # Создаем бэкап оригинальных масок
    if not os.path.exists(backup_dir):
        print("📦 Создаем бэкап оригинальных масок...")
        os.makedirs(backup_dir)

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
            mask = np.array(Image.open(mask_path))

            if len(mask.shape) == 3:
                mask = mask[:, :, 0]

            binary_mask = (mask > 0).astype(np.uint8)

            unique_vals = np.unique(binary_mask)
            if set(unique_vals) <= {0, 1}:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Конвертация RGB масок в бинарные")
    parser.add_argument("--mask_dir", type=str, required=True, help="Путь к папке с масками (.png)")
    args = parser.parse_args()

    print("🚨 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ МАСОК СЕГМЕНТАЦИИ")
    print("=" * 60)

    convert_masks_to_binary(args.mask_dir)

    print("\n🎯 МАСКИ ИСПРАВЛЕНЫ!")
