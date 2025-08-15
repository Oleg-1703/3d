#!/usr/bin/env python3
"""
Скрипт для ручного сохранения обученной модели
Используйте если модель обучилась но не сохранилась
"""

import sys
import torch
import os
from argparse import ArgumentParser, Namespace

sys.path.append('.')
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state

def save_model_manually(model_path, output_iteration=30000):
    """Ручное сохранение модели из памяти"""
    
    print(f"🔄 Загружаем модель из {model_path}...")
    
    # Создаем аргументы для загрузки
    class Args:
        source_path = "data/dataset"
        model_path = model_path
        images = "images"
        eval = False
        resolution = 1
        white_background = False
        data_device = "cuda"
        sh_degree = 3
    
    args = Args()
    
    try:
        # Загружаем модель
        gaussians = GaussianModel(args.sh_degree)
        scene = Scene(ModelParams().extract(args), gaussians, shuffle=False)
        
        print(f"✓ Модель загружена: {gaussians.get_xyz.shape[0]} точек")
        
        # Создаем папку для сохранения
        save_path = os.path.join(model_path, f"point_cloud/iteration_{output_iteration}")
        os.makedirs(save_path, exist_ok=True)
        
        # Сохраняем point cloud
        ply_path = os.path.join(save_path, "point_cloud.ply")
        gaussians.save_ply(ply_path)
        print(f"✓ Point cloud сохранен: {ply_path}")
        
        # Сохраняем с object IDs если есть
        try:
            ply_with_objects_path = os.path.join(save_path, "point_cloud_with_objects.ply")
            gaussians.save_ply_with_object_id(ply_with_objects_path)
            print(f"✓ Point cloud с object IDs сохранен: {ply_with_objects_path}")
        except Exception as e:
            print(f"⚠️  Не удалось сохранить с object IDs: {e}")
        
        # Проверяем что классификатор уже есть
        classifier_path = os.path.join(save_path, "classifier.pth")
        if os.path.exists(classifier_path):
            print(f"✓ Классификатор уже существует: {classifier_path}")
        else:
            print(f"❌ Классификатор отсутствует: {classifier_path}")
        
        print(f"\n🎉 МОДЕЛЬ УСПЕШНО СОХРАНЕНА!")
        print(f"📁 Папка: {save_path}")
        print(f"📄 Файлы:")
        for file in os.listdir(save_path):
            file_path = os.path.join(save_path, file)
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"   {file} ({size:.1f} MB)")
            
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True, help="Путь к модели")
    parser.add_argument("-i", "--iteration", type=int, default=30000, help="Итерация для сохранения")
    
    args = parser.parse_args()
    
    if save_model_manually(args.model_path, args.iteration):
        print("\n✅ Готово! Теперь можно запускать:")
        print(f"python3 render.py -m {args.model_path} --num_classes 2")
    else:
        print("\n❌ Сохранение не удалось")
