# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from sklearn.decomposition import PCA

def feature_to_rgb(features):
    """Преобразуем фичи в RGB для визуализации"""
    if features.dim() == 2:  # [H, W]
        # Для 2D создаем псевдо-фичи
        H, W = features.shape
        rgb_array = np.zeros((H, W, 3), dtype=np.uint8)
        # Простая визуализация на основе значений
        normalized = (features.cpu().numpy() * 255).astype(np.uint8)
        rgb_array[:, :, 0] = normalized  # Красный канал
        return rgb_array
    elif features.dim() == 3:  # [C, H, W]
        # Берем первые 3 канала или дублируем
        H, W = features.shape[1], features.shape[2]
        if features.shape[0] >= 3:
            rgb_array = features[:3].permute(1, 2, 0).cpu().numpy()
        else:
            # Дублируем первый канал
            single_channel = features[0].cpu().numpy()
            rgb_array = np.stack([single_channel] * 3, axis=-1)
        
        # Нормализация к [0, 255]
        rgb_array = 255 * (rgb_array - rgb_array.min()) / (rgb_array.max() - rgb_array.min() + 1e-8)
        return rgb_array.astype(np.uint8)
    else:
        # Fallback для других размерностей
        H, W = 58, 51  # Значения по умолчанию
        return np.zeros((H, W, 3), dtype=np.uint8)

def id2rgb(id, max_num_obj=256):
    """Преобразование ID в RGB цвет"""
    if not 0 <= id <= max_num_obj:
        return np.zeros((3,), dtype=np.uint8)
    
    if id == 0:  # Фон
        return np.zeros((3,), dtype=np.uint8)
    
    # Генерируем уникальный цвет для каждого ID
    golden_ratio = 1.6180339887
    h = ((id * golden_ratio) % 1)
    s = 0.5 + (id % 2) * 0.5
    l = 0.5
    
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    rgb = np.array([int(r*255), int(g*255), int(b*255)], dtype=np.uint8)
    return rgb

def visualize_obj(objects):
    """Создаем RGB маску из карты объектов"""
    rgb_mask = np.zeros((*objects.shape[-2:], 3), dtype=np.uint8)
    all_obj_ids = np.unique(objects)
    for id in all_obj_ids:
        colored_mask = id2rgb(id)
        rgb_mask[objects == id] = colored_mask
    return rgb_mask

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier):
    """Рендеринг набора изображений"""
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_feature16")
    gt_colormask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_objects_color")
    pred_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "objects_pred")
    
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(gt_colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)

    device = next(classifier.parameters()).device
    print(f"Классификатор на устройстве: {device}")

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Рендеринг
        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        
        print(f"Обработка изображения {idx}: rendering_obj shape = {rendering_obj.shape}")
        
        # КРИТИЧЕСКИ ВАЖНО: Обеспечиваем совместимость устройств
        rendering_obj = rendering_obj.to(device)
        
        # Приводим rendering_obj к правильному формату для классификатора
        # ТОЧНО КАК В TRAIN.PY
        if rendering_obj.dim() == 2:  # [H, W]
            rendering_obj_input = rendering_obj.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif rendering_obj.dim() == 3:  # [C, H, W]
            # Берем первый канал как объектную карту (как в train.py)
            rendering_obj_input = rendering_obj[0:1].unsqueeze(0)  # [1, 1, H, W]
            # Применяем ту же нормализацию что и в train.py
            rendering_obj_input = (rendering_obj_input - rendering_obj_input.mean()) / (rendering_obj_input.std() + 1e-8)
        else:
            rendering_obj_input = rendering_obj.float()
        
        # Убеждаемся что input на том же устройстве что и классификатор
        rendering_obj_input = rendering_obj_input.to(device).float()
        
        print(f"  Перед классификатором: input shape = {rendering_obj_input.shape}, device = {rendering_obj_input.device}")
        
        # Классификация
        try:
            logits = classifier(rendering_obj_input)
            print(f"  После классификатора: logits shape = {logits.shape}")
            
            # Получаем предсказания
            if logits.dim() == 4:  # [1, num_classes, H, W]
                pred_obj = torch.argmax(logits, dim=1).squeeze()  # [H, W]
            else:
                pred_obj = torch.argmax(logits, dim=0)  # [H, W]
                
            pred_obj_mask = visualize_obj(pred_obj.cpu().numpy().astype(np.uint8))
            
        except Exception as e:
            print(f"  Ошибка классификации: {e}")
            # Создаем пустую маску в случае ошибки
            H, W = rendering_obj.shape[-2:]
            pred_obj_mask = np.zeros((H, W, 3), dtype=np.uint8)

        # Ground truth объекты
        if hasattr(view, 'objects') and view.objects is not None:
            gt_objects = view.objects.cpu().numpy().astype(np.uint8)
            gt_rgb_mask = visualize_obj(gt_objects)
        else:
            # Создаем пустую GT маску если нет разметки
            H, W = rendering_obj.shape[-2:]
            gt_rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)

        # Визуализация фич
        rgb_mask = feature_to_rgb(rendering_obj)
        
        # Сохранение результатов
        try:
            Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
            Image.fromarray(gt_rgb_mask).save(os.path.join(gt_colormask_path, '{0:05d}'.format(idx) + ".png"))
            Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_path, '{0:05d}'.format(idx) + ".png"))
            
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            
            print(f"  ✓ Сохранены все изображения для кадра {idx}")
            
        except Exception as e:
            print(f"  Ошибка сохранения для кадра {idx}: {e}")

    # Создание видео (если все изображения сохранены)
    try:
        create_comparison_video(render_path, gts_path, gt_colormask_path, pred_obj_path, colormask_path)
        print("✓ Видео создано успешно")
    except Exception as e:
        print(f"Ошибка создания видео: {e}")

def create_comparison_video(render_path, gts_path, gt_colormask_path, pred_obj_path, colormask_path):
    """Создание сравнительного видео"""
    out_path = os.path.join(os.path.dirname(render_path), 'concat')
    makedirs(out_path, exist_ok=True)
    
    # Получаем список файлов
    gt_files = sorted(os.listdir(gts_path))
    if not gt_files:
        print("Нет файлов для создания видео")
        return
    
    # Определяем размер первого изображения
    first_gt = np.array(Image.open(os.path.join(gts_path, gt_files[0])))
    height, width = first_gt.shape[:2]
    
    # Настройки видео
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Используем mp4v вместо DIVX
    fps = 2.0  # Медленнее для лучшего просмотра
    video_width = width * 5  # 5 изображений в ряд
    
    video_path = os.path.join(out_path, 'comparison.mp4')
    writer = cv2.VideoWriter(video_path, fourcc, fps, (video_width, height))

    for file_name in gt_files:
        try:
            # Загружаем все изображения
            gt = np.array(Image.open(os.path.join(gts_path, file_name)))
            
            # Проверяем существование остальных файлов
            rgb_path = os.path.join(render_path, file_name)
            gt_obj_path = os.path.join(gt_colormask_path, file_name)
            pred_obj_path_full = os.path.join(pred_obj_path, file_name)
            render_obj_path = os.path.join(colormask_path, file_name)
            
            if os.path.exists(rgb_path):
                rgb = np.array(Image.open(rgb_path))
            else:
                rgb = np.zeros_like(gt)
                
            if os.path.exists(gt_obj_path):
                gt_obj = np.array(Image.open(gt_obj_path))
            else:
                gt_obj = np.zeros_like(gt)
                
            if os.path.exists(pred_obj_path_full):
                pred_obj = np.array(Image.open(pred_obj_path_full))
            else:
                pred_obj = np.zeros_like(gt)
                
            if os.path.exists(render_obj_path):
                render_obj = np.array(Image.open(render_obj_path))
            else:
                render_obj = np.zeros_like(gt)

            # Приводим все к одному размеру
            target_size = (height, width)
            gt = cv2.resize(gt, (width, height))
            rgb = cv2.resize(rgb, (width, height))
            gt_obj = cv2.resize(gt_obj, (width, height))
            pred_obj = cv2.resize(pred_obj, (width, height))
            render_obj = cv2.resize(render_obj, (width, height))

            # Объединяем горизонтально
            result = np.hstack([gt, rgb, gt_obj, pred_obj, render_obj])
            result = result.astype('uint8')

            # Сохраняем кадр изображения
            Image.fromarray(result).save(os.path.join(out_path, file_name))
            
            # Добавляем в видео (OpenCV ожидает BGR)
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            writer.write(result_bgr)
            
        except Exception as e:
            print(f"Ошибка обработки {file_name}: {e}")

    writer.release()
    print(f"Видео сохранено: {video_path}")

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    """Основная функция рендеринга"""
    with torch.no_grad():
        # Определяем устройство
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Используется устройство: {device}")
        
        # Загружаем модель
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        num_classes = dataset.num_classes
        print(f"Num classes: {num_classes}")

        # КРИТИЧЕСКИ ВАЖНО: Создаем классификатор ТОЧНО КАК В TRAIN.PY
        classifier = torch.nn.Conv2d(1, num_classes, kernel_size=1).to(device)
        
        # Загружаем веса классификатора
        classifier_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{scene.loaded_iter}", "classifier.pth")
        print(f"Загружаем классификатор из: {classifier_path}")
        
        try:
            classifier_state = torch.load(classifier_path, map_location=device)
            classifier.load_state_dict(classifier_state)
            print("✓ Классификатор загружен успешно")
        except Exception as e:
            print(f"❌ Ошибка загрузки классификатора: {e}")
            print(f"Ожидаемая архитектура: Conv2d(1, {num_classes})")
            print(f"Проверьте совместимость с train.py")
            return

        # Фон
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)

        # Рендеринг train сета
        if not skip_train:
            print("Рендеринг training изображений...")
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), 
                      gaussians, pipeline, background, classifier)

        # Рендеринг test сета
        if not skip_test and len(scene.getTestCameras()) > 0:
            print("Рендеринг test изображений...")
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), 
                      gaussians, pipeline, background, classifier)
        
        print("✓ Рендеринг завершен успешно!")

if __name__ == "__main__":
    # Парсер аргументов командной строки
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Инициализация системы (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)