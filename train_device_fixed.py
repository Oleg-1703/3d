#!/usr/bin/env python3
"""
Упрощённая версия обучения без сложной object loss для отладки
"""

import os
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import json
import time

sys.path.append('.')

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
import torch.nn as nn
from torch.nn import functional as F

# GPU ОПТИМИЗАЦИИ
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def move_camera_to_device(camera, device):
    """Переносит данные камеры на указанное устройство"""
    if hasattr(camera, 'original_image'):
        camera.original_image = camera.original_image.to(device)
    if hasattr(camera, 'objects') and camera.objects is not None:
        camera.objects = camera.objects.to(device)
    return camera


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, 
            checkpoint, debug_from, use_wandb):
    """Упрощённая функция обучения без сложной object loss"""
    
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Создаем модель
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # Перемещаем камеры на GPU
    print("Перемещение камер на GPU...")
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    
    for cam in train_cameras:
        move_camera_to_device(cam, device)
    
    for cam in test_cameras:
        move_camera_to_device(cam, device)
    
    print(f"✓ {len(train_cameras)} train камер и {len(test_cameras)} test камер на GPU")
    
    # Простой классификатор
    classifier = nn.Conv2d(1, dataset.num_classes, kernel_size=1, bias=True).to(device)
    nn.init.xavier_uniform_(classifier.weight, gain=0.01)
    nn.init.constant_(classifier.bias, 0.0)
    cls_optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-4)
    print(f"Классификатор создан: 1 -> {dataset.num_classes} классов")
    
    # Фон
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Батчинг
    batch_size = min(2, len(train_cameras))  # Уменьшаем batch size для стабильности
    print(f"Batch size: {batch_size}")
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    
    # Прогресс-бар
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Обучение")
    first_iter = 1

    # ГЛАВНЫЙ ЦИКЛ ОБУЧЕНИЯ
    for iteration in range(first_iter, opt.iterations + 1):
        
        start_time = time.time()

        gaussians.update_learning_rate(iteration)

        # Увеличиваем SH степень
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Выбираем камеры
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        
        current_batch = []
        for _ in range(min(batch_size, len(viewpoint_stack))):
            if not viewpoint_stack:
                viewpoint_stack = train_cameras.copy()
            cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            current_batch.append(cam)

        # Отладка
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # РЕНДЕРИНГ И ПОТЕРИ
        viewspace_tensors = []
        visibility_filters = []
        radii_list = []
        
        total_l1 = 0.0
        total_ssim_loss = 0.0
        total_obj_loss = 0.0
        valid_obj_count = 0
        
        for cam in current_batch:
            # Рендеринг
            render_pkg = render(cam, gaussians, pipe, background)
            
            # Данные для денсификации
            viewspace_tensors.append(render_pkg["viewspace_points"])
            visibility_filters.append(render_pkg["visibility_filter"])
            radii_list.append(render_pkg["radii"])
            
            # Основные потери
            image = render_pkg["render"]
            gt_image = cam.original_image
            
            # L1 loss
            l1 = l1_loss(image, gt_image)
            total_l1 += l1
            
            # SSIM loss
            ssim_loss = 1.0 - ssim(image, gt_image)
            total_ssim_loss += ssim_loss
            
            # УПРОЩЁННАЯ Object segmentation loss
            if hasattr(cam, 'objects') and cam.objects is not None and "render_object" in render_pkg:
                try:
                    gt_obj = cam.objects
                    objects = render_pkg["render_object"]
                    
                    # Простая MSE loss между rendered objects и ground truth
                    if objects.dim() == 3:  # [C, H, W]
                        objects_map = objects[0]  # Берем первый канал
                    else:
                        objects_map = objects
                    
                    # Приводим к одинаковым размерам
                    if objects_map.shape != gt_obj.shape:
                        gt_obj_resized = F.interpolate(
                            gt_obj.float().unsqueeze(0).unsqueeze(0),
                            size=objects_map.shape,
                            mode='nearest'
                        ).squeeze()
                    else:
                        gt_obj_resized = gt_obj.float()
                    
                    # Простая MSE loss
                    obj_loss = F.mse_loss(objects_map, gt_obj_resized, reduction='mean')
                    
                    total_obj_loss += obj_loss
                    valid_obj_count += 1
                    
                except Exception as e:
                    print(f"Ошибка в простой object loss: {e}")
                    continue

        # Усредняем потери
        if len(current_batch) == 0:
            print("Нет данных для обучения!")
            continue
            
        avg_l1 = total_l1 / len(current_batch)
        avg_ssim = total_ssim_loss / len(current_batch)
        avg_obj = total_obj_loss / max(valid_obj_count, 1) if valid_obj_count > 0 else torch.tensor(0.0, device=device)
        
        # Общая потеря (только L1 и SSIM для стабильности)
        loss = (1.0 - opt.lambda_dssim) * avg_l1 + opt.lambda_dssim * avg_ssim
        
        # Добавляем obj loss с маленьким весом
        if valid_obj_count > 0:
            loss = loss + 0.1 * avg_obj  # Маленький вес для object loss

        # Обратное распространение
        try:
            loss.backward()
        except Exception as e:
            print(f"Ошибка в backward: {e}")
            continue

        end_time = time.time()
        iter_time = end_time - start_time

        with torch.no_grad():
            # EMA потери
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            # Отчетность
            if iteration % 10 == 0:
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.4f}",
                    "L1": f"{avg_l1.item():.4f}",
                    "SSIM": f"{avg_ssim.item():.4f}",
                    "Obj": f"{avg_obj.item():.4f}" if valid_obj_count > 0 else "0.0",
                    "GPU": f"{gpu_memory:.1f}GB"
                })
                progress_bar.update(10)
            
            if iteration == opt.iterations:
                progress_bar.close()

            # Тестирование
            if iteration in testing_iterations:
                print(f"\n[ITER {iteration}] Быстрый тест:")
                if test_cameras and len(test_cameras) > 0:
                    test_cam = test_cameras[0]
                    with torch.no_grad():
                        test_render = render(test_cam, gaussians, pipe, background)
                        test_image = torch.clamp(test_render["render"], 0.0, 1.0)
                        test_gt = torch.clamp(test_cam.original_image, 0.0, 1.0)
                        test_l1 = l1_loss(test_image, test_gt).mean()
                        test_psnr = -10.0 * torch.log10(((test_image - test_gt) ** 2).mean())
                        print(f"Test L1: {test_l1:.4f}, PSNR: {test_psnr:.2f}")
            
            # Сохранение
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Сохранение модели")
                scene.save(iteration)
                
                classifier_path = os.path.join(scene.model_path, f"point_cloud/iteration_{iteration}")
                os.makedirs(classifier_path, exist_ok=True)
                torch.save(classifier.state_dict(), os.path.join(classifier_path, 'classifier.pth'))

            # ДЕНСИФИКАЦИЯ
            if iteration < opt.densify_until_iter:
                for vst, vf, radii in zip(viewspace_tensors, visibility_filters, radii_list):
                    # Убеждаемся в consistency устройств
                    radii = radii.to(gaussians.max_radii2D.device)
                    vf = vf.to(gaussians.max_radii2D.device)
                    
                    gaussians.max_radii2D[vf] = torch.max(gaussians.max_radii2D[vf], radii[vf])
                    gaussians.add_densification_stats(vst, vf)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 
                                              scene.cameras_extent, size_threshold)
                
                # Сброс прозрачности
                if iteration % (opt.opacity_reset_interval * 2) == 0:
                    gaussians.reset_opacity()

            # Шаги оптимизаторов
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            
            cls_optimizer.step()
            cls_optimizer.zero_grad(set_to_none=True)

        # Очистка кэша
        if iteration % 100 == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Упрощённое обучение")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 2500, 5000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 2500, 5000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config_file", type=str, default="config/gaussian_dataset/train_fast.json")

    args = parser.parse_args(sys.argv[1:])
    
    # Загрузка конфигурации
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
        print(f"✓ Конфигурация загружена из {args.config_file}")
    except:
        print("Используем значения по умолчанию")
        config = {}

    # Применяем настройки
    args.densify_until_iter = config.get("densify_until_iter", 3000)
    args.num_classes = config.get("num_classes", 2)
    args.save_iterations.append(args.iterations)
    
    # Создаём быструю конфигурацию если её нет
    if not os.path.exists(args.config_file):
        os.makedirs(os.path.dirname(args.config_file), exist_ok=True)
        with open(args.config_file, 'w') as f:
            json.dump({
                "densify_until_iter": 3000,
                "num_classes": 2,
                "iterations": 5000
            }, f, indent=2)
        print(f"✓ Создан файл конфигурации: {args.config_file}")
    
    print(f"✓ Оптимизация {args.model_path}")
    print(f"✓ Источник: {args.source_path}")
    print(f"✓ GPU: {torch.cuda.get_device_name()}")

    # Инициализация и запуск
    safe_state(False)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    try:
        training(lp.extract(args), op.extract(args), pp.extract(args), 
                args.test_iterations, args.save_iterations, [], 
                args.start_checkpoint, args.debug_from, False)
        
        print("\n✓ Обучение завершено!")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()   