#!/usr/bin/env python3
"""
Исправленная версия train_gpu_optimized.py с правильным управлением устройствами
"""

import os
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import json
import time

# Добавляем текущую директорию в путь для импортов
sys.path.append('.')

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
import torch.nn as nn
from torch.nn import functional as F

# GPU ОПТИМИЗАЦИИ
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class OptimizedSegmentationNetwork(nn.Module):
    """GPU-оптимизированная сеть для семантической сегментации"""
    def __init__(self, num_classes):
        super(OptimizedSegmentationNetwork, self).__init__()
        self.classifier = nn.Conv2d(1, num_classes, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.classifier.weight, gain=0.01)
        nn.init.constant_(self.classifier.bias, 0.0)
        
    def forward(self, x):
        return self.classifier(x)


def move_camera_to_device(camera, device):
    """Переносит данные камеры на указанное устройство"""
    if hasattr(camera, 'original_image'):
        camera.original_image = camera.original_image.to(device)
    if hasattr(camera, 'objects') and camera.objects is not None:
        camera.objects = camera.objects.to(device)
    return camera


def training_report_fast(iteration, loss_dict, elapsed, testing_iterations, scene, renderFunc, renderArgs, use_wandb=False):
    """Быстрая отчетность"""
    
    # Тестирование только на важных итерациях
    if iteration in testing_iterations:
        test_cameras = scene.getTestCameras()
        if test_cameras and len(test_cameras) > 0:
            test_sample = test_cameras[:min(4, len(test_cameras))]
            
            l1_test = 0.0
            psnr_test = 0.0
            
            with torch.no_grad():
                for viewpoint in test_sample:
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    
                    # ИСПРАВЛЕНИЕ: убеждаемся что gt_image на правильном устройстве
                    if hasattr(viewpoint, 'original_image'):
                        gt_image = torch.clamp(viewpoint.original_image.to(image.device), 0.0, 1.0)
                    else:
                        continue
                    
                    l1_test += l1_loss(image, gt_image).mean()
                    mse = ((image - gt_image) ** 2).mean()
                    psnr_test += -10.0 * torch.log10(mse)
            
            l1_test /= len(test_sample)
            psnr_test /= len(test_sample)
            
            print(f"\n[ITER {iteration}] Test: L1 {l1_test:.4f} PSNR {psnr_test:.2f}")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, 
            checkpoint, debug_from, use_wandb):
    """Исправленная функция обучения с правильным управлением устройствами"""
    
    # CUDA настройки
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Создаем модель
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # ИСПРАВЛЕНИЕ: Перемещаем камеры на GPU
    print("Перемещение камер на GPU...")
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    
    for cam in train_cameras:
        move_camera_to_device(cam, device)
    
    for cam in test_cameras:
        move_camera_to_device(cam, device)
    
    print(f"✓ {len(train_cameras)} train камер и {len(test_cameras)} test камер на GPU")
    
    # Классификатор
    classifier = OptimizedSegmentationNetwork(dataset.num_classes).to(device)
    cls_optimizer = torch.optim.AdamW(classifier.parameters(), lr=5e-4, weight_decay=1e-4)
    print(f"Классификатор создан: 1 -> {dataset.num_classes} классов")
    
    # Фон
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Батчинг
    batch_size = min(4, len(train_cameras))
    print(f"Batch size: {batch_size}")
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    
    # Прогресс-бар
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Обучение")
    first_iter = 1
    
    # Восстановление checkpoint
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # ГЛАВНЫЙ ЦИКЛ ОБУЧЕНИЯ
    for iteration in range(first_iter, opt.iterations + 1):
        
        start_time = time.time()

        gaussians.update_learning_rate(iteration)

        # Увеличиваем SH степень
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Выбираем камеры для батча
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

        # РЕНДЕРИНГ И ОБРАБОТКА БАТЧА
        render_results = []
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
            render_results.append(render_pkg)
            
            # Данные для денсификации
            viewspace_tensors.append(render_pkg["viewspace_points"])
            visibility_filters.append(render_pkg["visibility_filter"])
            radii_list.append(render_pkg["radii"])
            
            # ИСПРАВЛЕНИЕ: Обрабатываем потери с правильными устройствами
            image = render_pkg["render"]
            
            # Убеждаемся что original_image на правильном устройстве
            if hasattr(cam, 'original_image'):
                gt_image = cam.original_image  # Уже на device после move_camera_to_device
                if gt_image.device != image.device:
                    gt_image = gt_image.to(image.device)
            else:
                print(f"ВНИМАНИЕ: У камеры {cam.image_name} нет original_image!")
                continue
            
            # L1 loss
            l1 = l1_loss(image, gt_image)
            total_l1 += l1
            
            # SSIM loss
            ssim_loss = 1.0 - ssim(image, gt_image)
            total_ssim_loss += ssim_loss
            
            # Object segmentation loss
            if hasattr(cam, 'objects') and cam.objects is not None and "render_object" in render_pkg:
                try:
                    # objects уже на device после move_camera_to_device
                    gt_obj = cam.objects.long()
                    if gt_obj.device != image.device:
                        gt_obj = gt_obj.to(image.device)
                    
                    objects = render_pkg["render_object"]
                    
                    # Быстрая обработка
                    if objects.dim() == 3:
                        objects_input = objects[0:1].unsqueeze(0)
                    else:
                        objects_input = objects.unsqueeze(0).unsqueeze(0)
                    
                    # Нормализация
                    objects_input = (objects_input - objects_input.mean()) / (objects_input.std() + 1e-8)
                    
                    # Resize если нужно
                    if objects_input.shape[-2:] != gt_obj.shape[-2:]:
                        gt_obj = F.interpolate(
                            gt_obj.float().unsqueeze(0).unsqueeze(0),
                            size=objects_input.shape[-2:],
                            mode='nearest'
                        ).squeeze().long()
                    
                    # Классификация
                    logits = classifier(objects_input)
                    target = gt_obj.unsqueeze(0) if gt_obj.dim() == 2 else gt_obj
                    
                    # Object loss
                    if dataset.num_classes == 2:
                        logits_binary = logits[:, 1:2, :, :]
                        target_binary = (target > 0).float().unsqueeze(1)
                        obj_loss = F.binary_cross_entropy_with_logits(
                            logits_binary, target_binary, reduction='mean'
                        )
                    else:
                        obj_loss = F.cross_entropy(logits, target, reduction='mean')
                    
                    total_obj_loss += obj_loss
                    valid_obj_count += 1
                    
                except Exception as e:
                    print(f"Ошибка в object loss для камеры {cam.image_name}: {e}")
                    continue

        # Проверяем что у нас есть данные для обучения
        if len(current_batch) == 0:
            print("ОШИБКА: Нет данных для обучения в батче!")
            continue
            
        # Усредняем потери по батчу
        avg_l1 = total_l1 / len(current_batch)
        avg_ssim = total_ssim_loss / len(current_batch)
        avg_obj = total_obj_loss / max(valid_obj_count, 1) if valid_obj_count > 0 else torch.tensor(0.0, device=device)
        
        # Общая потеря
        loss = (1.0 - opt.lambda_dssim) * avg_l1 + opt.lambda_dssim * avg_ssim + avg_obj

        # Обратное распространение
        loss.backward()

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
                    "GPU": f"{gpu_memory:.1f}GB",
                    "Time": f"{iter_time:.1f}s"
                })
                progress_bar.update(10)
            
            if iteration == opt.iterations:
                progress_bar.close()

            # Подробная отчетность
            if iteration % 100 == 0:
                loss_dict = {
                    'l1': avg_l1,
                    'ssim': avg_ssim, 
                    'obj': avg_obj,
                    'total': loss
                }
                training_report_fast(iteration, loss_dict, iter_time, 
                                   testing_iterations, scene, render, (pipe, background), use_wandb)
            
            # Сохранение
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Сохранение модели")
                scene.save(iteration)
                
                classifier_path = os.path.join(scene.model_path, f"point_cloud/iteration_{iteration}")
                os.makedirs(classifier_path, exist_ok=True)
                torch.save(classifier.state_dict(), os.path.join(classifier_path, 'classifier.pth'))

            # ДЕНСИФИКАЦИЯ
            if iteration < opt.densify_until_iter:
                for i, (vst, vf, radii) in enumerate(zip(viewspace_tensors, visibility_filters, radii_list)):
                    # Убеждаемся в consistency устройств
                    radii = radii.to(gaussians.max_radii2D.device)
                    vf = vf.to(gaussians.max_radii2D.device)
                    
                    gaussians.max_radii2D[vf] = torch.max(gaussians.max_radii2D[vf], radii[vf])
                    gaussians.add_densification_stats(vst, vf)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 
                                              scene.cameras_extent, size_threshold)
                
                # Сброс прозрачности (реже)
                if iteration % (opt.opacity_reset_interval * 2) == 0:
                    gaussians.reset_opacity()

            # Шаги оптимизаторов
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            
            cls_optimizer.step()
            cls_optimizer.zero_grad(set_to_none=True)

        # Очистка кэша реже
        if iteration % 50 == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Парсер аргументов
    parser = ArgumentParser(description="Исправленное GPU обучение")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 5000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 5000])
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
    args.densify_until_iter = config.get("densify_until_iter", 5000)
    args.num_classes = config.get("num_classes", 2)
    args.reg3d_interval = config.get("reg3d_interval", 10)
    args.reg3d_k = config.get("reg3d_k", 3) 
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 1.0)
    args.save_iterations.append(args.iterations)
    
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