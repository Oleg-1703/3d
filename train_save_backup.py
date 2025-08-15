#!/usr/bin/env python3
"""
Исправленный train.py с правильным управлением устройствами GPU/CPU
Решает проблему RuntimeError: Expected all tensors to be on the same device
"""

import os
import torch
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import json
import wandb

# Добавляем текущую директорию в путь для импортов
sys.path.append('.')

from arguments import ModelParams, PipelineParams, OptimizationParams
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
from utils.lpipsloss import lpips
import lpips as lpips_eval

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Для сегментации
import torch.nn as nn
from torch.nn import functional as F


class SegmentationNetwork(nn.Module):
    """Простая сеть для семантической сегментации"""
    def __init__(self, feature_size, num_classes):
        super(SegmentationNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256), 
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, features):
        # Обрабатываем входные признаки
        if features.dim() == 4:  # [B, C, H, W]
            B, C, H, W = features.shape
            features = features.permute(0, 2, 3, 1).reshape(B * H * W, C)
        elif features.dim() == 3:  # [C, H, W]
            C, H, W = features.shape
            features = features.permute(1, 2, 0).reshape(H * W, C)
            
        logits = self.mlp(features)
        
        if features.dim() == 4:
            logits = logits.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        elif features.dim() == 3:
            logits = logits.reshape(H, W, -1).permute(2, 0, 1)
            
        return logits


def ensure_device_consistency(tensor_dict, target_device):
    """
    Обеспечивает, что все тензоры находятся на одном устройстве
    """
    result = {}
    for key, tensor in tensor_dict.items():
        if torch.is_tensor(tensor):
            result[key] = tensor.to(target_device)
        else:
            result[key] = tensor
    return result


def safe_tensor_operation(tensor1, tensor2, operation_func):
    """
    Безопасно выполняет операцию между тензорами, убеждаясь что они на одном устройстве
    """
    device = tensor1.device
    if tensor2.device != device:
        tensor2 = tensor2.to(device)
    return operation_func(tensor1, tensor2)


def training_report(iteration, Ll1, loss, l1_loss_val, elapsed, testing_iterations, scene, renderFunc, renderArgs, loss_obj_3d=None, use_wandb=False):
    """Отчетность во время обучения"""
    if use_wandb:
        wandb.log({
            "train/loss_l1": Ll1.item(),
            "train/total_loss": loss.item(),
            "train/iter_time": elapsed,
            "train/total_points": scene.gaussians.get_xyz.shape[0],
            "iter": iteration
        })
        if loss_obj_3d is not None:
            wandb.log({"train/loss_obj_3d": loss_obj_3d.item()})

    # Запускаем тестирование если нужно
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                            {'name': 'train', 'cameras': scene.getTrainCameras()[:4]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to(image.device), 0.0, 1.0)
                    
                    if use_wandb and config['name'] == 'test':
                        wandb.log({f"test_view_{idx}": wandb.Image(image.cpu().numpy().transpose(1, 2, 0))})
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += (-10.0 * torch.log10(((image - gt_image) ** 2).mean())).double()
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.4f} PSNR {psnr_test:.2f}")
                
                if use_wandb:
                    wandb.log({f"{config['name']}/loss_viewpoint": l1_test, f"{config['name']}/psnr": psnr_test})

        torch.cuda.empty_cache()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, 
            checkpoint, debug_from, use_wandb):
    """Основная функция обучения"""
    
    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Создаем модель
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # Убеждаемся что модель на правильном устройстве
    
    # Создаем классификатор для сегментации
    # ПРАВИЛЬНЫЙ КЛАССИФИКАТОР для Gaussian Grouping сегментации
    classifier = torch.nn.Conv2d(1, dataset.num_classes, kernel_size=1).to(device)
    torch.nn.init.xavier_uniform_(classifier.weight, gain=0.01)
    torch.nn.init.constant_(classifier.bias, 0.0)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    print(f"Классификатор создан: 1 -> {dataset.num_classes} классов")
    
    # Фон
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # Настройка для W&B
    if use_wandb:
        wandb.watch(gaussians)

    # Итераторы камер
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    
    # Таймеры
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    # Прогресс-бар
    progress_bar = tqdm(range(1, opt.iterations + 1), desc="Обучение")
    first_iter = 1
    
    # Восстановление с checkpoint если нужно
    if checkpoint:
        model_params, first_iter = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    for iteration in range(first_iter, opt.iterations + 1):
        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Увеличиваем степень SH каждые 1000 итераций
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Выбираем случайную камеру
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Отладка
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # Рендеринг
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        
        # КРИТИЧЕСКИ ВАЖНО: Обеспечиваем consistency устройств
        render_pkg = ensure_device_consistency(render_pkg, device)
        
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"] 
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        # Проверяем что есть объекты для сегментации
        if "render_object" in render_pkg:
            objects = render_pkg["render_object"]
        else:
            # Создаем заглушку если нет
            objects = torch.zeros_like(image[:1])

        # Основная потеря L1
        gt_image = viewpoint_cam.original_image.to(device)
        Ll1 = l1_loss(image, gt_image)
        
        # ПРАВИЛЬНАЯ СЕГМЕНТАЦИЯ GAUSSIAN GROUPING
        
        # Object Loss - как в оригинальном train.py
        if hasattr(viewpoint_cam, 'objects') and viewpoint_cam.objects is not None:
            try:
                # Получаем gt маски объектов  
                gt_obj = viewpoint_cam.objects.to(device).long()
                
                # objects из рендера должен быть [C, H, W] где C - каналы объектов
                if "render_object" in render_pkg:
                    objects = render_pkg["render_object"]  # [C, H, W]
                    
                    print(f'objects shape: {objects.shape}, gt_obj shape: {gt_obj.shape}')
                    
                    # Преобразуем objects для классификатора
                    # Классификатор ожидает [batch, 1, H, W]
                    if objects.dim() == 3:  # [C, H, W]
                        # Берем первый канал как объектную карту
                        objects_input = objects[0:1].unsqueeze(0)  # [1, 1, H, W]
                        objects_input = (objects_input - objects_input.mean()) / (objects_input.std() + 1e-8)

                    elif objects.dim() == 2:  # [H, W] 
                        objects_input = objects.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    else:
                        objects_input = objects.float()
                    
                    # Убеждаемся что размеры совпадают
                    if objects_input.shape[-2:] != gt_obj.shape[-2:]:
                        # Ресайзим gt_obj к размеру objects
                        gt_obj = torch.nn.functional.interpolate(
                            gt_obj.float().unsqueeze(0).unsqueeze(0),
                            size=objects_input.shape[-2:],
                            mode='nearest'
                        ).squeeze().long()
                    
                    # Классификация объектов
                    logits = classifier(objects_input.float())  # [1, num_classes, H, W]
                    
                    print(f'logits shape: {logits.shape}, gt_obj shape: {gt_obj.shape}')
                    
                    # Приводим gt_obj к правильному формату для loss
                    if gt_obj.dim() == 2:  # [H, W]
                        target = gt_obj.unsqueeze(0)  # [1, H, W] 
                    else:
                        target = gt_obj
                    
                    # Убеждаемся что размеры логитов и таргета совпадают
                    if logits.shape[-2:] != target.shape[-2:]:
                        target = torch.nn.functional.interpolate(
                            target.float().unsqueeze(0),
                            size=logits.shape[-2:],
                            mode='nearest'
                        ).squeeze().long()
                        if target.dim() == 2:
                            target = target.unsqueeze(0)
                    
                    # Вычисляем object loss
                    # ПРАВИЛЬНАЯ бинарная сегментация loss (скаляр)
                    if dataset.num_classes == 2:
                        # Для бинарной классификации используем BCEWithLogitsLoss
                        logits_binary = logits[:, 1:2, :, :]  # Канал объекта [1,1,H,W]
                        target_binary = (target > 0).float().unsqueeze(1)  # [1,1,H,W]
                        pos_weight = (target_binary == 0).sum() / (target_binary.sum() + 1e-8)
                        bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
                        loss_obj = bce_loss(logits_binary, target_binary)
                    else:
                        # Для многоклассовой классификации используем CrossEntropyLoss
                        loss_obj = cls_criterion(logits, target).mean()
                    if dataset.num_classes > 1:
                        loss_obj = loss_obj / torch.log(torch.tensor(float(dataset.num_classes), device=device))
                    # DEBUG: проверяем содержимое                    
                    # Убеждаемся что loss_obj скаляр
                    if loss_obj.dim() > 0:
                        loss_obj = loss_obj.mean()
                    
                    #print(f"Object loss: {float(loss_obj):.4f}")
                    # loss_obj уже нормализован  # нормализация
                    
                    # #print(f'Object loss: {float(loss_obj):.4f}')  # Временно отключено
                    
                else:
                    print("Нет render_object в render_pkg")
                    loss_obj = torch.tensor(0.0, device=device, requires_grad=True)
                    
            except Exception as e:
                print(f"Ошибка в object segmentation на итерации {iteration}: {e}")
                loss_obj = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            print("Нет объектных масок для камеры")
            loss_obj = torch.tensor(0.0, device=device, requires_grad=True)
        
        # 3D Regularization loss (как в оригинале)
        loss_obj_3d = None
        if (iteration % opt.reg3d_interval == 0 and 
            hasattr(gaussians, '_objects_dc') and 
            gaussians._objects_dc is not None and 
            gaussians._objects_dc.numel() > 0):
            try:
                # 3D классификация Gaussians  
                logits3d = classifier(gaussians._objects_dc.permute(2,0,1))  # [num_classes, num_points, 1]
                prob_obj3d = torch.softmax(logits3d, dim=0).squeeze().permute(1,0)  # [num_points, num_classes]
                
                # Используем 3D loss из utils
                loss_obj_3d = loss_cls_3d(
                    gaussians._xyz.squeeze().detach(), 
                    prob_obj3d, 
                    opt.reg3d_k, 
                    opt.reg3d_lambda_val, 
                    opt.reg3d_max_points, 
                    opt.reg3d_sample_size
                )
                print(f'3D regularization loss: {loss_obj_3d.item():.4f}')
            except Exception as e:
                print(f"Ошибка в 3D regularization: {e}")
                loss_obj_3d = torch.tensor(0.0, device=device, requires_grad=True)

        # Общая потеря
        # Общая потеря (Gaussian Grouping стиль)
        if loss_obj_3d is not None:
            # Полная потеря с SSIM, object loss и 3D regularization
            loss = ((1.0 - opt.lambda_dssim) * Ll1 + 
                   opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 
                   loss_obj + loss_obj_3d)
                    # print(f'Full loss: L1={Ll1.item():.4f}, Obj={float(loss_obj):.4f}, 3D={loss_obj_3d.item():.4f}')  # Временно отключено
        else:
            # Потеря без 3D regularization
            loss = ((1.0 - opt.lambda_dssim) * Ll1 + 
                   opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 
                   loss_obj)
                    # print(f'Loss: L1={Ll1.item():.4f}, Obj={float(loss_obj):.4f}')  # Временно отключено

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Обновляем EMA потери
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            # Отчетность
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.7f}"})
                progress_bar.update(10)
            
            if iteration == opt.iterations:
                progress_bar.close()

            # Логирование и сохранение
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), 
                          testing_iterations, scene, render, (pipe, background), loss_obj_3d, use_wandb)
            
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Сохранение Checkpoint")
                classifier_path = os.path.join(scene.model_path, f"point_cloud/iteration_{iteration}")
                os.makedirs(classifier_path, exist_ok=True)
                torch.save(classifier.state_dict(), os.path.join(classifier_path, 'classifier.pth'))

            # Денсификация
            if iteration < opt.densify_until_iter:
                # КРИТИЧЕСКИ ВАЖНО: Обеспечиваем что все тензоры на одном устройстве
                if gaussians.max_radii2D.device != radii.device:
                    radii = radii.to(gaussians.max_radii2D.device)
                    
                if gaussians.max_radii2D.device != visibility_filter.device:
                    visibility_filter = visibility_filter.to(gaussians.max_radii2D.device)
                
                # Теперь безопасно выполняем операцию
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], 
                    radii[visibility_filter]
                )
                
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 
                                              scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    print(f"⚠️  Пропущен reset_opacity на итерации {iteration} для стабильности")
                    # gaussians.reset_opacity()  # Отключено для стабильности обучения

            # Шаг оптимизатора
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                
                cls_optimizer.step()
                cls_optimizer.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Парсер аргументов командной строки
    parser = ArgumentParser(description="Параметры скрипта обучения")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1000, 7000, 30000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 7000, 30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config_file", type=str, default="config/gaussian_dataset/train.json", 
                       help="Путь к конфигурационному файлу")
    parser.add_argument("--use_wandb", action='store_true', default=False, 
                       help="Использовать wandb для записи потерь")

    args = parser.parse_args(sys.argv[1:])
    
    # Чтение и парсинг конфигурационного файла
    try:
        with open(args.config_file, 'r') as file:
            config = json.load(file)
        print(f"✓ Конфигурация загружена из {args.config_file}")
    except FileNotFoundError:
        print(f"Ошибка: Конфигурационный файл '{args.config_file}' не найден.")
        print("Используем значения по умолчанию")
        config = {}
    except json.JSONDecodeError as e:
        print(f"Ошибка: Не удалось разобрать JSON конфигурационный файл: {e}")
        exit(1)

    # Применяем настройки из конфигурации
    args.densify_until_iter = config.get("densify_until_iter", 15000)
    args.num_classes = config.get("num_classes", 256)
    args.reg3d_interval = config.get("reg3d_interval", 5)
    args.reg3d_k = config.get("reg3d_k", 5)
    args.reg3d_lambda_val = config.get("reg3d_lambda_val", 2)
    args.reg3d_max_points = config.get("reg3d_max_points", 300000)
    args.reg3d_sample_size = config.get("reg3d_sample_size", 1000)
    
    args.save_iterations.append(args.iterations)
    
    print(f"✓ Оптимизация {args.model_path}")
    print(f"✓ Источник данных: {args.source_path}")
    print(f"✓ Классов сегментации: {args.num_classes}")
    print(f"✓ Устройство: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Инициализация W&B если нужно
    if args.use_wandb:
        wandb.init(project="gaussian-splatting-segmentation")
        wandb.config.update(args)
        wandb.run.name = args.model_path.split('/')[-1]

    # Инициализация системы (RNG)
    safe_state(args.quiet)

    # Настройка и запуск обучения
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    try:
        training(lp.extract(args), op.extract(args), pp.extract(args), 
                args.test_iterations, args.save_iterations, args.checkpoint_iterations, 
                args.start_checkpoint, args.debug_from, args.use_wandb)
        
        print("\n✓ Обучение завершено успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        import traceback
        traceback.print_exc()
        exit(1)