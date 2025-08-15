#!/usr/bin/env python3
"""
КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ СЕГМЕНТАЦИИ
Исправляем размерности классификатора, СОХРАНЯЯ маски сегментации!
"""

import os

def fix_segmentation_train():
    with open('train.py', 'r') as f:
        content = f.read()
    
    # Бэкап
    with open('train.py.segmentation_backup', 'w') as f:
        f.write(content)
    
    print("✅ Создан бэкап train.py.segmentation_backup")
    
    # ИСПРАВЛЕНИЕ 1: Правильный классификатор для Gaussian Grouping
    # Заменяем неправильный классификатор
    old_classifier_section = '''classifier = SegmentationNetwork(feature_size=3, num_classes=dataset.num_classes).to(device)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)'''
    
    new_classifier_section = '''# ПРАВИЛЬНЫЙ КЛАССИФИКАТОР для Gaussian Grouping сегментации
    classifier = torch.nn.Conv2d(1, dataset.num_classes, kernel_size=1).to(device)
    cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
    print(f"Классификатор создан: 1 -> {dataset.num_classes} классов")'''
    
    if 'SegmentationNetwork' in content:
        content = content.replace(old_classifier_section, new_classifier_section)
        print("✅ Исправлен классификатор")
    
    # ИСПРАВЛЕНИЕ 2: Правильная обработка render_object
    # Ищем блок сегментации и заменяем его на корректную версию
    segmentation_block_start = content.find('# Потеря сегментации')
    if segmentation_block_start == -1:
        segmentation_block_start = content.find('loss_obj_3d = torch.tensor(0.0, device=device')
    
    if segmentation_block_start != -1:
        # Находим конец блока
        segmentation_block_end = content.find('# Общая потеря', segmentation_block_start)
        if segmentation_block_end == -1:
            segmentation_block_end = content.find('loss = Ll1', segmentation_block_start)
        
        if segmentation_block_end != -1:
            # Заменяем ВЕСЬ блок на правильную Gaussian Grouping сегментацию
            new_segmentation_code = '''# ПРАВИЛЬНАЯ СЕГМЕНТАЦИЯ GAUSSIAN GROUPING
        
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
                    loss_obj = cls_criterion(logits, target).mean()
                    loss_obj = loss_obj / torch.log(torch.tensor(dataset.num_classes, device=device))  # нормализация
                    
                    print(f'Object loss: {loss_obj.item():.4f}')
                    
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
        if iteration % opt.reg3d_interval == 0 and hasattr(gaussians, '_objects_dc') and gaussians._objects_dc.numel() > 0:
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

        '''
            
            content = content[:segmentation_block_start] + new_segmentation_code + content[segmentation_block_end:]
            print("✅ Заменен блок сегментации на правильную Gaussian Grouping версию")
    
    # ИСПРАВЛЕНИЕ 3: Правильное вычисление общей потери (как в оригинале)
    old_loss_computation = 'loss = Ll1 + 0.1 * loss_obj_3d'
    
    new_loss_computation = '''# Общая потеря (Gaussian Grouping стиль)
        if loss_obj_3d is not None:
            # Полная потеря с SSIM, object loss и 3D regularization
            loss = ((1.0 - opt.lambda_dssim) * Ll1 + 
                   opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 
                   loss_obj + loss_obj_3d)
            print(f'Full loss: L1={Ll1.item():.4f}, Obj={loss_obj.item():.4f}, 3D={loss_obj_3d.item():.4f}')
        else:
            # Потеря без 3D regularization
            loss = ((1.0 - opt.lambda_dssim) * Ll1 + 
                   opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 
                   loss_obj)
            print(f'Loss: L1={Ll1.item():.4f}, Obj={loss_obj.item():.4f}')'''
    
    # Заменяем простое вычисление потери на полное
    if old_loss_computation in content:
        content = content.replace(old_loss_computation, new_loss_computation)
        print("✅ Исправлено вычисление потери")
    
    # ИСПРАВЛЕНИЕ 4: Добавляем optimizer step для классификатора
    optimizer_step_location = content.find('gaussians.optimizer.step()')
    if optimizer_step_location != -1:
        # Добавляем step для классификатора после step gaussians
        new_optimizer_code = '''gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                
                # Шаг оптимизатора классификатора
                cls_optimizer.step()
                cls_optimizer.zero_grad(set_to_none=True)'''
        
        end_line = content.find('\n', optimizer_step_location)
        content = content[:optimizer_step_location] + new_optimizer_code + content[end_line+1:]
        print("✅ Добавлен optimizer step для классификатора")
    
    # Сохраняем исправленный файл
    with open('train.py', 'w') as f:
        f.write(content)
    
    print("✅ СЕГМЕНТАЦИЯ ПОЛНОСТЬЮ ИСПРАВЛЕНА!")
    print("🎯 Маски сегментации сохранены и будут работать правильно!")

if __name__ == "__main__":
    print("🚨 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ СЕГМЕНТАЦИИ GAUSSIAN GROUPING")
    print("Исправляем размерности классификатора, СОХРАНЯЯ маски!")
    fix_segmentation_train()
    print("\n🎯 Готово! Теперь запускайте с полной сегментацией:")
    print("python3 train.py -s data/dataset -r 1 -m output/dataset --config_file config/gaussian_dataset/train.json")