#!/usr/bin/env python3
"""
Исправление проблемы с пустым train_cam_infos
"""

def patch_readColmapSceneInfo():
    """Добавляет отладку и исправляет логику разделения камер"""
    
    # Читаем файл
    with open('scene/dataset_readers.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Создаем backup
    with open('scene/dataset_readers.py.train_cameras_fix.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Ищем место где создаются train_cam_infos и test_cam_infos
    old_split_logic = '''    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 4 != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 4 == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []'''
    
    new_split_logic = '''    print(f"🔍 Всего загружено камер: {len(cam_infos)}")
    for i, cam in enumerate(cam_infos[:3]):
        print(f"  Камера {i}: {cam.image_name}")
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 4 != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 4 == 0]
        print(f"📊 eval=True: train={len(train_cam_infos)}, test={len(test_cam_infos)}")
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        print(f"📊 eval=False: train={len(train_cam_infos)}, test={len(test_cam_infos)}")
    
    if len(train_cam_infos) == 0:
        print("❌ КРИТИЧЕСКАЯ ОШИБКА: train_cam_infos пустой!")
        print(f"   eval={eval}, len(cam_infos)={len(cam_infos)}")
        if eval and len(cam_infos) > 0:
            print("   Принудительно используем все камеры для обучения")
            train_cam_infos = cam_infos
        else:
            raise RuntimeError("Нет камер для обучения")'''
    
    # Заменяем логику
    if old_split_logic in content:
        content = content.replace(old_split_logic, new_split_logic)
        print("✅ Исправлена логика разделения камер")
    else:
        print("⚠️ Стандартная логика разделения не найдена, ищем альтернативный вариант...")
        
        # Альтернативный поиск с более гибким паттерном
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 4 != 0]' in line:
                # Нашли место, добавляем отладку перед этой строкой
                debug_lines = [
                    '    print(f"🔍 Всего загружено камер: {len(cam_infos)}")',
                    '    for j, cam in enumerate(cam_infos[:3]):',
                    '        print(f"  Камера {j}: {cam.image_name}")',
                    '    '
                ]
                
                # Вставляем отладку
                lines[i:i] = debug_lines
                
                # Находим следующие строки и добавляем проверку
                for j in range(i+len(debug_lines), min(i+len(debug_lines)+10, len(lines))):
                    if 'test_cam_infos = []' in lines[j]:
                        # Добавляем проверку после создания списков
                        check_lines = [
                            '    ',
                            '    print(f"📊 Камеры: train={len(train_cam_infos)}, test={len(test_cam_infos)}")',
                            '    if len(train_cam_infos) == 0:',
                            '        print("❌ КРИТИЧЕСКАЯ ОШИБКА: train_cam_infos пустой!")',
                            '        if eval and len(cam_infos) > 0:',
                            '            print("   Принудительно используем все камеры для обучения")',
                            '            train_cam_infos = cam_infos',
                            '        else:',
                            '            raise RuntimeError("Нет камер для обучения")'
                        ]
                        lines[j+1:j+1] = check_lines
                        break
                
                content = '\n'.join(lines)
                print("✅ Добавлена отладка и проверка камер")
                break
        else:
            print("❌ Не удалось найти место для исправления")
            return False
    
    # Также исправляем getNerfppNorm для обработки пустых списков
    old_getnerfpp = '''def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}'''
    
    new_getnerfpp = '''def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        if len(cam_centers) == 0:
            print("⚠️ Пустой список центров камер")
            return np.array([0.0, 0.0, 0.0]), 1.0
            
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    if len(cam_info) == 0:
        print("❌ getNerfppNorm: Нет камер для нормализации")
        return {"translate": np.array([0.0, 0.0, 0.0]), "radius": 1.0}

    cam_centers = []

    for cam in cam_info:
        try:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        except Exception as e:
            print(f"⚠️ Ошибка обработки камеры {getattr(cam, 'image_name', 'unknown')}: {e}")
            continue

    if len(cam_centers) == 0:
        print("❌ getNerfppNorm: Не удалось извлечь центры камер")
        return {"translate": np.array([0.0, 0.0, 0.0]), "radius": 1.0}

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    print(f"✅ NeRF нормализация: {len(cam_centers)} камер, радиус={radius:.3f}")
    return {"translate": translate, "radius": radius}'''
    
    # Заменяем getNerfppNorm
    if old_getnerfpp in content:
        content = content.replace(old_getnerfpp, new_getnerfpp)
        print("✅ Исправлена функция getNerfppNorm")
    else:
        print("⚠️ Функция getNerfppNorm не найдена для замены")
    
    # Записываем исправленный файл
    with open('scene/dataset_readers.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Исправления применены к scene/dataset_readers.py")
    return True

def check_train_args():
    """Проверяет аргументы обучения"""
    print("\n=== ПРОВЕРКА АРГУМЕНТОВ ОБУЧЕНИЯ ===")
    print("Команда: python3 train.py -s data/dataset1024 -r 1 -m output/dataset1024 --config_file config/gaussian_dataset/train.json")
    print("\nПараметры:")
    print("  --eval: НЕ УКАЗАН (eval=False)")
    print("  -r 1: разрешение = 1")
    print("  Значит train_cam_infos должен быть = cam_infos")
    print("\nЕсли после исправления все еще ошибка, попробуйте:")
    print("  python3 train.py -s data/dataset1024 -r 1 -m output/dataset1024 --config_file config/gaussian_dataset/train.json --eval")

if __name__ == "__main__":
    patch_readColmapSceneInfo()
    check_train_args()