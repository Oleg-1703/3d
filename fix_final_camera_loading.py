#!/usr/bin/env python3
"""
ОКОНЧАТЕЛЬНОЕ ИСПРАВЛЕНИЕ проблемы с загрузкой камер
"""

def fix_camera_loading():
    """Исправляет проблему с пустым scene.getTrainCameras()"""
    
    # Исправляем utils/camera_utils.py
    print("=== ИСПРАВЛЕНИЕ utils/camera_utils.py ===")
    
    # Читаем файл
    with open('utils/camera_utils.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Создаем backup
    with open('utils/camera_utils.py.final_fix.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Ищем функцию loadCam и добавляем проверки
    lines = content.split('\n')
    new_lines = []
    
    for i, line in enumerate(lines):
        new_lines.append(line)
        
        # Добавляем отладку в cameraList_from_camInfos
        if 'def cameraList_from_camInfos(cam_infos, resolution_scale, args):' in line:
            new_lines.extend([
                '    print(f"📸 cameraList_from_camInfos: получено {len(cam_infos)} cam_infos")',
                '    if len(cam_infos) == 0:',
                '        print("❌ КРИТИЧЕСКАЯ ОШИБКА: cam_infos пустой в cameraList_from_camInfos!")',
                '        return []',
                ''
            ])
        
        # Добавляем отладку в цикл загрузки камер
        elif 'for id, c in enumerate(cam_infos):' in line:
            new_lines.extend([
                '        print(f"  Обрабатываем камеру {id}: {c.image_name}")',
                ''
            ])
        
        # Добавляем отладку в конец функции cameraList_from_camInfos
        elif 'return camera_list' in line and 'cameraList_from_camInfos' in ''.join(lines[max(0, i-20):i]):
            new_lines.insert(-1, '    print(f"✅ cameraList_from_camInfos: создано {len(camera_list)} камер")')
    
    # Записываем исправленный файл
    with open('utils/camera_utils.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    print("✅ Добавлена отладка в utils/camera_utils.py")
    
    # Исправляем scene/__init__.py
    print("\n=== ИСПРАВЛЕНИЕ scene/__init__.py ===")
    
    with open('scene/__init__.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open('scene/__init__.py.final_fix.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Ищем место загрузки камер и добавляем отладку
    old_loading = '''        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)'''
    
    new_loading = '''        for resolution_scale in resolution_scales:
            print(f"Loading Training Cameras (scale={resolution_scale})")
            print(f"  scene_info.train_cameras: {len(scene_info.train_cameras)} элементов")
            if len(scene_info.train_cameras) == 0:
                print("❌ КРИТИЧЕСКАЯ ОШИБКА: scene_info.train_cameras пустой!")
                raise RuntimeError("Нет камер для обучения в scene_info")
            
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print(f"  Создано train_cameras: {len(self.train_cameras[resolution_scale])} камер")
            
            print(f"Loading Test Cameras (scale={resolution_scale})")
            print(f"  scene_info.test_cameras: {len(scene_info.test_cameras)} элементов")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            print(f"  Создано test_cameras: {len(self.test_cameras[resolution_scale])} камер")
            
            if len(self.train_cameras[resolution_scale]) == 0:
                print("❌ КРИТИЧЕСКАЯ ОШИБКА: self.train_cameras пустой после cameraList_from_camInfos!")
                raise RuntimeError("Камеры не загружены в Scene")'''
    
    if old_loading in content:
        content = content.replace(old_loading, new_loading)
        print("✅ Добавлена отладка загрузки камер")
    else:
        print("⚠️ Не найден блок загрузки камер для замены")
    
    # Также добавляем отладку в getTrainCameras
    old_get_train = '''    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]'''
    
    new_get_train = '''    def getTrainCameras(self, scale=1.0):
        cameras = self.train_cameras[scale]
        print(f"🎯 getTrainCameras(scale={scale}): возвращаем {len(cameras)} камер")
        if len(cameras) == 0:
            print("❌ КРИТИЧЕСКАЯ ОШИБКА: getTrainCameras возвращает пустой список!")
            print(f"   Доступные масштабы: {list(self.train_cameras.keys())}")
            for s, cams in self.train_cameras.items():
                print(f"     scale {s}: {len(cams)} камер")
        return cameras'''
    
    if old_get_train in content:
        content = content.replace(old_get_train, new_get_train)
        print("✅ Добавлена отладка в getTrainCameras")
    else:
        print("⚠️ Не найдена функция getTrainCameras для замены")
    
    # Записываем исправленный файл
    with open('scene/__init__.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Исправления применены к scene/__init__.py")
    
    # Исправляем train.py для лучшей обработки ошибок
    print("\n=== ИСПРАВЛЕНИЕ train.py ===")
    
    with open('train.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open('train.py.final_fix.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Ищем проблемное место и добавляем проверку
    old_viewpoint = '''        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))'''
    
    new_viewpoint = '''        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            print(f"🎲 Загружен viewpoint_stack: {len(viewpoint_stack)} камер")
            if len(viewpoint_stack) == 0:
                print("❌ КРИТИЧЕСКАЯ ОШИБКА: viewpoint_stack пустой!")
                print("   Это означает что scene.getTrainCameras() вернул пустой список")
                raise RuntimeError("Нет камер для обучения - проверьте загрузку данных")
        
        print(f"🎯 Выбираем случайную камеру из {len(viewpoint_stack)} доступных")
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        print(f"   Выбрана камера: {viewpoint_cam.image_name}")'''
    
    if old_viewpoint in content:
        content = content.replace(old_viewpoint, new_viewpoint)
        print("✅ Добавлена отладка выбора камеры")
    else:
        print("⚠️ Не найден блок выбора камеры для замены")
    
    # Записываем исправленный файл
    with open('train.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Исправления применены к train.py")
    
    print("\n🎯 ВСЕ ИСПРАВЛЕНИЯ ПРИМЕНЕНЫ!")
    print("\nТеперь запустите:")
    print("python3 train.py -s data/dataset1024 -r 1 -m output/dataset1024 --config_file config/gaussian_dataset/train.json")
    print("\nОтладочная информация покажет где именно теряются камеры.")

if __name__ == "__main__":
    fix_camera_loading()