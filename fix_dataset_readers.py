#!/usr/bin/env python3
"""
Патч для scene/dataset_readers.py - добавляет поддержку sparse_fixed папки
"""

def patch_dataset_readers():
    """Патчит readColmapSceneInfo для поддержки sparse_fixed"""
    
    patch_content = '''
def readColmapSceneInfo(path, images, eval, object_path, n_views=None, random_init=False, train_split=None):
    try:
        # Пробуем загрузить исправленную модель камеры
        cameras_extrinsic_file = os.path.join(path, "sparse_fixed/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse_fixed/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        reading_dir = "images" if images == None else images
        object_dir = "objects" if object_path == None else object_path
        print("✓ Используем исправленную модель камеры из sparse_fixed/0/")
    except:
        try:
            # Фоллбэк на оригинальную модель
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            reading_dir = "images" if images == None else images
            object_dir = "objects" if object_path == None else object_path
            print("⚠ Используем оригинальную модель камеры из sparse/0/")
        except:
            cam_extrinsics = read_extrinsics_text(os.path.join(path, "sparse/0", "images.txt"))
            cam_intrinsics = read_intrinsics_text(os.path.join(path, "sparse/0", "cameras.txt"))
            reading_dir = "images" if images == None else images
            object_dir = "objects" if object_path == None else object_path
            print("✓ Используем текстовые файлы из sparse/0/")

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), objects_folder=os.path.join(path, object_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 4 != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 4 == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # Здесь можно добавить ограничение количества изображений
    if n_views is not None:
        if train_split is not None:
            # Разделяем на train/test по процентному соотношению
            n_train = int(n_views * train_split)
            n_test = n_views - n_train
            train_cam_infos = train_cam_infos[:n_train]
            test_cam_infos = test_cam_infos[:n_test] if test_cam_infos else []
        else:
            train_cam_infos = train_cam_infos[:n_views]
            
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Загружаем точечное облако
    ply_path = os.path.join(path, "sparse_fixed/0/points3D.ply")
    if not os.path.exists(ply_path):
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
    if not os.path.exists(ply_path):
        print("Конвертируем points3D.bin в .ply...")
        try:
            bin_path = os.path.join(path, "sparse_fixed/0/points3D.bin")
            if not os.path.exists(bin_path):
                bin_path = os.path.join(path, "sparse/0/points3D.bin")
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            txt_path = os.path.join(path, "sparse/0/points3D.txt")
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                          train_cameras=train_cam_infos,
                          test_cameras=test_cam_infos,
                          nerf_normalization=nerf_normalization,
                          ply_path=ply_path)
    return scene_info
'''
    
    # Читаем оригинальный файл
    with open('scene/dataset_readers.py', 'r') as f:
        content = f.read()
    
    # Ищем функцию readColmapSceneInfo
    start = content.find('def readColmapSceneInfo(')
    if start == -1:
        print("❌ Функция readColmapSceneInfo не найдена")
        return False
    
    # Ищем конец функции (следующую def или EOF)
    end = content.find('\ndef ', start + 1)
    if end == -1:
        end = len(content)
    
    # Заменяем функцию
    new_content = content[:start] + patch_content.strip() + content[end:]
    
    # Создаем backup
    with open('scene/dataset_readers.py.backup', 'w') as f:
        f.write(content)
    
    # Записываем новый файл
    with open('scene/dataset_readers.py', 'w') as f:
        f.write(new_content)
    
    print("✓ Файл scene/dataset_readers.py обновлен")
    print("✓ Backup сохранен в scene/dataset_readers.py.backup")
    return True

if __name__ == "__main__":
    patch_dataset_readers()