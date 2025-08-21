#!/usr/bin/env python3
"""
Обновление scene/dataset_readers.py для автоматической работы с PINHOLE моделями
"""

def update_dataset_readers():
    """Обновляет readColmapSceneInfo для поддержки только PINHOLE моделей"""
    
    new_readColmapSceneInfo = '''
def readColmapSceneInfo(path, images, eval, object_path, n_views=None, random_init=False, train_split=None):
    """
    Загружает COLMAP сцену. Поддерживает только PINHOLE и SIMPLE_PINHOLE модели.
    Автоматически ищет правильную sparse модель.
    """
    
    def try_load_cameras(sparse_path_name):
        """Пытается загрузить камеры из указанной папки"""
        try:
            cameras_extrinsic_file = os.path.join(path, sparse_path_name, "images.bin")
            cameras_intrinsic_file = os.path.join(path, sparse_path_name, "cameras.bin")
            
            if not os.path.exists(cameras_extrinsic_file) or not os.path.exists(cameras_intrinsic_file):
                return None, None, sparse_path_name
                
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
            
            print(f"✅ Загружены камеры из {sparse_path_name}/")
            return cam_extrinsics, cam_intrinsics, sparse_path_name
        except Exception as e:
            print(f"⚠️ Не удалось загрузить из {sparse_path_name}/: {e}")
            return None, None, sparse_path_name
    
    # Пробуем загрузить камеры из разных папок в порядке приоритета
    cam_extrinsics, cam_intrinsics, used_sparse = None, None, None
    
    # 1. Сначала пробуем sparse/0 (результат image_undistorter)
    if cam_extrinsics is None:
        cam_extrinsics, cam_intrinsics, used_sparse = try_load_cameras("sparse/0")
    
    # 2. Если не получилось, пробуем sparse_fixed/0
    if cam_extrinsics is None:
        cam_extrinsics, cam_intrinsics, used_sparse = try_load_cameras("sparse_fixed/0")
    
    # 3. Последняя попытка - текстовые файлы
    if cam_extrinsics is None:
        try:
            cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
            used_sparse = "sparse/0 (text)"
            print(f"✅ Загружены камеры из текстовых файлов")
        except Exception as e:
            print(f"❌ Не удалось загрузить камеры: {e}")
            raise RuntimeError("Не найдены файлы COLMAP камер")
    
    # Проверяем совместимость моделей камер
    incompatible_cameras = []
    for camera_id, camera in cam_intrinsics.items():
        if camera.model not in ["PINHOLE", "SIMPLE_PINHOLE"]:
            incompatible_cameras.append(f"Камера {camera_id}: {camera.model}")
    
    if incompatible_cameras:
        error_msg = "Найдены несовместимые модели камер:\\n" + "\\n".join(incompatible_cameras)
        error_msg += "\\n\\nИспользуйте convert_fixed.py для создания правильных данных."
        raise RuntimeError(error_msg)
    
    reading_dir = "images" if images == None else images
    object_dir = "objects" if object_path == None else object_path
    
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, 
        cam_intrinsics=cam_intrinsics, 
        images_folder=os.path.join(path, reading_dir), 
        objects_folder=os.path.join(path, object_dir)
    )
    
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    
    if len(cam_infos) == 0:
        raise RuntimeError(f"Не загружено ни одной камеры из {used_sparse}")
    
    print(f"📷 Загружено {len(cam_infos)} камер из {used_sparse}")
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 4 != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % 4 == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # Ограничение количества изображений
    if n_views is not None:
        if train_split is not None:
            n_train = int(n_views * train_split)
            n_test = n_views - n_train
            train_cam_infos = train_cam_infos[:n_train]
            test_cam_infos = test_cam_infos[:n_test] if test_cam_infos else []
        else:
            train_cam_infos = train_cam_infos[:n_views]
    
    if len(train_cam_infos) == 0:
        raise RuntimeError("Нет камер для обучения")
            
    nerf_normalization = getNerfppNorm(train_cam_infos)

    # Загружаем точечное облако
    ply_path = None
    sparse_base = used_sparse.split()[0] if " " in used_sparse else used_sparse
    
    # Пробуем найти .ply файл
    for potential_ply in [
        os.path.join(path, sparse_base, "points3D.ply"),
        os.path.join(path, "sparse/0/points3D.ply"),
        os.path.join(path, "sparse_fixed/0/points3D.ply")
    ]:
        if os.path.exists(potential_ply):
            ply_path = potential_ply
            break
    
    # Если .ply не найден, конвертируем из .bin
    if ply_path is None:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        print("🔄 Конвертируем points3D.bin в .ply...")
        
        try:
            for potential_bin in [
                os.path.join(path, sparse_base, "points3D.bin"),
                os.path.join(path, "sparse/0/points3D.bin"),
                os.path.join(path, "sparse_fixed/0/points3D.bin")
            ]:
                if os.path.exists(potential_bin):
                    xyz, rgb, _ = read_points3D_binary(potential_bin)
                    storePly(ply_path, xyz, rgb)
                    print(f"✅ Конвертирован {potential_bin} -> {ply_path}")
                    break
            else:
                # Пробуем текстовый файл
                txt_path = os.path.join(path, "sparse/0/points3D.txt")
                if os.path.exists(txt_path):
                    xyz, rgb, _ = read_points3D_text(txt_path)
                    storePly(ply_path, xyz, rgb)
                    print(f"✅ Конвертирован {txt_path} -> {ply_path}")
                else:
                    print("⚠️ Файл points3D не найден, создаем случайные точки")
                    # Создаем случайные точки как fallback
                    num_pts = 1000
                    xyz = np.random.random((num_pts, 3)) * 2.0 - 1.0
                    rgb = np.random.randint(0, 255, (num_pts, 3))
                    storePly(ply_path, xyz, rgb)
        except Exception as e:
            print(f"⚠️ Ошибка конвертации points3D: {e}")

    try:
        pcd = fetchPly(ply_path)
        print(f"✅ Загружено точечное облако: {len(pcd.points)} точек")
    except Exception as e:
        print(f"⚠️ Не удалось загрузить точечное облако: {e}")
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )
    return scene_info
'''