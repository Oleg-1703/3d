#!/usr/bin/env python3
"""
Исправленный COLMAP конвертер для 3D Gaussian Splatting
Создает сразу правильные неискаженные данные (PINHOLE модель)
Поддерживает GPU ускорение
"""

import os
import logging
import shutil
import sys
from argparse import ArgumentParser
import subprocess

def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

def check_colmap_installation():
    """Проверка установки COLMAP"""
    try:
        result = subprocess.run(['colmap', '--help'], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def run_command(cmd, description):
    """Выполнение команды с обработкой ошибок"""
    logging.info(f"🔄 {description}")
    logging.info(f"Команда: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        if isinstance(cmd, str):
            exit_code = os.system(cmd)
        else:
            result = subprocess.run(cmd, check=True)
            exit_code = result.returncode
            
        if exit_code != 0:
            logging.error(f"❌ {description} завершилось с кодом {exit_code}")
            return False
        else:
            logging.info(f"✅ {description} выполнено успешно")
            return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ {description} не удалось: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ Ошибка при выполнении {description}: {e}")
        return False

def main():
    setup_logging()
    
    parser = ArgumentParser("COLMAP converter для 3D Gaussian Splatting")
    parser.add_argument("--no_gpu", action='store_true', help="Отключить GPU ускорение")
    parser.add_argument("--skip_matching", action='store_true', help="Пропустить feature matching")
    parser.add_argument("--source_path", "-s", required=True, type=str, help="Путь к датасету")
    parser.add_argument("--camera", default="PINHOLE", type=str, 
                       help="Модель камеры (PINHOLE рекомендуется для 3DGS)")
    parser.add_argument("--colmap_executable", default="", type=str, help="Путь к COLMAP")
    parser.add_argument("--resize", action="store_true", help="Создать уменьшенные копии")
    parser.add_argument("--magick_executable", default="", type=str, help="Путь к ImageMagick")
    parser.add_argument("--quality", default="high", choices=["low", "medium", "high"], 
                       help="Качество реконструкции")
    
    args = parser.parse_args()
    
    # Настройка команд
    colmap_command = f'"{args.colmap_executable}"' if args.colmap_executable else "colmap"
    magick_command = f'"{args.magick_executable}"' if args.magick_executable else "magick"
    use_gpu = 0 if args.no_gpu else 1
    
    # Проверка COLMAP
    if not check_colmap_installation():
        logging.error("❌ COLMAP не найден. Установите COLMAP или укажите путь через --colmap_executable")
        sys.exit(1)
    
    logging.info(f"🚀 Запуск COLMAP конвертера для 3DGS")
    logging.info(f"📂 Источник: {args.source_path}")
    logging.info(f"🎯 GPU ускорение: {'Включено' if use_gpu else 'Отключено'}")
    logging.info(f"📷 Модель камеры: {args.camera}")
    
    # Проверка входных данных
    input_path = os.path.join(args.source_path, "input")
    if not os.path.exists(input_path):
        logging.error(f"❌ Папка {input_path} не найдена")
        sys.exit(1)
    
    image_files = [f for f in os.listdir(input_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp'))]
    if len(image_files) == 0:
        logging.error(f"❌ В папке {input_path} не найдено изображений")
        sys.exit(1)
    
    logging.info(f"📸 Найдено изображений: {len(image_files)}")
    
    # === ЭТАП 1: FEATURE EXTRACTION И MATCHING ===
    if not args.skip_matching:
        os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)
        
        # Feature extraction с оптимизированными параметрами
        feat_extraction_cmd = f"{colmap_command} feature_extractor " \
            f"--database_path {args.source_path}/distorted/database.db " \
            f"--image_path {args.source_path}/input " \
            f"--ImageReader.single_camera 1 " \
            f"--ImageReader.camera_model {args.camera} " \
            f"--SiftExtraction.use_gpu {use_gpu} " \
            f"--SiftExtraction.max_image_size 3200 " \
            f"--SiftExtraction.max_num_features 8192"
        
        if not run_command(feat_extraction_cmd, "Feature extraction"):
            sys.exit(1)
        
        # Feature matching с GPU ускорением
        feat_matching_cmd = f"{colmap_command} exhaustive_matcher " \
            f"--database_path {args.source_path}/distorted/database.db " \
            f"--SiftMatching.use_gpu {use_gpu} " \
            f"--SiftMatching.max_ratio 0.8 " \
            f"--SiftMatching.max_distance 0.7"
        
        if not run_command(feat_matching_cmd, "Feature matching"):
            sys.exit(1)
        
        # Bundle adjustment (mapper) с оптимизированными параметрами
        mapper_cmd = f"{colmap_command} mapper " \
            f"--database_path {args.source_path}/distorted/database.db " \
            f"--image_path {args.source_path}/input " \
            f"--output_path {args.source_path}/distorted/sparse " \
            f"--Mapper.ba_refine_focal_length 1 " \
            f"--Mapper.ba_refine_principal_point 1 " \
            f"--Mapper.ba_refine_extra_params 1"
        
        if not run_command(mapper_cmd, "Bundle adjustment (Mapper)"):
            sys.exit(1)
    
    # === ЭТАП 2: IMAGE UNDISTORTION (КЛЮЧЕВОЙ ДЛЯ 3DGS) ===
    logging.info("🔧 Создание неискаженных изображений для 3DGS...")
    
    # Проверяем, есть ли результаты реконструкции
    sparse_input = os.path.join(args.source_path, "distorted/sparse/0")
    if not os.path.exists(sparse_input):
        logging.error(f"❌ Не найден результат реконструкции: {sparse_input}")
        sys.exit(1)
    
    # Image undistorter - создает PINHOLE модель без искажений
    img_undist_cmd = f"{colmap_command} image_undistorter " \
        f"--image_path {args.source_path}/input " \
        f"--input_path {sparse_input} " \
        f"--output_path {args.source_path} " \
        f"--output_type COLMAP " \
        f"--max_image_size 2048"
    
    if not run_command(img_undist_cmd, "Image undistortion"):
        sys.exit(1)
    
    # === ЭТАП 3: ОРГАНИЗАЦИЯ ФАЙЛОВ ДЛЯ 3DGS ===
    logging.info("📁 Организация файлов для 3DGS...")
    
    # Перемещаем файлы sparse модели в правильное место
    sparse_files = os.listdir(args.source_path + "/sparse")
    os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
    
    for file in sparse_files:
        if file == '0':
            continue
        source_file = os.path.join(args.source_path, "sparse", file)
        destination_file = os.path.join(args.source_path, "sparse", "0", file)
        if os.path.exists(source_file):
            shutil.move(source_file, destination_file)
            logging.info(f"📄 Перемещен: {file}")
    
    # Проверяем, что создались правильные файлы
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    sparse_0_path = os.path.join(args.source_path, "sparse/0")
    
    for req_file in required_files:
        file_path = os.path.join(sparse_0_path, req_file)
        if os.path.exists(file_path):
            logging.info(f"✅ Найден: {req_file}")
        else:
            logging.error(f"❌ Отсутствует: {req_file}")
    
    # Проверяем модель камеры
    logging.info("🔍 Проверка модели камеры...")
    try:
        import struct
        cameras_path = os.path.join(sparse_0_path, "cameras.bin")
        if os.path.exists(cameras_path):
            with open(cameras_path, 'rb') as f:
                num_cameras = struct.unpack('Q', f.read(8))[0]
                for i in range(num_cameras):
                    camera_id = struct.unpack('I', f.read(4))[0]
                    model_id = struct.unpack('I', f.read(4))[0]
                    width = struct.unpack('Q', f.read(8))[0]
                    height = struct.unpack('Q', f.read(8))[0]
                    
                    model_names = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 4: "OPENCV"}
                    model_name = model_names.get(model_id, f"UNKNOWN_{model_id}")
                    
                    if model_id in [0, 1]:  # SIMPLE_PINHOLE или PINHOLE
                        logging.info(f"✅ Камера {camera_id}: {model_name} - совместима с 3DGS")
                    else:
                        logging.warning(f"⚠️ Камера {camera_id}: {model_name} - может потребовать дополнительной обработки")
    except Exception as e:
        logging.warning(f"⚠️ Не удалось проверить модель камеры: {e}")
    
    # === ЭТАП 4: СОЗДАНИЕ УМЕНЬШЕННЫХ КОПИЙ (ОПЦИОНАЛЬНО) ===
    if args.resize:
        logging.info("🖼️ Создание уменьшенных копий изображений...")
        
        scales = [("images_2", "50%"), ("images_4", "25%"), ("images_8", "12.5%")]
        
        for folder, scale in scales:
            target_dir = os.path.join(args.source_path, folder)
            os.makedirs(target_dir, exist_ok=True)
            
            images_dir = os.path.join(args.source_path, "images")
            if os.path.exists(images_dir):
                image_files = os.listdir(images_dir)
                
                for image_file in image_files:
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp')):
                        source_file = os.path.join(images_dir, image_file)
                        dest_file = os.path.join(target_dir, image_file)
                        
                        # Копируем файл
                        shutil.copy2(source_file, dest_file)
                        
                        # Изменяем размер
                        resize_cmd = f"{magick_command} mogrify -resize {scale} {dest_file}"
                        if not run_command(resize_cmd, f"Изменение размера до {scale}"):
                            logging.warning(f"⚠️ Не удалось изменить размер {image_file}")
                
                logging.info(f"✅ Создана папка {folder} с изображениями {scale}")
    
    # === ФИНАЛЬНАЯ ПРОВЕРКА ===
    logging.info("🎯 Финальная проверка датасета...")
    
    # Проверяем структуру папок для 3DGS
    expected_structure = {
        "images": "Неискаженные изображения",
        "sparse/0": "COLMAP модель (PINHOLE)",
        "input": "Оригинальные изображения"
    }
    
    all_good = True
    for folder, description in expected_structure.items():
        path = os.path.join(args.source_path, folder)
        if os.path.exists(path):
            if folder == "images" or folder == "input":
                file_count = len([f for f in os.listdir(path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.bmp'))])
                logging.info(f"✅ {description}: {file_count} файлов")
            else:
                logging.info(f"✅ {description}: найден")
        else:
            logging.error(f"❌ {description}: не найден в {path}")
            all_good = False
    
    if all_good:
        logging.info("🎉 УСПЕХ! Датасет готов для 3D Gaussian Splatting")
        logging.info(f"📁 Запустите обучение: python3 train.py -s {args.source_path} -m output/model")
    else:
        logging.error("❌ Датасет не готов. Проверьте ошибки выше.")
        sys.exit(1)

if __name__ == "__main__":
    main()