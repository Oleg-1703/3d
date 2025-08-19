#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
from argparse import ArgumentParser
import shutil

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')
parser.add_argument("--skip_matching", action='store_true')
parser.add_argument("--source_path", "-s", required=True, type=str)
parser.add_argument("--output_path", "-o", type=str, default=None, help="Output path for results (default: same as source_path)")
parser.add_argument("--camera", default="OPENCV", type=str)
parser.add_argument("--colmap_executable", default="", type=str)
parser.add_argument("--resize", action="store_true")
parser.add_argument("--magick_executable", default="", type=str)
# Новые параметры для улучшенной обработки
parser.add_argument("--aggressive_matching", action="store_true", help="Use more aggressive matching parameters")
parser.add_argument("--max_images", type=int, default=None, help="Maximum number of images to process")
parser.add_argument("--vocab_tree_path", type=str, default="", help="Path to vocabulary tree for sequential matching")

args = parser.parse_args()

# Определяем выходную папку
output_path = args.output_path if args.output_path else args.source_path
if args.output_path:
    os.makedirs(output_path, exist_ok=True)
    print(f"Результаты будут сохранены в: {output_path}")
else:
    print(f"Результаты будут сохранены в источнике: {args.source_path}")

colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
use_gpu = 1 if not args.no_gpu else 0

# Проверим количество изображений
input_images = os.listdir(args.source_path + "/input")
print(f"Найдено изображений в input: {len(input_images)}")

if args.max_images and len(input_images) > args.max_images:
    print(f"Ограничиваем до {args.max_images} изображений")
    # Создаем временную папку с выбранными изображениями
    temp_input = args.source_path + "/input_selected"
    os.makedirs(temp_input, exist_ok=True)
    
    # Выбираем изображения равномерно
    step = len(input_images) // args.max_images
    selected_images = input_images[::step][:args.max_images]
    
    for img in selected_images:
        shutil.copy2(os.path.join(args.source_path, "input", img), 
                    os.path.join(temp_input, img))
    
    input_path = temp_input
else:
    input_path = args.source_path + "/input"

if not args.skip_matching:
    os.makedirs(output_path + "/distorted/sparse", exist_ok=True)

    ## Feature extraction с улучшенными параметрами
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + output_path + "/distorted/database.db \
        --image_path " + input_path + " \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera
    
    # Добавляем агрессивные параметры для извлечения особенностей
    if args.aggressive_matching:
        feat_extracton_cmd += " \
        --SiftExtraction.max_num_features 8192 \
        --SiftExtraction.peak_threshold 0.004 \
        --SiftExtraction.edge_threshold 5.0 \
        --SiftExtraction.max_num_orientations 2"
    else:
        feat_extracton_cmd += " \
        --SiftExtraction.max_num_features 4096"
    
    feat_extracton_cmd += " --SiftExtraction.use_gpu " + str(use_gpu)
    
    print("Запуск извлечения особенностей...")
    print(feat_extracton_cmd)
    exit_code = os.system(feat_extracton_cmd)
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## Feature matching с улучшенными параметрами
    if args.vocab_tree_path and os.path.exists(args.vocab_tree_path):
        # Используем vocabulary tree для sequential matching (лучше для большого количества изображений)
        feat_matching_cmd = colmap_command + " vocab_tree_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --VocabTreeMatching.vocab_tree_path " + args.vocab_tree_path + " \
        --SiftMatching.use_gpu " + str(use_gpu)
    else:
        # Используем exhaustive matcher с более мягкими параметрами
        feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
        
        if args.aggressive_matching:
            feat_matching_cmd += " \
            --SiftMatching.max_ratio 0.8 \
            --SiftMatching.max_distance 0.7 \
            --SiftMatching.cross_check 1 \
            --SiftMatching.max_num_matches 32768"
    
    print("Запуск сопоставления особенностей...")
    print(feat_matching_cmd)
    exit_code = os.system(feat_matching_cmd)
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### Bundle adjustment с улучшенными параметрами
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + input_path + " \
        --output_path "  + args.source_path + "/distorted/sparse")
    
    if args.aggressive_matching:
        mapper_cmd += " \
        --Mapper.init_min_tri_angle 4 \
        --Mapper.init_max_forward_motion 0.95 \
        --Mapper.init_min_num_inliers 100 \
        --Mapper.abs_pose_min_num_inliers 30 \
        --Mapper.abs_pose_min_inlier_ratio 0.25 \
        --Mapper.ba_local_max_num_iterations 50 \
        --Mapper.ba_global_max_num_iterations 100 \
        --Mapper.min_num_matches 15"
    
    print("Запуск bundle adjustment...")
    print(mapper_cmd)
    exit_code = os.system(mapper_cmd)
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

# Проверим результаты mapping
sparse_models = []
distorted_sparse_path = args.source_path + "/distorted/sparse"
if os.path.exists(distorted_sparse_path):
    sparse_models = [d for d in os.listdir(distorted_sparse_path) 
                    if os.path.isdir(os.path.join(distorted_sparse_path, d)) and d.isdigit()]

print(f"Найдено моделей после mapping: {len(sparse_models)}")
if len(sparse_models) == 0:
    print("ВНИМАНИЕ: Mapping не создал ни одной модели!")
    print("Попробуйте:")
    print("1. Запустить с --aggressive_matching")
    print("2. Уменьшить количество изображений с --max_images 50")
    print("3. Проверить качество изображений")
    exit(1)

# Выбираем модель с наибольшим количеством изображений
best_model = "0"
if len(sparse_models) > 1:
    model_sizes = {}
    for model in sparse_models:
        images_path = os.path.join(distorted_sparse_path, model, "images.bin")
        if os.path.exists(images_path):
            # Простая проверка размера файла как индикатор количества изображений
            model_sizes[model] = os.path.getsize(images_path)
    if model_sizes:
        best_model = max(model_sizes, key=model_sizes.get)
        print(f"Выбрана модель {best_model} с наибольшим количеством изображений")

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
best_model_path = os.path.join(distorted_sparse_path, best_model)
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + input_path + " \
    --input_path " + best_model_path + " \
    --output_path " + args.source_path + "\
    --output_type COLMAP")

print("Запуск undistortion...")
print(img_undist_cmd)
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"Image undistorter failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(args.source_path + "/sparse")
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

# Проверяем финальные результаты
final_images_path = args.source_path + "/sparse/0/images.bin"
if os.path.exists(final_images_path):
    from scene.colmap_loader import read_extrinsics_binary
    try:
        cam_extrinsics = read_extrinsics_binary(final_images_path)
        print(f"УСПЕХ: Зарегистрировано {len(cam_extrinsics)} изображений")
        print("Зарегистрированные изображения:")
        for key, img in cam_extrinsics.items():
            print(f"  {img.name}")
    except Exception as e:
        print(f"Ошибка чтения результатов: {e}")

# Очистка временных файлов
if args.max_images and os.path.exists(temp_input):
    shutil.rmtree(temp_input)

if(args.resize):
    print("Copying and resizing...")

    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)
    os.makedirs(args.source_path + "/images_4", exist_ok=True)
    os.makedirs(args.source_path + "/images_8", exist_ok=True)
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # Copy each file from the source directory to the destination directory
    for file in files:
        source_file = os.path.join(args.source_path, "images", file)

        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

print("Done.")