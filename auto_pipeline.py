#!/usr/bin/env python3
"""
Автоматизированный пайплайн для 3D Gaussian Splatting с сегментацией
Использование: python3 automated_3dgs_pipeline.py --dataset data/dataset512 --max_size 1980 --prompt central_object
"""

import os
import sys
import argparse
import subprocess
import shutil
from pathlib import Path
import json
import time


class Colors:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_step(step_num, title, message=""):
    """Print formatted step information"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}Шаг {step_num}: {title}{Colors.ENDC}")
    if message:
        print(f"{Colors.WHITE}{message}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")


def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.ENDC}")


def run_command(command, description, check_success=True, cwd=None):
    """Run shell command with error handling"""
    print(f"{Colors.WHITE}Выполняется: {description}{Colors.ENDC}")
    print(f"{Colors.CYAN}Команда: {command}{Colors.ENDC}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=cwd
        )
        
        if result.returncode == 0:
            print_success(f"{description} завершено успешно")
            if result.stdout.strip():
                print(f"Вывод: {result.stdout.strip()}")
            return True
        else:
            print_error(f"{description} завершилось с ошибкой (код: {result.returncode})")
            if result.stderr.strip():
                print(f"Ошибка: {result.stderr.strip()}")
            if result.stdout.strip():
                print(f"Вывод: {result.stdout.strip()}")
            if check_success:
                return False
            return True
            
    except Exception as e:
        print_error(f"Исключение при выполнении {description}: {e}")
        return False


def validate_environment():
    """Validate required environment and dependencies"""
    print_step(0, "Проверка окружения")
    
    # Check Python packages
    required_packages = ["torch", "torchvision", "cv2", "numpy", "PIL"]
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"Пакет {package} найден")
        except ImportError:
            print_error(f"Пакет {package} не найден")
            return False
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print_success(f"CUDA доступна: {torch.cuda.get_device_name()}")
        else:
            print_warning("CUDA недоступна, будет использоваться CPU")
    except:
        print_warning("Не удалось проверить CUDA")
    
    # Check required scripts and directories
    required_files = [
        "convert.py",
        "train.py", 
        "render.py",
        "resize_images.py",
        "Tracking-Anything-with-DEVA/demo/demo_with_text.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print_success(f"Файл найден: {file_path}")
        else:
            print_error(f"Файл не найден: {file_path}")
            return False
    
    return True


def setup_dataset_structure(dataset_path):
    """Setup proper dataset folder structure"""
    print_step(1, "Настройка структуры датасета", f"Путь: {dataset_path}")
    
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    input_dir = dataset_path / "input"
    
    # Check if images folder exists
    if not images_dir.exists():
        print_error(f"Папка с изображениями не найдена: {images_dir}")
        return False
    
    # Create input symlink for COLMAP (COLMAP expects 'input' folder)
    if input_dir.exists():
        if input_dir.is_symlink():
            input_dir.unlink()
        elif input_dir.is_dir():
            shutil.rmtree(input_dir)
    
    input_dir.symlink_to("images", target_is_directory=True)
    print_success(f"Создана символическая ссылка: input -> images")
    
    # Create other required directories
    required_dirs = ["distorted", "sparse", "output_sam"]
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        dir_path.mkdir(exist_ok=True)
        print_success(f"Создана директория: {dir_name}")
    
    return True


def resize_images(dataset_path, max_size):
    """Resize images if max_size is specified"""
    if max_size is None:
        print_warning("Размер изображений не изменяется (max_size не указан)")
        return True
    
    print_step(2, "Изменение размера изображений", f"Максимальный размер: {max_size}")
    
    command = f"python3 resize_images.py {dataset_path} --max_size={max_size}"
    return run_command(command, "Изменение размера изображений")


def run_colmap(dataset_path):
    """Run COLMAP for structure-from-motion"""
    print_step(3, "Запуск COLMAP", "Извлечение признаков и создание 3D модели")
    
    # Set environment variables for headless operation
    env_commands = [
        "export QT_QPA_PLATFORM=offscreen",
        "export DISPLAY=:99"
    ]
    
    for env_cmd in env_commands:
        run_command(env_cmd, f"Установка переменной окружения: {env_cmd}", check_success=False)
    
    # Run COLMAP conversion
    command = f"xvfb-run -a python3 convert.py -s {dataset_path}"
    success = run_command(command, "Обработка COLMAP")
    
    if not success:
        print_error("COLMAP завершился с ошибкой")
        return False
    
    # Move sparse data to correct location
    source_sparse = Path(dataset_path) / "distorted" / "sparse"
    target_sparse = Path(dataset_path) / "sparse"
    
    if source_sparse.exists():
        if target_sparse.exists():
            shutil.rmtree(target_sparse)
        shutil.move(str(source_sparse), str(target_sparse))
        print_success("Данные COLMAP перемещены в правильную директорию")
    else:
        print_error(f"Данные COLMAP не найдены в {source_sparse}")
        return False
    
    return True


def run_segmentation(dataset_path, prompt):
    """Run DEVA segmentation"""
    print_step(4, "Запуск сегментации", f"Промпт: {prompt}")
    
    # Change to DEVA directory
    deva_dir = "Tracking-Anything-with-DEVA"
    if not os.path.exists(deva_dir):
        print_error(f"Директория DEVA не найдена: {deva_dir}")
        return False
    
    # Prepare paths
    images_path = f"../{dataset_path}/images"
    output_path = f"../{dataset_path}/output_sam"
    
    # Run segmentation
    command = (f"python3 demo/demo_with_text.py "
               f"--chunk_size 4 "
               f"--img_path {images_path} "
               f"--amp "
               f"--temporal_setting semionline "
               f"--size 480 "
               f"--output {output_path} "
               f"--prompt {prompt}")
    
    success = run_command(command, "Сегментация с помощью DEVA", cwd=deva_dir)
    
    if not success:
        print_error("Сегментация завершилась с ошибкой")
        return False
    
    # Move annotation files to object_mask folder
    os.chdir("..")  # Return to main directory
    
    source_annotation = Path(dataset_path) / "output_sam" / "Annotation"
    target_mask = Path(dataset_path) / "object_mask"
    
    if source_annotation.exists():
        if target_mask.exists():
            shutil.rmtree(target_mask)
        shutil.move(str(source_annotation), str(target_mask))
        print_success("Маски перемещены в директорию object_mask")
    else:
        print_error(f"Аннотации не найдены в {source_annotation}")
        return False
    
    return True


def run_training(dataset_path):
    """Run 3D Gaussian Splatting training"""
    print_step(5, "Запуск тренировки 3DGS", "Обучение модели Gaussian Splatting")
    
    output_path = f"output/{Path(dataset_path).name}"
    config_path = "config/gaussian_dataset/train.json"
    
    # Check if config exists
    if not os.path.exists(config_path):
        print_error(f"Конфигурационный файл не найден: {config_path}")
        return False
    
    command = (f"python3 train.py "
               f"-s {dataset_path} "
               f"-r 1 "
               f"-m {output_path} "
               f"--config_file {config_path}")
    
    return run_command(command, "Тренировка 3DGS")


def run_rendering(dataset_path):
    """Run rendering with trained model"""
    print_step(6, "Запуск рендеринга", "Генерация результатов")
    
    output_path = f"output/{Path(dataset_path).name}"
    
    command = f"python3 render.py -m {output_path} --num_classes 2"
    return run_command(command, "Рендеринг результатов")


def cleanup_intermediate_files(dataset_path):
    """Clean up intermediate files to save space"""
    print_step(7, "Очистка временных файлов")
    
    dataset_path = Path(dataset_path)
    cleanup_dirs = ["distorted", "output_sam", "input"]
    
    for dir_name in cleanup_dirs:
        dir_path = dataset_path / dir_name
        if dir_path.exists():
            if dir_path.is_symlink():
                dir_path.unlink()
            else:
                shutil.rmtree(dir_path)
            print_success(f"Удалена временная директория: {dir_name}")


def generate_summary_report(dataset_path, start_time):
    """Generate summary report"""
    end_time = time.time()
    duration = end_time - start_time
    
    print_step(8, "Отчет о выполнении")
    
    dataset_path = Path(dataset_path)
    output_path = Path("output") / dataset_path.name
    
    print(f"{Colors.WHITE}Датасет: {Colors.CYAN}{dataset_path}{Colors.ENDC}")
    print(f"{Colors.WHITE}Время выполнения: {Colors.CYAN}{duration:.2f} секунд{Colors.ENDC}")
    print(f"{Colors.WHITE}Результаты сохранены в: {Colors.CYAN}{output_path}{Colors.ENDC}")
    
    # Check output files
    if output_path.exists():
        train_dir = output_path / "train"
        test_dir = output_path / "test"
        
        if train_dir.exists():
            train_files = list(train_dir.glob("*.png"))
            print_success(f"Создано {len(train_files)} изображений для тренировочного набора")
        
        if test_dir.exists():
            test_files = list(test_dir.glob("*.png"))
            print_success(f"Создано {len(test_files)} изображений для тестового набора")
    
    print(f"\n{Colors.GREEN}{'='*60}")
    print(f"🎉 ПАЙПЛАЙН ВЫПОЛНЕН УСПЕШНО! 🎉")
    print(f"{'='*60}{Colors.ENDC}")


def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(
        description="Автоматизированный пайплайн для 3D Gaussian Splatting с сегментацией",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python3 automated_3dgs_pipeline.py --dataset data/dataset512 --prompt central_object
  python3 automated_3dgs_pipeline.py --dataset data/my_scene --max_size 1920 --prompt "red car"
  python3 automated_3dgs_pipeline.py --dataset data/room --max_size 1080 --prompt "sofa" --no_cleanup
        """
    )
    
    parser.add_argument(
        "--dataset", 
        required=True, 
        help="Путь к датасету (должен содержать папку 'images')"
    )
    
    parser.add_argument(
        "--max_size", 
        type=int, 
        help="Максимальный размер изображений для ресайза (например, 1980)"
    )
    
    parser.add_argument(
        "--prompt", 
        default="central_object", 
        help="Промпт для сегментации объекта (по умолчанию: central_object)"
    )
    
    parser.add_argument(
        "--no_cleanup", 
        action="store_true", 
        help="Не удалять временные файлы после завершения"
    )
    
    parser.add_argument(
        "--skip_colmap", 
        action="store_true", 
        help="Пропустить этап COLMAP (если уже выполнен)"
    )
    
    parser.add_argument(
        "--skip_segmentation", 
        action="store_true", 
        help="Пропустить этап сегментации (если уже выполнен)"
    )
    
    args = parser.parse_args()
    
    # Start timing
    start_time = time.time()
    
    print(f"{Colors.MAGENTA}{'='*60}")
    print(f"🚀 ЗАПУСК АВТОМАТИЗИРОВАННОГО ПАЙПЛАЙНА 3DGS 🚀")
    print(f"{'='*60}{Colors.ENDC}")
    
    # Validate environment
    if not validate_environment():
        print_error("Проверка окружения не пройдена")
        sys.exit(1)
    
    # Setup dataset structure
    if not setup_dataset_structure(args.dataset):
        print_error("Ошибка настройки структуры датасета")
        sys.exit(1)
    
    # Resize images if needed
    if not resize_images(args.dataset, args.max_size):
        print_error("Ошибка изменения размера изображений")
        sys.exit(1)
    
    # Run COLMAP
    if not args.skip_colmap:
        if not run_colmap(args.dataset):
            print_error("Ошибка выполнения COLMAP")
            sys.exit(1)
    else:
        print_warning("COLMAP пропущен")
    
    # Run segmentation
    if not args.skip_segmentation:
        if not run_segmentation(args.dataset, args.prompt):
            print_error("Ошибка выполнения сегментации")
            sys.exit(1)
    else:
        print_warning("Сегментация пропущена")
    
    # Run training
    if not run_training(args.dataset):
        print_error("Ошибка выполнения тренировки")
        sys.exit(1)
    
    # Run rendering
    if not run_rendering(args.dataset):
        print_error("Ошибка выполнения рендеринга")
        sys.exit(1)
    
    # Cleanup
    if not args.no_cleanup:
        cleanup_intermediate_files(args.dataset)
    else:
        print_warning("Очистка временных файлов пропущена")
    
    # Generate summary
    generate_summary_report(args.dataset, start_time)


if __name__ == "__main__":
    main()