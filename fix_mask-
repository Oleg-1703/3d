#!/usr/bin/env python3
"""
Исправление путей к маскам в dataset_readers.py
"""

def fix_mask_paths():
    """Исправляет пути к маскам в readColmapSceneInfo"""
    
    # Читаем файл
    with open('scene/dataset_readers.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Создаем backup
    with open('scene/dataset_readers.py.mask_fix.backup', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Исправляем путь к маскам
    # Заменяем 'objects' на 'object_mask' в строке object_dir
    old_line = "object_dir = 'object_mask' if object_path == None else object_path"
    new_line = "object_dir = 'object_mask' if object_path == None else object_path"
    
    # Также исправляем другую возможную строку
    content = content.replace(
        "object_dir = \"objects\" if object_path == None else object_path",
        "object_dir = \"object_mask\" if object_path == None else object_path"
    )
    
    content = content.replace(
        "object_dir = 'objects' if object_path == None else object_path", 
        "object_dir = 'object_mask' if object_path == None else object_path"
    )
    
    # Записываем исправленный файл
    with open('scene/dataset_readers.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Исправлен путь к маскам: objects -> object_mask")
    return True

def create_objects_symlink():
    """Создает символическую ссылку objects -> object_mask для совместимости"""
    import os
    
    dataset_path = "data/dataset1024"
    object_mask_path = os.path.join(dataset_path, "object_mask")
    objects_path = os.path.join(dataset_path, "objects")
    
    if os.path.exists(object_mask_path) and not os.path.exists(objects_path):
        try:
            os.symlink("object_mask", objects_path)
            print(f"✅ Создана символическая ссылка: {objects_path} -> object_mask")
            return True
        except Exception as e:
            print(f"⚠️ Не удалось создать символическую ссылку: {e}")
            return False
    elif os.path.exists(objects_path):
        print(f"✅ Папка {objects_path} уже существует")
        return True
    else:
        print(f"❌ Папка {object_mask_path} не найдена")
        return False

def check_mask_files():
    """Проверяет наличие файлов масок"""
    import os
    
    dataset_path = "data/dataset1024"
    object_mask_path = os.path.join(dataset_path, "object_mask")
    
    if not os.path.exists(object_mask_path):
        print(f"❌ Папка {object_mask_path} не найдена")
        return False
    
    mask_files = [f for f in os.listdir(object_mask_path) if f.endswith('.png')]
    print(f"📁 Найдено масок в object_mask: {len(mask_files)}")
    
    if len(mask_files) > 0:
        print(f"📄 Примеры масок: {mask_files[:5]}")
        return True
    else:
        print("❌ Маски не найдены")
        return False

def main():
    print("=== ИСПРАВЛЕНИЕ ПУТЕЙ К МАСКАМ ===")
    
    # 1. Проверяем наличие масок
    print("\n1. ПРОВЕРКА МАСОК:")
    has_masks = check_mask_files()
    
    if not has_masks:
        print("❌ Сначала создайте маски с помощью SAM")
        return
    
    # 2. Исправляем код
    print("\n2. ИСПРАВЛЕНИЕ КОДА:")
    fix_mask_paths()
    
    # 3. Создаем символическую ссылку для совместимости
    print("\n3. СОЗДАНИЕ СИМВОЛИЧЕСКОЙ ССЫЛКИ:")
    create_objects_symlink()
    
    print("\n✅ Исправления применены!")
    print("\nТеперь запустите обучение:")
    print("python3 train.py -s data/dataset1024 -r 1 -m output/dataset1024 --config_file config/gaussian_dataset/train.json")

if __name__ == "__main__":
    main()