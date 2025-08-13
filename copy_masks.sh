#!/bin/bash

# Скрипт поиска и копирования масок сегментации

DATASET_NAME="dataset"
PROJECT_ROOT="/root/autodl-tmp/3d_v2"

echo "=== Поиск и копирование масок сегментации ==="

cd "$PROJECT_ROOT"

DATASET_PATH="data/$DATASET_NAME"
TARGET_MASK_DIR="$DATASET_PATH/object_mask"

# Создание целевой папки если не существует
mkdir -p "$TARGET_MASK_DIR"

# Поиск масок в различных возможных местах
SEARCH_PATHS=(
    "Tracking-Anything-with-DEVA/output_text_seg/$DATASET_NAME/Annotations"
    "Tracking-Anything-with-DEVA/example/output_gaussian_dataset/$DATASET_NAME/Annotations" 
    "data/$DATASET_NAME/object_mask"
    "$DATASET_PATH/Annotations"
)

echo "Поиск масок в следующих местах:"
for path in "${SEARCH_PATHS[@]}"; do
    echo "- $path"
done

echo ""

found_masks=false
masks_found_in=""

for search_path in "${SEARCH_PATHS[@]}"; do
    if [ -d "$search_path" ]; then
        mask_count=$(ls "$search_path"/*.png 2>/dev/null | wc -l)
        if [ $mask_count -gt 0 ]; then
            echo "✓ Найдено $mask_count масок в: $search_path"
            
            if [ "$search_path" != "$TARGET_MASK_DIR" ]; then
                echo "  Копирование в $TARGET_MASK_DIR..."
                cp "$search_path"/*.png "$TARGET_MASK_DIR/"
                if [ $? -eq 0 ]; then
                    echo "  ✓ Маски скопированы успешно"
                    found_masks=true
                    masks_found_in="$search_path"
                else
                    echo "  ❌ Ошибка копирования"
                fi
            else
                echo "  ✓ Маски уже в целевой папке"
                found_masks=true
                masks_found_in="$search_path"
            fi
            break
        else
            echo "- Пусто: $search_path"
        fi
    else
        echo "- Не существует: $search_path"
    fi
done

echo ""

if [ "$found_masks" = true ]; then
    final_count=$(ls "$TARGET_MASK_DIR"/*.png 2>/dev/null | wc -l)
    echo "🎉 Успешно! Найдено $final_count масок в $TARGET_MASK_DIR"
    
    # Показать несколько примеров
    echo ""
    echo "Примеры файлов масок:"
    ls "$TARGET_MASK_DIR"/*.png 2>/dev/null | head -5
    
    # Проверка соответствия с изображениями
    if [ -d "$DATASET_PATH/images" ]; then
        image_count=$(ls "$DATASET_PATH/images"/*.jpg "$DATASET_PATH/images"/*.png "$DATASET_PATH/images"/*.JPG "$DATASET_PATH/images"/*.PNG 2>/dev/null | wc -l)
        echo ""
        echo "Проверка соответствия:"
        echo "- Изображения: $image_count"
        echo "- Маски: $final_count"
        
        if [ $image_count -eq $final_count ]; then
            echo "✓ Количество совпадает"
        else
            echo "⚠️  Количество не совпадает"
            
            # Показать несоответствия
            echo ""
            echo "Детальная проверка названий файлов..."
            
            # Создать списки файлов для сравнения
            (cd "$DATASET_PATH/images" && ls *.jpg *.png *.JPG *.PNG 2>/dev/null | sed 's/\.[^.]*$//' | sort) > /tmp/images_list.txt
            (cd "$TARGET_MASK_DIR" && ls *.png 2>/dev/null | sed 's/\.png$//' | sort) > /tmp/masks_list.txt
            
            echo "Изображения без соответствующих масок:"
            comm -23 /tmp/images_list.txt /tmp/masks_list.txt | head -10
            
            echo "Маски без соответствующих изображений:"
            comm -13 /tmp/images_list.txt /tmp/masks_list.txt | head -10
            
            rm /tmp/images_list.txt /tmp/masks_list.txt 2>/dev/null
        fi
    fi
    
else
    echo "❌ Маски сегментации не найдены!"
    echo ""
    echo "Возможные причины:"
    echo "1. Сегментация не была запущена"
    echo "2. Сегментация завершилась с ошибкой"
    echo "3. Маски находятся в неожиданном месте"
    echo ""
    echo "Проверьте вывод сегментации:"
    echo "ls -la Tracking-Anything-with-DEVA/output_text_seg/"
    echo "ls -la Tracking-Anything-with-DEVA/example/output_gaussian_dataset/"
    echo ""
    echo "Для повторного запуска сегментации:"
    echo "cd Tracking-Anything-with-DEVA"
    echo "python3 demo/demo_with_text.py --chunk_size 4 --img_path ../data/$DATASET_NAME/images --amp --temporal_setting semionline --size 480 --output ./output_text_seg/$DATASET_NAME --prompt \"vase\""
    
    exit 1
fi