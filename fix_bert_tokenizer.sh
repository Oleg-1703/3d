#!/bin/bash

# Скрипт исправления get_tokenlizer.py для работы с локальной bert-base-uncased

echo "=== Исправление GroundingDINO tokenizer для локальной модели ==="

TOKENIZER_FILE="/root/autodl-tmp/3d_v2/Tracking-Anything-with-DEVA/Grounded-Segment-Anything/GroundingDINO/groundingdino/util/get_tokenlizer.py"
BERT_PATH="/root/autodl-tmp/3d_v2/bert-base-uncased"

# Проверка существования файла tokenizer
if [ ! -f "$TOKENIZER_FILE" ]; then
    echo "Ошибка: файл $TOKENIZER_FILE не найден"
    exit 1
fi

# Проверка существования локальной модели bert
echo "Проверка локальной модели bert-base-uncased..."
if [ ! -d "$BERT_PATH" ]; then
    echo "Ошибка: папка с bert-base-uncased не найдена в $BERT_PATH"
    echo "Убедитесь, что модель скачана и находится в правильном месте"
    exit 1
fi

# Проверка необходимых файлов в модели
REQUIRED_FILES=("config.json" "pytorch_model.bin" "tokenizer.json" "vocab.txt")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$BERT_PATH/$file" ]; then
        echo "Предупреждение: файл $file не найден в $BERT_PATH"
    else
        echo "✓ Найден: $file"
    fi
done

# Создание резервной копии
echo "Создание резервной копии..."
cp "$TOKENIZER_FILE" "${TOKENIZER_FILE}.backup"

# Создание исправленного файла
echo "Создание исправленного get_tokenlizer.py..."
cat > "$TOKENIZER_FILE" << 'EOF'
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for dealing with the GroundingDINO tokenizer.
"""
from transformers import AutoTokenizer, BertModel, BertTokenizer


# Абсолютный путь к локальной модели bert-base-uncased
LOCAL_BERT_PATH = "/root/autodl-tmp/3d_v2/bert-base-uncased"


def get_tokenlizer(text_encoder_type):
    if text_encoder_type == "bert-base-uncased":
        # Используем локальную модель
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
    else:
        # Для других типов токенизаторов (если они есть)
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_type, local_files_only=True)
    return tokenizer


def get_pretrained_language_model(text_encoder_type):
    if text_encoder_type == "bert-base-uncased":
        # Используем локальную модель
        return BertModel.from_pretrained(LOCAL_BERT_PATH, local_files_only=True)
    else:
        # Для других типов моделей (если они есть)
        return BertModel.from_pretrained(text_encoder_type, local_files_only=True)
EOF

echo "✓ Файл исправлен"

# Установка переменных окружения для offline режима
echo "Настройка переменных окружения..."
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Добавление в bashrc для постоянного использования
echo "export TRANSFORMERS_OFFLINE=1" >> ~/.bashrc
echo "export HF_DATASETS_OFFLINE=1" >> ~/.bashrc
echo "export HF_HUB_OFFLINE=1" >> ~/.bashrc

echo "✓ Переменные окружения настроены"

# Тестирование исправления
echo ""
echo "=== Тестирование исправления ==="

cd /root/autodl-tmp/3d_v2/Tracking-Anything-with-DEVA

# Тест 1: Проверка импорта tokenizer
echo "Тест 1: Проверка импорта tokenizer..."
python3 -c "
import sys
sys.path.append('/root/autodl-tmp/3d_v2/Tracking-Anything-with-DEVA/Grounded-Segment-Anything/GroundingDINO')
try:
    from groundingdino.util.get_tokenlizer import get_tokenlizer, get_pretrained_language_model
    print('✓ Импорт успешный')
    
    # Тест создания токенизатора
    tokenizer = get_tokenlizer('bert-base-uncased')
    print('✓ Токенизатор создан')
    
    # Тест создания модели
    model = get_pretrained_language_model('bert-base-uncased')
    print('✓ Модель BERT загружена')
    
    print('Все тесты прошли успешно!')
    
except Exception as e:
    print(f'✗ Ошибка: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ Токенизатор исправлен успешно!"
    echo ""
    echo "Теперь можно запускать текстовую сегментацию:"
    echo "python3 demo/demo_with_text.py --chunk_size 4 --img_path ../data/dataset/images --amp --temporal_setting semionline --size 480 --output ./output_text_seg/dataset --prompt \"vase\""
else
    echo "❌ Исправление не удалось. Восстанавливаем резервную копию..."
    cp "${TOKENIZER_FILE}.backup" "$TOKENIZER_FILE"
    echo "Резервная копия восстановлена"
    exit 1
fi