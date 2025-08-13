#!/bin/bash

# Скрипт проверки корректности локальной модели bert-base-uncased

BERT_PATH="/root/autodl-tmp/3d_v2/bert-base-uncased"

echo "=== Проверка модели bert-base-uncased ==="
echo "Путь: $BERT_PATH"
echo ""

# Проверка существования директории
if [ ! -d "$BERT_PATH" ]; then
    echo "❌ Папка с моделью не найдена: $BERT_PATH"
    echo ""
    echo "Найденные папки с bert в корне:"
    find /root/autodl-tmp/3d_v2 -name "*bert*" -type d 2>/dev/null
    echo ""
    echo "Возможные решения:"
    echo "1. Переместите папку с моделью в $BERT_PATH"
    echo "2. Создайте символическую ссылку: ln -s /путь/к/вашей/bert-модели $BERT_PATH"
    exit 1
fi

echo "✓ Папка модели найдена"

# Проверка структуры модели
echo ""
echo "Структура модели:"
ls -la "$BERT_PATH"

echo ""
echo "Проверка обязательных файлов:"

# Список обязательных файлов для BERT
REQUIRED_FILES=(
    "config.json"
    "pytorch_model.bin"
    "tokenizer.json" 
    "vocab.txt"
    "tokenizer_config.json"
)

OPTIONAL_FILES=(
    "special_tokens_map.json"
    "pytorch_model.bin.index.json"
    "model.safetensors"
)

missing_files=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$BERT_PATH/$file" ]; then
        echo "✓ $file"
    else
        echo "❌ $file (обязательный)"
        missing_files=$((missing_files + 1))
    fi
done

echo ""
echo "Дополнительные файлы:"
for file in "${OPTIONAL_FILES[@]}"; do
    if [ -f "$BERT_PATH/$file" ]; then
        echo "✓ $file"
    else
        echo "- $file (отсутствует)"
    fi
done

if [ $missing_files -gt 0 ]; then
    echo ""
    echo "❌ Отсутствует $missing_files обязательных файлов"
    echo "Модель может работать некорректно"
    echo ""
    echo "Попробуйте скачать полную модель с всеми файлами"
    exit 1
fi

# Проверка размеров файлов
echo ""
echo "Размеры файлов:"
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$BERT_PATH/$file" ]; then
        size=$(stat -c%s "$BERT_PATH/$file" 2>/dev/null || echo "неизвестно")
        echo "$file: $size байт"
    fi
done

# Проверка содержимого config.json
echo ""
echo "Проверка config.json:"
if [ -f "$BERT_PATH/config.json" ]; then
    python3 -c "
import json
try:
    with open('$BERT_PATH/config.json', 'r') as f:
        config = json.load(f)
    
    print('✓ config.json валиден')
    print(f'  Модель: {config.get(\"model_type\", \"неизвестно\")}')
    print(f'  Архитектура: {config.get(\"architectures\", [\"неизвестно\"])[0]}')
    print(f'  Vocab size: {config.get(\"vocab_size\", \"неизвестно\")}')
    
except Exception as e:
    print(f'❌ Ошибка чтения config.json: {e}')
    exit(1)
"
else
    echo "❌ config.json не найден"
    exit 1
fi

# Тестирование загрузки модели
echo ""
echo "=== Тестирование загрузки модели ==="

python3 -c "
import torch
from transformers import AutoTokenizer, BertModel

bert_path = '$BERT_PATH'

try:
    print('Загрузка токенизатора...')
    tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
    print('✓ Токенизатор загружен')
    
    print('Загрузка модели...')
    model = BertModel.from_pretrained(bert_path, local_files_only=True)
    print('✓ Модель загружена')
    
    # Тест работы
    print('Тестирование работы...')
    test_text = 'Hello world'
    inputs = tokenizer(test_text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    print('✓ Модель работает корректно')
    print(f'  Входной текст: \"{test_text}\"')
    print(f'  Размер выхода: {outputs.last_hidden_state.shape}')
    
    print('')
    print('🎉 Модель bert-base-uncased готова к использованию!')
    
except Exception as e:
    print(f'❌ Ошибка: {e}')
    print('')
    print('Возможные проблемы:')
    print('1. Неполная или поврежденная модель')
    print('2. Несовместимая версия transformers')
    print('3. Недостаточно памяти')
    exit(1)
"

echo ""
echo "=== Проверка завершена ==="