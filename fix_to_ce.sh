#!/bin/bash

# Скрипт для перехода с BCE на CrossEntropy loss в train.py
# Для стабильного обучения Gaussian Grouping с бинарными масками

echo "🔧 Переход с BCE на CrossEntropy loss для Gaussian Grouping..."

# Проверяем что файл train.py существует
if [ ! -f "train.py" ]; then
    echo "❌ Ошибка: файл train.py не найден в текущей директории"
    exit 1
fi

# Создаем бэкап
cp train.py train_bce_backup.py
echo "✓ Создан бэкап: train_bce_backup.py"

# Создаем временный Python скрипт для замены
cat > fix_train_crossentropy.py << 'EOF'
#!/usr/bin/env python3
import re

def fix_train_file():
    with open('train.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Заменяем весь блок с BCE на CrossEntropy
    bce_pattern = r'''                    # Правильная бинарная сегментация для 2 классов
                    if dataset\.num_classes == 2:
                        # Для бинарной классификации используем BCEWithLogitsLoss
                        logits_binary = logits\[:, 1:2, :, :\]  # Канал объекта \[1,1,H,W\]
                        target_binary = \(target > 0\)\.float\(\)\.unsqueeze\(1\)  # \[1,1,H,W\]
                        .*?pos_weight = .*?
                        bce_loss = torch\.nn\.BCEWithLogitsLoss\(reduction='mean', pos_weight=pos_weight\)
                        loss_obj = bce_loss\(logits_binary, target_binary\)
                    else:
                        # Для многоклассовой классификации используем CrossEntropyLoss
                        loss_obj = cls_criterion\(logits, target\)\.mean\(\)
                    
                    # Нормализация object loss
                    if dataset\.num_classes > 1:
                        loss_obj = loss_obj / torch\.log\(torch\.tensor\(float\(dataset\.num_classes\), device=device\)\)'''

    crossentropy_replacement = '''                    # Оригинальный Gaussian Grouping object loss с CrossEntropy
                    # Убеждаемся что target правильный формат для CrossEntropy (0=фон, 1=объект)
                    if target.max() > 1:
                        # Если маски имеют значения >1, приводим к бинарному
                        target = (target > 0).long()
                    
                    # CrossEntropy ожидает target как [batch, H, W] с классами 0,1
                    loss_obj = cls_criterion(logits, target)  # [1, H, W] 
                    loss_obj = loss_obj.mean()  # Скаляр
                    
                    # Нормализация для 2 классов как в оригинальном Gaussian Grouping
                    loss_obj = loss_obj / torch.log(torch.tensor(2.0, dtype=torch.float32, device=device))
                    
                    print(f"CrossEntropy loss (2 classes): {loss_obj.item():.4f}")'''
    
    # Заменяем с учетом многострочности и пробелов
    content = re.sub(bce_pattern, crossentropy_replacement, content, flags=re.DOTALL | re.MULTILINE)
    
    # 2. Убираем все debug принты BCE
    content = re.sub(r'\s*print\(f"  logits_binary shape:.*?\)\n', '', content)
    content = re.sub(r'\s*print\(f"  target_binary shape:.*?\)\n', '', content)
    content = re.sub(r'\s*print\(f"  BCEWithLogitsLoss result:.*?\)\n', '', content)
    
    # 3. Упрощаем входные данные для классификатора (убираем нормализацию)
    content = content.replace(
        'objects_input = objects[0:1].unsqueeze(0)  # [1, 1, H, W]\n'
        '                    # Нормализация входных данных\n'
        '                    objects_input = (objects_input - objects_input.mean()) / (objects_input.std() + 1e-8)',
        'objects_input = objects[0:1].unsqueeze(0)  # [1, 1, H, W]'
    )
    
    # 4. Возвращаем стандартный learning rate
    content = content.replace(
        'cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-5)',
        'cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)'
    )
    
    # 5. Убираем инициализацию весов классификатора (если есть)
    content = re.sub(r'# Правильная инициализация весов.*?\n.*?torch\.nn\.init\.constant_\(classifier\.bias, 0\.0\)\n', '', content, flags=re.DOTALL)
    
    # Записываем исправленный файл
    with open('train.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✓ Заменен BCE loss на CrossEntropy loss")
    print("✓ Удалены debug принты BCE")
    print("✓ Упрощены входные данные классификатора")
    print("✓ Восстановлен стандартный learning rate")

if __name__ == "__main__":
    fix_train_file()
EOF

# Запускаем Python скрипт для исправления
python3 fix_train_crossentropy.py

# Удаляем временный скрипт
rm fix_train_crossentropy.py

echo ""
echo "🎯 ИСПРАВЛЕНИЯ ПРИМЕНЕНЫ:"
echo "✓ BCE loss заменен на CrossEntropy loss"
echo "✓ Убраны все debug принты BCE"
echo "✓ Упрощена обработка входных данных"
echo "✓ Восстановлен стандартный learning rate (5e-4)"
echo ""
echo "📊 ОЖИДАЕМЫЕ РЕЗУЛЬТАТЫ:"
echo "• Object loss должен быть в диапазоне 0.1-0.7"
echo "• Более стабильное обучение" 
echo "• Совместимость с оригинальной архитектурой Gaussian Grouping"
echo ""
echo "🚀 Теперь можете запускать обучение:"
echo "python3 train.py -s data/dataset -r 1 -m output/dataset --config_file config/gaussian_dataset/train.json"
echo ""
echo "📁 Создан бэкап BCE версии: train_bce_backup.py"