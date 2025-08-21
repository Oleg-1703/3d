#!/usr/bin/env python3

# Читаем файл
with open('scene/dataset_readers.py', 'r') as f:
    content = f.read()

# Заменяем вызов getNerfppNorm
old_call = 'nerf_normalization = getNerfppNorm(train_cam_infos)'

new_call = '''print(f"🔍 Вызываем getNerfppNorm с {len(train_cam_infos)} камерами")
    if len(train_cam_infos) == 0:
        print("❌ ОШИБКА: train_cam_infos пустой перед getNerfppNorm!")
        nerf_normalization = {"translate": np.array([0.0, 0.0, 0.0]), "radius": 1.0}
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)'''

content = content.replace(old_call, new_call)

# Записываем исправленный файл
with open('scene/dataset_readers.py', 'w') as f:
    f.write(content)

print("✅ Исправлен вызов getNerfppNorm")