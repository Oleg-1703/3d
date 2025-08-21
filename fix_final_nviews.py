#!/usr/bin/env python3

# Исправляем логику n_views в scene/dataset_readers.py
with open('scene/dataset_readers.py', 'r') as f:
    content = f.read()

# Создаем backup
with open('scene/dataset_readers.py.nviews_final.backup', 'w') as f:
    f.write(content)

# Заменяем проблемную логику
old_logic = '''            elif isinstance(n_views,int):
                idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views) # 3views
                idx_sub = [round(i) for i in idx_sub]
                train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
                print(train_cam_infos)'''

new_logic = '''            elif isinstance(n_views,int):
                print(f"🔍 n_views={n_views}, доступно камер: {len(train_cam_infos)}")
                if n_views >= len(train_cam_infos):
                    print("✅ n_views >= количества камер, используем все")
                    # Используем все камеры
                    pass
                else:
                    print(f"📊 Выбираем {n_views} камер из {len(train_cam_infos)}")
                    idx_sub = np.linspace(0, len(train_cam_infos)-1, n_views)
                    idx_sub = [round(i) for i in idx_sub]
                    train_cam_infos = [train_cam_infos[i_sub] for i_sub in idx_sub]
                    print(f"   Выбраны индексы: {idx_sub}")'''

content = content.replace(old_logic, new_logic)

# Записываем исправленный файл
with open('scene/dataset_readers.py', 'w') as f:
    f.write(content)

print("✅ Исправлена логика n_views фильтрации")
print("Теперь при n_views >= количества камер будут использоваться все камеры")