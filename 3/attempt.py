import librosa
import numpy as np

maj_ctr=0
min_ctr=0

def detect_key_and_scale(file_path):
    # 1. Загружаем аудиофайл
    y, sr = librosa.load(file_path)  # Достаточно первых 30 секунд

    # 2. Извлекаем признаки (хромограмму)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

    # 3. Вычисляем среднее значение для каждого из 12 полутонов
    mean_chroma = np.mean(chromagram, axis=1)

    # 4. Профили мажорных и минорных аккордов по методу Krumhansl-Schmuckler
    maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    # 5. Находим корреляцию между нашим треком и профилями
    maj_corrs = [np.corrcoef(mean_chroma, np.roll(maj_profile, i))[0, 1] for i in range(12)]
    min_corrs = [np.corrcoef(mean_chroma, np.roll(min_profile, i))[0, 1] for i in range(12)]

    # 6. Определяем индекс наилучшего совпадения
    max_maj_idx = np.argmax(maj_corrs)
    max_min_idx = np.argmax(min_corrs)

    # 7. Ноты
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # 8. Сравниваем силу корреляции для мажора и минора
    if maj_corrs[max_maj_idx] >= min_corrs[max_min_idx]:
        return f"{os.path.basename(file_path)}: {notes[max_maj_idx]} мажор (Major)"
    else:
        return f"{os.path.basename(file_path)}: {notes[max_min_idx]} минор (Minor)"


# Пример использования:
# result = detect_key_and_scale("albums/05 - 5 Лет (2008)/03. Пьян.mp3")
# print(result)

import os

folder = 'albums'
info={}

for foldername in os.listdir(folder):
    folderpath = os.path.join(folder, foldername)
    songs_info=[]
    for filename in os.listdir(folderpath):
        file_path = os.path.join(folderpath, filename)
        songs_info.append(detect_key_and_scale(file_path))

    info[foldername] = songs_info

for foldername in info.keys():
    print(foldername)
    print('\n'.join(info[foldername]))

print(info)