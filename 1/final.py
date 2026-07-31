f=open('results.txt', 'r', encoding='utf-8')

all_data=[]

for line in f:
    all_data.append(line[:-1])
f.close()

options_positive=['positive-positive', 'neutral-positive', 'positive-neutral', 'neutral-neutral']
options_negative=['negative-negative', 'neutral-negative', 'negative-neutral', 'neutral-neutral']
options_controversial=['positive-negative', 'negative-positive']

statistics={'match positive':[], 'match negative':[], 'strange_minor':[], 'strange_major':[], 'strange_major_neg':[], 'strange_minor_pos':[]}
for song in all_data:
    song=song.split(' - ')
    if song[2]=='Major':
        if song[1] in options_positive:
            statistics['match positive'].append(song[0])
        elif song[1] in options_negative:
            statistics['strange_major_neg'].append(song[0])
        else:
            statistics['strange_major'].append(song[0])
    elif song[2]=='Minor':
        if song[1] in options_negative:
            statistics['match negative'].append(song[0])
        elif song[1] in options_positive:
            statistics['strange_minor_pos'].append(song[0])
        else:
            statistics['strange_minor'].append(song[0])

for result in statistics:
    print(f'{result}: {statistics[result]}')

import matplotlib.pyplot as plt


# Превращаем списки песен в их количество
sentiment_counts = {key: len(songs) for key, songs in statistics.items()}

# Строим базовый график и сохраняем его в переменную bars
bars = plt.bar(sentiment_counts.keys(), sentiment_counts.values())

# Добавляем подписи цифр над столбцами (работает в любом современном matplotlib)
plt.bar_label(bars, padding=3)

# Стандартные подписи осей
plt.title('Распределение треков по категориям')
plt.xlabel('Категории')
plt.ylabel('Количество треков')

# Легкий наклон подписей, чтобы не слипались
plt.xticks(rotation=30, ha='right')

plt.tight_layout()
plt.show()