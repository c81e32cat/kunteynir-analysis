from twp import twp

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud
from sklearn.decomposition import PCA

from gensim import corpora
from gensim.models import LdaModel

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


# ============================================
# 2. Проверка входных данных
# ============================================
# Ожидается, что tokenized_texts уже существует:


print("Количество документов:", len(twp))
print("Пример документа:")
print(twp[0][:30])

structural_noise = ['быть', 'хотеть', 'давать', 'мочь', 'хуй', 'ебать', 'уже', 'жопа', 'пизда']

twp = [
    [lemma for lemma in track if lemma.lower() not in structural_noise]
    for track in twp
]


# ============================================
# 3. Словарь и корпус
# =======g=====================================
# dictionary: отображение token -> id
dictionary = corpora.Dictionary(twp)

# Фильтруем слишком редкие и слишком частые слова
# no_below=5   -> слово должно встретиться минимум в 5 документах
# no_above=0.4 -> удаляем слова, встречающиеся более чем в 40% документов
# keep_n=10000 -> максимум 10k токенов в словаре
dictionary.filter_extremes(no_below=3, no_above=0.55, keep_n=3000)
# dictionary.filter_extremes(no_below=5, no_above=0.2, keep_n=10000)
dictionary.compactify()

print("Размер словаря после фильтрации:", len(dictionary))

# corpus: каждый документ в формате bag-of-words
# [(token_id, count), (token_id, count), ...]
corpus = [dictionary.doc2bow(doc) for doc in twp]

print("Пример документа в BoW-формате:")
print(corpus[0][:10])


# ============================================
# 4. Обучение LDA
# ============================================
# num_topics можно менять: 5 или 6 обычный кейс "навскидку"
num_topics = 3

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    passes=20,
    iterations=200,
    alpha='auto',
    eta='auto'
)

print("\nLDA обучена.\n")


# ============================================
# 5. Печать тем в текстовом виде
# ============================================
# Каждая тема = набор слов с весами
print("Топ-словa по темам:\n")

for topic_id in range(num_topics):
    terms = lda_model.show_topic(topic_id, topn=30)
    terms_str = ", ".join([f"{word} ({weight:.3f})" for word, weight in terms])
    print(f"Тема {topic_id}: {terms_str}")
    print()


# ============================================
# 6. Таблица топ-слов по темам
# ============================================
rows = []
for topic_id in range(num_topics):
    for word, weight in lda_model.show_topic(topic_id, topn=15):
        rows.append({
            "topic": topic_id,
            "word": word,
            "weight": weight
        })

df_topics = pd.DataFrame(rows)
df_topics.head()


# ============================================
# 9. Матрица document-topic
# ============================================
# Для каждого документа получаем распределение по темам
# например [0.1, 0.6, 0.05, 0.2, 0.05, 0.0]

doc_topic_matrix = []

for bow in corpus:
    topic_probs = lda_model.get_document_topics(bow, minimum_probability=0)
    # topic_probs = [(0, p0), (1, p1), ...]
    topic_vector = [prob for _, prob in sorted(topic_probs, key=lambda x: x[0])]
    doc_topic_matrix.append(topic_vector)

doc_topic_matrix = np.array(doc_topic_matrix)

print("Форма матрицы document-topic:", doc_topic_matrix.shape)
print("Пример распределения тем по документу:")
print(doc_topic_matrix[0])
dominant_topic = doc_topic_matrix.argmax(axis=1)

print(dominant_topic)

df_docs = pd.DataFrame({
    "doc_id": np.arange(len(twp)),
    "dominant_topic": dominant_topic
})

for topic_id in range(num_topics):
    df_docs[f"topic_{topic_id}"] = doc_topic_matrix[:, topic_id]

print(df_docs.head())

# ============================================
# 11. Визуализация 3: карта документов по темам
# ============================================
# Снижаем document-topic в 2D через PCA
# pca = PCA(n_components=2, random_state=42)
# docs_2d = pca.fit_transform(doc_topic_matrix)
#
# df_docs["x"] = docs_2d[:, 0]
# df_docs["y"] = docs_2d[:, 1]
#
# plt.figure(figsize=(12, 8))
#
# for topic_id in range(num_topics):
#     subset = df_docs[df_docs["dominant_topic"] == topic_id]
#     plt.scatter(
#         subset["x"],
#         subset["y"],
#         label=f"Тема {topic_id}",
#         alpha=0.15
#     )
#
# plt.title("Документы в пространстве тем LDA")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()
#
# vis = gensimvis.prepare(lda_model, corpus, dictionary)
# html_filename = 'lda_visualization.html'
#
# # Сохраняем тяжелую интерактивную схему в файл
# pyLDAvis.save_html(vis, html_filename)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for topic_id in range(num_topics):
    topic_terms = dict(lda_model.show_topic(topic_id, topn=40))

    wc = WordCloud(
        width=800,
        height=400,
        background_color="white",
        collocations=False
    ).generate_from_frequencies(topic_terms)

    axes[topic_id].imshow(wc, interpolation="bilinear")
    axes[topic_id].set_title(f"Тема {topic_id}")
    axes[topic_id].axis("off")

# если тем меньше 6, лишние оси выключим
for i in range(num_topics, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.show()