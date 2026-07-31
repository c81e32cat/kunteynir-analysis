from gensim.models import Word2Vec
from twp import twp
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from wordcloud import WordCloud

w2v_model = Word2Vec(
    sentences=twp,
    vector_size=50, # может поменять как-то?..
    window=5,
    min_count=3,
    workers=4,
    sg=1,
    epochs=100
)

token_counts = Counter(
    tok
    for tokens in twp
    for tok in tokens
)

candidate_tokens = [
    tok for tok, cnt in token_counts.items()
    if cnt >= 3 and tok in w2v_model.wv
]

X = np.array([w2v_model.wv[tok] for tok in candidate_tokens])

n_clusters = 3 # еще на 4 кластера для частотности 7 было мило - именно через вордклауд, по картинке плохо было

kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=20
)

labels = kmeans.fit_predict(X)


df_clusters = pd.DataFrame({
    "token": candidate_tokens,
    "cluster": labels,
    "freq": [token_counts[t] for t in candidate_tokens]
})

centers = kmeans.cluster_centers_
distances = []

for token, cluster_id in zip(candidate_tokens, labels):
    vec = w2v_model.wv[token]
    center = centers[cluster_id]
    dist = np.linalg.norm(vec - center)
    distances.append(dist)

df_clusters["dist_to_center"] = distances

print(df_clusters.head())

for cluster_id in range(n_clusters): # ПЕРВЫЙ КЛАСТЕР ОЧЕНЬ НЕКУЛЬТУРНЫЙ!!!!!! остальные тоже но меньше.
    print("\n" + "=" * 90)
    print(f"КЛАСТЕР {cluster_id}")
    print("=" * 90)

    cluster_df = df_clusters[df_clusters["cluster"] == cluster_id].copy()

    top_freq = cluster_df.sort_values("freq", ascending=False).head(20)
    top_center = cluster_df.sort_values("dist_to_center", ascending=True).head(20)

    print("\nСамые частотные токены:")
    print(", ".join(top_freq["token"].tolist()))

    print("\nСамые типичные токены (ближе всего к центру):")
    print(", ".join(top_center["token"].tolist()))

#
# pca = PCA(n_components=2, random_state=42)
# X_2d = pca.fit_transform(X)
#
# df_clusters["x"] = X_2d[:, 0]
# df_clusters["y"] = X_2d[:, 1]
#
# plt.figure(figsize=(12, 8))
#
# for cluster_id in range(n_clusters):
#     cluster_df = df_clusters[df_clusters["cluster"] == cluster_id]
#     plt.scatter(
#         cluster_df["x"],
#         cluster_df["y"],
#         label=f"Cluster {cluster_id}",
#         alpha=0.7
#     )
#
#
# # top_for_labels = df_clusters.sort_values("freq", ascending=False).head(80)
#
# # for _, row in top_for_labels.iterrows():
# #     plt.text(row["x"], row["y"], row["token"], fontsize=8)
#
# plt.title("Кластеры токенов в пространстве Word2Vec (PCA)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()
#
# fig, axes = plt.subplots(2, 3, figsize=(18, 10))
# axes = axes.flatten()
#
# for cluster_id in range(n_clusters):
#     cluster_df = df_clusters[df_clusters["cluster"] == cluster_id]
#
#     freqs = {
#         row["token"].replace("_", " "): row["freq"]
#         for _, row in cluster_df.iterrows()
#     }
#
#     if not freqs:
#         axes[cluster_id].axis("off")
#         continue
#
#     wc = WordCloud(
#         width=800,
#         height=400,
#         background_color="white",
#         collocations=False
#     ).generate_from_frequencies(freqs)
#
#     axes[cluster_id].imshow(wc, interpolation="bilinear")
#     axes[cluster_id].set_title(f"Cluster {cluster_id}")
#     axes[cluster_id].axis("off")
#
# plt.tight_layout()
# plt.show()
# я пока это убрала а то первый (нулевой) кластер все еще неприличный....
token_to_cluster = dict(zip(df_clusters["token"], df_clusters["cluster"]))

songs_results = []

doms=[]

for i, song_tokens in enumerate(twp):
    # Собираем кластеры только для тех слов песни, которые выжили после фильтрации и есть в модели
    song_word_clusters = [token_to_cluster[tok] for tok in song_tokens if tok in token_to_cluster]

    if song_word_clusters:
        # Считаем, сколько раз каждый кластер встретился в треке
        cluster_counts = Counter(song_word_clusters)
        total_valid_words = len(song_word_clusters)

        # Определяем доминирующий кластер (мажоритарный выбор)
        dominant_cluster = cluster_counts.most_common(1)[0][0]

        doms.append(dominant_cluster)

        # Считаем долю каждого кластера в песне (в процентах)
        share_c0 = round((cluster_counts[0] / total_valid_words) * 100, 1)
        share_c1 = round((cluster_counts[1] / total_valid_words) * 100, 1)
        share_c2 = round((cluster_counts[2] / total_valid_words) * 100, 1)
    else:
        # Если в песню не попало ни одно ключевое слово
        dominant_cluster = -1
        share_c0 = share_c1 = share_c2 = 0.0

    songs_results.append({
        "song_id": i + 1,
        "dominant_cluster": dominant_cluster,
        "cluster_0_%": share_c0,
        "cluster_1_%": share_c1,
        "cluster_2_%": share_c2,
        "text_preview": " ".join(song_tokens[:10]) + "..."
    })

# Создаем датафрейм с результатами по трекам
df_songs = pd.DataFrame(songs_results)

print("\n" + "=" * 90)
print("РАСПРЕДЕЛЕНИЕ КЛАСТЕРОВ ПО ПЕСНЯМ (ПЕРВЫЕ 10 ТРЕКОВ):")
print("=" * 90)
print(df_songs.head(10).to_string(index=False))

print("\n" + "=" * 90)
print("ОБЩАЯ СТАТИСТИКА ДИСКОГРАФИИ:")
print("=" * 90)
print(df_songs["dominant_cluster"].value_counts().to_string())

# Сохраняем итоговую разметку треков в CSV
df_songs.to_csv("kunteynir_w2v_songs.csv", index=False, encoding="utf-8-sig")
print("\n[УСПЕХ] Разметка песен сохранена в файл kunteynir_w2v_songs.csv")

print(doms)