from twp import twp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from transformers import pipeline

vectorizer = TfidfVectorizer(
    max_features=1700,
    min_df=4,
    max_df=0.6,
    sublinear_tf=True
)

twp_strings = [" ".join(text) for text in twp]

# Теперь на вход идет список строк
X_tfidf = vectorizer.fit_transform(twp_strings)

print("Размер матрицы TF-IDF:", X_tfidf.shape)

feature_names = np.array(vectorizer.get_feature_names_out())
mean_tfidf = np.asarray(X_tfidf.mean(axis=0)).ravel()

top_idx = mean_tfidf.argsort()[::-1][:40]

top_terms = pd.DataFrame({
    "term": feature_names[top_idx],
    "mean_tfidf": mean_tfidf[top_idx]
})

print(top_terms)

sentiment_model = pipeline(
    task="sentiment-analysis",
    model="cointegrated/rubert-tiny-sentiment-balanced",
    tokenizer="cointegrated/rubert-tiny-sentiment-balanced"
)

def predict_sentiment(text, max_chars=500):
    text = text[:max_chars]   # oбрезаем текст до 500 символов
    result = sentiment_model(text)[0]
    return result["label"], result["score"]

print("Запускаем анализ тональности для 230 треков...")
results = [predict_sentiment(text) for text in twp_strings]

# Собираем красивый датафрейм
res = pd.DataFrame({
    "text": twp_strings,
    "sentiment": [r[0] for r in results],
    "sentiment_score": [r[1] for r in results]
})

print("\nРезультаты анализа тональности:")
print(res.head(20))

print(res['sentiment'].value_counts())

# import numpy as np
# import pandas as pd
# from transformers import pipeline
# from cleaning import clean_data  # Твои данные
#
# # Собираем токены обратно в строки для контекстного анализа
# twp_strings = [song['lyrics'].replace('\n', ' ') for song in clean_data]
#
# # ============================================
# # 1. Загрузка точной модели от s-nlp
# # ============================================
# print("Загружаем модель s-nlp/rubert-base-corruption-detector...")
#
# corruption_pipeline = pipeline(
#     task="text-classification",
#     model="s-nlp/rubert-base-corruption-detector",
#     tokenizer="s-nlp/rubert-base-corruption-detector",
#     top_k=None  # Забираем вероятности абсолютно всех внутренних классов
# )
#
# # ============================================
# # 2. Обработка корпуса с динамической сборкой признаков
# # ============================================
# print(f"Запускаем анализ деконструкции смыслов для {len(twp_strings)} треков...")
#
# results = []
# for i, text in enumerate(twp_strings):
#     text_cut = text[:500]  # Ограничение по длине для Берта
#     outputs = corruption_pipeline(text_cut)[0]
#
#     # Создаем базовую строчку для таблицы
#     row = {
#         "track_id": i + 1,
#         "text_preview": text[:100] + "...",  # Короткое превью для консоли
#         "full_text_analyzed": text_cut  # Полный кусок, который ушел в нейросеть
#     }
#
#     # Динамически вытаскиваем ВСЕ лейблы, которые зашила команда s-nlp,
#     # и создаем под них персональные столбцы в таблице
#     for pred in outputs:
#         row[pred['label']] = pred['score']
#
#     # Дополнительно определяем самый сильный класс для этой песни
#     dominant = max(outputs, key=lambda x: x['score'])
#     row["dominant_class"] = dominant['label']
#     row["confidence"] = dominant['score']
#
#     results.append(row)
#
#     if (i + 1) % 20 == 0:
#         print(f"Успешно обработано: {i + 1}/{len(twp_strings)} треков")
#
# # Переводим всё собранное добро в структуру DataFrame
# df_results = pd.DataFrame(results)
#
# # ============================================
# # 3. Вывод превью в консоль PyCharm
# # ============================================
# print("\n=== ПЕРВЫЕ 10 СТРОК СФОРМИРОВАННОЙ ТАБЛИЦЫ ===")
# # Дропаем длинный текст при выводе в консоль, чтобы не захламлять экран
# print(df_results.drop(columns=["full_text_analyzed"]).head(10))
#
# print("\n=== ИТОГОВОЕ РАСПРЕДЕЛЕНИЕ СТИЛЕЙ ===")
# print(df_results["dominant_class"].value_counts())
#
# # ============================================
# # 4. ЭКСПОРТ ВСЕХ РЕЗУЛЬТАТОВ В ФАЙЛ
# # ============================================
# output_filename = "kunteynir_corruption_analysis.csv"
#
# # encoding="utf-8-sig" — это магия, благодаря которой русский язык в Excel
# # откроется сразу красивыми буквами, а не странными иероглифами
# df_results.to_csv(output_filename, index=False, encoding="utf-8-sig")
#
# print("\n" + "=" * 50)
# print(f"[УСПЕХ] Абсолютно все данные успешно посчитаны и экспортированы!")
# print(f"Ищи файл здесь: {os.path.abspath(output_filename)}")
# print("=" * 50)