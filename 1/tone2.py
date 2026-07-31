from twp import twp
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from transformers import pipeline
from collections import Counter

vectorizer = TfidfVectorizer(
    max_features=1700,
    min_df=4,
    max_df=0.6,
    sublinear_tf=True
)

twp_strings = [" ".join(text) for text in twp]

emotion_model = pipeline("text-classification", model="seara/rubert-tiny2-russian-emotion-detection-ru-go-emotions", top_k=None)

labels=[]
for text in twp_strings:
    labels.append(emotion_model(text)[0][0]['label'])

counts = Counter(labels)

for label in counts:
    print(label+': '+str(counts[label]))