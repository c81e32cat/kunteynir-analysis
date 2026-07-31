from pymystem3 import Mystem
from cleaning import clean_data
import re
from gensim.models.phrases import Phrases, Phraser


m = Mystem()

def tokenize(text):
    words = re.compile(r'[a-zA-Zа-яА-Я]+(?:-[a-zA-Zа-яА-ЯёЁ]+)*') # убираю знаки препинания и числа

    text = text.lower()
    text = (text
            .replace('\n', ' ')
            .replace('ё', 'е'))
    text = ' '.join(words.findall(text)) # привожу текст в единообразный и приемлемый вид

    analysis = m.analyze(text) # только mystem смог понять, что "мерс" и "мерса" - это одно и то же...

    clean_words = []

    for word in analysis:
        if 'analysis' in word and word['analysis']: # пробелы и прочее он тоже пытается проанализировать, нам не надо - там analysis пустой
            lemma = word['analysis'][0]['lex']
            grammar = word['analysis'][0]['gr'] # библиотека выдает какую-то длинную строку о том, что именно перед нами лежит, мне это не надо
            pos = re.match(r'\w+', grammar).group(0)

            if (len(lemma)>1):
                if pos not in ['ADVPRO', 'APRO', 'COM', 'CONJ', 'INTJ', 'NUM', 'PART', 'PR', 'SPRO']:
                    # я решила, что лучше, чем оставлять допустимые части речи, лучше убрать то, что ТОЧНО не нужно, а что-то спорное - оставить.
                    # зато не пришлось выписывать стоп-слова и вспоминать ВСЕ местоимения в русском
                    # доп. чистка не нужна вообще!!!! я проверила
                    clean_words.append(lemma)
    return clean_words

tokenized_texts=[]
for lyrics in clean_data:
    tokenized_texts.append(tokenize(lyrics['lyrics']))

# ну тут все как обычно - только, быть может, стоит поменять threshold...?
phrases_model = Phrases(
    tokenized_texts,
    min_count=3,
    threshold=10
)

bigram_phraser = Phraser(phrases_model)
tokenized_with_phrases = [bigram_phraser[tokens] for tokens in tokenized_texts]

print(tokenized_with_phrases)
