import re
import spacy
# natasha + пиво в подарок
# structured output на локальной модели (просить грамматическую основу) выдели все подлежащие и сказуемые и выдай стркутуру (получу json)


with open('processed_lyrics.txt', 'r', encoding='utf-8') as file:
    text = file.read()

text = [' '.join(lyrics.split('\n\n')[1:]) for lyrics in text.split('\n\n\n')]

text=[lyrics.split('.') for lyrics in text]

nlp = spacy.load("ru_core_news_lg")

hyphen = re.compile(r'[а-яА-ЯёЁa-zA-Z]-[а-яА-ЯёЁa-zA-Z]') # спейси как-то криво обрабатывало слово жар-птица
word_pattern = re.compile(r'[a-zA-Zа-яА-ЯёЁ]+') # не хочу иметь дел со знаками препинания

total_positions = 0
total_len_sent = 0
total_sent = 0
for lyrics in text:
    for sentence in lyrics:
        if sentence.strip() and not hyphen.search(sentence):
            doc = nlp(sentence)

            words = [t for t in doc if word_pattern.match(t.text)] # опять же, Я НЕ ХОЧУ ИМЕТЬ ДЕЛ СО ЗНАКАМИ ПРЕПИНАНИЯ

            if words:
                for i, word in enumerate(words):
                    if word.dep_ == 'ROOT':
                        total_positions += (i+1) # в предложении нет нулевой позиции?..
                        # total_pos+= (i+1)/len(words) - относительная позиция в предложении
                        total_len_sent += len(words)
                        total_sent += 1
                        break


print(f"предложений: {total_sent}")
print(f"позиция корня: {total_positions / total_sent:.2f}")
print(f"длина предложения: {total_len_sent / total_sent:.2f}")

# догрузить другие тексты + сравнить результаты с другими библиотеками?
# структура строчек (согласно mystem, видимо), структура предложений (что выкинуть??)
