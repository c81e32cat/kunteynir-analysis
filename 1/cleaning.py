import json
import re

def fix_lyrics(lyrics):
    lyrics = (lyrics.replace('\n\n', '.\n')
              .split('\n'))

    new_text = []
    forbidden_starts = ('—', '-', '" -', '\" -', '\"...', '"...') # удаляем (большинство) скитов и цитат

    for text_string in lyrics:
        text_string = text_string.strip()
        text_string = re.sub(r"'(?=[а-яА-ЯёЁ])[а-яА-ЯёЁ]+\b", "", text_string) # убираю суффиксы
        # помимо чистки скитов, убираем повторяющиеся строчки в припевах и так далее
        if text_string and not text_string.startswith(forbidden_starts) and text_string not in new_text:
            new_text.append(text_string.strip())

    lyrics = '\n'.join(new_text)
    lyrics = re.sub(r'\(.*?\)', '', lyrics) # удаляем повторения слов бэк-вокалом, обозначающиеся (вот так)
    lyrics=re.sub(r'\*[^*]+\*', '', lyrics) # текст внутри этих зведлочек зачастую тоже не нужен
    lyrics = (lyrics
                  .replace('\u00a0', ' ')  # NBSP
                  .replace('\u205f', ' ')  # MMSP
                  .replace('\u2005', ' ')  # 4/MSP
                  .replace('\u202f', ' ')  # NNBSP
                  .replace('\u200b', '')  # ZWSP
                  )
    lyrics = re.sub(r'[ ]{2,}', ' ', lyrics) # удаляем лишние двойные пробелы, которые могли образоваться где не надо
    lyrics=re.sub(r'\n{2,}', '\n', lyrics) # конец куплета или припева -- точно конец предложения!

    return lyrics

def fix_title(title): # если потом захочу проанализировать альбомы
    title = re.sub(r'\(.*?\)', '', title)
    title = (title
              .replace('\u00a0', ' ')  # NBSP (неразрывный пробел)
              .replace('\u205f', ' ')  # MMSP (средний математический)
              .replace('\u2005', ' ')  # 4/MSP (четвертной пробел)
              .replace('\u202f', ' ')  # NNBSP (узкий неразрывный)
              .replace('\u200b', '')  # ZWSP (нулевой ширины — просто удаляем)
              )
    return title

# тут гемини помог мне очистить весь лишний шлак
with open('Lyrics_Kunteynir_Fixed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

f.close()

clean_data = []
for song in data['songs']:
    clean_data.append({
        'title': song['title'],
        'lyrics': song['lyrics']
    })

f=open('clean_lyrics.txt', 'w', encoding='utf-8')

indices_to_remove = [] # чистим инструменталы, которые гениус пропустил
for i in range(len(clean_data)):
    clean_data[i]['title']=fix_title(clean_data[i]['title'])
    clean_data[i]['lyrics']=fix_lyrics(clean_data[i]['lyrics'])
    if clean_data[i]['lyrics']=='':
        indices_to_remove.append(i)

# удаляю пустые строки
for index in range(len(indices_to_remove)-1, 0, -1):
    clean_data.pop(indices_to_remove[index])
# красиво записываю в файл, который скормлю claude
for i in range(len(clean_data)):
    f.write(clean_data[i]['title']+'\n\n'+clean_data[i]['lyrics']+'\n\n\n')

f.close()

# eng={}
#
# for i in range(len(clean_data)):
#     english_words = re.findall(r'\b[a-zA-Z]{2,}\b', clean_data[i]['lyrics'].lower())
#
#     for word in english_words:
#         eng[word] = eng.get(word, 0) + 1

# for word in eng.keys():
#     print(word, eng[word])