import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pickle
# import sklearn
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from pymystem3 import Mystem
from functools import lru_cache
import string
import faiss
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
eng_stop_words = stopwords.words('english')
with open('russian.txt', 'r') as f:
    ru_stop_words = f.read()
ru_stop_words=ru_stop_words.split('\n')
allow="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя0123456789-' \n\t"
#Задаём стеммер
m= Mystem()
def embed_bert_cls(text, model=model, tokenizer=tokenizer)->np.array:
    """
    Встраивает входной текст с использованием модели на основе BERT.

    Аргументы:
        text (str): Входной текст для встраивания.
        model (torch.nn.Module): Модель на основе BERT для использования при встраивании.
        tokenizer (transformers.PreTrainedTokenizer): Токенизатор для токенизации текста.

    Возвращает:
        numpy.ndarray: Встроенное представление входного текста.
    """
    # Токенизируем текст и преобразуем его в PyTorch тензоры
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

    # Отключаем вычисление градиентов
    with torch.no_grad():
        # Пропускаем тензоры через модель
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})

    # Извлекаем последний скрытый состояние из выходных данных модели
    embeddings = model_output.last_hidden_state[:, 0, :]

    # Нормализуем встроенные представления
    embeddings = torch.nn.functional.normalize(embeddings)
    embeddings=embeddings[0].cpu().numpy()
    
    # Преобразуем встроенные представления в массив numpy и возвращаем первый элемент
    return embeddings

def lems_eng(text):
    if type(text)==type('text'):
        text=text.split()  
    wnl= WordNetLemmatizer()
    lemmatized= []
    pos_map = {
        'NN': 'n',  # существительное
        'NNS': 'n',  # существительное (множественное число)
        'NNP': 'n',  # собственное имя (единственное число)
        'NNPS': 'n',  # собственное имя (множественное число)
        'VB': 'v',  # глагол (инфинитив)
        'VBD': 'v',  # глагол (прошедшее время)
        'VBG': 'v',  # глагол (настоящее причастие/герундий)
        'VBN': 'v',  # глагол (прошедшее причастие)
        'JJ': 'a',  # прилагательное
        'JJR': 'a',  # прилагательное (сравнительная степень)
        'JJS': 'a',  # прилагательное (превосходная степень)
        'RB': 'r',  # наречие
        'RBR': 'r',  # наречие (сравнительная степень)
        'RBS': 'r',  # наречие (превосходная степень)
        'PRP': 'n',  # личное местоимение
        'PRP$': 'n',  # притяжательное местоимение
        'DT': 'n'  # определитель
    }
    pos_tags = pos_tag(text)
    lemmas = []
    for token, pos in pos_tags:
        pos = pos_map.get(pos,'n')
        lemma = wnl.lemmatize(token, pos=pos)
        lemmas.append(lemma)
    return ' '.join(lemmas)

def lems_rus(texts):
    if type(texts)==type([]):
        texts=' '.join(texts)
    #lemmatized =[]
    lemmas = m.lemmatize(texts)
    return ''.join(lemmas)
def clean(text: str)-> str: 

    
    text = ''.join(c for c in text if c in allow)
    text= text.split()
    text = [word for word in text if word.lower() not in ru_stop_words]
    text = [word for word in text if word.lower() not in eng_stop_words]
    return ' '.join(text)


def improved_lemmatizer(texts,batch_size=1000):
    if type(texts)==type('text'):
        texts=texts.split()







#Читаем датасет книжек
df=pd.read_csv('final+lem.csv',index_col=0).reset_index(drop=True)

# embs=[]
# for i in tqdm(df.index):
#     embs.append(embed_bert_cls(df['lemmatized'][i]))

# with open('embs+lem.pickle', 'wb') as f:
#     pickle.dump(embs, f)



#Читаем эмбединги
with open('embs+lem.pickle', 'rb') as f:
    embs = pickle.load(f)
#df['']
embs =np.array(embs)
print('Тип выхода:',type(embs),'Размер выхода: ',embs.shape)

#Читаем стоп-слова

index=faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
@lru_cache()
def find_similar(text, k=10):
    """
    Находит похожие тексты на основе косинусного сходства.

    Аргументы:
        text (str): Входной текст для поиска похожих текстов.
        embeddings (numpy.ndarray): Предварительно вычисленные встроенные представления текстов.
        threshold (float): Порог, выше которого тексты считаются похожими.

    Возвращает:
        numpy.ndarray: Сходства между входным текстом и каждым текстом во встроенных представлениях.
    """
    
    # Встраиваем входной текст
    text_emb = embed_bert_cls(text)
    text_emb = np.expand_dims(text_emb, axis=0)
    print(f'Тип поискового запроса: {type(text_emb)}\nРазмер полученного запроса: {text_emb.shape}')#\nСам запрос:\n{text_emb}\n')
    dist,idx=index.search(text_emb,k)
    
    return dist.squeeze(),idx.squeeze()#,idx
@lru_cache()
def find_unsimilar(text,n=10, d=embs.shape[0]):
    """
    Находит похожие тексты на основе косинусного сходства.

    Аргументы:
        text (str): Входной текст для поиска похожих текстов.
        embeddings (numpy.ndarray): Предварительно вычисленные встроенные представления текстов.
        threshold (float): Порог, выше которого тексты считаются похожими.

    Возвращает:
        numpy.ndarray: Сходства между входным текстом и каждым текстом во встроенных представлениях.
    """
    
    # Встраиваем входной текст
    text_emb = embed_bert_cls(text)
    text_emb = np.expand_dims(text_emb, axis=0)
    print(f'Тип поискового запроса: {type(text_emb)}\nРазмер полученного запроса: {text_emb.shape}')#\nСам запрос:\n{text_emb}\n')
    dist,idx=index.search(text_emb,d)
    dist=dist.flatten()[::-1]
    idx=idx.flatten()[::-1]
    
    return dist[:n],idx[:n]#,idx