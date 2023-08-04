import streamlit as st
import pandas as pd
import random as rd
import webbrowser as wb
import numpy as np
from find import find_similar,df,lems_eng,lems_rus,clean,find_unsimilar

st.set_page_config(
    page_title="Умный поиск книг",
    page_icon="📖",
    layout="wide",
    #initial_sidebar_state="expanded"
)
with st.expander('Исходный датафрейм'):
    if st.checkbox('С лемматизацией'):
        df
    else:
        df.iloc[:,:-1]
st.title('Умный поиск книг')
#negability= st.checkbox('Негативный промт (beta)')
with st.form(key='search_form'):
    
    input=st.text_input('Введите поисковый запрос','Пример запроса')

    # if negability:
    #     neg=st.text_input('Введите отрицательный запрос')


    search_but=st.form_submit_button('Искать')

items_per_page=st.number_input('Количество книг на странице',min_value=1,max_value=10,value=5)
# if search_but:
#     st.session_state.clicked = True
#st.toast('Уфф')
#@st.cache_data(experimental_allow_widgets=True)
def books_show(books_idx,sim,n=items_per_page):
    col=[]
    books=df.copy().iloc[books_idx][:n]
    for author in books['author']:
        if author.find('Донцова')!=-1:
            #st.toast('Уфф')
            pass
    books['sims']=sims[:n]
    with st.expander('Датафрейм с результатами'):
        books
    #print(books.index)
    for i,book_id in enumerate(books_idx[:n]):
        pic_col,text_col=st.columns([0.2,0.8])
        '---'
        
        url=books.loc[book_id][0]
        #url
        pic_col.image(books.loc[book_id,'image_url'],use_column_width=True)
        pic_col.markdown(f'<a href={url} target="_blank">Ссылка на книгу</a>', unsafe_allow_html=True)
        pic_col.markdown(f'**Степень похожести:** {books.loc[book_id,"sims"]:.4f}')

        #col[i][0].button('Купить',key=books['page_url'][i],on_click=lambda: wb.open_new_tab(books['page_url'][i]))

        text_col.markdown('## ' + books.loc[book_id, 'title'])
        text_col.markdown('**Автор:** ' + books.loc[book_id, 'author'])
        text_col.markdown('**Жанр:** ' + books.loc[book_id, 'genre'])
        text_col.markdown('**Аннотация:** ' + books.loc[book_id, 'annotation'])
  
if search_but:
    neg_mark=input.find(' -')
    cleaned_input=clean(lems_eng(lems_rus(input[:neg_mark])))
    cleaned_neg=clean(lems_eng(lems_rus(input[neg_mark+2:])))
    #print(cleaned_neg.split(),df.loc[15390,'lemmatized'].split())
    with st.spinner('Wait for it...'):
        if neg_mark!=-1:
            st.markdown(f'**Лемматизированный запрос:** {cleaned_input} \n\n **Лемматизированый негативный запрос:** {cleaned_neg}')
            sims,books_idx=find_similar(cleaned_input,50)
            for book in books_idx:
                if any(word in cleaned_neg.split() for word in df.loc[book,'lemmatized'].split()):
                    books_idx=np.delete(books_idx,np.where(books_idx==book))        
        else:
            st.markdown(f'**Лемматизированный запрос:** {cleaned_input}')
            sims,books_idx=find_similar(cleaned_input)
        print(f'Похожести:\n{sims}\nИндексы:\n{books_idx}')
        books_show(books_idx,sims)
