import streamlit as st
import pandas as pd
import random as rd
st.set_page_config(
    page_title="Умный поиск книг",
    page_icon="path/to/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)
df=pd.read_csv('books_sample.csv',index_col=0)
df
st.title('Умный поиск книг')
form = st.form(key='search')
form.text_input('Введите поисковый запрос')
search=form.form_submit_button('Искать')
df= pd.read_csv('books_sample.csv')

pic_col,text_col=st.columns(2)
items_per_page=st.number_input('Количество книг на странице',min_value=1,max_value=10,value=5)
if search:
    samp=df.sample(n=items_per_page).reset_index()
    for i in range(len(samp)):
        #samp['image_url'][i]#='https://picsum.photos/200/300?random='+str(rd.randint(1,1000))']
        pic_col.image(samp['image_url'][i],width=200)
        text_col.markdown('## '+ f"Название:  {samp['title'][i]}")
        text_col.markdown('**Автор** ' +samp['author'][i])
        text_col.markdown('**Аннотация**'  +samp['annotation'][i])
        pic_col.write('-'*100)
        text_col.write('-'*100)
        # Аннотация: {samp['annotation'][i]}

   
    # form.dataframe(
    #     df.sample(n=items_per_page),
    #     hide_index=True,
    #     column_config={'num':None,
    #         'title':st.column_config.TextColumn("Название"),
    #                    'page_url':st.column_config.LinkColumn("Купить"),
    #                    'image_url':st.column_config.ImageColumn("Обложка"),
    #                    'author':st.column_config.TextColumn("Автор"),
    #                    'annotaton':st.column_config.TextColumn("Аннотация"),
    #              },
    #              #height=2000
    #             use_container_width=True
    #              )