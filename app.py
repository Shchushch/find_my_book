import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Умный поиск книг",
    page_icon="path/to/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title('Умный поиск книг')
form = st.form(key='search')
form.text_input('Введите поисковый запрос')
search=form.form_submit_button('Искать')
df= pd.read_csv('books_sample.csv')
st.number_input('Количество книг на странице',min_value=1,max_value=12)
if search:
    