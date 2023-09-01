import streamlit as st

from conclusao_projeto import conclusao
from estatistica import estatisticas
from estudo_de_dados import graficos
from modelo_predicao import predicao
from problema_resolvido import problema_ser_resolvido

st.set_page_config(page_title='Projeto DataScience - Veículos', layout="wide")

st.sidebar.markdown("**Bem-vindo ao Projeto de Data Science** 🇧🇷")
nome = "Renato Moraes"
titulo = "Machine Learning 🤖 | Data Scientis 🧑‍🔬"
linkedin_url = 'https://linkedin.com/in/renato-moraes-11b546272'
github_url = 'https://github.com/RenatoDev4'


st.sidebar.markdown("**Sobre o Autor:**")
st.sidebar.text(f'Nome: {nome}')
st.sidebar.markdown('**Áreas de Especialização:**')
st.sidebar.text(titulo)
st.sidebar.markdown('**Conexões Profissionais:**')
st.sidebar.markdown(f'**[LinkedIn]( {linkedin_url} )** e **[Github]( {github_url} )** 🔗')

st.sidebar.divider()

# Add selectbox
opcoes = ["❓ Problema a ser resolvido", "📝 Estatisticas do dataframe",
          "👩‍🏭 Estudo dos dados",
          "🛠️ Modelo de predição", "💵 Conclusão"]
selecao = st.sidebar.selectbox(
    "Quais informações você quer verificar?",
    (opcoes)
)


if selecao == "❓ Problema a ser resolvido":
    problema_ser_resolvido()


if selecao == "📝 Estatisticas do dataframe":
    estatisticas()


if selecao == '👩‍🏭 Estudo dos dados':
    graficos()


if selecao == '🛠️ Modelo de predição':
    predicao()


if selecao == '💵 Conclusão':
    conclusao()