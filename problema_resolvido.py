import streamlit as st


def problema_ser_resolvido() -> None:
    """
    This function generates a function comment for a given function body in a markdown code block with the correct language syntax.
    
    Args:
        None
    
    Returns:
        str: The function comment in markdown format.
    """
    st.title('Problema')
    st.subheader('Uma empresa de revenda de carros nos contratou para resolver o seguinte problema:') # noqa
    st.write('Qual é o melhor lugar para se comprar carros para revenda e quais são os melhores carros para se comprar para se obter o maior lucro?') # noqa

    st.divider()

    st.title("Abordagem")
    st.subheader('Para resolver esse problema, vamos realizar as seguintes etapas:') # noqa
    st.write('1. **Análise do mercado de carros usados.**')
    st.write('Vamos analisar os dados de preços de carros usados no Brasil, para entender quais são os modelos e marcas mais populares e com maior valor de revenda.')# noqa
    st.write('2. **Identificação dos melhores cidades para comprar carros usados.**') # noqa
    st.write('Vamos identificar as principais cidades com menor valor de carros usados.') # noqa
    st.write('3. **Avaliação dos fatores que influenciam o lucro na revenda de carros.**')# noqa
    st.write('Vamos identificar os principais fatores que influenciam o lucro na revenda de carros, como o preço de compra, o custo de manutenção e os custos operacionais.') # noqa

    st.divider()

    st.title('Resultados esperados')
    st.subheader('Com essas etapas, esperamos encontrar as seguintes respostas para o problema:') # noqa
    st.write("* Os melhores lugares para se comprar carros para revenda, considerando fatores como variedade de modelos, preços competitivos e etc.")# noqa
    st.write("* Os carros com maior potencial de lucro, considerando fatores como preço de compra, custo de manutenção e demanda do mercado.")# noqa
    st.write("* Um modelo de machine learning para estimar o valor de revenda de um carro, com base em dados históricos de preços e características do carro.")# noqa