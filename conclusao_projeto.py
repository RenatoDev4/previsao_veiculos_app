import streamlit as st


def conclusao():
    """
    Function to display the conclusion of the research.

    This function displays the conclusion of the research based on the analysis of the data. It provides insights on the patterns observed in different cities and their influence on the car market. It also identifies cities with good opportunities for buying vehicles at a lower cost. The function highlights the preference for cars manufactured between 2010 and 2014 and the impact of optional features on the price of popular cars. The conclusion acknowledges that the dataset is constantly updated, and credits the author of the web scraping, research, graphs, and study.

    Parameters:
    None

    Returns:
    None
    """
    st.title('Conclusão de pesquisa:')
    st.write('Notamos um padrão intrigante durante nosso estudo dos dados. Nas metrópoles como **São Paulo**, **Curitiba** e **Rio de Janeiro**, observamos um volume significativamente maior de carros disponíveis para venda. Essas cidades densamente povoadas e economicamente ativas parecem ser pontos quentes para transações de veículos, muito provavelmente devido à alta demanda e oferta em suas agitadas áreas urbanas. A abundância de veículos à venda nessas cidades aponta para uma dinâmica de mercado especial, influenciada pelo número de habitantes e pela intensa atividade econômica.') # noqa
    st.write('Identificamos que as cidades mais propícias para encontrar **boas oportunidades de compra de veículos** com excelente **custo-benefício** são **Bauru-SP**, **Belo Horizonte-MG** e **Sorocaba-SP**. Essas cidades oferecem **preços mais atraentes** para carros populares em comparação com outras regiões.') # noqa
    st.write('É interessante notar que cidades com muitos carros à venda nem sempre têm os **preços mais baixos**. Existem diferenças notáveis nos preços médios entre diferentes cidades.') # noqa
    st.write('Uma tendência interessante é que as pessoas estão mais interessadas em **carros populares** fabricados entre **2010** e **2014**. Essa preferência sugere um equilíbrio entre custo e benefício, indicando que carros desse período possuem características atraentes em termos de desempenho, consumo de combustível, tecnologia e valor.') # noqa
    st.write('Claramente, o preço de um veículo é determinado pela seguinte ordem de influência: **Modelo** -> **Ano do Carro** -> **Quilometragem** -> **Acessórios** -> **Tamanho do Motor** -> **Tipo de Câmbio**. ')
    st.write('Fica evidente que a maior discrepância de preço entre os carros com opcionais chega a impressionantes **27,88%**. Esse insight nos sugere que, em geral, os **carros populares** com um maior número de opcionais tendem a ser substancialmente mais caros. Isso implica que, se um cliente busca um veículo mais confortável, poderia considerar cuidadosamente se valeria a pena pagar a mais por tais opcionais, uma vez que existem modelos de veículos que já incluem esses recursos de fábrica sem acréscimo de preço.') # noqa

    st.caption('A conclusão pode mudar a qualquer momento, uma vez que o dataset está em constante atualização. :flag-br: :car:') # noqa
    st.caption('Web scraping, pesquisa, gráficos e estudo feito por Renato Moraes') # noqa