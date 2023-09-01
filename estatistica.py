import streamlit as st

from config import DADOS


def formata_numero(valor: float, prefixo: str = '') -> str:
    """
    Formats a number with a specified prefix.

    Args:
        valor (float): The number to be formatted.
        prefixo (str, optional): The prefix to be added to the formatted number. Defaults to ''.

    Returns:
        str: The formatted number as a string.

    Example:
        >>> formata_numero(12345.6789, 'R$')
        'R$ 12,34 mil'
    """
    for unidade in ['', 'mil']:
        if valor < 1000:
            return f'{prefixo} {valor:.2f} {unidade}'
        valor /= 1000
    return f'{prefixo} {valor:.2f}'


def estatisticas() -> None:
    """
    Function to generate statistics and display them using the Streamlit library.

    This function generates various statistics about a dataset called DADOS and displays them using the Streamlit library. The statistics include general information about the dataset, such as the total number of vehicles, the most sold car model, and the most used fuel type. It also includes statistics about the price of the vehicles, such as the minimum, average, and maximum prices. Additionally, it provides statistics about the year and mileage of the vehicles.

    Parameters:
    None

    Returns:
    None
    """
    st.title('Análise descritiva')

    st.subheader("Estatísticas gerais")
    coluna11, coluna12, coluna13 = st.columns(3)
    with coluna11:
        qtd_veiculos = DADOS.shape[0]
        st.metric('Quantidade total de veiculos :', qtd_veiculos)
    with coluna12:
        modelo_mais_vendido = DADOS['modelo'].value_counts().idxmax()
        st.metric('Carro mais vendido:', modelo_mais_vendido)
    with coluna13:
        tipo_combustivel = DADOS['combustivel'].value_counts().idxmax()
        st.metric('Tipo de combustivel mais usado pelos veículos:',
                  tipo_combustivel)

    coluna14, coluna15 = st.columns(2)
    with coluna14:
        cor_mais_comum = DADOS['cor'].value_counts().idxmax()
        st.metric('Cor mais comum entre os veículos', cor_mais_comum)

    st.divider()

    st.subheader("Estatísticas para o preço")
    coluna1, coluna2, coluna3 = st.columns(3)
    with coluna1:
        preco_min = DADOS['preco'].min()
        st.metric('Veículo mais barato:', formata_numero(preco_min, 'R$'))
    with coluna2:
        preco_med = DADOS['preco'].mean()
        st.metric('Preço médio dos veículos:', formata_numero(preco_med, 'R$'))
    with coluna3:
        preco_max = DADOS['preco'].max()
        st.metric('Veículo mais caro:', formata_numero(preco_max, 'R$'))

    st.divider()

    st.subheader("Estatísticas do ano dos veículos")
    coluna4, coluna5, coluna6 = st.columns(3)
    with coluna4:
        menor_ano_veiculos = DADOS['ano'].min()
        st.metric('Veículo mais antigo:', int(menor_ano_veiculos))
    with coluna5:
        media_ano_veiculos = DADOS['ano'].mean()
        st.metric('Ano médio dos veículos:', int(media_ano_veiculos))
    with coluna6:
        maior_ano_veiculos = DADOS['ano'].max()
        st.metric('Veículo mais novo:', int(maior_ano_veiculos))

    st.divider()

    st.subheader("Estatísticas de quilometragem dos veículos")
    coluna7, coluna8, coluna9 = st.columns(3)
    with coluna7:
        menor_km = int(DADOS['km'].min())
        st.metric('Veículo com a quilometragem mais baixa:', menor_km)
    with coluna8:
        media_km = round(DADOS['km'].mean())
        DADOS['km'].fillna(media_km, inplace=True)
        st.metric('Média de quilometragem dos veículos:',
                f'{media_km:.2f} mil')
    with coluna9:
        maior_km = round(DADOS['km'].max())
        st.metric('Veículo com a quilometragem mais alta:',
                f'{maior_km:.2f} mil')

    st.divider()