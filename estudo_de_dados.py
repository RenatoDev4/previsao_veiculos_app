import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from config import DADOS, MAPA

dados = DADOS


def graficos():
    """
    Generate various data visualizations and insights based on the provided data.

    This function creates and displays several graphs and charts to visualize data related to vehicle prices,
    features, and trends. It includes bar charts, maps, correlation matrices, and more, providing valuable insights
    into the dataset.

    Returns:
        None
    """
    # Gráfico barras (Variação de preço por ano de fabricação do veículo)
    filtro_ano = DADOS[(DADOS['ano'] >= 2013) & (DADOS['ano'] <= 2023)]
    preco_por_ano = filtro_ano.groupby('ano')['preco'].mean().round(2). \
        reset_index()

    fig = px.bar(preco_por_ano, x='ano', y='preco',
                 labels={'ano': 'Ano', 'preco': 'Preço Médio'},
                 text='preco', height=600, width=1000)

    fig.update_traces(texttemplate='R$ %{text:.2f}', textposition='outside')
    fig.update_layout(title='Variação de preço por ano', yaxis_title='',
                      xaxis_title_font_size=16,
                      font=dict(family="Helvetica", size=16))
    fig.update_layout()

    st.header("**Variação de preço por ano de fabricação do veículo**")
    st.markdown("No gŕafico abaixo claramente podemos notar um aumento significativo nos preços dos veículos a partir de **2019**, possivelmente em razão da escassez de matérias-primas para sua fabricação e do início da pandemia de **COVID-19**.") # noqa
    st.plotly_chart(fig)

    st.divider()

    # Gráfico mapa
    st.header("**Gráfico de mapa - veículos por estado**")
    st.markdown("Nesse gráfico de mapa interativo que destaca as cidades com o maior número de veículos à venda, observamos um padrão interessante. As maiores capitais, como **São Paulo**, **Curitiba** e **Rio de Janeiro**, apresentam uma concentração significativamente maior de veículos disponíveis para venda. Essas cidades metropolitanas e economicamente ativas parecem atrair um maior volume de transações de veículos, o que pode ser reflexo da maior demanda e oferta nesses centros urbanos. A quantidade substancial de veículos à venda nessas cidades sugere uma dinâmica de mercado diferenciada, onde a disponibilidade de veículos parece estar correlacionada com a densidade populacional e a atividade econômica das regiões.") # noqa
    components.html(MAPA, height=600, width=1000)

    # Linha formatação
    st.divider()

    # Gráfico de correlação
    st.header("**Gráfico de matriz de correlação**")
    st.markdown("**O que podemos inferir desta matriz de correlação em relação ao preço do veiculo?**") # noqa
    st.markdown("**Preço e ano do carro:** Se o ano em que o carro foi fabricado é mais recente, geralmente o preço é mais alto. Isso faz sentido, certo? Carros mais novos costumam ser mais caros. Mas não parece haver uma ligação muito forte entre o preço e a quilometragem ou o tipo de câmbio ou até mesmo o motor.") # noqa
    st.markdown("**Ano do carro e quilometragem:** Carros mais antigos geralmente têm mais quilômetros rodados, o que também faz sentido. No entanto, parece que o ano do carro importa mais para o preço do que a quilometragem. Ou seja, mesmo se um carro for mais antigo, mas tiver rodado poucos quilômetros, ainda pode valer mais.") # noqa
    st.markdown("**Quilometragem e preço:** Embora carros com menos quilômetros tendam a ser mais caros, essa relação não é tão forte. Algumas vezes, mesmo com mais quilômetros, um carro pode ter um preço alto por causa de outros fatores, como ser mais novo.") # noqa
    st.markdown("**Tipo de câmbio e motor:** O tipo de câmbio do carro (se é automático ou manual) parece estar relacionado ao tipo de motor. Isso significa que, dependendo de como o motor é, o carro pode ter um tipo de câmbio específico.") # noqa
    st.markdown("**Tamanho do motor e tipo de câmbio:** A grandeza do motor do carro está ligada ao tipo de câmbio. Isso indica que o tamanho do motor pode influenciar em qual tipo de câmbio funciona melhor no carro.") # noqa
    st.markdown("**Acessórios em veículos:** Um único acessório tem um pequeno efeito no preço do veículo. No entanto, à medida que mais acessórios são adicionados ao veículo, o preço final pode aumenta significativamente.") # noqa
    st.markdown("**Acessórios que têm o maior impacto no preço são:** **freios ABS, direção elétrica, controle de tração e bancos de couro.**") # noqa

    def preprocess_data(df: pd.DataFrame) -> None:
        """
        Preprocesses the given DataFrame by replacing all occurrences of 'N/D' with pd.NA and converting the specified numeric columns to their respective numeric data types.

        Parameters:
            - df (pd.DataFrame): The DataFrame to be preprocessed.

        Returns:
            None
        """
        df.replace('N/D', pd.NA)
        numeric_columns = ['preco', 'km']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric,
                                                        errors='coerce')

    def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the correlation matrix for the given DataFrame.
     
        Parameters:
            df (DataFrame): The input DataFrame containing the data for which the correlation matrix is to be calculated.
     
        Returns:
            correlation_matrix (DataFrame): The resulting correlation matrix of the specified columns of interest.
        """
        colunas_interesse = ['preco', 'ano', 'km', 'cambio',
                             'airbag motorista', 'freios ABS',
                             'airbag passageiro', 'ar-condicionado',
                             'direção elétrica',
                             'volante com regulagem de altura',
                             'travas elétricas',
                             'cd player com MP3',
                             'entrada USB', 'vidros elétricos dianteiros',
                             'limajuste de alturap. traseiro',
                             'desemb. traseiro', 'alarme',
                             'ajuste de altura',
                             'controle de tração', 'retrovisores elétricos',
                             'piloto automático',
                             'Kit Multimídia', 'bancos de couro',
                             'limp. traseiro', 'motor']
     
        correlation_matrix = df[colunas_interesse].corr()
       
        return correlation_matrix
 
    correlation_matrix = calculate_correlation_matrix(DADOS)

    fig = px.imshow(correlation_matrix,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    zmin=-1,
                    color_continuous_scale='RdBu',
                    zmax=1)
    fig.update_layout(height=600, width=1000, xaxis_title_font_size=16,
                      font=dict(family="Helvetica", size=16))
    st.plotly_chart(fig)

    st.divider()

    # Gráfico de barras (Opcionais mais comuns entre veículos)
    caracteristicas = [
                    'ar-condicionado', 'direção elétrica', 'travas elétricas',
                    'cd player com MP3', 'entrada USB',
                    'vidros elétricos dianteiros',
                    'limajuste de alturap. traseiro',
                    'desemb. traseiro', 'alarme',
                    'ajuste de altura',
                    'distribuição eletrônica de frenagem,',
                    'controle de tração',
                    'retrovisores elétricos', 'piloto automático',
                    'Kit Multimídia', 'bancos de couro', 'limp. traseiro']
    contagem_caracteristicas = dados[caracteristicas].sum(). \
        sort_values(ascending=False).head(10)
    df_caracteristicas = pd.DataFrame({'Característica': contagem_caracteristicas.index, 'Contagem': contagem_caracteristicas.values}) # noqa
    fig = px.bar(df_caracteristicas, x='Característica', y='Contagem',
                 labels={'Característica': 'Opcionais', 'Contagem': 'Contagem'}, # noqa
                 text='Contagem', height=600, width=1000)

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_title='', xaxis_title_font_size=16,
                      yaxis_title_font_size=16,
                      font=dict(family="Helvetica", size=14))

    st.header("**Opcionais mais comuns entre os veículos**")
    st.markdown("Esses dados podem ser úteis para entender quais características são mais comuns e desejadas em veículos, bem como quais características são menos frequentes. Eles também podem fornecer informações valiosas para o desenvolvimento e aprimoramento de produtos automotivos, além de orientar estratégias de marketing. Lembre-se de que essas observações são baseadas nos dados fornecidos e podem variar dependendo do contexto e do mercado.") # noqa
    st.plotly_chart(fig)

    # Linha formatação
    st.divider()

    # Gráfico de barras (Modelo com maior número de opcionais)
    dados['total_caracteristicas'] = dados[caracteristicas].sum(axis=1)
    top_models = dados.nlargest(10, 'total_caracteristicas')
    top_models = top_models.sort_values('total_caracteristicas')
    fig = px.bar(top_models, x='total_caracteristicas', y='modelo',
                 labels={'total_caracteristicas': 'Opcionais',
                         'modelo': 'Veículo'},
                 text='total_caracteristicas', height=600, width=1000)
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_title='', xaxis_title_font_size=14,
                      yaxis_title_font_size=14, font=dict(family="Helvetica",
                                                          size=12))

    st.header("**Modelos com maior número de opcionais**")
    st.markdown("Esses dados podem ser úteis para entender quais características são mais comuns e desejadas em veículos, bem como quais características são menos frequentes. Eles também podem fornecer informações valiosas para o desenvolvimento e aprimoramento de produtos automotivos, além de orientar estratégias de marketing. Lembre-se de que essas observações são baseadas nos dados fornecidos e podem variar dependendo do contexto e do mercado.") # noqa
    st.plotly_chart(fig)

    # Linha formatação
    st.divider()

    # Gráfico scatter (Preço médio dos carros nas cidades com mais veículos à venda) # noqa
    contagem_por_cidade = dados['cidade'].value_counts()
    cidades_mais_veiculos = contagem_por_cidade.nlargest(20).index
    data_mais_veiculos = dados[dados['cidade'].isin(cidades_mais_veiculos)]
    preco_medio_por_cidade = data_mais_veiculos.groupby('cidade')['preco'].mean() # noqa

    preco_formatado = preco_medio_por_cidade.apply(lambda x: f'R$ {x:.2f}')
    data_plot = pd.DataFrame({'cidade': preco_medio_por_cidade.index,
                              'preco_medio': preco_medio_por_cidade.values,
                              'preco_formatado': preco_formatado.values})
    
    fig = px.scatter(data_plot, x='cidade', y='preco_medio',
                     text='preco_formatado', height=600, width=1000)

    fig.update_layout(xaxis_title='Cidades', yaxis_title='Preço Médio',
                      title='Preço médio por idade')
    for trace in fig.data:
        trace.textposition = 'top center'  # type: ignore

    st.header("**Preço médio dos carros nas cidades com mais veículos à venda**") # noqa
    st.markdown("**Variação de preços por cidade:** O gráfico de pontos revela que as cidades com maior quantidade de veículos à venda não necessariamente têm os preços mais altos. Existem variações notáveis nos preços médios entre diferentes cidades.") # noqa
    st.markdown("**Cidades com preços elevados:** Notamos que cidades como **Campinas** e **Rio de Janeiro** possuem preços médios relativamente **elevados** em comparação com outras cidades do grupo. Isso pode estar relacionado a características econômicas locais e preferências dos consumidores.") # noqa
    st.markdown("**Cidades com preços elevados²:** Creio que isso deve-se ao dataset estar em construção e ter poucos veículos na cidade de Sao Carlos, por enquanto desconsiderar") # noqa
    st.plotly_chart(fig)

    # Linha formatação
    st.divider()

    # Gráfico mais vendidos por cidade (Preço médio dos carros nas cidades com mais veículos à venda) # noqa
    contagem_por_cidade = dados['cidade'].value_counts()
    top_10_cidades = contagem_por_cidade.nlargest(15).index
    data_top_10_cidades = dados[dados['cidade'].isin(top_10_cidades)]
    carro_mais_vendido_por_cidade = data_top_10_cidades.groupby('cidade')['modelo'].apply(lambda x: x.value_counts().index[0]).reset_index() # noqa
    df_carro_mais_vendido_por_cidade = pd.DataFrame(carro_mais_vendido_por_cidade) # noqa
    fig = px.bar(df_carro_mais_vendido_por_cidade, x='cidade', y='modelo',
                 labels={'cidade': 'Cidade', 'modelo': 'Modelo mais vendido'},
                 height=600, width=1000)

    # Ajustar legendas dos eixos
    fig.update_layout(xaxis_title='', yaxis_title='', xaxis_tickangle=-45,
                      font=dict(family="Helvetica", size=12))

    st.header("**Preço médio dos carros nas cidades com mais veículos à venda**") # noqa
    st.markdown("**Custo-benefício:** Nota-se que a seleção de modelos mais populares recai principalmente em veículos fabricados entre os anos de **2010 e 2014**. Esta faixa temporal sugere uma tendência em busca do equilíbrio entre custo e benefício, indicando que carros desse período possuem características atrativas em termos de desempenho, consumo de combustivel, tecnologia e valor.") # noqa
    st.markdown("Além disso, merece destaque o fato de que a maioria desses carros pertence à categoria de veículos populares. Ademais, é relevante notar que muitos dos modelos mais vendidos pertencem à categoria de veículos compactos, refletindo a preferência por automóveis que combinam praticidade e eficiência, especialmente em ambientes urbanos congestionados.") # noqa

    st.plotly_chart(fig)

    # Linha formatação
    st.divider()

    # Gráfico de linha (Mudanças nos opcionais dos veículos entre 2013 e 2023)
    colunas_caracteristicas = ['ano', 'freios ABS', 'airbag motorista',
                               'controle de tração',
                               'distribuição eletrônica de frenagem,']
    st.header(f"**Mudanças nos opcionais dos veículos entre os anos**")
    st.markdown("**Avanços nos recursos de segurança nos carros populares:** É evidente que a grande maioria dos carros populares contemporâneos está equipada com um recurso essencial para garantir a segurança do motorista e dos passageiros: os **Freios ABS**. Em 2022, impressionantes **82%** dos veículos ofereciam esse recurso fundamental como opcional de série.") # noqa
    st.markdown("No entanto, a incorporação de elementos cruciais, como o **Controle de tração** (presente em apenas **21%** dos automóveis), desempenhando um papel fundamental na prevenção de acidentes, ou mesmo os **Airbags** (disponíveis em apenas **35%** dos automóveis), que têm o potencial de evitar tragédias no caso de colisões, ainda é relativamente escassa.") # noqa
    st.markdown("Outro item extreamente importante é a **Distribuição eletrônica de frenagem**, mas esse é praticamente não existe em carros populares mesmo nos dias de hoje") # noqa
    st.markdown('**Escolha o ano mínimo e máximo para ajustar o gráfico**:')
    ano_minimo = st.slider("Ano Mínimo", min_value=2000, max_value=2023,
                           value=2013)
    ano_maximo = st.slider("Ano Máximo", min_value=2000, max_value=2023,
                           value=2023)
    data_caracteristicas = dados[colunas_caracteristicas]
    data_caracteristicas_filtrado = data_caracteristicas[data_caracteristicas['ano'].between(ano_minimo, ano_maximo)] # noqa
    media_caracteristicas_por_ano = data_caracteristicas_filtrado.groupby('ano').mean().reset_index() # noqa
    melted_data = media_caracteristicas_por_ano.melt(id_vars='ano',
                                                     var_name='Opcionais',
                                                     value_name='Média')
    melted_data['Média'] *= 100
    fig = px.line(melted_data, x='ano', y='Média', color='Opcionais',
                  markers=True,
                  height=600, width=1200)

    fig.update_layout(xaxis_title='Ano',
                      yaxis_title='Porcentagem de veiculo com opcionais',
                      font=dict(family="Helvetica", size=14))

    st.plotly_chart(fig)

    # Linha formatação
    st.divider()

    # Gráfico de barras (Preço médio do veículo mais vendido (Volkswagen Gol))
    top_cidades = dados['cidade'].value_counts().head(15).index
    data_top_cidades = dados[dados['cidade'].isin(top_cidades)]
    veiculo_mais_vendido_global = dados['modelo'].value_counts().idxmax()
    data_veiculo_mais_vendido = data_top_cidades[data_top_cidades['modelo'] == veiculo_mais_vendido_global] # noqa
    preco_medio_por_cidade = data_veiculo_mais_vendido.groupby('cidade')['preco'].mean().reset_index() # noqa

    fig = px.bar(preco_medio_por_cidade, x='cidade', y='preco',
                 labels={'cidade': 'Cidades', 'preco': 'Preços'},
                 text='preco', height=600, width=1000)
    fig.update_traces(texttemplate='R$ %{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_title='', xaxis_title_font_size=16,
                      yaxis_title_font_size=16)

    st.header(f'Preço médio do veículo mais vendido ({veiculo_mais_vendido_global})') # noqa
    st.markdown(f"No gráfico, destacam-se as cidades mais favoráveis para adquirir o veículo mais popular do ICarros **{veiculo_mais_vendido_global}**:") # noqa
    st.markdown("**Belo Horizonte, Bauro-SP e Sorocaba-SP**, estas cidades apresentam os preços mais atrativos dentre todas as demais.") # noqa

    st.plotly_chart(fig)

    st.divider()

    # Gráfico de dispersão opicionais
    modelo_selecionado = dados['modelo'].value_counts().idxmax()

    dados_modelo_selecionado = dados[dados['modelo'] == modelo_selecionado]

    colunas_opcionais_dict = {'airbag motorista': 'Airbag do motorista',
                              'freios ABS': 'Freios ABS',
                              'ar-condicionado': 'Ar-Condicionado',
                              'Kit Multimídia': 'Kit Multimídia',
                              'bancos de couro': 'Bancos de couro'}
    
    st.header('Diferença de preços entre o mesmo veículo com e sem opcional')
    st.write(f'Veículo utilizado: **{modelo_selecionado}**')

    opcional_selecionado = st.selectbox('Selecione um opcional para comparar os preços:', # noqa
                                        list(colunas_opcionais_dict.values()))
    
    coluna_original = [coluna for coluna,
                       nome in colunas_opcionais_dict.items() if nome == opcional_selecionado][0] # noqa

    dados_com_opcional = dados_modelo_selecionado[dados_modelo_selecionado[coluna_original] == 1] # noqa
    dados_sem_opcional = dados_modelo_selecionado[dados_modelo_selecionado[coluna_original] == 0] # noqa

    media_preco_com_opcional = dados_com_opcional['preco'].mean()
    media_preco_sem_opcional = dados_sem_opcional['preco'].mean()

    diferenca_percentual = ((media_preco_com_opcional - media_preco_sem_opcional) / media_preco_sem_opcional) * 100 # noqa

    data = {
        'Possui opcional?': ['Com Opcional', 'Sem Opcional'],
        'Preço Médio': [media_preco_com_opcional, media_preco_sem_opcional]
    }
    df = pd.DataFrame(data)

    fig = px.line(df, x='Possui opcional?',
                  y='Preço Médio', height=600, width=1000)
    fig.add_annotation(
        text=f'Diferença: {diferenca_percentual:.2f}%',
        x='Com Opcional',
        y=media_preco_com_opcional,
        showarrow=True,
        arrowhead=1
    )

    fig.update_layout(font=dict(family="Helvetica", size=14))

    st.plotly_chart(fig)
    st.subheader('Observações')
    st.write('Ao examinar o gráfico, fica evidente que a maior discrepância de preço entre os carros com opcionais chega a impressionantes **27,88%**. Esse insight nos sugere que, em geral, os **carros populares** com um maior número de opcionais tendem a ser substancialmente mais caros. Isso implica que, se um cliente busca um veículo mais confortável, poderia considerar cuidadosamente se valeria a pena pagar a mais por tais opcionais, uma vez que existem modelos de veículos que já incluem esses recursos de fábrica sem acréscimo de preço.') # noqa