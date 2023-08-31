import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from category_encoders import TargetEncoder
from sklearn.preprocessing import FunctionTransformer

dados = pd.read_csv('precos_carros_tratados.csv', sep=';', na_values=['N/D'])

with open('modelo_rf_otimizado_target.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


dataframe = pd.read_csv('dataframe_let.csv', sep=';')
modelo_unico = dataframe['modelo'].unique()
combustivel_unico = dataframe['combustivel'].unique()
ano_escolha = [2000, 2005, 2010, 2015, 2020, 2023]
km_escolha = [0, 1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000,
              90000, 100000, 150000, 200000]
cores_escolha = ['Branco', 'Preto', 'Prata', 'Cinza']
cidade_unico = sorted(dataframe['cidade'].unique())
motor_unico = dataframe['motor'].unique()


# Set page layout as wide
st.set_page_config(layout="wide")

# Add selectbox
opcoes = ["Problema a ser resolvido", "Estatisticas do dataframe", "Estudo dos dados",
           "Modelo de predição", "Conclusão"]
selecao = st.sidebar.selectbox(
    "Quais informações você quer verificar?",
    (opcoes)
)
dados['preco'] = dados['preco'].str.replace('[^\d.]', '', regex=True).astype(float)*1000


def formata_numero(valor, prefixo=''):
    for unidade in ['', 'mil']:
        if valor < 1000:
            return f'{prefixo} {valor:.2f} {unidade}' 
        valor /= 1000
    return f'{prefixo} {valor:.2f}'

if selecao == "Problema a ser resolvido":
    st.title('Problema')
    st.subheader('Uma empresa de revenda de carros nos contratou para resolver o seguinte problema:')
    st.write('Qual é o melhor lugar para se comprar carros para revenda e quais são os melhores carros para se comprar para se obter o maior lucro?')

    st.divider()
 
    st.title("Abordagem")
    st.subheader('Para resolver esse problema, vamos realizar as seguintes etapas:')
    st.write('1. **Análise do mercado de carros usados.**')
    st.write('Vamos analisar os dados de preços de carros usados no Brasil, para entender quais são os modelos e marcas mais populares e com maior valor de revenda.')
    st.write('2. **Identificação dos melhores cidades para comprar carros usados.**')
    st.write('Vamos identificar as principais cidades com menor valor de carros usados.')
    st.write('3. **Avaliação dos fatores que influenciam o lucro na revenda de carros.**')
    st.write('Vamos identificar os principais fatores que influenciam o lucro na revenda de carros, como o preço de compra, o custo de manutenção e os custos operacionais.')
    
    st.divider()

    st.title('Resultados esperados')
    st.subheader('Com essas etapas, esperamos encontrar as seguintes respostas para o problema:')
    st.write( "* Os melhores lugares para se comprar carros para revenda, considerando fatores como variedade de modelos, preços competitivos e etc.")
    st.write( "* Os carros com maior potencial de lucro, considerando fatores como preço de compra, custo de manutenção e demanda do mercado.")
    st.write( "* Um modelo de machine learning para estimar o valor de revenda de um carro, com base em dados históricos de preços e características do carro.")



if selecao == "Estatisticas do dataframe":
    # Titulo
    st.title('Análise descritiva')

    st.subheader("Estatísticas gerais")
    coluna11, coluna12, coluna13 = st.columns(3)
    with coluna11:
        qtd_veiculos = dados.shape[0]
        st.metric('Quantidade total de veiculos :', qtd_veiculos)
    with coluna12:
        modelo_mais_vendido = dados['modelo'].value_counts().idxmax()
        modelo_mais_vendido_str = str(modelo_mais_vendido)
        modelo_mais_vendido_resumido = modelo_mais_vendido_str.split('(')[0]
        st.metric('Carro mais vendido:', modelo_mais_vendido_resumido)
    with coluna13:
        tipo_combustivel = dados['combustivel'].value_counts().idxmax()
        st.metric('Tipo de combustivel mais usado pelos veículos:', tipo_combustivel)

    coluna14, coluna15 = st.columns(2)
    with coluna14:
        cor_mais_comum = dados['cor'].value_counts().idxmax()
        st.metric('Cor mais comum entre os veículos', cor_mais_comum)

    st.divider()

    st.subheader("Estatísticas para o preço")
    # Visualização no Streamlits
    coluna1, coluna2, coluna3 = st.columns(3)
    with coluna1:
        st.metric('Carro mais barato:', formata_numero(round(dados['preco'].min(), 2), 'R$'))
    with coluna2:
        st.metric('Preço médio dos carros:', formata_numero(round(dados['preco'].mean(), 2), 'R$'))
    with coluna3:
        st.metric('Carro mais caro:', formata_numero(round(dados['preco'].max(), 2), 'R$'))

    st.divider()

    st.subheader("Estatísticas do ano dos veículos")
    coluna4, coluna5, coluna6 = st.columns(3)
    with coluna4:
        menor_ano_veiculos = dados['ano'].min()
        st.metric('Veículo mais antigo:', int(menor_ano_veiculos))
    with coluna5:
        media_ano_veiculos = dados['ano'].mean()
        st.metric('Ano médio dos veículos:', int(media_ano_veiculos))
    with coluna6:
        maior_ano_veiculos = dados['ano'].max()
        st.metric('Veículo mais novo:', int(maior_ano_veiculos))

    st.divider()

    st.subheader("Estatísticas de quilometragem dos veículos")
    coluna7, coluna8, coluna9 = st.columns(3)
    with coluna7:
        menor_km = int(dados['km'].min())
        st.metric('Veículo com a quilometragem mais baixa:', menor_km)
    with coluna8:
        media_km = round(dados['km'].mean())
        dados['km'].fillna(media_km, inplace=True)
        st.metric('Média de quilometragem dos veículos:', f'{media_km:.2f} mil')
    with coluna9:
        maior_km = round(dados['km'].max())
        st.metric('Veículo com a quilometragem mais alta:', f'{maior_km:.2f} mil')

    st.divider()


if selecao == 'Estudo dos dados':
    # Gráfico barras
    filtro_ano = dados[(dados['ano'] >= 2013) & (dados['ano'] <= 2023)]
    preco_por_ano = filtro_ano.groupby('ano')['preco'].mean().round(2).reset_index()

    fig = px.bar(preco_por_ano, x='ano', y='preco',
                labels={'ano': 'Ano', 'preco': 'Preço Médio'},
                text='preco', height=600, width=1000)

    fig.update_traces(texttemplate='R$ %{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_title='', xaxis_title_font_size=14, yaxis_title_font_size=14)

    st.header("**Variação de preço por ano de fabricação do veículo**")
    st.markdown("No gŕafico abaixo claramente podemos notar um aumento significativo nos preços dos veículos a partir de **2019**, possivelmente em razão da escassez de matérias-primas para sua fabricação e do início da pandemia de **COVID-19**.")
    st.plotly_chart(fig)

    # Linha formatação
    st.divider()

    # Gráfico mapa
    with open('mapa_veiculos.html', 'r') as file:
        mapa = file.read()
    st.header("**Gráfico de mapa - veículos por estado**")
    st.markdown("Nesse gráfico de mapa interativo que destaca as cidades com o maior número de veículos à venda, observamos um padrão interessante. As maiores capitais, como **São Paulo**, **Curitiba** e **Rio de Janeiro**, apresentam uma concentração significativamente maior de veículos disponíveis para venda. Essas cidades metropolitanas e economicamente ativas parecem atrair um maior volume de transações de veículos, o que pode ser reflexo da maior demanda e oferta nesses centros urbanos. A quantidade substancial de veículos à venda nessas cidades sugere uma dinâmica de mercado diferenciada, onde a disponibilidade de veículos parece estar correlacionada com a densidade populacional e a atividade econômica das regiões.")
    components.html(mapa, height=600, width=1000)

    # Linha formatação
    st.divider()

    # Gráfico de correlação
    st.header("**Gráfico de matriz de correlação**")
    st.markdown("**O que podemos inferir desta matriz de correlação em relação ao preço do veiculo?**")
    st.markdown("**Preço e ano do carro:** Se o ano em que o carro foi fabricado é mais recente, geralmente o preço é mais alto. Isso faz sentido, certo? Carros mais novos costumam ser mais caros. Mas não parece haver uma ligação muito forte entre o preço e a quilometragem ou o tipo de câmbio ou até mesmo o motor.")
    st.markdown("**Ano do carro e quilometragem:** Carros mais antigos geralmente têm mais quilômetros rodados, o que também faz sentido. No entanto, parece que o ano do carro importa mais para o preço do que a quilometragem. Ou seja, mesmo se um carro for mais antigo, mas tiver rodado poucos quilômetros, ainda pode valer mais.")
    st.markdown("**Quilometragem e preço:** Embora carros com menos quilômetros tendam a ser mais caros, essa relação não é tão forte. Algumas vezes, mesmo com mais quilômetros, um carro pode ter um preço alto por causa de outros fatores, como ser mais novo.")
    st.markdown("**Tipo de câmbio e motor:** O tipo de câmbio do carro (se é automático ou manual) parece estar relacionado ao tipo de motor. Isso significa que, dependendo de como o motor é, o carro pode ter um tipo de câmbio específico.")
    st.markdown("**Tamanho do motor e tipo de câmbio:** A grandeza do motor do carro está ligada ao tipo de câmbio. Isso indica que o tamanho do motor pode influenciar em qual tipo de câmbio funciona melhor no carro.")
    st.markdown("**Acessórios em veículos:** Um único acessório tem um pequeno efeito no preço do veículo. No entanto, à medida que mais acessórios são adicionados ao veículo, o preço final pode aumenta significativamente.")
    st.markdown("**Acessórios que têm o maior impacto no preço são:** **freios ABS, direção elétrica, controle de tração e bancos de couro.**")

    dados = dados.replace('N/D', pd.NA)
    dados['preco'] = pd.to_numeric(dados['preco'], errors='coerce')
    dados['km'] = pd.to_numeric(dados['km'], errors='coerce')
    correlation_matrix = dados[['preco', 'ano', 'km', 'cambio',
       'airbag motorista', 'freios ABS', 'airbag passageiro',
       'ar-condicionado', 'direção elétrica',
       'volante com regulagem de altura', 'travas elétricas',
       'cd player com MP3', 'entrada USB', 'vidros elétricos dianteiros',
       'limajuste de alturap. traseiro', 'desemb. traseiro', 'alarme',
       'ajuste de altura',
       'distribuição eletrônica de frenagem,', 'controle de tração',
       'retrovisores elétricos', 'piloto automático', 'Kit Multimídia',
       'bancos de couro', 'limp. traseiro', 'cilindrada']].corr()
    

    fig = px.imshow(correlation_matrix,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.columns,
                    zmin=-1,
                    color_continuous_scale='RdBu',
                    zmax=1)
    fig.update_layout(height=600, width=1000)
    st.plotly_chart(fig)

    # Linha formatação
    st.divider()

    caracteristicas = [
    'ar-condicionado', 'direção elétrica', 'travas elétricas',
    'cd player com MP3', 'entrada USB', 'vidros elétricos dianteiros',
    'limajuste de alturap. traseiro', 'desemb. traseiro', 'alarme',
    'câmbio automático', 'ajuste de altura',
    'distribuição eletrônica de frenagem,', 'controle de tração',
    'retrovisores elétricos', 'piloto automático', 'Kit Multimídia',
    'bancos de couro', 'limp. traseiro'
]
    contagem_caracteristicas = dados[caracteristicas].sum().sort_values(ascending=False).head(10)
    df_caracteristicas = pd.DataFrame({'Característica': contagem_caracteristicas.index, 'Contagem': contagem_caracteristicas.values})
    fig = px.bar(df_caracteristicas, x='Característica', y='Contagem',
                labels={'Característica': 'Opcionais', 'Contagem': 'Contagem'},
                text='Contagem', height=600, width=1000)

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_title='', xaxis_title_font_size=14, yaxis_title_font_size=14)

    st.header("**Opcionais mais comuns entre os veículos**")
    st.markdown("Esses dados podem ser úteis para entender quais características são mais comuns e desejadas em veículos, bem como quais características são menos frequentes. Eles também podem fornecer informações valiosas para o desenvolvimento e aprimoramento de produtos automotivos, além de orientar estratégias de marketing. Lembre-se de que essas observações são baseadas nos dados fornecidos e podem variar dependendo do contexto e do mercado.")
    st.plotly_chart(fig)

    # Linha formatação
    st.divider()

    dados['total_caracteristicas'] = dados[caracteristicas].sum(axis=1)
    top_models = dados.nlargest(10, 'total_caracteristicas')
    top_models = top_models.sort_values('total_caracteristicas')
    fig = px.bar(top_models, x='total_caracteristicas', y='modelo',
                 labels={'total_caracteristicas': 'Opcionais', 'modelo': 'Veículo'},
                 text='total_caracteristicas', height=600, width=1000)
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_title='', xaxis_title_font_size=14, yaxis_title_font_size=14)

    st.header("**Modelos com maior número de opcionais**")
    st.markdown("Esses dados podem ser úteis para entender quais características são mais comuns e desejadas em veículos, bem como quais características são menos frequentes. Eles também podem fornecer informações valiosas para o desenvolvimento e aprimoramento de produtos automotivos, além de orientar estratégias de marketing. Lembre-se de que essas observações são baseadas nos dados fornecidos e podem variar dependendo do contexto e do mercado.")
    st.plotly_chart(fig)

    # Linha formatação
    st.divider()

    # Gráfico scatter
    contagem_por_cidade = dados['cidade'].value_counts()
    cidades_mais_veiculos = contagem_por_cidade.nlargest(20).index
    data_mais_veiculos = dados[dados['cidade'].isin(cidades_mais_veiculos)]
    preco_medio_por_cidade = data_mais_veiculos.groupby('cidade')['preco'].mean()

    preco_formatado = preco_medio_por_cidade.apply(lambda x: f'R$ {x:.2f}')
    data_plot = pd.DataFrame({'cidade': preco_medio_por_cidade.index, 'preco_medio': preco_medio_por_cidade.values,
                            'preco_formatado': preco_formatado.values})
    
    fig = px.scatter(data_plot, x='cidade', y='preco_medio', text='preco_formatado', height=600, width=1000)

    fig.update_layout(xaxis_title='Cidades', yaxis_title='Preço Médio', title='Preço médio por idade')
    for trace in fig.data:
        trace.textposition = 'top center'

    st.header("**Preço médio dos carros nas cidades com mais veículos à venda**")
    st.markdown("**Variação de preços por cidade:** O gráfico de pontos revela que as cidades com maior quantidade de veículos à venda não necessariamente têm os preços mais altos. Existem variações notáveis nos preços médios entre diferentes cidades.")
    st.markdown("**Cidades com preços elevados:** Notamos que cidades como **Campinas** e **Rio de Janeiro** possuem preços médios relativamente **elevados** em comparação com outras cidades do grupo. Isso pode estar relacionado a características econômicas locais e preferências dos consumidores.")
    st.markdown("**Cidades com preços elevados²:** Creio que isso deve-se ao dataset estar em construção e ter poucos veículos na cidade de Sao Carlos, por enquanto desconsiderar")
    st.plotly_chart(fig)

    # Linha formatação
    st.divider()

    # Gráfico mais vendidos por cidade
    contagem_por_cidade = dados['cidade'].value_counts()
    top_10_cidades = contagem_por_cidade.nlargest(15).index
    data_top_10_cidades = dados[dados['cidade'].isin(top_10_cidades)]
    carro_mais_vendido_por_cidade = data_top_10_cidades.groupby('cidade')['modelo'].apply(lambda x: x.value_counts().index[0]).reset_index()
    df_carro_mais_vendido_por_cidade = pd.DataFrame(carro_mais_vendido_por_cidade)
    fig = px.bar(df_carro_mais_vendido_por_cidade, x='cidade', y='modelo',
             labels={'cidade': 'Cidade', 'modelo': 'Modelo mais vendido'},
             height=600, width=1000)

    # Ajustar legendas dos eixos
    fig.update_layout(xaxis_title='', yaxis_title='', xaxis_tickangle=-45)


    st.header("**Preço médio dos carros nas cidades com mais veículos à venda**")
    st.markdown("**Custo-benefício:** Nota-se que a seleção de modelos mais populares recai principalmente em veículos fabricados entre os anos de **2010 e 2014**. Esta faixa temporal sugere uma tendência em busca do equilíbrio entre custo e benefício, indicando que carros desse período possuem características atrativas em termos de desempenho, consumo de combustivel, tecnologia e valor.")
    st.markdown("Além disso, merece destaque o fato de que a maioria desses carros pertence à categoria de veículos populares. Ademais, é relevante notar que muitos dos modelos mais vendidos pertencem à categoria de veículos compactos, refletindo a preferência por automóveis que combinam praticidade e eficiência, especialmente em ambientes urbanos congestionados.")

    st.plotly_chart(fig)

    # Linha formatação
    st.divider()

    colunas_caracteristicas = ['ano', 'freios ABS', 'airbag motorista', 'controle de tração', 'distribuição eletrônica de frenagem,' ]
    data_caracteristicas = dados[colunas_caracteristicas]
    data_caracteristicas_filtrado = data_caracteristicas[data_caracteristicas['ano'].between(2013, 2023)]
    media_caracteristicas_por_ano = data_caracteristicas_filtrado.groupby('ano').mean().reset_index()
    melted_data = media_caracteristicas_por_ano.melt(id_vars='ano', var_name='Opcionais', value_name='Média')
    melted_data['Média'] *= 100
    fig = px.line(melted_data, x='ano', y='Média', color='Opcionais', markers=True,
                height=600, width=1200)

    fig.update_layout(xaxis_title='Ano', yaxis_title='Porcentagem de veiculo com opcionais')

    st.header("**Mudanças nos opcionais dos veículos entre 2013 e 2023**")
    st.markdown("**Avanços nos recursos de segurança nos carros populares:** É evidente que a grande maioria dos carros populares contemporâneos está equipada com um recurso essencial para garantir a segurança do motorista e dos passageiros: os **Freios ABS**. Em 2022, impressionantes **82%** dos veículos ofereciam esse recurso fundamental como opcional de série.")
    st.markdown("No entanto, a incorporação de elementos cruciais, como o **Controle de tração** (presente em apenas **21%** dos automóveis), desempenhando um papel fundamental na prevenção de acidentes, ou mesmo os **Airbags** (disponíveis em apenas **35%** dos automóveis), que têm o potencial de evitar tragédias no caso de colisões, ainda é relativamente escassa.")
    st.markdown("Outro item extreamente importante é a **Distribuição eletrônica de frenagem**, mas esse é praticamente não existe em carros populares mesmo nos dias de hoje")
    st.plotly_chart(fig)

    # Linha formatação
    st.divider()

    # Gráfico preço médio carro mais vendido
    top_cidades = dados['cidade'].value_counts().head(100).index
    data_top_cidades = dados[dados['cidade'].isin(top_cidades)]
    veiculo_mais_vendido_global = dados['modelo'].value_counts().idxmax()
    data_veiculo_mais_vendido = data_top_cidades[data_top_cidades['modelo'] == veiculo_mais_vendido_global]
    preco_medio_por_cidade = data_veiculo_mais_vendido.groupby('cidade')['preco'].mean().reset_index()

    fig = px.bar(preco_medio_por_cidade, x='cidade', y='preco', labels={'cidade': 'Cidades', 'preco': 'Preços'},
                text='preco', height=600, width=1000)
    fig.update_traces(texttemplate='R$ %{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_title='', xaxis_title_font_size=15, yaxis_title_font_size=15)

    st.header(f'Preço médio do veículo mais vendido ({veiculo_mais_vendido_global})')
    st.markdown("No gráfico, destacam-se as cidades mais favoráveis para adquirir o veículo mais popular do ICarros **(VW Fox 1.0)**:")
    st.markdown("**Belo Horizonte, Curitiba e Santo André-SP**, estas cidades apresentam os preços mais atrativos dentre todas as demais.")
 
    st.plotly_chart(fig)

    st.divider()

    # Gráfico de dispersão opicionais
    modelo_selecionado = dados['modelo'].value_counts().idxmax()

    dados_modelo_selecionado = dados[dados['modelo'] == modelo_selecionado]

    colunas_opcionais = ['airbag motorista', 'freios ABS', 'ar-condicionado',
                        'Kit Multimídia', 'bancos de couro']
    
    st.header('Diferença de preços entre o mesmo veículo com e sem opcional')
    st.write(f'Veículo utilizado: **{modelo_selecionado}**')

    opcional_selecionado = st.selectbox('Selecione um opcional para comparar os preços:',
                                        colunas_opcionais) 

    dados_com_opcional = dados_modelo_selecionado[dados_modelo_selecionado[opcional_selecionado] == 1]
    dados_sem_opcional = dados_modelo_selecionado[dados_modelo_selecionado[opcional_selecionado] == 0]

    media_preco_com_opcional = dados_com_opcional['preco'].mean()
    media_preco_sem_opcional = dados_sem_opcional['preco'].mean()

    diferenca_percentual = ((media_preco_com_opcional - media_preco_sem_opcional) / media_preco_sem_opcional) * 100

    data = {
        'Possui opcional?': ['Com Opcional', 'Sem Opcional'],
        'Preço Médio': [media_preco_com_opcional, media_preco_sem_opcional]
    }
    df = pd.DataFrame(data)

    fig = px.line(df, x='Possui opcional?', y='Preço Médio', height=600, width=1000)
    fig.add_annotation(
        text=f'Diferença: {diferenca_percentual:.2f}%',
        x='Com Opcional',
        y=media_preco_com_opcional,
        showarrow=True,
        arrowhead=1
    )

    st.plotly_chart(fig)
    st.subheader('Observações')
    st.write('Ao analisarmos o gráfico, percebemos que a maior diferença de preço em relação aos carros com opcionais é de **5.23%**, enquanto a menor diferença é de **1.73%**. Isso nos indica que os carros com mais opcionais tendem a ser um pouco mais caros, mas a diferença **não é grande**. Isso significa que se o cliente quiser um carro mais confortável, pode escolher um com mais opcionais pagando só um pouquinho a mais. As pessoas estão dispostas a gastar mais para ter um carro mais confortável e seguro.')

if selecao == 'Modelo de predição':

    columns = ['modelo', 'combustivel', 'ano', 'km', 'cor', 'cambio',
    'cidade', 'airbag motorista', 'freios ABS', 'airbag passageiro',
    'ar-condicionado', 'direção elétrica',
    'volante com regulagem de altura', 'travas elétricas',
    'cd player com MP3', 'entrada USB', 'vidros elétricos dianteiros',
    'limajuste de alturap. traseiro', 'desemb. traseiro', 'alarme',
    'ajuste de altura',
    'distribuição eletrônica de frenagem,', 'controle de tração',
    'retrovisores elétricos', 'piloto automático', 'Kit Multimídia',
    'bancos de couro', 'limp. traseiro', 'motor']

    with open('modelo_rf_otimizado_target.pkl', 'rb') as model_file:
        modelo_predicao = pickle.load(model_file)

    def transfomar_data(user_input):
        novo_veiculo_df = pd.DataFrame([user_input], columns=columns)
        transformer = FunctionTransformer(np.log1p, validate=True)
        dados_transformados = transformer.transform(novo_veiculo_df.select_dtypes(exclude=['object']))
        colunas_dados_transformados = novo_veiculo_df.select_dtypes(exclude=['object']).columns
        novo_veiculo_transformado = pd.concat([
            novo_veiculo_df.select_dtypes(include=['object']),
            pd.DataFrame(dados_transformados, columns=colunas_dados_transformados)
        ], axis=1)
        dados_limpos = pd.read_csv('dataframe_let.csv', sep=';').drop('Unnamed: 0', axis=1)
        encoder = TargetEncoder()
        variaveis_categoricas = ['modelo', 'combustivel', 'cor', 'cidade']
        encoder.fit(dados_limpos[variaveis_categoricas], dados_limpos['preco'])
        novo_veiculo_transformado[variaveis_categoricas] = encoder.transform(dados_limpos[variaveis_categoricas])

        return novo_veiculo_transformado

    def make_prediction(data):
        nova_previsao = model.predict(data)
        nova_previsao_valor_original = np.expm1(nova_previsao)
        return nova_previsao_valor_original

    def main():
        st.title('Previsão de Valor de Veículo')
        st.write('Insira os detalhes do veículo para obter a previsão de valor.')

        user_input = {}
        colunas_renomeadas = {
            'modelo': 'Modelo',
            'combustivel': 'Combustível',
            'ano': 'Ano',
            'km': 'Quilometragem',
            'cor': 'Cor',
            'cambio': 'Cambio',
            'cidade': 'Cidade',
            'motor': 'Motorização (Cilindradas*)'
        }
        prefill_columns = ['airbag motorista', 'freios ABS', 'airbag passageiro', 'ar-condicionado', 'direção elétrica',
        'volante com regulagem de altura', 'travas elétricas', 'cd player com MP3', 'entrada USB', 'vidros elétricos dianteiros',
        'limajuste de alturap. traseiro', 'desemb. traseiro', 'alarme', 'ajuste de altura',
        'distribuição eletrônica de frenagem', 'controle de tração', 'retrovisores elétricos', 'piloto automático', 'Kit Multimídia',
        'bancos de couro', 'limp. traseiro', 'distribuição eletrônica de frenagem,']

        todos_campos_preenchidos = True

        for col in columns:
            col_label = colunas_renomeadas.get(col, col)
            if col == 'modelo':
                user_input[col] = st.selectbox(col_label, modelo_unico)
            elif col == 'combustivel':
                user_input[col] = st.selectbox(col_label, combustivel_unico)
            elif col == 'ano':
                user_input[col] = st.selectbox(col_label, ano_escolha)
            elif col == 'km':
                user_input[col] = st.selectbox(col_label, km_escolha)
            elif col == 'cor':
                user_input[col] = st.selectbox(col_label, cores_escolha)
            elif col == 'cidade':
                user_input[col] = st.selectbox(col_label, cidade_unico)
            elif col == 'cambio':
                opcoes_cambio = {'Manual': 0, 'Automático': 1}
                opcao_selecionada = st.selectbox(col_label, list(opcoes_cambio.keys()))
                user_input[col] = opcoes_cambio[opcao_selecionada]
            elif col == 'motor':
                user_input[col] = st.selectbox(col_label, motor_unico)
            else:
                user_input[col] = 0 if col in prefill_columns else st.number_input(col_label)
                if col not in prefill_columns and user_input[col] == 0:
                    todos_campos_preenchidos = False

        if st.button('Fazer Previsão'):
            if todos_campos_preenchidos:
                for col in prefill_columns:
                    user_input[col] = 0
                transformed_data = transfomar_data(user_input)
                prediction = make_prediction(transformed_data)
                valor_formatado = "R${:,.2f}".format(prediction[0])
                st.write(f"Valor predito: {valor_formatado}")
            else:
                st.error('Preencha todos os campos obrigatórios antes de fazer a previsão.')

    main()

if selecao == 'Conclusão':
    st.title('Conclusão de pesquisa:')
    st.write('Identificamos um padrão interessante em nossa pesquisa. Nas grandes cidades, como **São Paulo**, **Curitiba** e **Rio de Janeiro**, notamos uma quantidade significativamente maior de carros disponíveis para venda. Essas cidades maiores e economicamente ativas parecem atrair mais transações de carros, provavelmente devido à maior procura e oferta nessas áreas urbanas movimentadas. A presença abundante de carros à venda nessas cidades sugere uma forma especial de funcionamento do mercado. Isso está ligado à quantidade de pessoas vivendo nessas áreas e à atividade econômica intensa.')
    st.write('Descobrimos que as **melhores cidades** para procurar oportunidades de compra de carros com **melhor custo** são: **São Bernardo do Campo-SP**, **Curitiba-PR**, **Sao José do Rio Preto-SP** e **Jundiaí-SP**. Essas cidades oferecem preços mais atrativos em comparação com outras.')
    st.write('Uma observação importante é que cidades com muitos carros à venda nem sempre têm os preços mais altos. Existem diferenças notáveis nos preços médios entre diferentes cidades.')
    st.write('Uma tendência interessante é que as pessoas estão mais interessadas em **carros populares** fabricados entre **2010** e **2014**. Essa preferência sugere um equilíbrio entre custo e benefício, indicando que carros desse período possuem características atraentes em termos de desempenho, consumo de combustível, tecnologia e valor.')
    st.write('Ao analisar carros com ou sem opcionais, observamos que a maior diferença de preço é de **5.23%**, enquanto a menor diferença é de **1.73%**. Isso nos indica que carros com mais opcionais tendem a ser um pouco mais caros, mas a diferença **não é significativa**. Isso significa que, se um cliente desejar um carro mais confortável, optar por um com mais opcionais implica um custo adicional pequeno. Isso reflete que as pessoas estão dispostas a investir um pouco mais para obter um carro mais **confortável** e **seguro**.')

    st.caption('Minha conclusão pode mudar a qualquer momento, uma vez que o dataset está em constante atualização. :flag-br: :car:')
    st.caption('Web scraping, pesquisa, gráficos e estudo feito por Renato Moraes')
