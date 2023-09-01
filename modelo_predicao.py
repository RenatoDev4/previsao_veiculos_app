import numpy as np
import pandas as pd
import streamlit as st
from category_encoders import TargetEncoder
from sklearn.preprocessing import FunctionTransformer

from config import (ANO_ESCOLHA, CIDADE_UNICO, COMBUSTIVEL_UNICO,
                    CORES_ESCOLHA, DADOS_MACHINE_LEARNING, KM_ESCOLHA, MODELO,
                    MODELO_UNICO, MOTOR_UNICO)

dataframe_machine_learning = DADOS_MACHINE_LEARNING
modelo_unico = MODELO_UNICO
combustivel_unico = COMBUSTIVEL_UNICO
ano_escolha = ANO_ESCOLHA
km_escolha = KM_ESCOLHA
cores_escolha = CORES_ESCOLHA
cidade_unico = CIDADE_UNICO
motor_unico = MOTOR_UNICO


def predicao():
    """
    Runs the main function of the program.
    This function displays a title and a description on the screen.
    It then prompts the user to input details about a vehicle to obtain a value prediction.
    The function collects the user input and stores it in a dictionary.
    It renames the columns of the input dictionary according to a predefined mapping.
    The function initializes a list of columns that should be pre-filled with default values.
    It also initializes a boolean variable to track if all fields have been filled.
    The function iterates over each column and prompts the user to input a value for that column.
    The user input is stored in the dictionary based on the column name.
    If the column is a special case (e.g., 'modelo', 'combustivel', etc.), a dropdown is displayed instead of a number input.
    If the user does not fill in a non-prefill column, the function sets the 'todos_campos_preenchidos' variable to False.
    When the user clicks the 'Fazer Previsão' button, the function checks if all fields have been filled.
    If all fields are filled, the prefill columns are set to zero and the input data is transformed.
    The transformed data is then used to make a prediction using a separate prediction function.
    The predicted value is formatted as a currency and displayed on the screen.
    If not all fields are filled, an error message is displayed.
    """
    columns = ['modelo', 'combustivel', 'ano', 'km', 'cor', 'cambio',
               'cidade', 'airbag motorista', 'freios ABS', 'airbag passageiro',
               'ar-condicionado', 'direção elétrica',
               'volante com regulagem de altura', 'travas elétricas',
               'cd player com MP3', 'entrada USB',
               'vidros elétricos dianteiros',
               'limajuste de alturap. traseiro',
               'desemb. traseiro', 'alarme',
               'ajuste de altura',
               'distribuição eletrônica de frenagem,',
               'controle de tração',
               'retrovisores elétricos', 'piloto automático', 'Kit Multimídia',
               'bancos de couro', 'limp. traseiro', 'motor']

    def transform_data(user_input: list) -> pd.DataFrame:
        """
        Transforms user input data into a transformed dataframe.

        Parameters:
            user_input (list): A list containing user input data.

        Returns:
            pandas.DataFrame: A dataframe containing the transformed data.
        """
        novo_veiculo_df = pd.DataFrame([user_input], columns=columns)
        transformer = FunctionTransformer(np.log1p, validate=True)
        dados_transformados = transformer.transform(novo_veiculo_df.select_dtypes(exclude=['object'])) # noqa
        colunas_dados_transformados = novo_veiculo_df.select_dtypes(exclude=['object']).columns # noqa
        novo_veiculo_transformado = pd.concat([
            novo_veiculo_df.select_dtypes(include=['object']),
            pd.DataFrame(dados_transformados,
                         columns=colunas_dados_transformados)
        ], axis=1)
        dados_limpos = pd.read_csv('dataframe_let.csv', sep=';').drop('Unnamed: 0', axis=1) # noqa
        encoder = TargetEncoder()
        variaveis_categoricas = ['modelo', 'combustivel', 'cor', 'cidade']
        encoder.fit(dados_limpos[variaveis_categoricas], dados_limpos['preco'])
        novo_veiculo_transformado[variaveis_categoricas] = encoder.transform(dados_limpos[variaveis_categoricas]) # noqa

        return novo_veiculo_transformado

    def make_prediction(data: object) -> object:
        """
        Make a prediction using the given data.

        Parameters:
            data (object): The data to be used for making the prediction.

        Returns:
            object: The predicted value after applying the model.
        """
        nova_previsao = MODELO.predict(data)
        nova_previsao_valor_original = np.expm1(nova_previsao)
        return nova_previsao_valor_original

    def main() -> None:
        """
        Runs the main function of the program.
        This function displays a title and a description on the screen.
        It then prompts the user to input details about a vehicle to obtain a value prediction.
        The function collects the user input and stores it in a dictionary.
        It renames the columns of the input dictionary according to a predefined mapping.
        The function initializes a list of columns that should be pre-filled with default values.
        It also initializes a boolean variable to track if all fields have been filled.
        The function iterates over each column and prompts the user to input a value for that column.
        The user input is stored in the dictionary based on the column name.
        If the column is a special case (e.g., 'modelo', 'combustivel', etc.), a dropdown is displayed instead of a number input.
        If the user does not fill in a non-prefill column, the function sets the 'todos_campos_preenchidos' variable to False.
        When the user clicks the 'Fazer Previsão' button, the function checks if all fields have been filled.
        If all fields are filled, the prefill columns are set to zero and the input data is transformed.
        The transformed data is then used to make a prediction using a separate prediction function.
        The predicted value is formatted as a currency and displayed on the screen.
        If not all fields are filled, an error message is displayed.
        """
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
        prefill_columns = ['airbag motorista', 'freios ABS',
                           'airbag passageiro', 'ar-condicionado',
                           'direção elétrica',
                           'volante com regulagem de altura',
                           'travas elétricas', 'cd player com MP3',
                           'entrada USB', 'vidros elétricos dianteiros',
                           'limajuste de alturap. traseiro',
                           'desemb. traseiro', 'alarme', 'ajuste de altura',
                           'distribuição eletrônica de frenagem',
                           'controle de tração', 'retrovisores elétricos',
                           'piloto automático', 'Kit Multimídia',
                           'bancos de couro', 'limp. traseiro',
                           'distribuição eletrônica de frenagem,']

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
                opcao_selecionada = st.selectbox(col_label, list(opcoes_cambio.keys()))  # noqa
                user_input[col] = opcoes_cambio[opcao_selecionada]  # type:ignore # noqa
            elif col == 'motor':
                user_input[col] = st.selectbox(col_label, motor_unico)
            else:
                user_input[col] = 0 if col in prefill_columns else st.number_input(col_label) # noqa
                if col not in prefill_columns and user_input[col] == 0:
                    todos_campos_preenchidos = False

        if st.button('Fazer Previsão'):
            if todos_campos_preenchidos:
                for col in prefill_columns:
                    user_input[col] = 0
                transformed_data = transform_data(user_input)  # type:ignore
                prediction = make_prediction(transformed_data)
                valor_formatado = "**R${:,.2f}**".format(prediction[0])  # type:ignore # noqa
                st.success(f"Valor predito: {valor_formatado}")
            else:
                st.error('Preencha todos os campos obrigatórios antes de fazer a previsão.') # noqa

    main()
