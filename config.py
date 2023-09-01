import pickle

import pandas as pd

DADOS = pd.read_csv('dataframe_let.csv', sep=';', na_values=['N/D'])
DADOS_MACHINE_LEARNING = pd.read_csv('dataframe_let.csv', sep=';')
MODELO_UNICO = DADOS_MACHINE_LEARNING['modelo'].unique()
COMBUSTIVEL_UNICO = DADOS_MACHINE_LEARNING['combustivel'].unique()
ANO_ESCOLHA = [2000, 2005, 2010, 2015, 2020, 2023]
KM_ESCOLHA = [0, 1000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000,
              90000, 100000, 150000, 200000]
CORES_ESCOLHA = ['Branco', 'Preto', 'Prata', 'Cinza']
CIDADE_UNICO = sorted(DADOS_MACHINE_LEARNING['cidade'].unique())
MOTOR_UNICO = DADOS_MACHINE_LEARNING['motor'].unique()

with open('modelo_rf_otimizado_target.pkl', 'rb') as model_file:
    MODELO = pickle.load(model_file)

with open('mapa_veiculos.html', 'r') as file:
    MAPA = file.read()
