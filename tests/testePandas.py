import pandas as pd

#infos_camadas.csv

dados_camadas = pd.read_csv('infos_camadas.csv')

print(dados_camadas.shape)

print(dados_camadas.dtypes)

print(dados_camadas.head())

print(dados_camadas.describe())

# infos_materiais.csv

dados_materiais = pd.read_csv('infos_materiais.csv')

print(dados_materiais.shape)

print(dados_materiais.dtypes)

print(dados_materiais.head())

print(dados_materiais.describe())


