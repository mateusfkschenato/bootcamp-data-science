import pandas as pd

dados_materiais = pd.read_csv('infos_materiais.csv')
print(dados_materiais.shape)

# print(dados_materiais.describe)

dados_camadas = pd.read_csv('infos_camadas.csv')
print(dados_camadas.head)