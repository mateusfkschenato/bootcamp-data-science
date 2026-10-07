import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Carregando os três datasets ---

materiais = pd.read_csv('infos_materiais.csv')
camadas = pd.read_csv('infos_camadas.csv')
resposta = pd.read_csv('VariavelResposta.csv')
dicionario = pd.read_csv('dicionario_tipo_material.csv')

# ==========================
# Entendendo os dados
# ==========================

print(f"materiais: {materiais.shape[0]} linhas x {materiais.shape[1]} colunas")
print(f"camadas: {camadas.shape[0]} linhas x {camadas.shape[1]} colunas")
print(f"resposta: {resposta.shape[0]} linhas x {resposta.shape[1]} colunas")
print(f"dicionario: {dicionario.shape[0]} linhas x {dicionario.shape[1]} colunas")

# --- Inspecao rapida de cada dataset ---
for nome, df in [('materiais', materiais), ('camadas', camadas), ('resposta', resposta), ('dicionario', dicionario)]:
    print(f"\n{'='*50}")
    print(f" {nome.upper()}")
    print(f"{'='*50}")
    print(df.info())
    print(f"\nPrimeiras linhas:\n{df.head()}")
    print(f"\nValores nulos por coluna:\n{df.isnull().sum()}")
    print(f"\nEstatisticas descritivas:\n{df.describe()}")
# Investigue: quais colunas vieram com tipo errado?
# Investigue: quais colunas tem valores nulos e o que eles significam?

# =================
# Identificação e Limpeza
# =================
# 1. Corrigir separador decimal em infos_materiais
# As colunas numericas usam virgula: "0,40" em vez de "0.40"
# Dica: releia o CSV com decimal=',' ou converta manualmente:
# materiais['coluna'] = materiais['coluna'].str.replace(',', '.').astype(float)
# 2. Separar materiais metalicos de insumos
# Algumas linhas sao insumos (CAL, COQUE), nao sucata metalica
# Identifique-as: quais linhas tem null em Energia e Rendimento?
# 3. Converter datas
# camadas['dt_hora_consumo'] = pd.to_datetime(camadas['dt_hora_consumo'])
# 4. Verificar duplicatas
# camadas.duplicated().sum()
# resposta.duplicated(subset='cd_corrida').sum()
# 5. Quantas corridas unicas existem em cada dataset?
# Compare: as corridas de camadas estao todas em resposta?
# corridas_camadas = set(camadas['cd_corrida'].unique())
# corridas_resposta = set(resposta['cd_corrida'].unique())
# print(f"So em camadas: {len(corridas_camadas - corridas_resposta)}")
# print(f"So em resposta: {len(corridas_resposta - corridas_camadas)}")

# ==========================
# Análise univariada
# ==========================

# ==========================
# Outliers
# ==========================

# ==========================
# Análise bivariada
# ==========================

# ==========================
# Análise multivariada
# ==========================

