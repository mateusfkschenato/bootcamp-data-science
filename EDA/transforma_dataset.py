# =====================================================================
# [NOTA DE ORGANIZAÇÃO — não faz parte do código original]
# Este arquivo é a PARTE A (PIPELINE) do script, em um .py próprio.
# Parte B (eda_analise.py/EDA.py) não guarda mais nada em memória vindo
# daqui: cada execução começa do zero e troca informação por arquivo.
#
# Rode este arquivo PRIMEIRO, do início ao fim, antes do EDA.py (Parte
# B). Ele parte dos 4 arquivos brutos do projeto (infos_camadas.csv,
# infos_materiais.csv, dicionario_tipo_material.csv, VariavelResposta.csv)
# e produz/atualiza, em disco:
#   - base.csv            (construído do zero no bloco A.0, depois
#                           limpo e com as colunas novas: cd_baia,
#                           dt_hora_consumo_estimada, peso_estimado,
#                           Densidade/Rendimento como número, peso_t,
#                           volume_m3)
#   - dataset_modelagem.csv  (uma linha por cestão, com as features
#                           agregadas + a variável resposta)
#
# O conteúdo dos blocos A.1 a A.7 não foi alterado - é exatamente o que
# já estava na versão anterior do script reorganizado. O que muda é o
# novo bloco A.0, que antes não existia (a base.csv já vinha pronta).
# =====================================================================


# -----------------------------------------------------------------
# A.0) Construção da base.csv a partir dos 4 arquivos brutos
# -----------------------------------------------------------------
# =====================================================================
# Junta infos_camadas.csv (granularidade de linha: corrida + carregamento
# + camada + material) com infos_materiais.csv (propriedades por
# material - a coluna "Metálico" nesse arquivo é, na prática, o código
# do material, apesar do nome) e dicionario_tipo_material.csv
# (classificação Leve/Misto/Pesado). Os dois primeiros arquivos
# compartilham a coluna ds_descricao_material com o dicionário, então o
# merge gera os sufixos _x (de infos_camadas) e _y (do dicionário) -
# isso é esperado, o resto do pipeline já lida com essas duas colunas.
#
# IMPORTANTE: infos_camadas.csv usa o texto literal "null" (não um
# campo vazio) para marcar ausência. Sem o na_values abaixo, o pandas
# leria isso como a STRING "null", não como NaN - e todo o .isnull()
# usado no resto do pipeline pararia de detectar essas ausências.
# =====================================================================
#%%
import pandas as pd

infos_camadas = pd.read_csv("infos_camadas.csv", na_values=["null"])
infos_materiais = pd.read_csv("infos_materiais.csv", na_values=["null"])
dicionario_tipo_material = pd.read_csv("dicionario_tipo_material.csv", na_values=["null"])

base = infos_camadas.merge(
    infos_materiais, left_on="cd_codigo_material", right_on="Metálico", how="left"
)
base = base.merge(dicionario_tipo_material, on="cd_codigo_material", how="left")

# ---------- Conferência ----------
print("base.csv construído:", base.shape)
print(base.columns.tolist())
print("\nNulos por coluna (confere se o na_values=['null'] funcionou):")
print(base.isnull().sum())

# ---------- Salvar ----------
base.to_csv("base.csv", index=False)
print("\nbase.csv salvo.")


# -----------------------------------------------------------------
# A.1) LIMPEZA - cd_baia e dt_hora_consumo
# -----------------------------------------------------------------
# =====================================================================
# LIMPEZA - cd_baia e dt_hora_consumo
# Decisões tomadas (ver conversa/documentação do grupo):
#   1) cd_baia nulo (100% GUSL)      -> marcador "SEM_BAIA_GUSL"
#   2) dt_hora_consumo nulo (GUSL)   -> mantém NaT (ausência estrutural)
#   3) dt_hora_consumo nulo (falha pontual, 152 linhas, 13 corridas)
#      -> estimado com a hora do carregamento irmão (1 ou 2) da MESMA
#         corrida, marcado em dt_hora_consumo_estimada = True
# =====================================================================
#%%
import pandas as pd

base = pd.read_csv("base.csv")
base["dt_hora_consumo"] = pd.to_datetime(base["dt_hora_consumo"], utc=True, format="ISO8601")

eh_gusl = base["cd_codigo_material"] == "GUSL"

# ---------- 1) cd_baia: marcador pro GUSL ----------
# Converte pra texto primeiro (evita erro de tipo ao misturar número com
# texto), preservando os códigos de baia existentes sem a casa decimal.
# Só reprocessa se ainda não tiver sido convertido (evita erro ao rodar 2x)
ja_convertido = base["cd_baia"].astype(str).eq("SEM_BAIA_GUSL").any()

if not ja_convertido:
    base["cd_baia"] = base["cd_baia"].apply(
        lambda v: "SEM_BAIA_GUSL" if pd.isnull(v) else str(int(v))
    )
# ---------- 2) e 3) dt_hora_consumo ----------
# Coluna de controle: True só nas linhas onde a hora foi estimada
# (não é um dado medido de verdade).
base["dt_hora_consumo_estimada"] = False

# Linhas problemáticas: nulo, não é GUSL, é carregamento 1 ou 2
mascara_suspeitos = (
    base["dt_hora_consumo"].isnull()
    & ~eh_gusl
    & base["nu_carregamento"].isin([1, 2])
)

# Hora "representativa" de cada corrida, usando só carregamentos 1/2
# com hora registrada (a mediana evita que um valor isolado puxe o
# resultado, caso existam várias camadas com horários um pouco
# diferentes dentro do mesmo carregamento)
hora_por_corrida = (
    base.loc[base["dt_hora_consumo"].notnull() & base["nu_carregamento"].isin([1, 2])]
    .groupby("cd_corrida")["dt_hora_consumo"]
    .median()
)

base.loc[mascara_suspeitos, "dt_hora_consumo"] = base.loc[mascara_suspeitos, "cd_corrida"].map(hora_por_corrida)
base.loc[mascara_suspeitos, "dt_hora_consumo_estimada"] = True

# dt_hora_consumo nulo do GUSL fica como está (NaT) - não precisa de ação,
# já é o comportamento padrão do pandas.

# ---------- Conferência ----------
print("cd_baia nulos restantes:", base["cd_baia"].isnull().sum())
print("dt_hora_consumo nulos restantes:", base["dt_hora_consumo"].isnull().sum())
print("  (esperado: só os do GUSL)")
print(base[base["dt_hora_consumo"].isnull()]["cd_codigo_material"].value_counts())
print("Linhas com hora estimada:", base["dt_hora_consumo_estimada"].sum())

# ---------- Salvar ----------
base.to_csv("base.csv", index=False)
print("\nbase.csv atualizado e salvo.")


# -----------------------------------------------------------------
# A.2) Remoção das duplicatas 100% iguais
# -----------------------------------------------------------------
#%%
import pandas as pd

base = pd.read_csv("base.csv")

antes = len(base)
base = base.drop_duplicates(keep="first")
depois = len(base)

print(f"Linhas removidas: {antes - depois}")

base.to_csv("base.csv", index=False)
print("base.csv atualizado e salvo.")


# -----------------------------------------------------------------
# A.3) Imputação do peso zero pela mediana do material
# -----------------------------------------------------------------
#%%
import pandas as pd

base = pd.read_csv("base.csv")

# Mediana de peso por material, calculada só com as linhas que têm peso real (>0)
mediana_por_material = (
    base.loc[base["qt_peso_carregamento"] > 0]
    .groupby("cd_codigo_material")["qt_peso_carregamento"]
    .median()
)

# Linhas com problema de pesagem (peso registrado como zero)
mascara_zero = base["qt_peso_carregamento"] == 0

# Coluna de controle: True só nas linhas onde o peso foi estimado
# (não é um dado medido de verdade)
base["peso_estimado"] = False

# Substitui o peso zero pela mediana do material e marca a flag
base.loc[mascara_zero, "qt_peso_carregamento"] = base.loc[mascara_zero, "cd_codigo_material"].map(mediana_por_material)
base.loc[mascara_zero, "peso_estimado"] = True

# ---------- Conferência ----------
print("Linhas marcadas como peso_estimado:", base["peso_estimado"].sum())
print("Peso zero restante:", (base["qt_peso_carregamento"] == 0).sum())
print("Peso nulo (sem mediana p/ imputar):", base["qt_peso_carregamento"].isnull().sum())

# ---------- Salvar ----------
base.to_csv("base.csv", index=False)
print("\nbase.csv atualizado e salvo.")


# -----------------------------------------------------------------
# A.4) Conversão permanente de Densidade e Rendimento Metálico
# -----------------------------------------------------------------
#%%
import pandas as pd

base = pd.read_csv("base.csv")

for col in ["Densidade t/m3", "Rendimento Metálico %"]:
    if not pd.api.types.is_numeric_dtype(base[col]):
        base[col] = base[col].str.replace(",", ".").astype(float)

base.to_csv("base.csv", index=False)
print("Densidade e Rendimento convertidos e salvos como número.")
print(base[["Densidade t/m3", "Rendimento Metálico %"]].dtypes)


# -----------------------------------------------------------------
# A.5) Criação e gravação de volume_m3 em base.csv
# -----------------------------------------------------------------
#%%
import pandas as pd

base = pd.read_csv("base.csv")

# Conversão vírgula -> ponto, caso você ainda não tenha rodado a limpeza permanente
for col in ["Densidade t/m3", "Rendimento Metálico %"]:
    if not pd.api.types.is_numeric_dtype(base[col]):
        base[col] = base[col].str.replace(",", ".").astype(float)

base["peso_t"] = base["qt_peso_carregamento"] / 1000

# ---------- volume_m3: por linha, vai direto pro base.csv ----------
base["volume_m3"] = base["peso_t"] / base["Densidade t/m3"]
# OBS: RECG e RECC não têm densidade (já sabemos a causa), então essas
# 10 linhas ficam com volume_m3 = NaN - é esperado, não é erro.

base.to_csv("base.csv", index=False)
print("volume_m3 criada e salva no base.csv.")
print("Linhas sem volume_m3 (RECG/RECC):", base["volume_m3"].isnull().sum())


# -----------------------------------------------------------------
# A.6) Montagem do dataframe "cestao" (features agregadas por
#      corrida + carregamento) — usado no fechamento (A.7)
# -----------------------------------------------------------------
#%%
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
import matplotlib
import matplotlib.pyplot as plt

base = pd.read_csv("base.csv")
resposta = pd.read_csv("VariavelResposta.csv")

for col in ["Densidade t/m3", "Rendimento Metálico %"]:
    if not pd.api.types.is_numeric_dtype(base[col]):
        base[col] = base[col].str.replace(",", ".").astype(float)

base["peso_t"] = base["qt_peso_carregamento"] / 1000
base["volume_m3"] = base["peso_t"] / base["Densidade t/m3"]
base["dt_hora_consumo"] = pd.to_datetime(base["dt_hora_consumo"], utc=True, format="ISO8601")

# ---------- Monta a base por cestão (features + resposta) ----------
num_camadas = base.groupby(["cd_corrida", "nu_carregamento"])["nu_camada"].nunique().rename("num_camadas")
# num_materiais: quantos códigos de material DIFERENTES entram no cestão
# (o Guia, seção 8.2, cita isso como feature candidata)
num_materiais = base.groupby(["cd_corrida", "nu_carregamento"])["cd_codigo_material"].nunique().rename("num_materiais")
peso_por_classe = (
    base.groupby(["cd_corrida", "nu_carregamento", "tp_material"])["peso_t"].sum()
    .unstack("tp_material", fill_value=0)
    .rename(columns={"Leve": "peso_leve", "Misto": "peso_misto", "Pesado": "peso_pesado"})
)
peso_total = base.groupby(["cd_corrida", "nu_carregamento"])["peso_t"].sum().rename("peso_t_total")
volume_total = base.groupby(["cd_corrida", "nu_carregamento"])["volume_m3"].sum().rename("volume_m3_total")
primeiro_horario = base.groupby(["cd_corrida", "nu_carregamento"])["dt_hora_consumo"].min().rename("dt_inicio")

cestao = pd.concat([num_camadas, num_materiais, peso_por_classe, peso_total, volume_total, primeiro_horario], axis=1).reset_index()
cestao["prop_leve"] = cestao["peso_leve"] / cestao["peso_t_total"]


# -----------------------------------------------------------------
# A.7) FECHAMENTO — merge final e gravação de dataset_modelagem.csv
# -----------------------------------------------------------------
#%%
resposta_long = pd.concat([
    resposta[["cd_corrida", "is_carga_alta_carregamento_1", "qt_segundos_duracao_parada_carregamento_1"]]
        .rename(columns={"is_carga_alta_carregamento_1": "carga_alta", "qt_segundos_duracao_parada_carregamento_1": "duracao_parada_segundos"})
        .assign(nu_carregamento=1),
    resposta[["cd_corrida", "is_carga_alta_carregamento_2", "qt_segundos_duracao_parada_carregamento_2"]]
        .rename(columns={"is_carga_alta_carregamento_2": "carga_alta", "qt_segundos_duracao_parada_carregamento_2": "duracao_parada_segundos"})
        .assign(nu_carregamento=2),
])

dataset_modelagem = cestao[cestao["nu_carregamento"].isin([1, 2])].merge(
    resposta_long, on=["cd_corrida", "nu_carregamento"], how="left"
)

dataset_modelagem.to_csv("dataset_modelagem.csv", index=False)
print("dataset_modelagem.csv salvo com", len(dataset_modelagem), "linhas e", dataset_modelagem.shape[1], "colunas.")
print(dataset_modelagem.columns.tolist())
