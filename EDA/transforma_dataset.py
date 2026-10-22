# =====================================================================
# [NOTA DE ORGANIZAÇÃO — não faz parte do código original]
# Este arquivo é a PARTE A (PIPELINE) do script, em um .py próprio.
# Parte B (eda_analise.py/EDA.py) não guarda mais nada em memória vindo
# daqui: cada execução começa do zero e troca informação por arquivo.
#
# Rodar este arquivo PRIMEIRO, do início ao fim, antes do EDA.py (Parte
# B). Ele parte dos 4 arquivos brutos do projeto (infos_camadas.csv,
# infos_materiais.csv, dicionario_tipo_material.csv, VariavelResposta.csv)
# e produz:
#   - base.csv            (construído do zero no bloco A.0, depois
#                           limpo e com as colunas novas: cd_baia,
#                           dt_hora_consumo_estimada, sem as camadas
#                           abaixo de 2.000 kg (peso zero incluído),
#                           Densidade/Rendimento como número, peso_t,
#                           volume_m3)
#   - dataset_modelagem.csv  (uma linha por cestão, com as features
#                           agregadas + a variável resposta)

# =====================================================================


# -----------------------------------------------------------------
# A.0a) Construção da base.csv a partir dos 4 arquivos brutos
# -----------------------------------------------------------------
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


# -----------------------------------------------------------------
# A.0b) Propriedades estimadas para RECC e RECG (sem correspondência
# em infos_materiais.csv). Pesquisado pelo Lucas
# -----------------------------------------------------------------
# =====================================================================

propriedades_estimadas_dict = {
    "RECC": {"Energia Elétrica": 400, "Densidade t/m3": "2,6", "Rendimento Metálico %": "0,70"},
    "RECG": {"Energia Elétrica": 370, "Densidade t/m3": "2,8", "Rendimento Metálico %": "0,80"},
}

base["propriedades_estimadas"] = False

for codigo, props in propriedades_estimadas_dict.items():
    mascara = base["cd_codigo_material"] == codigo
    for coluna, valor in props.items():
        base.loc[mascara, coluna] = valor
    base.loc[mascara, "propriedades_estimadas"] = True

# ---------- Conferência ----------
print("base.csv construído:", base.shape)
print(base.columns.tolist())
print("\nNulos por coluna (confere se o na_values=['null'] funcionou):")
print(base.isnull().sum())
print("\nLinhas com propriedades estimadas (RECC/RECG):", base["propriedades_estimadas"].sum())

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
# A.3) Remoção das camadas abaixo de 2.000 kg (peso zero incluído)
# -----------------------------------------------------------------
# DECISÃO REVISTA DUAS VEZES com a EVCOMX (out/2026):
#
# 1) Após a apresentação da EDA: antes imputávamos o peso zero pela
#    mediana do material. Isso estava ERRADO. Peso zero significa que a
#    camada estava na "receita" do cestão, mas o gancho não pegou o
#    material - a camada NUNCA entrou no cestão. Imputar a mediana
#    colocava no cestão material que não existia (em média +2,3 t em 40%
#    dos cestões). Correto: remover essas linhas.
#
# 2) Resposta do Guima por e-mail (21/out/2026), sobre as camadas de
#    10 kg e 100 kg: "também podem ser desconsideradas na montagem,
#    dificilmente temos camadas com menos de 2000kg. Podem usar esse
#    valor de corte." Então o corte passou de "== 0" para "< 2.000 kg".
#
# Atenção: o corte remove a CAMADA, nunca o cestão inteiro. Nenhum
# cestão fica vazio (todos têm pelo menos uma camada >= 2.000 kg).
# Como nem todo registro abaixo de 2 t é "gancho vazio" (a mediana entre
# 100 kg e 2 t é ~1,6 t), o efeito disso no modelo foi conferido na
# verificação (ver verificacao_validacao/README.md).
#%%
import pandas as pd

PESO_MINIMO_CAMADA_KG = 2000

base = pd.read_csv("base.csv")
peso = base["qt_peso_carregamento"]
chave_cestao = ["cd_corrida", "nu_carregamento"]

mascara_zero = peso == 0
mascara_baixo = (peso > 0) & (peso < PESO_MINIMO_CAMADA_KG)
mascara_remover = mascara_zero | mascara_baixo

print(f"Camadas removidas (peso < {PESO_MINIMO_CAMADA_KG} kg):", mascara_remover.sum(),
      f"({mascara_remover.mean():.1%} das camadas)")
print("  peso zero:", mascara_zero.sum())
print("  entre 0 e 100 kg:", ((peso > 0) & (peso < 100)).sum(), "| exatamente 10 kg:", (peso == 10).sum())
print("  entre 100 e 2.000 kg:", ((peso >= 100) & (peso < PESO_MINIMO_CAMADA_KG)).sum())
print("  por carregamento:", base.loc[mascara_remover, "nu_carregamento"].value_counts().sort_index().to_dict())
print("  cestões afetados:", base.loc[mascara_remover, chave_cestao].drop_duplicates().shape[0])
print(f"  peso removido: {peso[mascara_remover].sum() / 1000:,.0f} t "
      f"({peso[mascara_remover].sum() / peso.sum():.1%} do peso total)")

cestoes_antes = base[chave_cestao].drop_duplicates().shape[0]
base = base.loc[~mascara_remover].reset_index(drop=True)
cestoes_depois = base[chave_cestao].drop_duplicates().shape[0]

# ---------- Conferência ----------
print(f"Camadas abaixo de {PESO_MINIMO_CAMADA_KG} kg restantes:",
      (base["qt_peso_carregamento"] < PESO_MINIMO_CAMADA_KG).sum())
print("Cestões antes / depois do corte:", cestoes_antes, "/", cestoes_depois)
assert cestoes_antes == cestoes_depois, "O corte apagou um cestão inteiro: rever a regra!"

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