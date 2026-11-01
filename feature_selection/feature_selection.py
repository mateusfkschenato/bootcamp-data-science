# %% [markdown]
# # Feature Engineering & Feature Selection — Predição de Carga Alta (EVCOMX)
#
# **Conjunto oficial: 121 features** (as 84 candidatas da engenharia de features + 36 novas de
# histórico, gusa líquida, ritmo do forno e regime recente + `carga_alta_c1`, o resultado do
# cestão 1). É o conjunto do **modelo oficial do projeto: a média de 5 Gradient Boosting**.
#
# > **Antes (out/2026), o conjunto era de 6 features**, escolhidas por votação entre seis métodos
# > (filtro, RFE, L1, Elastic Net, permutação, Boruta) para uma Random Forest. Esse caminho está
# > guardado em `historico_rf6/` (script, notebook, CSVs e gráficos). Ele foi superado quando a
# > EVCOMX deu o critério de negócio (**menos minutos de forno perdidos**) e confirmou que o
# > resultado do cestão 1 é conhecido a tempo de mexer no cestão 2.
#
# Segue a aula de Feature Selection, o Guia de FE/FS da EVCOMX e o Guia de Modelagem (§1 a §3).
#
# | Etapa | Seção |
# |---|---|
# | Base limpa da EDA, unidade de análise e split 70/15/15 temporal | 0–1 |
# | Feature engineering: massa/volume, composição, ordem/ciclos, tempo, interações, contexto | 2 |
# | Seleção na mão: IDs, constantes, vazamento → 84 candidatas | 3 |
# | Features novas: histórico rico, gusa líquida, ritmo, regime recente, resultado do cestão 1 | 4 |
# | Grupos de features | 5 |
# | Filtro: correlação e informação mútua (e alarme de feature "boa demais") | 6 |
# | Redundância: correlação > 0,90 (e por que não cortamos para árvores) | 7 |
# | Embedded: permutação por grupo na validação | 8 |
# | Wrapper: tirar um grupo de cada vez, no walk-forward, medindo minutos e PR-AUC | 9 |
# | Mudança de regime em maio/2026 | 10 |
# | Respostas às perguntas do Guia §1.3 | 11 |
# | Arquivos para a modelagem e tabela das features | 12 |
#
# **Toda escolha aqui usa só treino + validação (o "dev").** O teste não é aberto nesta etapa.
#
# **Rode antes o `EDA/transforma_dataset.py`**: a feature selection parte do `EDA/base.csv`.
# Leva ~15 min (a seção 9 treina 5 Gradient Boosting × 8 meses para cada conjunto testado).

# %%
import io
import json
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, mannwhitneyu

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score

RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / "EDA").is_dir() and (p / "original_dataset").is_dir())
sys.path.insert(0, str(RAIZ))
from comum.protocolo import (MESES_CUSTO, VARIANTES, FINAL6, familia, prever, oof_dev, resumo_wf, RF6, carregar)

SEED = 42
pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
AZUL, VERMELHO, CINZA, LARANJA, VERDE = "#2b6cb0", "#c0392b", "#7f8c8d", "#dd8a2e", "#4a8c4a"

# %% [markdown]
# ## 0. Ler a base limpa da EDA
#
# A feature selection **não refaz a limpeza**: lê o `EDA/base.csv` gerado pelo
# `EDA/transforma_dataset.py`. Decisões de limpeza que já vêm aplicadas:
# - duplicatas 100% iguais removidas;
# - **camadas abaixo de 2.000 kg removidas, peso zero incluído** (EVCOMX: peso zero = o gancho
#   não pegou o material, a camada nunca entrou no cestão; e, por e-mail, camadas de 10 ou 100 kg
#   também podem ser desconsideradas, "dificilmente temos camadas com menos de 2000kg");
# - hora das 152 camadas sem horário estimada pela corrida;
# - propriedades de RECC/RECG estimadas (pesquisa do Lucas);
# - densidade e rendimento como número.
#
# A variável resposta vem de `original_dataset/` (somente leitura).

# %%
# A raiz do repositório foi achada na célula de cima (funciona rodando de feature_selection/ ou da raiz)
PASTA_EDA, PASTA_BRUTOS = RAIZ / "EDA", RAIZ / "original_dataset"
PASTA_SAIDA = RAIZ / "feature_selection"
PASTA_GRAF = PASTA_SAIDA / "graficos"
PASTA_GRAF.mkdir(exist_ok=True)

def salvar(nome):
    plt.tight_layout()
    plt.savefig(PASTA_GRAF / f"{nome}.png", dpi=130, bbox_inches="tight")
    plt.show()

arq_base, arq_pipeline = PASTA_EDA / "base.csv", PASTA_EDA / "transforma_dataset.py"
assert arq_base.exists(), "EDA/base.csv não existe: rode o EDA/transforma_dataset.py primeiro!"
if arq_pipeline.exists() and arq_pipeline.stat().st_mtime > arq_base.stat().st_mtime:
    print("ATENÇÃO: transforma_dataset.py foi alterado depois do base.csv ser gerado.\n"
          "         Rode o pipeline de novo para usar a limpeza atual.")

base = pd.read_csv(arq_base)
resposta = pd.read_csv(PASTA_BRUTOS / "VariavelResposta.csv")
dataset_eda = pd.read_csv(PASTA_EDA / "dataset_modelagem.csv")   # só para conferir que batemos com a EDA

base["dt_hora_consumo"] = pd.to_datetime(base["dt_hora_consumo"], utc=True, format="ISO8601")
base["cd_codigo_material"] = base["cd_codigo_material"].astype(str)
cam = base[base.nu_carregamento.isin([1, 2])].rename(columns={
    "Energia Elétrica": "energia_kwh_t", "Densidade t/m3": "densidade_t_m3",
    "Rendimento Metálico %": "rendimento"}).copy()

print("Raiz do projeto:", RAIZ)
print("Camadas dos cestões 1 e 2:", cam.shape)
print("Camadas abaixo de 2.000 kg (deve ser 0):", int((cam.qt_peso_carregamento < 2000).sum()))
print("Sem densidade / sem classe:", cam.densidade_t_m3.isna().sum(), "/", cam.tp_material.isna().sum())

# %% [markdown]
# ## 1. Unidade de análise e split (Guia §2)
#
# **Unidade: o carregamento (cestão 1 ou 2).** Justificativa:
# - é a pergunta operacional: "este cestão vai dar carga alta?" — dá para agir no cestão antes
#   de despejá-lo;
# - o alvo já vem separado por cestão (`is_carga_alta_carregamento_1/2`);
# - dobra o número de exemplos (~3.500 cestões contra ~1.800 corridas);
# - o cestão 2 pode usar o que se sabe do cestão 1 (seção 2.5).
#
# O carregamento 3 (gusa líquido) não entra: nunca gera carga alta.
#
# **Split 70/15/15, temporal, definido pelo professor.** As 70% corridas mais antigas são o
# treino, as 15% seguintes a validação e as 15% mais recentes o teste. Os dois cestões da mesma
# corrida ficam sempre do mesmo lado. Split aleatório vazaria informação do futuro (Guia §2.2).
# - **Treino + validação (o "dev"):** onde tudo é escolhido. Os filtros rodam só no treino; a
#   permutação mede na validação; a decisão final (seção 9) usa o walk-forward mensal no dev.
# - **Teste:** não é aberto nesta etapa (é aberto uma única vez, na seleção de algoritmos).

# %%
chave = ["cd_corrida", "nu_carregamento"]
inicio_cestao = cam.groupby(chave)["dt_hora_consumo"].min()
inicio_corrida = inicio_cestao.groupby(level=0).min().sort_values()
n_corr = len(inicio_corrida)
corte_val = inicio_corrida.iloc[int(n_corr * 0.70)]
corte_teste = inicio_corrida.iloc[int(n_corr * 0.85)]
conjunto_corrida = pd.Series(
    np.select([inicio_corrida < corte_val, inicio_corrida < corte_teste], ["treino", "validacao"], "teste"),
    index=inicio_corrida.index)
print(f"Validação a partir de {corte_val.tz_convert('America/Sao_Paulo'):%d/%m/%Y}, "
      f"teste a partir de {corte_teste.tz_convert('America/Sao_Paulo'):%d/%m/%Y}")
print(conjunto_corrida.value_counts())

# %% [markdown]
# ## 2. Feature engineering
#
# Transformar várias linhas de camadas em **uma linha por cestão** (Guia §1.2). As features
# estão em 5 grupos, com a intuição física de cada uma:
#
# | Grupo | Features | Intuição |
# |---|---|---|
# | 2.1 Massa e volume (Guia §3) | `peso_t_total`, `volume_m3_total`, `densidade_media`, `razao_vol_peso`, peso e proporção por classe, `rendimento_medio`, `energia_estimada_mwh`, `energia_por_ton` | quanto material e quanto espaço ele ocupa |
# | 2.2 Composição (Guia §4) | `num_materiais`, `num_camadas`, `num_baias`, `hhi_materiais`, `prop_<material>` | o quê e quanta variedade |
# | 2.3 Ordem, ciclos e posição (Guia §5) | `n_ciclos`, blocos, `fundo_*`, `topo_*`, peso e leve nos 30% de cima/baixo, `pos_media_pesado`, `violacao_regra` | **a sequência leve → pesado → misto → leve se repete várias vezes no cestão** (correção da EVCOMX) |
# | 2.4 Tempo (Guia §5.3) | `tempo_montagem_min`, `intervalo_medio_min`, `turno_A/B`, `hora`, `dia_semana`, `fim_de_semana` | ritmo e equipe |
# | 2.5 Interações e contexto (Guia §6) | `vol_x_prop_leve`, `peso_x_hhi`, `peso_por_ciclo_t`, `peso_leve_topo_t`, dados do cestão 1 no cestão 2, taxa recente de carga alta | combinações com sentido físico e "estado do forno" |

# %%
cam = cam.sort_values(chave + ["nu_camada"]).reset_index(drop=True)
g = cam.groupby(chave)
cam["ordem"] = g.cumcount()
cam["n_cam"] = g["nu_camada"].transform("size")
cam["pos_rel"] = cam["ordem"] / (cam["n_cam"] - 1).clip(lower=1)      # 0 = fundo, 1 = topo
feat = pd.DataFrame(index=inicio_cestao.index)

def por_cestao(serie, preencher=0):
    """Reindexa um resultado agregado por cestão para todas as linhas de `feat`."""
    return serie.reindex(feat.index).fillna(preencher)

# %% [markdown]
# ### 2.1 Massa e volume (Guia §3)

# %%
feat["peso_t_total"] = g["peso_t"].sum()
feat["volume_m3_total"] = g["volume_m3"].sum()
feat["ocupacao_cestao"] = feat["volume_m3_total"] / 78                  # capacidade: 78 m³
feat["densidade_media"] = feat["peso_t_total"] / feat["volume_m3_total"]
feat["razao_vol_peso"] = feat["volume_m3_total"] / feat["peso_t_total"]  # Guia pergunta: é 1/densidade? (sim)
feat["peso_max_camada_t"] = g["peso_t"].max()
feat["peso_medio_camada_t"] = g["peso_t"].mean()
for classe, k in [("Leve", "leve"), ("Misto", "misto"), ("Pesado", "pesado")]:
    m = cam["tp_material"] == classe
    feat[f"peso_{k}"] = por_cestao(cam[m].groupby(chave)["peso_t"].sum())
    feat[f"vol_{k}_m3"] = por_cestao(cam[m].groupby(chave)["volume_m3"].sum())
    feat[f"prop_peso_{k}"] = feat[f"peso_{k}"] / feat["peso_t_total"]
    feat[f"prop_vol_{k}"] = feat[f"vol_{k}_m3"] / feat["volume_m3_total"]
feat = feat.rename(columns={"prop_peso_leve": "prop_leve"})             # mesmo nome da EDA

feat["rendimento_medio"] = (cam["rendimento"] * cam["peso_t"]).groupby([cam.cd_corrida, cam.nu_carregamento]).sum() / feat["peso_t_total"]
feat["energia_estimada_mwh"] = (cam["energia_kwh_t"] * cam["peso_t"]).groupby([cam.cd_corrida, cam.nu_carregamento]).sum() / 1000
feat["energia_por_ton"] = feat["energia_estimada_mwh"] * 1000 / feat["peso_t_total"]   # kWh/t

# %% [markdown]
# ### 2.2 Composição (Guia §4)
#
# Proporção **em peso** de cada material. Materiais presentes em menos de 5% dos cestões
# viram `prop_outros`, para evitar colunas quase sempre zero (Guia §4.3). A frequência é
# calculada só com o treino (o critério é escolhido sem olhar validação/teste).

# %%
feat["num_camadas"] = g["nu_camada"].nunique()
feat["num_materiais"] = g["cd_codigo_material"].nunique()
feat["num_baias"] = g["cd_baia"].nunique()

peso_mat = cam.pivot_table(index=chave, columns="cd_codigo_material", values="peso_t",
                           aggfunc="sum", fill_value=0).reindex(feat.index).fillna(0)
prop_mat = peso_mat.div(peso_mat.sum(axis=1), axis=0)
feat["hhi_materiais"] = (prop_mat ** 2).sum(axis=1)

no_treino = feat.index.get_level_values("cd_corrida").map(conjunto_corrida) == "treino"
freq_mat = (peso_mat[no_treino] > 0).mean()
comuns = freq_mat[freq_mat >= 0.05].index
print("Materiais com coluna própria (>= 5% dos cestões do treino):", list(comuns))
print("Agrupados em 'outros':", list(freq_mat[freq_mat < 0.05].index))
for mat in comuns:
    feat[f"prop_{mat}"] = prop_mat[mat]
feat["prop_outros"] = prop_mat.drop(columns=comuns).sum(axis=1)

# %% [markdown]
# ### 2.3 Ordem, ciclos e posição das camadas (Guia §5)
#
# **Correção da EVCOMX:** a ordem leve → pesado → misto → leve **se repete em vários ciclos
# dentro do mesmo cestão**. Camadas seguidas da mesma classe formam um **bloco**; cada ciclo tem
# um bloco de pesado, então `n_ciclos` = nº de blocos de pesado. Nos dados o ciclo aparece tanto
# como L→P→M quanto L→M→P; `n_ciclos_LPM` conta só os que seguem exatamente L→P→M.
#
# Posição: os 30% de camadas de baixo e de cima (Guia §5.1) e a posição média dos pesados
# ponderada pelo peso. `violacao_regra` = pesado na primeira ou na última camada.

# %%
cam["novo_bloco"] = cam["tp_material"] != cam.groupby(chave)["tp_material"].shift()
cam["id_bloco"] = cam.groupby(chave)["novo_bloco"].cumsum()
blocos = (cam.groupby(chave + ["id_bloco"])
          .agg(classe=("tp_material", "first"), peso_t=("peso_t", "sum"), volume_m3=("volume_m3", "sum"))
          .reset_index())
feat["n_blocos"] = blocos.groupby(chave).size()
for classe, k in [("Leve", "leve"), ("Misto", "misto"), ("Pesado", "pesado")]:
    bc = blocos[blocos["classe"] == classe].groupby(chave)
    feat[f"n_blocos_{k}"] = por_cestao(bc.size())
    feat[f"max_bloco_{k}_m3"] = por_cestao(bc["volume_m3"].max())
feat["n_ciclos"] = feat.pop("n_blocos_pesado")
sigla = blocos["classe"].map({"Leve": "L", "Pesado": "P", "Misto": "M"})
seq_blocos = sigla.groupby([blocos.cd_corrida, blocos.nu_carregamento]).agg("".join)
feat["n_ciclos_LPM"] = seq_blocos.str.count("(?=LPM)")

primeira = g.head(1).set_index(chave)["tp_material"]
ultima = g.tail(1).set_index(chave)["tp_material"]
for classe, k in [("Leve", "leve"), ("Misto", "misto"), ("Pesado", "pesado")]:
    feat[f"fundo_{k}"] = (primeira == classe).astype(int)
    feat[f"topo_{k}"] = (ultima == classe).astype(int)
feat["violacao_regra"] = ((primeira == "Pesado") | (ultima == "Pesado")).astype(int)

topo30, base30 = cam["pos_rel"] >= 0.7, cam["pos_rel"] <= 0.3
eh_leve, eh_pesado = cam["tp_material"] == "Leve", cam["tp_material"] == "Pesado"
feat["peso_topo_30pct_t"] = por_cestao(cam[topo30].groupby(chave)["peso_t"].sum())
feat["peso_base_30pct_t"] = por_cestao(cam[base30].groupby(chave)["peso_t"].sum())
feat["peso_leve_topo_t"] = por_cestao(cam[topo30 & eh_leve].groupby(chave)["peso_t"].sum())
feat["prop_leve_topo"] = feat["peso_leve_topo_t"] / feat["peso_topo_30pct_t"]
feat["prop_leve_base"] = por_cestao(cam[base30 & eh_leve].groupby(chave)["peso_t"].sum()) / feat["peso_base_30pct_t"]

def pos_ponderada(mascara):
    d = cam[mascara]
    return (d["pos_rel"] * d["peso_t"]).groupby([d.cd_corrida, d.nu_carregamento]).sum() / d.groupby(chave)["peso_t"].sum()
feat["pos_media_pesado"] = pos_ponderada(eh_pesado).reindex(feat.index)   # NaN se não há pesado
feat["pos_media_leve"] = pos_ponderada(eh_leve).reindex(feat.index)

# %% [markdown]
# ### 2.4 Tempo (Guia §5.3)
#
# Turno no padrão da EVCOMX: A (06–14h), B (14–22h), C (22–06h, a referência).
#
# A data/hora em si (`inicio`) **não é feature**: com split temporal, todo valor da validação e
# do teste é maior que qualquer valor do treino, e o modelo só aprenderia "em que época estamos",
# o que não serve para o futuro. Ela fica no dataset só como metadado. Hora, dia da semana e
# turno se repetem e podem entrar.

# %%
ini_local = inicio_cestao.dt.tz_convert("America/Sao_Paulo")
fim_local = g["dt_hora_consumo"].max().dt.tz_convert("America/Sao_Paulo")
feat["inicio"] = ini_local                                               # metadado, não é feature
feat["tempo_montagem_min"] = (fim_local - ini_local).dt.total_seconds() / 60
feat["intervalo_medio_min"] = feat["tempo_montagem_min"] / (feat["num_camadas"] - 1).replace(0, np.nan)
feat["hora"] = ini_local.dt.hour
feat["turno_A"] = feat["hora"].between(6, 13).astype(int)
feat["turno_B"] = feat["hora"].between(14, 21).astype(int)
feat["dia_semana"] = ini_local.dt.dayofweek
feat["fim_de_semana"] = (feat["dia_semana"] >= 5).astype(int)

# %% [markdown]
# ### 2.5 Interações e contexto (Guia §6)
#
# **Interações** (poucas e com sentido físico, Guia §6.1):
# - `vol_x_prop_leve`: volume alto com muito leve;
# - `peso_x_hhi`: muito peso concentrado em poucos materiais;
# - `peso_por_ciclo_t`: quanto material entra por ciclo (ciclos "grossos" = camadas mal intercaladas).
#
# **Contexto do cestão 1 para o cestão 2** (Guia §6.2): quando o cestão 2 é despejado, o forno já
# contém o cestão 1, então os dados do cestão 1 são informação disponível. No cestão 1 essas
# colunas valem 0 e `eh_cestao_2` distingue os casos.
#
# **Histórico recente:** taxa de carga alta das últimas N corridas, com `shift(1)` para não usar a
# corrida atual (sem vazamento).
#
# **`carga_alta_c1` (se o cestão 1 teve carga alta)** é calculada aqui, mas só entra no conjunto na
# seção 4: o Guia pede para discutir o timing, e a EVCOMX confirmou em 31/out/2026 que o resultado do
# cestão 1 é conhecido quando o cestão 2 ainda pode ser mexido.

# %%
feat["vol_x_prop_leve"] = feat["volume_m3_total"] * feat["prop_leve"]
feat["peso_x_hhi"] = feat["peso_t_total"] * feat["hhi_materiais"]
feat["peso_por_ciclo_t"] = feat["peso_t_total"] / feat["n_ciclos"].clip(lower=1)

feat = feat.reset_index()
feat["eh_cestao_2"] = (feat["nu_carregamento"] == 2).astype(int)
c1 = feat[feat.nu_carregamento == 1].set_index("cd_corrida")
eh2 = feat["eh_cestao_2"] == 1
feat["volume_c1_m3"] = np.where(eh2, feat["cd_corrida"].map(c1["volume_m3_total"]), 0)
feat["peso_c1_t"] = np.where(eh2, feat["cd_corrida"].map(c1["peso_t_total"]), 0)
feat["vol_acumulado_m3"] = feat["volume_m3_total"] + feat["volume_c1_m3"].fillna(0)
feat["delta_peso_12_t"] = np.where(eh2, feat["peso_t_total"] - feat["peso_c1_t"], 0)
mat_c1 = cam[cam.nu_carregamento == 1].groupby("cd_corrida")["cd_codigo_material"].agg(frozenset)
mat_c2 = cam[cam.nu_carregamento == 2].groupby("cd_corrida")["cd_codigo_material"].agg(frozenset)
feat["mudou_composicao"] = np.where(eh2, (feat["cd_corrida"].map(mat_c1) != feat["cd_corrida"].map(mat_c2)).astype(int), 0)

# histórico: taxa de carga alta das N corridas anteriores (ordem cronológica, shift(1))
hist = resposta[["cd_corrida", "is_carga_alta"]].assign(t=lambda d: d.cd_corrida.map(inicio_corrida)).sort_values("t")
for n in [5, 10, 20]:
    hist[f"taxa_ca_ult{n}"] = hist["is_carga_alta"].shift(1).rolling(n, min_periods=1).mean()
for n in [5, 10, 20]:
    feat[f"taxa_ca_ult{n}"] = feat["cd_corrida"].map(hist.set_index("cd_corrida")[f"taxa_ca_ult{n}"])

# conferência: as 8 features que já existiam na EDA batem com o dataset_modelagem do grupo?
conf = feat.merge(dataset_eda, on=chave, suffixes=("", "_eda"))
for c in ["num_camadas", "num_materiais", "peso_leve", "peso_misto", "peso_pesado",
          "peso_t_total", "volume_m3_total", "prop_leve"]:
    print(f"{c:16s} diferença máxima para a EDA: {(conf[c] - conf[c + '_eda']).abs().max():.6f}")
print("Tabela de features:", feat.shape)

# %% [markdown]
# ### Juntar com a variável resposta e com o split

# %%
alvo = pd.concat([
    resposta.assign(nu_carregamento=1.0, carga_alta=resposta.is_carga_alta_carregamento_1,
                    segundos_parada=resposta.qt_segundos_duracao_parada_carregamento_1),
    resposta.assign(nu_carregamento=2.0, carga_alta=resposta.is_carga_alta_carregamento_2,
                    segundos_parada=resposta.qt_segundos_duracao_parada_carregamento_2),
])[chave + ["carga_alta", "segundos_parada"]]
c1_alvo = alvo[alvo.nu_carregamento == 1].set_index("cd_corrida")["carga_alta"]

df = feat.merge(alvo, on=chave, how="inner")
df["conjunto"] = df["cd_corrida"].map(conjunto_corrida)
df["carga_alta_c1"] = np.where(df.eh_cestao_2 == 1, df["cd_corrida"].map(c1_alvo), 0)   # entra na seção 4
print("Cestões:", len(df), "| sem camadas do cestão 2 mas com carga alta nele (fora):",
      int(alvo[(alvo.nu_carregamento == 2) & (alvo.carga_alta == 1)]
          .merge(feat[chave], on=chave, how="left", indicator=True)["_merge"].eq("left_only").sum()))
print(df.groupby("conjunto").agg(cestoes=("carga_alta", "size"), taxa_carga_alta=("carga_alta", "mean"),
                                 de=("inicio", "min"), ate=("inicio", "max")).loc[["treino", "validacao", "teste"]])

treino = df[df.conjunto == "treino"].reset_index(drop=True)
valid = df[df.conjunto == "validacao"].reset_index(drop=True)
teste = df[df.conjunto == "teste"].reset_index(drop=True)

# %% [markdown]
# ## 3. Seleção na mão (slides 13–16)
#
# Perguntas: faz sentido? Vou ter na hora de prever? É confiável? Repete outra? Tudo calculado
# **só no treino**.

# %%
metadados = ["cd_corrida", "nu_carregamento", "carga_alta", "conjunto", "inicio", "carga_alta_c1"]
descartes = {
    "segundos_parada": "vazamento: a duração da parada só existe depois da carga alta",
}
candidatas = [c for c in df.columns if c not in metadados and c not in descartes]
for c in candidatas:
    moda = treino[c].value_counts(normalize=True, dropna=False).iloc[0]
    if treino[c].nunique(dropna=False) <= 1:
        descartes[c] = "constante"
    elif moda > 0.99:
        descartes[c] = f"quase constante ({moda:.1%} igual)"
    elif treino[c].isna().mean() > 0.5:
        descartes[c] = f"quase vazia ({treino[c].isna().mean():.0%} nulo)"
candidatas = [c for c in candidatas if c not in descartes]
print(pd.Series(descartes).to_string())
print(f"\nSobraram {len(candidatas)} candidatas")
print("Com vazios no treino (% ):", (treino[candidatas].isna().mean() * 100).loc[lambda s: s > 0].round(1).to_dict())

# %% [markdown]
# ### 3b. As 84 candidatas da engenharia de features
#
# A partir daqui, as features "da receita" estão prontas. As 84 vão para o
# `dataset_features_completo.csv` junto com as novas da seção 4. Antes disso, passam por um CSV em
# memória: é o mesmo arredondamento do arquivo que gerou o modelo oficial, então o modelo sai
# **idêntico** (a seção 12 confere).

# %%
meta = ["cd_corrida", "nu_carregamento", "conjunto", "inicio"]
_buf = io.StringIO()
df[meta + candidatas + ["carga_alta_c1", "carga_alta"]].to_csv(_buf, index=False)
_buf.seek(0)
d = pd.read_csv(_buf)
print("84 candidatas:", len(candidatas) == 84, "| tabela:", d.shape)

# %% [markdown]
# ## 4. Features novas (depois do critério de minutos da EVCOMX, out/2026)
#
# Na rodada da Random Forest, o histórico era só `taxa_ca_ult5/10/20`. A otimização por minutos
# testou 168 configurações (registro em `otimizacao_minutos/experimentos/`) e ficaram **36 features
# novas** em quatro famílias, mais o resultado do cestão 1:
#
# | Família | Features | Por que | Quando está disponível |
# |---|---|---|---|
# | Histórico rico | `h_taxa3/5/10/20/50`, `h_ewm2/5/15`, `h_ult1`, `h_ult1_c1`, `h_ult2_c2`, `h_desde_ultima`, `h_min_parada10`, `h_taxa_8h/24h` | carga alta puxa carga alta; janelas curtas reagem rápido, longas são estáveis | quando a corrida anterior termina (`shift(1)`, nunca a atual) |
# | Gusa líquida | `gusl_t`, `tem_gusl`, `carga3_solida_t` | a gusa do carregamento 3 muda o espaço e o banho no forno | na programação da corrida |
# | Ritmo do forno | `gap_corrida_min`, `gap_c2ant_c1_min`, `h_n_corridas_8h/24h` | forno acelerado tem menos tempo de fusão entre cestões | no início da montagem |
# | Regime recente (`rel_`, `rel2_`) | a feature menos a média dos 100 cestões anteriores (ou dos 60 do mesmo número de cestão) | depois de maio/26 o perfil mudou; o que importa é o cestão ser "fofo para a época" | após a montagem (usa só features passadas, nunca o alvo) |
# | **Resultado do cestão 1** | `carga_alta_c1` (no cestão 1 vale 0; `eh_cestao_2` separa os casos) | o cestão 2 dá **menos** carga alta quando o 1 deu (10,6% × 19,2% no dev) | **a EVCOMX confirmou (31/out/2026)** que é conhecido quando o cestão 2 ainda pode ser mexido |
#
# Também se calculam versões `_lag2` do histórico (pulando a corrida anterior), para o caso de o
# resultado dela ainda não ser conhecido. A duração da parada (`min_parada`) vai junto **só para
# medir minutos; não é feature**.

# %%
b = pd.read_csv(arq_base)
r = resposta.copy()

# ---- alvo e duração da parada (duração NÃO é feature; só para medir minutos)
alvo_min = pd.concat([
    r.assign(nu_carregamento=1.0, seg=r.qt_segundos_duracao_parada_carregamento_1),
    r.assign(nu_carregamento=2.0, seg=r.qt_segundos_duracao_parada_carregamento_2)])[chave + ["seg"]]
d = d.merge(alvo_min, on=chave, how="left")
d["min_parada"] = d["seg"] / 60
d = d.drop(columns="seg")
d["t"] = pd.to_datetime(d["inicio"], utc=True)

# ---- tempos de montagem por cestão
b["t"] = pd.to_datetime(b["dt_hora_consumo"], utc=True, format="mixed")
tc = b[b.nu_carregamento < 3].groupby(chave)["t"].agg(t_ini="min", t_fim="max").reset_index()
d = d.merge(tc, on=chave, how="left")

# ---- gusa líquida (carregamento 3) da corrida
gu = b[(b.nu_carregamento == 3) & (b.cd_codigo_material == "GUSL")].groupby("cd_corrida")["peso_t"].sum()
d["gusl_t"] = d["cd_corrida"].map(gu).fillna(0)
d["tem_gusl"] = (d["gusl_t"] > 0).astype(int)
outro3 = b[(b.nu_carregamento == 3) & (b.cd_codigo_material != "GUSL")].groupby("cd_corrida")["peso_t"].sum()
d["carga3_solida_t"] = d["cd_corrida"].map(outro3).fillna(0)

# ---- ordem das corridas e histórico
corr = (d[d.nu_carregamento == 1][["cd_corrida", "t_ini", "t_fim"]]
        .rename(columns={"t_ini": "c1_ini", "t_fim": "c1_fim"}))
c2 = d[d.nu_carregamento == 2][["cd_corrida", "t_fim"]].rename(columns={"t_fim": "c2_fim"})
corr = corr.merge(c2, on="cd_corrida", how="left")
corr = corr.merge(r[["cd_corrida", "is_carga_alta", "is_carga_alta_carregamento_1",
                     "is_carga_alta_carregamento_2", "qt_segundos_parada_carga_alta"]], on="cd_corrida")
corr = corr.sort_values("c1_ini").reset_index(drop=True)
corr["ordem"] = np.arange(len(corr))

y_corr = corr["is_carga_alta"].astype(float)
yprev = y_corr.shift(1)          # corrida anterior
for lag_nome, yy in [("", yprev), ("_lag2", y_corr.shift(2))]:   # _lag2 = versão "segura": pula a corrida anterior
    for n in [3, 5, 10, 20, 50]:
        corr[f"h_taxa{n}{lag_nome}"] = yy.rolling(n, min_periods=1).mean()
    for hl in [2, 5, 15]:
        corr[f"h_ewm{hl}{lag_nome}"] = yy.ewm(halflife=hl, min_periods=1).mean()
corr["h_ult1"] = yprev
corr["h_ult2_c2"] = corr["is_carga_alta_carregamento_2"].shift(1)  # cestão 2 da corrida anterior
corr["h_ult1_c1"] = corr["is_carga_alta_carregamento_1"].shift(1)
cnt, ultimo = [], None                                              # corridas desde a última carga alta
for i, v in enumerate(y_corr):
    cnt.append(np.nan if ultimo is None else i - ultimo)
    if v == 1:
        ultimo = i
corr["h_desde_ultima"] = pd.Series(cnt).clip(upper=50)
corr["h_min_parada10"] = (corr["qt_segundos_parada_carga_alta"] / 60).shift(1).rolling(10, min_periods=1).sum()

# janela de tempo: taxa de carga alta nas corridas que COMEÇARAM nas 24 h / 8 h antes (excluindo a atual)
ts = corr["c1_ini"].values.astype("datetime64[s]").astype(np.int64)
yy = y_corr.values
for horas in [8, 24]:
    w_s = horas * 3600
    taxa, nn, j0 = [], [], 0
    for i in range(len(corr)):
        while ts[j0] < ts[i] - w_s:
            j0 += 1
        seg = yy[j0:i]
        taxa.append(seg.mean() if len(seg) else np.nan)
        nn.append(len(seg))
    corr[f"h_taxa_{horas}h"] = taxa
    corr[f"h_n_corridas_{horas}h"] = nn

# ritmo do forno
corr["gap_corrida_min"] = (corr["c1_ini"] - corr["c1_ini"].shift(1)).dt.total_seconds() / 60
corr["gap_c2ant_c1_min"] = (corr["c1_ini"] - corr["c2_fim"].shift(1)).dt.total_seconds() / 60

hcols = [c for c in corr.columns if c.startswith("h_")] + ["gap_corrida_min", "gap_c2ant_c1_min", "ordem"]
d = d.merge(corr[["cd_corrida"] + hcols], on="cd_corrida", how="left")

# ---- composição relativa ao "regime" recente (média móvel das features dos 100 cestões anteriores)
d = d.sort_values(["t_ini"]).reset_index(drop=True)
for c in ["densidade_media", "prop_peso_pesado", "peso_por_ciclo_t", "peso_t_total", "volume_m3_total",
          "prop_leve", "n_ciclos"]:
    m = d[c].shift(1).rolling(100, min_periods=20).mean()
    d[f"rel_{c}"] = d[c] - m
    m2 = d.groupby("nu_carregamento")[c].transform(lambda s: s.shift(1).rolling(60, min_periods=15).mean())
    d[f"rel2_{c}"] = d[c] - m2

d["mes"] = d["t"].dt.tz_convert("America/Sao_Paulo").dt.strftime("%Y-%m")
d = d.copy()

NOVAS = ["gusl_t", "tem_gusl", "carga3_solida_t", "h_taxa3", "h_taxa5", "h_taxa10", "h_taxa20", "h_taxa50",
         "h_ewm2", "h_ewm5", "h_ewm15", "h_ult1", "h_ult2_c2", "h_ult1_c1", "h_desde_ultima", "h_min_parada10",
         "h_taxa_8h", "h_n_corridas_8h", "h_taxa_24h", "h_n_corridas_24h", "gap_corrida_min", "gap_c2ant_c1_min",
         "rel_densidade_media", "rel2_densidade_media", "rel_prop_peso_pesado", "rel2_prop_peso_pesado",
         "rel_peso_por_ciclo_t", "rel2_peso_por_ciclo_t", "rel_peso_t_total", "rel2_peso_t_total",
         "rel_volume_m3_total", "rel2_volume_m3_total", "rel_prop_leve", "rel2_prop_leve", "rel_n_ciclos", "rel2_n_ciclos"]
F_PRINCIPAL = candidatas + NOVAS                                    # 120 (a ordem das colunas é a do modelo oficial)
F_OFICIAL = F_PRINCIPAL + ["carga_alta_c1"]                         # 121
HIST_LAG1 = [c for c in F_PRINCIPAL if c.startswith(("taxa_ca_ult", "h_")) and not c.startswith("h_n_corridas")] + \
            ["h_n_corridas_8h", "h_n_corridas_24h"]
F_LAG2 = [c for c in F_PRINCIPAL if c not in HIST_LAG1] + [f"h_taxa{n}_lag2" for n in [3, 5, 10, 20, 50]] + \
         [f"h_ewm{h}_lag2" for h in [2, 5, 15]]
_ant = RAIZ / "otimizacao_minutos" / "features_finais.json"
if _ant.exists():
    _j = json.load(open(_ant, encoding="utf-8"))
    assert _j["features_principal"] == F_PRINCIPAL and _j["features_lag2"] == F_LAG2, "listas diferentes da otimização"
print(f"{len(NOVAS)} novas | principal {len(F_PRINCIPAL)} | oficial {len(F_OFICIAL)} | lag2 {len(F_LAG2)} | tabela {d.shape}")

c2d = d[d.eh_cestao_2 == 1]
for parte, nome in [(c2d[c2d.conjunto != "teste"], "dev")]:
    print(f"Cestão 2 no {nome}: carga alta {parte[parte.carga_alta_c1 == 1].carga_alta.mean():.1%} quando o cestão 1 deu, "
          f"{parte[parte.carga_alta_c1 == 0].carga_alta.mean():.1%} quando não deu")

# %% [markdown]
# ## 5. Grupos de features
#
# Com 121 colunas, a pergunta útil não é "qual coluna", e sim **"qual tipo de informação"**. Cada
# feature cai em exatamente um grupo; a seleção (seções 8 e 9) e a interpretação trabalham com eles.

# %%
def _grupo(c):
    if c == "carga_alta_c1":
        return "resultado do cestão 1"
    if c in ("gusl_t", "tem_gusl", "carga3_solida_t"):
        return "gusa líquida"
    if c in ("gap_corrida_min", "gap_c2ant_c1_min", "h_n_corridas_8h", "h_n_corridas_24h"):
        return "ritmo do forno"
    if c.startswith(("taxa_ca_ult", "h_")):
        return "histórico do forno"
    if c.startswith(("rel_", "rel2_")):
        return "regime recente (rel_)"
    if c in ("eh_cestao_2", "volume_c1_m3", "peso_c1_t", "vol_acumulado_m3", "delta_peso_12_t", "mudou_composicao"):
        return "contexto do cestão 1 (receita)"
    if c in ("tempo_montagem_min", "intervalo_medio_min", "hora", "turno_A", "turno_B", "dia_semana", "fim_de_semana"):
        return "tempo e turno"
    if c.startswith("prop_") and c not in ("prop_leve", "prop_vol_leve", "prop_peso_misto", "prop_vol_misto",
                                           "prop_peso_pesado", "prop_vol_pesado", "prop_leve_topo", "prop_leve_base"):
        return "materiais (prop_<código>)"
    if c in ("num_materiais", "num_baias", "hhi_materiais"):
        return "materiais (prop_<código>)"
    if c.startswith(("n_blocos", "max_bloco", "n_ciclos", "fundo_", "topo_", "pos_media", "peso_topo", "peso_base",
                     "peso_leve_topo", "prop_leve_topo", "prop_leve_base")) or c in ("violacao_regra", "num_camadas",
                                                                                     "peso_por_ciclo_t"):
        return "ordem, ciclos e posição"
    return "massa, volume e densidade"


GRUPOS = {}
for c in F_OFICIAL:
    GRUPOS.setdefault(_grupo(c), []).append(c)
assert sum(len(v) for v in GRUPOS.values()) == len(F_OFICIAL) == 121
DISPONIVEL = {
    "massa, volume e densidade": "após a montagem do cestão (pesagem das camadas)",
    "materiais (prop_<código>)": "após a montagem do cestão",
    "ordem, ciclos e posição": "após a montagem do cestão (sequência das camadas)",
    "tempo e turno": "após a montagem do cestão",
    "contexto do cestão 1 (receita)": "no cestão 2, a receita do cestão 1 já é conhecida; no cestão 1 vale 0",
    "histórico do forno": "quando a corrida anterior termina (shift(1); versão _lag2 pula a anterior)",
    "gusa líquida": "na programação da corrida (carregamento 3)",
    "ritmo do forno": "no início da montagem (horários das corridas anteriores)",
    "regime recente (rel_)": "após a montagem (média das features dos cestões anteriores; nunca o alvo)",
    "resultado do cestão 1": "no cestão 2: confirmado pela EVCOMX em 31/out/2026; no cestão 1 vale 0",
}
print(pd.Series({g: len(v) for g, v in GRUPOS.items()}).to_string())

# %% [markdown]
# ## 6. Filtro: cada feature sozinha (slides 18–21, Guia §7.1)
#
# Só no treino. Correlação point-biserial, Mann-Whitney com Bonferroni e informação mútua, como na
# rodada anterior, agora para as 121. Também medimos a PR-AUC de cada feature sozinha: uma feature
# **"boa demais"** (PR-AUC isolada muito acima do resto) seria o primeiro sinal de vazamento
# (armadilha do Guia §3.2).

# %%
treino_d = d[d.conjunto == "treino"].reset_index(drop=True)
y_tr = treino_d["carga_alta"]
X_tr121 = treino_d[F_OFICIAL]
med = X_tr121.median()
linhas_f = []
for c in F_OFICIAL:
    x = X_tr121[c].fillna(med[c])
    r_, _ = pointbiserialr(y_tr, x) if x.nunique() > 1 else (0.0, 1.0)
    _, p_mw = mannwhitneyu(x[y_tr == 0], x[y_tr == 1])
    pr = average_precision_score(y_tr, x if r_ >= 0 else -x)
    linhas_f.append({"feature": c, "grupo": _grupo(c), "corr": r_, "p_bonferroni": min(p_mw * len(F_OFICIAL), 1),
                     "PR-AUC_sozinha_treino": pr})
filtro = pd.DataFrame(linhas_f).set_index("feature")
filtro["info_mutua"] = mutual_info_classif(X_tr121.fillna(med), y_tr, random_state=SEED)
filtro = filtro.sort_values("info_mutua", ascending=False)
print(f"Chute no treino: PR-AUC {y_tr.mean():.3f}. Melhor feature sozinha: "
      f"{filtro['PR-AUC_sozinha_treino'].idxmax()} ({filtro['PR-AUC_sozinha_treino'].max():.3f})")
print(f"Mann-Whitney significativo (Bonferroni): {(filtro.p_bonferroni < 0.05).sum()} de 121")
print(filtro.head(20).round(4).to_string())

CORES_G = {"histórico do forno": VERMELHO, "resultado do cestão 1": "#8e44ad", "gusa líquida": LARANJA,
           "ritmo do forno": "#b7950b", "regime recente (rel_)": VERDE, "massa, volume e densidade": AZUL,
           "materiais (prop_<código>)": "#5d6d7e", "ordem, ciclos e posição": "#1abc9c", "tempo e turno": CINZA,
           "contexto do cestão 1 (receita)": "#34495e"}
fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
top = filtro.head(25)
axes[0].barh(top.index[::-1], top["info_mutua"][::-1], color=[CORES_G[g] for g in top["grupo"][::-1]])
axes[0].set_title("Informação mútua no treino (top 25)\ncor = grupo", loc="left")
t2 = filtro.reindex(filtro["corr"].abs().sort_values(ascending=False).index).head(25)
axes[1].barh(t2.index[::-1], t2["corr"][::-1], color=[CORES_G[g] for g in t2["grupo"][::-1]])
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_title("Correlação com carga alta (top 25 em |r|)\n+ aumenta o risco", loc="left")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=v, label=k) for k, v in CORES_G.items()], loc="lower center", ncol=5, fontsize=8,
           frameon=False)
plt.tight_layout(rect=(0, 0.07, 1, 1))
plt.savefig(PASTA_GRAF / "01_filtro_121_features.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 7. Redundância (Guia §7.2)
#
# Há muita redundância estrutural (ex.: `h_taxa5` e `taxa_ca_ult5` são a mesma conta; `rel_` é a
# feature menos uma média). Na rodada da RF, a regra |r| > 0,90 + VIF < 10 tirava as repetidas.
# **Para o Gradient Boosting não cortamos**, por três motivos:
# 1. árvores não têm coeficientes para "inflar": entre duas colunas iguais, cada divisão escolhe uma;
# 2. a regra do VIF foi feita para modelos lineares (o próprio Guia associa o VIF à logística);
# 3. o teste que importa é o de desempenho: na seção 9, tirar grupos inteiros não melhora os
#    minutos nem a PR-AUC além do ruído.
# O custo é na interpretação: a importância de uma informação se divide entre suas cópias. Por isso
# a permutação (seção 8) e a interpretação (Aula 08) embaralham **grupos** juntos.

# %%
corr_abs = X_tr121.corr().abs()
pares = (corr_abs.where(np.triu(np.ones(corr_abs.shape, dtype=bool), k=1)).stack()
         .loc[lambda s: s > 0.90].sort_values(ascending=False))
print(f"Pares com |r| > 0,90 no treino: {len(pares)} (envolvendo {len(set(pares.index.get_level_values(0)) | set(pares.index.get_level_values(1)))} features)")
print(pares.head(15).round(3).to_string())

# %% [markdown]
# ## 8. Embedded: permutação por grupo na validação (Guia §8.1, passo 4 do Guia de Modelagem)
#
# O modelo oficial (5 HGB, 121 features) treina no **treino** e prevê a **validação**. Depois
# embaralhamos as linhas de **um grupo inteiro de colunas de uma vez** (assim as cópias não se
# "cobrem" umas às outras) e medimos a queda da PR-AUC. 20 repetições por grupo.
# Também a permutação feature a feature, para ver as 20 de maior queda.

# %%
valid_d = d[d.conjunto == "validacao"].reset_index(drop=True)
treinados_tv = [(fab(), cols) for fab, cols in familia(F_OFICIAL)]
for m, cols in treinados_tv:
    m.fit(treino_d[cols], y_tr)
def _p(X):
    return np.mean([m.predict_proba(X[cols])[:, 1] for m, cols in treinados_tv], axis=0)
y_va = valid_d["carga_alta"].values
base_va = average_precision_score(y_va, _p(valid_d))
rng = np.random.default_rng(SEED)
perm_g = {}
for gname, cols in GRUPOS.items():
    quedas = []
    for _ in range(20):
        Xp = valid_d.copy()
        Xp[cols] = Xp[cols].values[rng.permutation(len(Xp))]
        quedas.append(base_va - average_precision_score(y_va, _p(Xp)))
    perm_g[gname] = (np.mean(quedas), np.std(quedas))
perm_g = pd.DataFrame(perm_g, index=["queda_PR-AUC", "dp"]).T.sort_values("queda_PR-AUC", ascending=False)
perm_f = {}
for c in F_OFICIAL:
    quedas = []
    for _ in range(5):
        Xp = valid_d.copy(); Xp[c] = Xp[c].values[rng.permutation(len(Xp))]
        quedas.append(base_va - average_precision_score(y_va, _p(Xp)))
    perm_f[c] = np.mean(quedas)
perm_f = pd.Series(perm_f).sort_values(ascending=False)
print(f"PR-AUC na validação (treino → validação): {base_va:.3f} | chute {y_va.mean():.3f}")
print(perm_g.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].barh(perm_g.index[::-1], perm_g["queda_PR-AUC"][::-1], xerr=perm_g["dp"][::-1],
             color=[CORES_G[g] for g in perm_g.index[::-1]])
axes[0].axvline(0, color="k", lw=0.8)
axes[0].set_title("Embaralhando um GRUPO inteiro\nqueda da PR-AUC na validação", loc="left")
t = perm_f.head(20)
axes[1].barh(t.index[::-1], t[::-1], color=[CORES_G[_grupo(c)] for c in t.index[::-1]])
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_title("Embaralhando UMA feature (top 20)\na queda é menor: as cópias cobrem", loc="left")
plt.tight_layout(); plt.savefig(PASTA_GRAF / "02_permutacao_por_grupo.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 9. Wrapper: tirar um grupo de cada vez, no walk-forward (Guia §8.2 e §9.2)
#
# É a decisão de fato. Mesmo protocolo da escolha do modelo: **walk-forward mensal** no dev (para
# cada mês de nov/25 a jun/26, treina com tudo antes e prevê o mês) e **minutos perdidos de jan a
# jun/26**, com o limiar de cada mês vindo só dos meses anteriores (950 min de parada sem modelo).
# Algoritmo fixo: os 5 HGB do modelo oficial. Também entram:
# - as **6 features da votação antiga** (com os 5 HGB, e a RF antiga como referência);
# - as **84 candidatas** sem as features novas;
# - o oficial **+ as 3 features do Guia de Modelagem** (violação fundo/topo, colchão, volume no centro).
#
# **Regra:** um grupo sai só se tirá-lo melhorar **além do ruído** os minutos (mais de 30 min a menos)
# **e** a PR-AUC (+0,02). A faixa de ruído vem das 5 variantes de hiperparâmetro, que sozinhas já
# espalham 600–635 min e ±0,02 de PR-AUC.
#
# **Transparência:** esta ablação já tinha sido feita na rodada 2 da otimização (31/out/2026,
# `otimizacao_minutos/experimentos/`), com a conclusão "tudo dentro do ruído". Aqui ela é refeita
# de forma reproduzível, com a mesma conclusão.

# %%
ultimo_bloco = blocos.groupby(chave).tail(1).set_index(chave)
centro = ~topo30 & ~base30
extra = pd.DataFrame({
    "violacao_regra": ((primeira == "Pesado") | (ultima == "Pesado")).astype(int),
    "violacao_fundo_topo": ((primeira != "Leve") | (ultima != "Leve")).astype(int),
    "colchao_topo_m3": ultimo_bloco["volume_m3"].where(ultimo_bloco["classe"] == "Leve", 0),
    "prop_vol_centro": cam[centro].groupby(chave)["volume_m3"].sum().reindex(primeira.index).fillna(0)
                       / g["volume_m3"].sum(),
}).reset_index()
EXTRA = ["violacao_fundo_topo", "colchao_topo_m3", "prop_vol_centro"]

# Os modelos leem o dataset do CSV (seção 12); aqui o mesmo arredondamento é feito em memória,
# para os números serem idênticos aos da seleção de algoritmos.
_buf2 = io.StringIO(); d.to_csv(_buf2, index=False); _buf2.seek(0)
_, DEV, _ = carregar(_buf2)
DEV = DEV.merge(extra[chave + EXTRA], on=chave, how="left")
SEM_MODELO = DEV[DEV.mes.isin(MESES_CUSTO)].min_parada.sum()

conjuntos = {"OFICIAL (121)": F_OFICIAL, "sem o resultado do cestão 1 (120)": F_PRINCIPAL}
for gname, cols in GRUPOS.items():
    if gname != "resultado do cestão 1":
        conjuntos[f"sem {gname} (−{len(cols)})"] = [c for c in F_OFICIAL if c not in cols]
conjuntos["84 candidatas + cestão 1"] = candidatas + ["carga_alta_c1"]
conjuntos["6 da votação antiga + cestão 1"] = FINAL6 + ["carga_alta_c1"]
conjuntos["6 da votação antiga"] = FINAL6
conjuntos["OFICIAL + 3 do Guia de Modelagem"] = F_OFICIAL + EXTRA

abl = []
for nome, cols in conjuntos.items():
    w = oof_dev(familia(cols), DEV)
    abl.append({"conjunto": nome, "n_features": len(cols), **resumo_wf(w)})
    print(f"{nome:48s} PR-AUC WF {abl[-1]['PR-AUC WF']:.3f} | min jan–jun {abl[-1]['min WF jan-jun']:.0f}")
w = oof_dev([(RF6, FINAL6)], DEV)
abl.append({"conjunto": "Antigo: RF 6 features (plano A)", "n_features": 6, **resumo_wf(w)})
abl = pd.DataFrame(abl).set_index("conjunto")
ref = abl.loc["OFICIAL (121)"]
abl["Δ min vs oficial"] = abl["min WF jan-jun"] - ref["min WF jan-jun"]
abl["Δ PR-AUC vs oficial"] = abl["PR-AUC WF"] - ref["PR-AUC WF"]
abl["sem modelo jan-jun"] = SEM_MODELO
melhoram = abl[(abl["Δ min vs oficial"] < -30) & (abl["Δ PR-AUC vs oficial"] > 0.02) & abl.index.str.startswith("sem ")]
pioram = abl[(abl["Δ min vs oficial"] > 30) | (abl["Δ PR-AUC vs oficial"] < -0.02)]
print(f"\nSem modelo: {SEM_MODELO:.0f} min. Grupos cuja retirada melhora minutos E PR-AUC além do ruído: "
      f"{list(melhoram.index) or 'nenhum'}")
print("Conjuntos claramente piores que o oficial:", list(pioram.index))
abl.round(3)

# %%
o = abl.drop(index="OFICIAL (121)").sort_values("Δ min vs oficial")
fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
cor = [VERMELHO if v > 30 else (VERDE if v < -30 else CINZA) for v in o["Δ min vs oficial"]]
axes[0].barh(o.index[::-1], o["Δ min vs oficial"][::-1], color=cor[::-1])
axes[0].axvspan(-30, 30, color=CINZA, alpha=0.12, label="ruído (±30 min)")
axes[0].axvline(0, color="k", lw=0.8); axes[0].legend(fontsize=8, loc="lower right")
axes[0].set_title(f"Minutos a mais que o oficial (jan–jun/26)\noficial: {ref['min WF jan-jun']:.0f} de {SEM_MODELO:.0f} min",
                  loc="left")
cor = [VERMELHO if v < -0.02 else (VERDE if v > 0.02 else CINZA) for v in o["Δ PR-AUC vs oficial"]]
axes[1].barh(o.index[::-1], o["Δ PR-AUC vs oficial"][::-1], color=cor[::-1])
axes[1].axvspan(-0.02, 0.02, color=CINZA, alpha=0.12)
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_title(f"PR-AUC (walk-forward) − oficial\noficial: {ref['PR-AUC WF']:.3f}", loc="left")
plt.tight_layout(); plt.savefig(PASTA_GRAF / "03_ablacao_por_grupo_walkforward.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 10. Mudança de regime em maio/2026
#
# O treino vai até 25/04/2026; validação e teste caem num período em que o processo mudou. Por isso
# existem as features `rel_` (a composição relativa à época) e o walk-forward re-treina todo mês.

# %%
dm = df.copy()
dm["mes"] = dm["inicio"].dt.strftime("%Y-%m")
mensal = dm.groupby("mes").agg(cestoes=("carga_alta", "size"), taxa_carga_alta=("carga_alta", "mean"),
                               densidade_media=("densidade_media", "mean"),
                               prop_peso_pesado=("prop_peso_pesado", "mean"), ciclos_medios=("n_ciclos", "mean"),
                               corr_pesado_x_alvo=("prop_peso_pesado", lambda s: s.corr(dm.loc[s.index, "carga_alta"])))
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
axes[0].plot(mensal.index, mensal["prop_peso_pesado"], marker="o", color=VERMELHO, label="prop. peso pesado")
axes[0].plot(mensal.index, mensal["taxa_carga_alta"], marker="o", color=CINZA, label="taxa de carga alta")
axes[0].set_title("Composição muda, taxa de carga alta não", loc="left"); axes[0].legend(fontsize=8)
axes[1].plot(mensal.index, mensal["ciclos_medios"], marker="o", color=AZUL)
axes[1].set_title("Ciclos de montagem por cestão", loc="left")
axes[2].bar(mensal.index, mensal["corr_pesado_x_alvo"], color=AZUL); axes[2].axhline(0, color="k", lw=0.8)
axes[2].set_title("Correlação prop. pesado × carga alta", loc="left")
for ax in axes:
    ax.tick_params(axis="x", rotation=60)
    ax.axvspan("2026-04", "2026-08", color=LARANJA, alpha=0.08)
salvar("04_mudanca_de_regime")
mensal.round(3)

# %% [markdown]
# ## 11. Respostas às perguntas do Guia §1.3
#
# (Os números são calculados aqui; a interpretação está no README da pasta.)

# %%
print("1) Que tipo de informação descreve um cestão propenso a carga alta? (permutação por grupo, validação)")
for gname, v in perm_g["queda_PR-AUC"].items():
    print(f"   {gname:32s} queda PR-AUC {v:+.3f}")
print("\n2) Volume x peso — correlação com carga alta no treino:")
for c in ["volume_m3_total", "peso_t_total", "densidade_media", "prop_peso_pesado"]:
    print(f"   {c:18s} {filtro.loc[c, 'corr']:+.3f} | PR-AUC sozinha {filtro.loc[c, 'PR-AUC_sozinha_treino']:.3f}")
print(f"\n3) Ordem e ciclos agregam? Tirar o grupo no walk-forward: "
      f"Δ min {abl.loc[[i for i in abl.index if i.startswith('sem ordem')][0], 'Δ min vs oficial']:+.0f}, "
      f"Δ PR-AUC {abl.loc[[i for i in abl.index if i.startswith('sem ordem')][0], 'Δ PR-AUC vs oficial']:+.3f}")
print("\n4) Que conjunto rende melhor fora da amostra (walk-forward)?")
print(abl[["n_features", "PR-AUC WF", "min WF jan-jun"]].sort_values("min WF jan-jun").round(3).to_string())

# %% [markdown]
# ## 12. Arquivos para a modelagem
#
# - **`dataset_modelagem_fs.csv`**: o dataset de modelagem do modelo oficial. Metadados
#   (`cd_corrida`, `nu_carregamento`, `conjunto`, `inicio`, `t`, `t_ini`, `t_fim`, `mes`), as **121
#   features oficiais**, `carga_alta` e `min_parada` (**só para medir minutos; não é feature**);
# - `dataset_features_completo.csv`: tudo o que foi calculado (inclui as versões `_lag2`);
# - `features_finais.json`: as listas de features (oficial, sem o cestão 1, lag2, as 6 antigas) e os grupos;
# - `tabela_features_finais.csv`: o entregável do Guia de Modelagem (§3.2), feature a feature;
# - `grupos_features.csv`: resumo por grupo (o que é, quando está disponível, permutação, ablação);
# - `filtro_features.csv`, `ablacao_grupos_walkforward.csv`, `features_guia_modelagem.csv`;
# - `graficos/`.

# %%
meta_out = ["cd_corrida", "nu_carregamento", "conjunto", "inicio", "t", "t_ini", "t_fim", "mes"]
d.to_csv(PASTA_SAIDA / "dataset_features_completo.csv", index=False)
d[meta_out + F_OFICIAL + ["carga_alta", "min_parada"]].to_csv(PASTA_SAIDA / "dataset_modelagem_fs.csv", index=False)
json.dump({"features_oficial": F_OFICIAL, "features_principal": F_PRINCIPAL, "features_lag2": F_LAG2,
           "features_rf6_antigo": FINAL6, "grupos": GRUPOS,
           "variantes_hgb": VARIANTES},
          open(PASTA_SAIDA / "features_finais.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
filtro.to_csv(PASTA_SAIDA / "filtro_features.csv")
abl.to_csv(PASTA_SAIDA / "ablacao_grupos_walkforward.csv")
extra.to_csv(PASTA_SAIDA / "features_guia_modelagem.csv", index=False)

DESC_GRUPO = {
    "massa, volume e densidade": "peso, volume estimado (peso ÷ densidade do material), densidade média, peso/volume por classe, rendimento, energia típica e interações",
    "materiais (prop_<código>)": "proporção em peso de cada material comum (≥ 5% dos cestões do treino), nº de materiais e baias, concentração (HHI)",
    "ordem, ciclos e posição": "blocos e ciclos leve→pesado→misto, maior bloco, o que está no fundo/topo e nos 30% de baixo/cima, posição média",
    "tempo e turno": "tempo de montagem, intervalo entre camadas, hora, turno, dia da semana",
    "contexto do cestão 1 (receita)": "no cestão 2: volume e peso do cestão 1, volume acumulado, diferença de peso, se a composição mudou",
    "histórico do forno": "taxa de carga alta nas últimas 3–50 corridas, médias exponenciais, última corrida e seus cestões, corridas desde a última carga alta, minutos parados nas últimas 10, taxa nas últimas 8 h e 24 h",
    "gusa líquida": "toneladas de gusa líquida (GUSL) e de carga sólida no carregamento 3 da corrida",
    "ritmo do forno": "minutos desde o início da corrida anterior e desde o fim do cestão 2 dela; nº de corridas nas últimas 8 h e 24 h",
    "regime recente (rel_)": "a feature menos a média dos 100 cestões anteriores (rel_) ou dos 60 do mesmo número de cestão (rel2_)",
    "resultado do cestão 1": "no cestão 2: 1 se o cestão 1 da mesma corrida deu carga alta; no cestão 1 vale 0",
}
POR_QUE = {
    "massa, volume e densidade": "cestão 'fofo' (pouca tonelada por m³) ocupa mais altura para o mesmo peso",
    "materiais (prop_<código>)": "alguns materiais são volumosos para o peso (ex.: pacotes de chaparia)",
    "ordem, ciclos e posição": "camadas mal intercaladas deixam leve acumulado sem pesado em cima",
    "tempo e turno": "ritmo e equipe da montagem",
    "contexto do cestão 1 (receita)": "o cestão 2 entra num forno que já contém o cestão 1",
    "histórico do forno": "carga alta puxa carga alta: é o sinal mais forte do problema",
    "gusa líquida": "a gusa muda o espaço e o banho no forno; corridas sem gusa quase não dão carga alta",
    "ritmo do forno": "forno acelerado tem menos tempo de fusão entre cestões",
    "regime recente (rel_)": "depois de maio/26 o perfil dos cestões mudou; conta ser fofo 'para a época'",
    "resultado do cestão 1": "quando o cestão 1 deu carga alta, o cestão 2 dá menos (a parada ajuda a fundir, ou o pátio corrige)",
}
tabela = pd.DataFrame([{
    "feature": c, "grupo": _grupo(c), "como_e_calculada": DESC_GRUPO[_grupo(c)], "por_que_entra": POR_QUE[_grupo(c)],
    "quando_esta_disponivel": DISPONIVEL[_grupo(c)], "sinal (corr. no treino)": f"{filtro.loc[c, 'corr']:+.2f}",
    "info_mutua_treino": round(filtro.loc[c, "info_mutua"], 4),
    "queda_PR-AUC_permutacao_validacao": round(perm_f[c], 4)} for c in F_OFICIAL])
tabela.to_csv(PASTA_SAIDA / "tabela_features_finais.csv", index=False)
resumo_g = pd.DataFrame([{
    "grupo": gname, "n_features": len(cols), "o_que_e": DESC_GRUPO[gname], "quando_esta_disponivel": DISPONIVEL[gname],
    "queda_PR-AUC_permutacao_grupo": round(perm_g.loc[gname, "queda_PR-AUC"], 3),
    "Δ_min_sem_o_grupo_WF": round(abl.loc[[i for i in abl.index if i.startswith(f"sem {gname}") or
                                          (gname == "resultado do cestão 1" and i.startswith("sem o resultado"))][0],
                                  "Δ min vs oficial"], 1),
    "Δ_PR-AUC_sem_o_grupo_WF": round(abl.loc[[i for i in abl.index if i.startswith(f"sem {gname}") or
                                             (gname == "resultado do cestão 1" and i.startswith("sem o resultado"))][0],
                                     "Δ PR-AUC vs oficial"], 3)} for gname, cols in GRUPOS.items()])
resumo_g.to_csv(PASTA_SAIDA / "grupos_features.csv", index=False)
print(resumo_g.drop(columns=["o_que_e"]).to_string(index=False))

# Conferência: o dataset é o mesmo que gerou o modelo oficial na otimização (out/2026)?
antigo = RAIZ / "otimizacao_minutos" / "dataset_minutos.csv"
if antigo.exists():
    a = pd.read_csv(antigo); n = pd.read_csv(PASTA_SAIDA / "dataset_features_completo.csv")
    iguais = list(a.columns) == list(n.columns) and a.shape == n.shape and \
        all(a[c].equals(n[c]) or np.allclose(a[c], n[c], equal_nan=True, rtol=0, atol=0)
            if a[c].dtype.kind in "fi" else a[c].astype(str).equals(n[c].astype(str)) for c in a.columns)
    print("Dataset idêntico ao da otimização por minutos:", iguais)
print("Salvos em", PASTA_SAIDA)
