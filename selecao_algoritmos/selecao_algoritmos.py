# %% [markdown]
# # Seleção de Algoritmos e Modelagem — Predição de Carga Alta (EVCOMX)
#
# Segue a **Aula 05 do bootcamp** (slides citados como "slide N") e o Guia de Modelagem da EVCOMX
# (§4), e continua `feature_selection/`.
#
# **Algoritmo oficial: média das probabilidades de 5 `HistGradientBoostingClassifier`** com
# hiperparâmetros levemente diferentes, sem pesos de classe, com as 121 features oficiais
# (120 + `carga_alta_c1`, o resultado do cestão 1).
#
# > **Antes (out/2026), o algoritmo escolhido era uma Random Forest com 6 features** (plano A),
# > escolhida pela PR-AUC na validação. Ela está guardada em `historico_rf6/`. A troca veio quando
# > a EVCOMX deu o critério de negócio: **o melhor modelo é o que faz perder menos minutos de forno**
# > (carga alta não detectada = a duração da parada; alarme falso = 1 min). A RF continua aqui como
# > linha de comparação.
#
# | Passo da aula (slide 36) | Seção |
# |---|---|
# | 1. Regressão ou classificação? | 1 |
# | 2. Métrica pelo custo do erro (minutos, com a PR-AUC junto) | 2 |
# | 3. Separação dos dados e baselines | 3 e 4 |
# | 4. Pipeline e algoritmos | 5 |
# | 5. Comparação justa: walk-forward mensal no dev | 6 |
# | Hiperparâmetros e por que a média de 5 (slide 24) | 7 |
# | Desbalanceamento: pesos × sem peso (Guia §4.4) | 7b |
# | 6. A escada e a regra de escolha declarada antes (slide 33) | 8 |
# | Limiar de decisão pelo custo (slide 16) | 9 |
# | 7. Conferir no teste, uma única vez (slide 35) | 10 |
# | Modelo para uso e arquivos | 11 |
#
# **Rode antes** `EDA/transforma_dataset.py` e `feature_selection/feature_selection.py`.
# Leva ~15 min (cada algoritmo treina 8 vezes no walk-forward).

# %%
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "comum").is_dir())
sys.path.insert(0, str(RAIZ))
from comum.protocolo import (ARQ_COMPLETO, C_FP, FINAL6, MESES_CUSTO, MESES_WF, NOME_OFICIAL, NOME_RESERVA,
                             NOME_REGRA, NOME_RF6, RF6, VARIANTES, HGB, carregar, familia, ler_features,
                             melhor_limiar, minutos, minutos_wf, oof_dev, prever, resumo_wf)

PASTA = RAIZ / "selecao_algoritmos"
(PASTA / "graficos").mkdir(exist_ok=True)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6, "axes.axisbelow": True})
AZUL, VERMELHO, CINZA, LARANJA, VERDE = "#2b6cb0", "#c0392b", "#7f8c8d", "#dd8a2e", "#4a8c4a"

# %% [markdown]
# ## 0. Ler os arquivos da feature selection
#
# - `dataset_features_completo.csv`: as 121 features oficiais e as alternativas (versões `_lag2`),
#   o split (`conjunto`), o mês (`mes`), o alvo e a duração da parada (`min_parada`, só para medir);
# - `features_finais.json`: as listas de features e as 5 variantes de hiperparâmetro.

# %%
FEAT = ler_features()
F_OF, F_PRINC, F_LAG2 = FEAT["features_oficial"], FEAT["features_principal"], FEAT["features_lag2"]
D, DEV, TESTE = carregar(ARQ_COMPLETO)
print("scikit-learn", sklearn.__version__, "| dev:", DEV.shape, "| teste:", TESTE.shape,
      "| features oficiais:", len(F_OF))
print(D.groupby("conjunto").agg(cestoes=("carga_alta", "size"), taxa=("carga_alta", "mean"),
                                de=("t_ini", "min"), ate=("t_ini", "max")).loc[["treino", "validacao", "teste"]])

# %% [markdown]
# ## 1. Que tipo de pergunta? (slides 5 e 6)
#
# "Este cestão vai dar carga alta?" é sim ou não: **classificação**. A duração da parada não é
# prevista (na otimização, nenhuma feature tem |Spearman| > 0,1 com ela); ela entra só como o
# **peso** de cada carga alta perdida na régua de minutos.
#
# ## 2. A métrica, escolhida pelo custo do erro (slides 32 e 34)
#
# Resposta da EVCOMX (Guima, out/2026): **o melhor modelo é o que faz perder menos minutos de forno.**
#
# | Situação | Minutos perdidos |
# |---|---|
# | Carga alta sem alarme (FN) | a duração real daquela parada |
# | Alarme falso (FP) | 1 min (reajustar a montagem) |
# | Alarme certo (TP) | 0 min no cenário principal; 1 min no cenário "TP=1" |
# | Sem modelo | a soma de todas as paradas |
#
# - **Métrica de escolha: minutos perdidos** no walk-forward (seção 6).
# - **PR-AUC junto**, como a métrica principal do grupo: ela mede a ordem dos cestões sem depender do
#   limiar. Acurácia engana (quem sempre diz "normal" acerta ~83%).
# - Com uma parada média de ~4 min e alarme de 1 min, compensa alarmar quando a chance passa de
#   ~20%: **o limiar não é 0,5** (seção 9).

# %%
par = DEV[DEV.carga_alta == 1].min_parada
print(f"Paradas no dev: média {par.mean():.1f} min, mediana {par.median():.1f} min. "
      f"Ponto de equilíbrio aproximado: p > {C_FP / (C_FP + par.mean()):.2f}")

# %% [markdown]
# ## 3. Separação dos dados
#
# - **Split 70/15/15 temporal por corrida** (professor), da coluna `conjunto`.
# - **Dev = treino + validação** (ago/25 a jun/26). Toda escolha usa só o dev.
# - **Walk-forward mensal** (adaptação do slide 30 para dados no tempo): para cada mês de nov/25 a
#   jun/26, o modelo treina com tudo o que veio antes e prevê o mês. É mais exigente que o `cv=5`
#   da aula, que sorteia linhas e treinaria com o futuro. Os minutos contam de jan a jun/26
#   (nov e dez só "aquecem" o limiar, que em cada mês vem só dos meses anteriores).
# - **A prova de sempre:** treina no treino e prevê a validação (PR-AUC).
# - **Teste (24/jun a 4/ago/26): aberto uma única vez**, na seção 10.

# %%
for mth in MESES_WF:
    n_tr = (DEV.t_ini < DEV[DEV.mes == mth].t_ini.min()).sum()
    print(f"walk-forward {mth}: treina com {n_tr:4d} cestões, prevê {(DEV.mes == mth).sum():3d}")
SEM_DEV = DEV[DEV.mes.isin(MESES_CUSTO)].min_parada.sum()
print(f"Sem modelo, jan–jun/26: {SEM_DEV:.0f} min de parada")

# %% [markdown]
# ## 4. Baselines: as réguas mínimas (slides 8 a 11, Guia §4.2)
#
# - **Sem modelo** (nunca alarmar): perde todas as paradas. **Maioria** (`DummyClassifier`): a mesma
#   chance para todos, PR-AUC = taxa de carga alta;
# - **física, regra de volume:** quanto maior o volume estimado, maior o risco (é o que a operação
#   faria sem ML; Guia §4.2);
# - **regra do histórico:** a taxa de carga alta nas 5 corridas anteriores. Sem modelo nenhum, é a
#   régua que de fato aperta (análise de modelos, rodada da RF);
# - **estatística:** a regressão logística (seção 5).

# %%
def wf_score(col):
    w = DEV[DEV.mes.isin(MESES_WF)].copy(); w["p"] = w[col].fillna(0).values
    return w


resultados, oofs = [], {}
for nome, col in [("Baseline: maioria (DummyClassifier)", None), ("Baseline físico: regra de volume", "volume_m3_total"),
                  (NOME_REGRA, "taxa_ca_ult5")]:
    if col is None:
        w = DEV[DEV.mes.isin(MESES_WF)].copy()
        w["p"] = DummyClassifier(strategy="prior").fit(DEV[F_OF], DEV.carga_alta).predict_proba(w[F_OF])[:, 1]
        pv = np.full((DEV.conjunto == "validacao").sum(), DEV[DEV.conjunto == "treino"].carga_alta.mean())
    else:
        w = wf_score(col); pv = DEV[DEV.conjunto == "validacao"][col].fillna(0).values
    oofs[nome] = w
    resultados.append({"modelo": nome, "familia": "baseline", **resumo_wf(w),
                       "PR-AUC validacao": average_precision_score(DEV[DEV.conjunto == "validacao"].carga_alta, pv)})
pd.DataFrame(resultados).round(3)

# %% [markdown]
# ## 5. Pipeline e algoritmos (slides 13 a 29, trajetória do Guia §4.3)
#
# Todos com as **121 features oficiais**, menos a referência antiga:
# - **regressão logística** (mediana para vazios + padronização);
# - **árvore de decisão** (rasa, `max_depth=5`);
# - **Random Forest** com as 121, e a **RF antiga com 6 features** (plano A, `class_weight="balanced_subsample"`);
# - **Extra Trees** (florestas com cortes sorteados);
# - **Gradient Boosting**: um `HistGradientBoostingClassifier` sozinho e a **média de 5** (o oficial).
#   O HGB lida com vazios sozinho e não precisa de padronização.

# %%
def LOGIT():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))


def ARVORE():
    return make_pipeline(SimpleImputer(strategy="median"),
                         DecisionTreeClassifier(max_depth=5, min_samples_leaf=40, random_state=42))


def RF121():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=42, n_jobs=-1))


def ET121():
    return make_pipeline(SimpleImputer(strategy="median"),
                         ExtraTreesClassifier(n_estimators=500, min_samples_leaf=5, random_state=42, n_jobs=-1))


ALGORITMOS = {
    "Regressão logística (121)": ("linear", [(LOGIT, F_OF)]),
    "Árvore de decisão rasa (121)": ("árvore", [(ARVORE, F_OF)]),
    "Random Forest (121)": ("floresta", [(RF121, F_OF)]),
    NOME_RF6: ("floresta", [(RF6, FINAL6)]),
    "Extra Trees (121)": ("floresta", [(ET121, F_OF)]),
    "Gradient Boosting, 1 modelo (121)": ("boosting", [(HGB(**VARIANTES["l2=5"]), F_OF)]),
    NOME_OFICIAL: ("boosting", familia(F_OF)),
    NOME_RESERVA: ("boosting", familia(F_PRINC)),
    "5 HGB, histórico sem a corrida anterior (lag2)": ("boosting", familia(F_LAG2)),
}

# %% [markdown]
# ## 6. Comparação justa: walk-forward mensal no dev (slide 30)
#
# Cada algoritmo treina 8 vezes (uma por mês). Para cada um: PR-AUC no walk-forward, minutos perdidos
# de jan a jun/26 (limiar de cada mês vindo dos meses anteriores), economia contra "sem modelo" e a
# PR-AUC na prova de sempre (treino → validação).

# %%
VAL = DEV[DEV.conjunto == "validacao"]
for nome, (fam, membros) in ALGORITMOS.items():
    w = oof_dev(membros, DEV); oofs[nome] = w
    pv = prever(membros, DEV[DEV.conjunto == "treino"], VAL)
    resultados.append({"modelo": nome, "familia": fam, **resumo_wf(w),
                       "PR-AUC validacao": average_precision_score(VAL.carga_alta, pv)})
    r = resultados[-1]
    print(f"{nome:50s} WF PR-AUC {r['PR-AUC WF']:.3f} | {r['min WF jan-jun']:.0f} min | val {r['PR-AUC validacao']:.3f}")
comp = pd.DataFrame(resultados).set_index("modelo")
comp["sem modelo jan-jun"] = SEM_DEV
comp.to_csv(PASTA / "comparacao_algoritmos.csv")
comp.round(3)

# %%
o = comp.sort_values("min WF jan-jun", ascending=False)
cores = [AZUL if i == NOME_OFICIAL else (LARANJA if i == NOME_RF6 else CINZA) for i in o.index]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
axes[0].barh(o.index, o["min WF jan-jun"], color=cores)
axes[0].axvline(SEM_DEV, ls="--", color=CINZA); axes[0].text(SEM_DEV, -0.9, f"sem modelo: {SEM_DEV:.0f}", ha="right", fontsize=8)
for i, v in enumerate(o["min WF jan-jun"]):
    axes[0].text(v + 5, i, f"{v:.0f}", va="center", fontsize=8)
axes[0].set_xlabel("minutos perdidos, jan–jun/26 (menor é melhor)")
axes[0].set_title("Minutos perdidos no walk-forward (dev)", loc="left")
axes[1].barh(o.index, o["PR-AUC WF"], color=cores)
for i, v in enumerate(o["PR-AUC WF"]):
    axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
axes[1].set_xlabel("PR-AUC no walk-forward (maior é melhor)")
axes[1].set_title("PR-AUC no walk-forward (dev)", loc="left")
fig.suptitle("Azul: oficial (5 Gradient Boosting + cestão 1). Laranja: RF 6 features (antigo plano A).", x=0.01, ha="left",
             fontsize=9, color="#52514e")
plt.tight_layout(); plt.savefig(PASTA / "graficos/01_algoritmos_minutos_e_prauc.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 7. Hiperparâmetros: por que a média de 5 (slide 24)
#
# Na otimização foram testadas 168 configurações (`otimizacao_minutos/experimentos/`). A família
# "Gradient Boosting com todas as features" deu os menores minutos em **todas** as variações de
# hiperparâmetro. Escolher a melhor delas seria otimista (o melhor de muitos testes tende a ter
# sorte). Por isso o oficial é a **média de 5 variações próximas** (`learning_rate` 0,03–0,05,
# 8–16 folhas, `min_samples_leaf` 20–40, `l2_regularization=5`): o resultado depende menos de um
# ajuste específico. Abaixo, cada variação sozinha e a média.

# %%
var = []
for k, kw in VARIANTES.items():
    w = oof_dev([(HGB(**kw), F_OF)], DEV)
    var.append({"variante": k, **{f"{a}": b for a, b in kw.items()}, **resumo_wf(w)})
var.append({"variante": "MÉDIA DAS 5 (oficial)", **resumo_wf(oofs[NOME_OFICIAL])})
var = pd.DataFrame(var).set_index("variante")
var.to_csv(PASTA / "variantes_hgb.csv")
print(var[["PR-AUC WF", "min WF jan-jun", "economia WF %"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.scatter(var["PR-AUC WF"][:-1], var["min WF jan-jun"][:-1], color=CINZA, s=40, label="uma variante")
ax.scatter(var["PR-AUC WF"][-1:], var["min WF jan-jun"][-1:], color=AZUL, s=90, marker="D", label="média das 5 (oficial)")
for k, r in var.iloc[:-1].iterrows():
    ax.annotate(k, (r["PR-AUC WF"], r["min WF jan-jun"]), fontsize=8, xytext=(4, 3), textcoords="offset points")
ax.axhline(comp.loc[NOME_RF6, "min WF jan-jun"], color=LARANJA, ls="--", lw=1, label="RF 6 features (antigo)")
ax.set_xlabel("PR-AUC no walk-forward"); ax.set_ylabel("minutos perdidos jan–jun")
ax.set_title("As 5 variações de hiperparâmetro e a média", loc="left"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(PASTA / "graficos/02_variantes_hgb.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 7b. Desbalanceamento: pesos de classe × sem peso (Guia de Modelagem §4.4)
#
# Só ~17% dos cestões têm carga alta. A RF antiga usava `class_weight="balanced_subsample"`, e por
# isso as chances saíam infladas (a análise precisava calibrar). O oficial **não usa pesos**:

# %%
desb = []
for nome, cw in [("sem peso (oficial)", None), ("pesos de classe (balanced)", "balanced")]:
    membros = [((lambda kw=kw, cw=cw: HGB(**kw)().set_params(class_weight=cw)), F_OF) for kw in VARIANTES.values()]
    w = oofs[NOME_OFICIAL] if cw is None else oof_dev(membros, DEV)
    desb.append({"estrategia": nome, **resumo_wf(w), "chance media WF": w.p.mean(), "taxa real WF": w.carga_alta.mean()})
desb = pd.DataFrame(desb).set_index("estrategia")
desb.to_csv(PASTA / "desbalanceamento.csv")
desb.round(3)

# %% [markdown]
# - **Na ordem dos cestões (PR-AUC), as duas estratégias ficam próximas.**
# - **A diferença está na escala da chance.** Sem peso, a chance média fica perto da taxa real, e o
#   limiar escolhido no passado continua valendo no modelo re-treinado. Com peso, a chance infla.
# - Não usamos SMOTE: ele cria cestões sintéticos que podem ser receitas impossíveis (Guia §4.4).
#
# ## 8. A escada e a regra de escolha (slide 33)
#
# **Regra 1, declarada antes de abrir o teste** (`otimizacao_minutos/finalistas_dev_pre_registro.csv`):
# entre os **finalistas** registrados, fica o de **menos minutos no walk-forward**, desde que a PR-AUC
# no walk-forward seja pelo menos a da RF do grupo. O resultado do cestão 1 entrou com a condição
# "se a EVCOMX confirmar o timing", e a EVCOMX confirmou (31/out/2026).
#
# **Regra 2, a escada (slide 33):** um degrau mais simples (linear < árvore < floresta < boosting)
# toma o lugar do escolhido se **empatar nas duas réguas** dentro do ruído: minutos a até 30 min
# **e** PR-AUC no walk-forward a até 0,02. As faixas de ruído vêm das 5 variantes de hiperparâmetro,
# que sozinhas espalham ~575–645 min e ±0,02 de PR-AUC (seção 7). "1 HGB" e "5 HGB" são o mesmo
# degrau: a média existe para não escolher a variante de mais sorte.

# %%
PRE_REGISTRO = [NOME_OFICIAL, NOME_RESERVA, "5 HGB, histórico sem a corrida anterior (lag2)", NOME_RF6]
fin = comp.loc[PRE_REGISTRO]
fin = fin[fin["PR-AUC WF"] >= comp.loc[NOME_RF6, "PR-AUC WF"]]
escolhido = fin["min WF jan-jun"].idxmin()
print("Regra 1 (finalistas pré-registrados):")
print(fin[["PR-AUC WF", "min WF jan-jun", "economia WF %"]].sort_values("min WF jan-jun").round(3).to_string())
print("→ escolhido:", escolhido)
assert escolhido == NOME_OFICIAL, "a regra declarada não aponta mais para o modelo oficial: revise antes de seguir"

DEGRAU = {"linear": 1, "árvore": 2, "floresta": 3, "boosting": 4}
ref = comp.loc[escolhido]
escada = comp[comp.familia != "baseline"].copy()
escada["degrau"] = escada.familia.map(DEGRAU)
escada["Δ min"] = escada["min WF jan-jun"] - ref["min WF jan-jun"]
escada["Δ PR-AUC WF"] = escada["PR-AUC WF"] - ref["PR-AUC WF"]
escada["empata nos minutos (±30)"] = escada["Δ min"].abs() <= 30
escada["empata na PR-AUC (±0,02)"] = escada["Δ PR-AUC WF"] >= -0.02
escada["toma o lugar?"] = (escada.degrau < DEGRAU["boosting"]) & escada["empata nos minutos (±30)"] & escada["empata na PR-AUC (±0,02)"]
escada.to_csv(PASTA / "escada_de_complexidade.csv")
print("\nRegra 2 (escada):")
print(escada[["familia", "min WF jan-jun", "Δ min", "PR-AUC WF", "Δ PR-AUC WF", "PR-AUC validacao", "toma o lugar?"]]
      .sort_values("min WF jan-jun").round(3).to_string())
assert not escada["toma o lugar?"].any(), "um modelo mais simples empata nas duas réguas: revise a escolha"

# %% [markdown]
# **Leitura da escada.**
# - **A regressão logística empata nos minutos** (a diferença está dentro do ruído), **mas perde na
#   ordem dos cestões**: PR-AUC 0,46 contra 0,50 no walk-forward e 0,35 contra 0,45 na validação.
#   Com o limiar de ~0,18, as duas acertam parecido "na média"; a logística erra mais quando o
#   orçamento de alarmes aperta (análise, seção 7). Ela não toma o lugar, mas é a **alternativa
#   simples** a reavaliar quando houver um lote novo de dados. Ela não é medida no teste: não era
#   finalista, e o teste não serve para escolher.
# - A árvore rasa e as florestas ficam bem atrás nos minutos.
# - O degrau "1 HGB → média de 5" quase não muda a média; ele existe para **não depender da sorte**
#   de uma variante (seção 7: a melhor variante sozinha teria 576 min, a pior 644).
#
# ## 9. O limiar de decisão pelo custo (slide 16)
#
# O limiar é o que **minimiza os minutos** nas previsões do walk-forward (dev). Dois cenários:
# alarme certo custa 0 min (principal) ou 1 min (TP=1, conservador). A curva abaixo mostra o
# custo do dev conforme o limiar.

# %%
w = oofs[NOME_OFICIAL]
lim0 = melhor_limiar(w.carga_alta, w.p, w.min_parada, C_FP, 0.0)
lim1 = melhor_limiar(w.carga_alta, w.p, w.min_parada, C_FP, 1.0)
grade = np.round(np.arange(0.02, 0.81, 0.01), 2)
curva = pd.DataFrame({"limiar": grade,
                      "min_dev_TP0": [minutos(w.carga_alta, w.min_parada, w.p > t) for t in grade],
                      "min_dev_TP1": [minutos(w.carga_alta, w.min_parada, w.p > t, C_FP, 1.0) for t in grade],
                      "alarmes_por_100": [100 * (w.p > t).mean() for t in grade],
                      "recall": [((w.p > t) & (w.carga_alta == 1)).sum() / w.carga_alta.sum() for t in grade]})
curva.to_csv(PASTA / "custo_por_limiar_dev.csv", index=False)
lims = pd.DataFrame([{"cenario": "alarme certo custa 0 min (principal)", "limiar": lim0},
                     {"cenario": "alarme certo também custa 1 min (TP=1)", "limiar": lim1}])
lims.to_csv(PASTA / "limiares_dev.csv", index=False)
print(lims.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.plot(curva.limiar, curva.min_dev_TP0, color=AZUL, label="alarme certo = 0 min")
ax.plot(curva.limiar, curva.min_dev_TP1, color=LARANJA, label="alarme certo = 1 min (TP=1)")
ax.axhline(w.min_parada[w.carga_alta == 1].sum(), color=CINZA, ls="--", lw=1, label="sem modelo (nov–jun)")
for lim, cor in [(lim0, AZUL), (lim1, LARANJA)]:
    ax.axvline(lim, color=cor, ls=":", lw=1.2)
ax.set_xlabel("limiar de alarme (chance prevista)"); ax.set_ylabel("minutos perdidos no dev (nov–jun)")
ax.set_title(f"Custo conforme o limiar: menor em {lim0:.2f} (TP=0) e {lim1:.2f} (TP=1), longe de 0,5", loc="left")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(PASTA / "graficos/03_custo_por_limiar_dev.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 10. Conferência no TESTE — uma única vez (slides 35 e 36)
#
# Cada finalista treina no **dev inteiro** e prevê o teste (24/jun a 4/ago/26: 515 cestões, 75
# cargas altas). O limiar vem do walk-forward do dev.
#
# **Transparência.** Este teste foi aberto uma vez na otimização por minutos (31/out/2026), só para
# os finalistas registrados antes (`otimizacao_minutos/finalistas_dev_pre_registro.csv`). O código
# aqui refaz exatamente a mesma conta, com os mesmos números. Os algoritmos que não eram finalistas
# (logística, árvore, florestas com 121, 1 HGB) **não são medidos no teste**: o teste não serve
# para escolher. O grupo já tinha visto este teste nas rodadas da RF; por isso nada foi escolhido
# olhando para ele.

# %%
FINALISTAS = [NOME_OFICIAL, NOME_RESERVA, "5 HGB, histórico sem a corrida anterior (lag2)", NOME_RF6, NOME_REGRA]
SEM_TESTE = TESTE.min_parada.sum()
y_te = TESTE.carga_alta.values.astype(bool)
teste_lin, prev_teste = [], {}
for nome in FINALISTAS:
    p = TESTE["taxa_ca_ult5"].fillna(0).values if nome == NOME_REGRA else prever(ALGORITMOS[nome][1], DEV, TESTE)
    prev_teste[nome] = p
    w = oofs[nome]
    lin = {"modelo": nome, "PR-AUC teste": average_precision_score(y_te, p), "ROC-AUC teste": roc_auc_score(y_te, p)}
    for c_tp, suf in [(0.0, ""), (1.0, " (TP=1)")]:
        lim = melhor_limiar(w.carga_alta, w.p, w.min_parada, C_FP, c_tp)
        a = p > lim
        lin[f"limiar{suf}"] = lim
        lin[f"min perdidos teste{suf}"] = minutos(y_te, TESTE.min_parada, a, C_FP, c_tp)
        if suf == "":
            lin.update({"alarmes/100": 100 * a.mean(), "recall": (a & y_te).sum() / y_te.sum(),
                        "precisão": (a & y_te).sum() / max(a.sum(), 1), "FN": int((~a & y_te).sum()),
                        "FP": int((a & ~y_te).sum())})
    teste_lin.append(lin)
res = pd.DataFrame(teste_lin).set_index("modelo")
res["min sem modelo (teste)"] = SEM_TESTE
res["economia teste %"] = 100 * (1 - res["min perdidos teste"] / SEM_TESTE)
res["economia teste % (TP=1)"] = 100 * (1 - res["min perdidos teste (TP=1)"] / SEM_TESTE)
res.to_csv(PASTA / "resultado_teste.csv")
print(f"Teste: {len(TESTE)} cestões, {y_te.sum()} cargas altas, {SEM_TESTE:.0f} min de parada sem modelo")
res.round(3)

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
o = res.sort_values("min perdidos teste")
ROT = {NOME_OFICIAL: "OFICIAL: 5 HGB + cestão 1", NOME_RESERVA: "Reserva: 5 HGB sem cestão 1",
       "5 HGB, histórico sem a corrida anterior (lag2)": "5 HGB, histórico lag2", NOME_RF6: "RF 6 features (antigo)",
       NOME_REGRA: "Regra do histórico (sem modelo)"}
axes[0].barh([ROT[i] for i in o.index], o["min perdidos teste"],
             color=[AZUL if i == NOME_OFICIAL else (LARANJA if i == NOME_RF6 else CINZA) for i in o.index])
axes[0].axvline(SEM_TESTE, color=CINZA, ls="--"); axes[0].invert_yaxis()
for i, (v, e) in enumerate(zip(o["min perdidos teste"], o["economia teste %"])):
    axes[0].text(v + 3, i, f"{v:.0f} min ({e:.0f}% a menos)", va="center", fontsize=8)
axes[0].set_xlim(0, SEM_TESTE * 1.35)
axes[0].set_title(f"Minutos perdidos no teste (sem modelo: {SEM_TESTE:.0f} min em 40 dias)", loc="left")
cm = confusion_matrix(y_te, prev_teste[NOME_OFICIAL] > res.loc[NOME_OFICIAL, "limiar"])
axes[1].imshow(cm, cmap="Blues"); axes[1].grid(False)
for (i, j), v in np.ndenumerate(cm):
    axes[1].text(j, i, v, ha="center", va="center", fontsize=13, color="white" if v > cm.max() / 2 else "black")
axes[1].set_xticks([0, 1], ["previsto: normal", "previsto: carga alta"]); axes[1].set_yticks([0, 1], ["real: normal", "real: carga alta"])
axes[1].set_title(f"Oficial no teste (limiar {res.loc[NOME_OFICIAL, 'limiar']:.2f})", loc="left")
plt.tight_layout(); plt.savefig(PASTA / "graficos/04_teste_minutos_e_matriz.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 11. Modelo para uso e arquivos gerados
#
# - **`modelo_final.joblib`**: dois pacotes re-treinados com **todos** os dados (treino + validação +
#   teste), cada um com os 5 modelos, as features e os limiares de menor minutos (calculados nas
#   previsões do walk-forward + teste, ou seja, sempre fora da amostra):
#   - `com_cestao1`: o **oficial**;
#   - `sem_cestao1_reserva`: a reserva, para quando o resultado do cestão 1 faltar.
# - `modelo_dev.joblib`: o oficial treinado só no dev (é o que a análise e a interpretação usam
#   para explicar o teste, que ele nunca viu);
# - `modelo_final.json`: receita, features e limiares em texto;
# - `previsoes_dev_teste.csv`: chance de cada cestão no walk-forward (dev) e no teste, para cada finalista;
# - `comparacao_algoritmos.csv`, `variantes_hgb.csv`, `desbalanceamento.csv`, `custo_por_limiar_dev.csv`,
#   `limiares_dev.csv`, `resultado_teste.csv`, `graficos/`.

# %%
TUDO = pd.concat([DEV, TESTE], ignore_index=True)


def empacotar(nome, features):
    modelos = []
    for fab, cols in ALGORITMOS[nome][1]:
        m = fab(); m.fit(TUDO[cols], TUDO["carga_alta"]); modelos.append(m)
    w = oofs[nome]
    pas_y = np.r_[w.carga_alta.values, TESTE.carga_alta.values]
    pas_p = np.r_[w.p.values, prev_teste[nome]]
    pas_d = np.r_[w.min_parada.values, TESTE.min_parada.values]
    return {"modelos": modelos, "features": features,
            "limiar_minutos_TP0": melhor_limiar(pas_y, pas_p, pas_d, C_FP, 0.0),
            "limiar_minutos_TP1": melhor_limiar(pas_y, pas_p, pas_d, C_FP, 1.0)}


pacote = {"com_cestao1": empacotar(NOME_OFICIAL, F_OF),
          "sem_cestao1_reserva": empacotar(NOME_RESERVA, F_PRINC),
          "como_usar": ("p = média de m.predict_proba(X[features])[:,1] para m em modelos; alarmar se p > limiar. "
                        "No cestão 2, carga_alta_c1 = resultado do cestão 1 da mesma corrida; no cestão 1, 0."),
          "versao_sklearn": sklearn.__version__}
joblib.dump(pacote, PASTA / "modelo_final.joblib")
dev_modelos = [(fab(), cols) for fab, cols in ALGORITMOS[NOME_OFICIAL][1]]
for m, cols in dev_modelos:
    m.fit(DEV[cols], DEV["carga_alta"])
joblib.dump({"modelos": [m for m, _ in dev_modelos], "features": F_OF, "limiar_minutos_TP0": res.loc[NOME_OFICIAL, "limiar"],
             "limiar_minutos_TP1": res.loc[NOME_OFICIAL, "limiar (TP=1)"], "treinado_em": "treino + validação (dev)",
             "versao_sklearn": sklearn.__version__}, PASTA / "modelo_dev.joblib")
json.dump({"modelo_oficial": {"descricao": "média das probabilidades de 5 HistGradientBoostingClassifier, sem pesos de classe",
                              "hiperparametros_base": {"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 8,
                                                       "min_samples_leaf": 40, "l2_regularization": 1.0, "random_state": 42},
                              "variantes": VARIANTES, "n_features": len(F_OF), "features": F_OF,
                              "limiar_minutos_TP0": pacote["com_cestao1"]["limiar_minutos_TP0"],
                              "limiar_minutos_TP1": pacote["com_cestao1"]["limiar_minutos_TP1"],
                              "limiar_do_dev_usado_no_teste": res.loc[NOME_OFICIAL, "limiar"]},
           "reserva_sem_cestao1": {"n_features": len(F_PRINC),
                                   "limiar_minutos_TP0": pacote["sem_cestao1_reserva"]["limiar_minutos_TP0"],
                                   "limiar_minutos_TP1": pacote["sem_cestao1_reserva"]["limiar_minutos_TP1"]},
           "treinado_em": "treino + validação + teste (todos os dados)", "versao_sklearn": sklearn.__version__,
           "antigo_planoA": "Random Forest, 6 features: ver historico_rf6/"},
          open(PASTA / "modelo_final.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

prev = []
for nome in FINALISTAS:
    w = oofs[nome]
    prev.append(pd.DataFrame({"cd_corrida": w.cd_corrida, "nu_carregamento": w.nu_carregamento, "conjunto": w.conjunto,
                              "mes": w.mes, "parte": "walk-forward (dev)", "modelo": nome, "p": w.p,
                              "carga_alta": w.carga_alta, "min_parada": w.min_parada}))
    prev.append(pd.DataFrame({"cd_corrida": TESTE.cd_corrida, "nu_carregamento": TESTE.nu_carregamento, "conjunto": "teste",
                              "mes": TESTE.mes, "parte": "teste", "modelo": nome, "p": prev_teste[nome],
                              "carga_alta": TESTE.carga_alta, "min_parada": TESTE.min_parada}))
pd.concat(prev).to_csv(PASTA / "previsoes_dev_teste.csv", index=False)
for k in ["com_cestao1", "sem_cestao1_reserva"]:
    print(k, "limiares (TP=0, TP=1):", round(pacote[k]["limiar_minutos_TP0"], 4), round(pacote[k]["limiar_minutos_TP1"], 4))

# conferência com o pacote da otimização (mesma receita, mesmos dados → mesmas chances)
antigo = RAIZ / "otimizacao_minutos" / "modelo_minutos.joblib"
if antigo.exists():
    pa = joblib.load(antigo)
    X = TUDO
    pn = np.mean([m.predict_proba(X[F_OF])[:, 1] for m in pacote["com_cestao1"]["modelos"]], axis=0)
    po = np.mean([m.predict_proba(X[pa["com_cestao1"]["features"]])[:, 1] for m in pa["com_cestao1"]["modelos"]], axis=0)
    print("Mesmas chances do modelo da otimização:", np.allclose(pn, po, atol=1e-12),
          "| mesmos limiares:", np.isclose(pacote["com_cestao1"]["limiar_minutos_TP0"], pa["com_cestao1"]["limiar_minutos_TP0"]))
