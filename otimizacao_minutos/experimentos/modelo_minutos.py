# %% [markdown]
# # Modelo de minutos: menos minutos de forno perdidos (etapa otimizacao_minutos/)
#
# **Pergunta de negócio (EVCOMX):** o melhor modelo é o que faz perder MENOS MINUTOS, não o que
# acha mais cargas altas.
# - carga alta não detectada (FN): perde a duração real daquela parada;
# - alarme falso (FP): perde 1 min (reajustar a montagem, resposta do Guima);
# - alarme certo (TP): 0 min no cenário principal; 1 min no cenário conservador ("TP=1"), em que
#   o reajuste também custa 1 min. Sem modelo, perde-se a soma de todas as paradas.
#
# **Protocolo honesto:**
# 1. Toda escolha (features, algoritmo, hiperparâmetros, limiar) usa só treino + validação (dev):
#    walk-forward mensal (para cada mês de nov/25 a jun/26, treina com tudo antes do mês e prevê o
#    mês) e a prova de sempre (treino → validação).
# 2. O limiar de alarme de cada mês é o de menor minutos nos meses ANTERIORES.
# 3. Os finalistas foram declarados (`finalistas_dev_pre_registro.csv`) antes de abrir o teste.
# 4. Teste (jun–ago/26) aberto uma única vez: treina no dev inteiro, limiar vindo do dev.
#
# Rodar antes: `python otimizacao_minutos/features_minutos.py`. Este script leva ~6 min.

# %%
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
D = pd.read_csv(AQUI / "dataset_minutos.csv")
for c in ["t", "t_ini", "t_fim"]:
    D[c] = pd.to_datetime(D[c], utc=True, format="mixed")
D["min_parada"] = D["min_parada"].fillna(0)
DEV = D[D.conjunto != "teste"].sort_values("t_ini").reset_index(drop=True)
TESTE = D[D.conjunto == "teste"].sort_values("t_ini").reset_index(drop=True)
MESES_WF = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
MESES_CUSTO = MESES_WF[2:]          # minutos contados de jan a jun (nov/dez só aquecem o limiar)
print("sklearn", sklearn.__version__, "| dev:", DEV.shape, "| teste:", TESTE.shape)

# %% [markdown]
# ## 1. Minutos perdidos e limiar de menor custo

# %%
def minutos(y, dur, alarme, c_fp=1.0, c_tp=0.0):
    y = np.asarray(y).astype(bool); a = np.asarray(alarme).astype(bool); dur = np.asarray(dur)
    return float(dur[y & ~a].sum() + c_fp * (a & ~y).sum() + c_tp * (a & y).sum())


def melhor_limiar(y, p, dur, c_fp=1.0, c_tp=0.0):
    """Limiar que minimiza os minutos perdidos (varre todos os cortes possíveis)."""
    p = np.asarray(p); y = np.asarray(y).astype(bool); dur = np.asarray(dur)
    o = np.argsort(-p); ps, ys, ds = p[o], y[o], dur[o]
    custo = (dur[y].sum() - np.r_[0, np.cumsum(np.where(ys, ds, 0))]
             + c_fp * np.r_[0, np.cumsum(~ys)] + c_tp * np.r_[0, np.cumsum(ys)])
    custo = np.where(np.r_[True, ps[1:] != ps[:-1], True], custo, np.inf)
    k = int(np.argmin(custo))
    return np.inf if k == 0 else (-np.inf if k == len(ps) else (ps[k - 1] + ps[k]) / 2)

# %% [markdown]
# ## 2. Features e modelos finalistas
#
# O conjunto principal tem 120 features: as 84 candidatas da feature selection + 36 novas
# (histórico, gusa líquido, ritmo do forno, composição relativa). O modelo é a média das
# probabilidades de 5 Gradient Boosting (`HistGradientBoostingClassifier`) com hiperparâmetros
# levemente diferentes. A média deixa o resultado menos dependente de um ajuste específico.
#
# Por que Gradient Boosting com muitas features e não a RF com 6? No walk-forward, a família
# "HGB com todas as features" deu os menores minutos em todas as 6 variações de hiperparâmetro
# testadas (600–635 min, contra 751 da RF do grupo), porque é melhor justamente na faixa de
# recall 60–80%, onde o limiar de menor custo cai (ver README).

# %%
FEAT = json.load(open(AQUI / "features_finais.json", encoding="utf-8"))
F_PRINCIPAL, F_LAG2 = FEAT["features_principal"], FEAT["features_lag2"]
FINAL6 = ["densidade_media", "prop_peso_pesado", "peso_por_ciclo_t", "taxa_ca_ult5", "taxa_ca_ult10", "prop_PACB"]
S2 = ["densidade_media", "prop_peso_pesado", "peso_por_ciclo_t", "prop_PACB", "h_taxa3", "h_taxa5", "h_taxa10",
      "h_taxa20", "gusl_t", "h_ewm2", "h_ewm5", "h_ult1", "h_taxa_24h", "eh_cestao_2", "tempo_montagem_min",
      "rendimento_medio"]
VARIANTES = {"l2=5": dict(l2=5.0), "lr.05": dict(lr=0.05, l2=5.0), "leaves16": dict(leaves=16, l2=5.0),
             "leaf20": dict(leaf=20, l2=5.0), "it150": dict(it=150, l2=5.0)}


def HGB(lr=0.03, it=300, leaves=8, leaf=40, l2=1.0):
    return lambda: HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves,
                                                  min_samples_leaf=leaf, l2_regularization=l2, random_state=42)


def RF6():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample",
                                                random_state=42, n_jobs=-1))


def familia(cols):
    return [(HGB(**kw), cols) for kw in VARIANTES.values()]


FINALISTAS = {
    "PRINCIPAL: média de 5 HGB, 120 features (histórico lag1)": familia(F_PRINCIPAL),
    "Equilíbrio: 5 HGB 120 feat + 5 HGB 16 feat": familia(F_PRINCIPAL) + familia(S2),
    "ESCOLHIDO: principal + resultado do cestão 1 (confirmado pela EVCOMX)": familia(F_PRINCIPAL + ["carga_alta_c1"]),
    "PRINCIPAL com histórico seguro (lag2)": familia(F_LAG2),
    "Referência: RF 6 features (plano A do grupo)": [(RF6, FINAL6)],
}


def prever(membros, tr, te):
    ps = []
    for fab, cols in membros:
        m = fab(); m.fit(tr[cols], tr["carga_alta"]); ps.append(m.predict_proba(te[cols])[:, 1])
    return np.mean(ps, axis=0)


def oof_dev(membros):
    """Walk-forward mensal no dev: previsões fora da amostra de nov/25 a jun/26."""
    oof = pd.Series(np.nan, index=DEV.index)
    for mth in MESES_WF:
        te = DEV[DEV.mes == mth]
        oof[te.index] = prever(membros, DEV[DEV.t_ini < te.t_ini.min()], te)
    w = DEV[DEV.mes.isin(MESES_WF)].copy(); w["p"] = oof[w.index]
    return w


def minutos_wf(w, c_tp=0.0):
    """Minutos de jan a jun no dev, com o limiar de cada mês vindo só dos meses anteriores."""
    tot, alarmes = 0.0, 0
    for mth in MESES_CUSTO:
        g = w[w.mes == mth]; pas = w[w.t_ini < g.t_ini.min()]
        a = g.p.values > melhor_limiar(pas.carga_alta, pas.p, pas.min_parada, 1.0, c_tp)
        tot += minutos(g.carga_alta, g.min_parada, a, 1.0, c_tp); alarmes += a.sum()
    return tot, 100 * alarmes / w.mes.isin(MESES_CUSTO).sum()

# %% [markdown]
# ## 3. Dev (walk-forward) e teste, uma única vez

# %%
linhas, prev_teste, limiares, oofs = [], {}, {}, {}
regra = "Regra: só o histórico (taxa_ca_ult5)"
for nome, membros in list(FINALISTAS.items()) + [(regra, None)]:
    if membros is None:
        w = DEV[DEV.mes.isin(MESES_WF)].copy(); w["p"] = w["taxa_ca_ult5"].fillna(0)
        p = TESTE["taxa_ca_ult5"].fillna(0).values
    else:
        w = oof_dev(membros); p = prever(membros, DEV, TESTE)
    oofs[nome], prev_teste[nome] = w, p
    wf_min, wf_al = minutos_wf(w)
    lin = {"modelo": nome,
           "PR-AUC dev WF": average_precision_score(w.carga_alta, w.p),
           "min dev WF jan-jun": wf_min, "economia dev %": 100 * (1 - wf_min / w[w.mes.isin(MESES_CUSTO)].min_parada.sum()),
           "PR-AUC teste": average_precision_score(TESTE.carga_alta, p),
           "ROC-AUC teste": roc_auc_score(TESTE.carga_alta, p)}
    for c_tp, suf in [(0.0, ""), (1.0, " (TP=1)")]:
        lim = melhor_limiar(w.carga_alta, w.p, w.min_parada, 1.0, c_tp)
        a = p > lim; y = TESTE.carga_alta.values.astype(bool)
        lin[f"limiar{suf}"] = lim
        lin[f"min perdidos teste{suf}"] = minutos(y, TESTE.min_parada, a, 1.0, c_tp)
        if suf == "":
            limiares[nome] = lim
            lin.update({"alarmes/100": 100 * a.mean(), "recall": (a & y).sum() / y.sum(),
                        "precisão": (a & y).sum() / max(a.sum(), 1)})
    linhas.append(lin)
    print(f"{nome[:58]:58s} teste: PR-AUC {lin['PR-AUC teste']:.3f} | min {lin['min perdidos teste']:.0f}")

SEM = TESTE.min_parada.sum()
res = pd.DataFrame(linhas)
res["min sem modelo (teste)"] = SEM
res["economia teste %"] = 100 * (1 - res["min perdidos teste"] / SEM)
res["economia teste % (TP=1)"] = 100 * (1 - res["min perdidos teste (TP=1)"] / SEM)
res.to_csv(AQUI / "resultado_teste_final.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res.round(3).to_string())
print(f"Teste: {len(TESTE)} cestões, {int(TESTE.carga_alta.sum())} cargas altas, {SEM:.0f} min de parada em "
      f"{(TESTE.t_ini.max() - TESTE.t_ini.min()).days} dias")

# %% [markdown]
# ## 4. Incerteza: bootstrap por dia no teste (diferença contra a RF do grupo)

# %%
ref = "Referência: RF 6 features (plano A do grupo)"
TESTE["dia"] = TESTE.t_ini.dt.tz_convert("America/Sao_Paulo").dt.date
dias = TESTE.dia.unique(); idx = {d: np.where(TESTE.dia.values == d)[0] for d in dias}
rng = np.random.default_rng(0); y = TESTE.carga_alta.values; dur = TESTE.min_parada.values
boot = []
for nome in res.modelo:
    dpr, dmin, prs = [], [], []
    for _ in range(2000):
        s = np.concatenate([idx[d] for d in rng.choice(dias, len(dias))])
        if y[s].sum() == 0:
            continue
        prs.append(average_precision_score(y[s], prev_teste[nome][s]))
        dpr.append(prs[-1] - average_precision_score(y[s], prev_teste[ref][s]))
        dmin.append(minutos(y[s], dur[s], prev_teste[nome][s] > limiares[nome])
                    - minutos(y[s], dur[s], prev_teste[ref][s] > limiares[ref]))
    boot.append({"modelo": nome, "PR-AUC teste IC90": np.round(np.percentile(prs, [5, 95]), 3),
                 "dif PR-AUC vs RF grupo": np.mean(dpr), "IC90 dif PR-AUC": np.round(np.percentile(dpr, [5, 95]), 3),
                 "dif minutos vs RF grupo": np.mean(dmin), "IC90 dif minutos": np.round(np.percentile(dmin, [5, 95]), 1)})
bt = pd.DataFrame(boot); bt.to_csv(AQUI / "bootstrap_teste_final.csv", index=False)
print(bt.round(3).to_string())

# %% [markdown]
# ## 5. Gráfico: minutos perdidos no teste

# %%
ordem = res.sort_values("min perdidos teste", ascending=True)
rotulos = {"PRINCIPAL: média de 5 HGB, 120 features (histórico lag1)": "Principal, sem o resultado do cestão 1",
           "Equilíbrio: 5 HGB 120 feat + 5 HGB 16 feat": "Equilíbrio (10 HGB)",
           "ESCOLHIDO: principal + resultado do cestão 1 (confirmado pela EVCOMX)": "Escolhido: principal + resultado do cestão 1",
           "PRINCIPAL com histórico seguro (lag2)": "Principal, histórico sem a corrida anterior",
           "Referência: RF 6 features (plano A do grupo)": "RF 6 features (plano A atual)",
           regra: "Regra: só histórico (sem modelo)"}
AZUL, CINZA, TXT, TXT2 = "#2a78d6", "#b4b2a9", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(9, 4.4), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
cores = [AZUL if m.startswith("ESCOLHIDO") else CINZA for m in ordem.modelo]
ax.barh([rotulos[m] for m in ordem.modelo], ordem["min perdidos teste"], color=cores, height=0.6)
ax.axvline(SEM, color=TXT2, lw=1, ls="--", zorder=0)
ax.text(SEM + 3, -0.75, f"sem modelo: {SEM:.0f} min", ha="left", va="center", color=TXT2, fontsize=9)
for i, (v, e) in enumerate(zip(ordem["min perdidos teste"], ordem["economia teste %"])):
    ax.text(v + 3, i, f"{v:.0f} min ({e:.0f}% a menos)", va="center", color=TXT, fontsize=9,
            bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1))
ax.set_ylim(len(ordem) - 0.5, -1.1); ax.set_xlim(0, SEM * 1.3)
ax.set_xlabel("minutos de forno perdidos no teste (24/jun–4/ago/2026)", color=TXT2)
ax.set_title("Minutos perdidos no teste: paradas não evitadas + 1 min por alarme falso", loc="left", color=TXT, fontsize=11)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.tick_params(colors=TXT2); ax.grid(axis="x", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
fig.text(0.01, 0.01, "A EVCOMX confirmou (31/out) que o resultado do cestão 1 é conhecido quando o cestão 2 ainda pode ser mexido.\n"
         "Diferenças de até ~40 min estão dentro do ruído (75 cargas altas).", fontsize=8, color=TXT2)
fig.tight_layout(rect=(0, 0.07, 1, 1))
(AQUI / "graficos").mkdir(exist_ok=True)
fig.savefig(AQUI / "graficos/01_minutos_perdidos_teste.png", dpi=150)

# %% [markdown]
# ## 6. Modelo para uso: re-treinado com treino + validação + teste
#
# A EVCOMX confirmou (31/out/2026) que, quando o cestão 2 ainda pode ser mexido, já se sabe se o
# cestão 1 da mesma corrida deu carga alta. Por isso o modelo de uso é o finalista **com o
# resultado do cestão 1** (`carga_alta_c1`; no próprio cestão 1 essa coluna vale 0 e
# `eh_cestao_2` = 0). Ele foi registrado como finalista antes de abrir o teste, com essa condição.
# A versão sem o cestão 1 vai junto como reserva, para quando esse resultado faltar.
# Os dois são re-treinados com todos os dados; o limiar vem das previsões walk-forward (dev) + teste.

# %%
TUDO = pd.concat([DEV, TESTE.drop(columns="dia")], ignore_index=True)


def empacotar(nome, features):
    modelos = []
    for fab, cols in FINALISTAS[nome]:
        m = fab(); m.fit(TUDO[cols], TUDO["carga_alta"]); modelos.append(m)
    w = oofs[nome]
    pas_y = np.r_[w.carga_alta.values, TESTE.carga_alta.values]
    pas_p = np.r_[w.p.values, prev_teste[nome]]
    pas_d = np.r_[w.min_parada.values, TESTE.min_parada.values]
    return {"modelos": modelos, "features": features,
            "limiar_minutos_TP0": melhor_limiar(pas_y, pas_p, pas_d, 1.0, 0.0),
            "limiar_minutos_TP1": melhor_limiar(pas_y, pas_p, pas_d, 1.0, 1.0)}


pacote = {"com_cestao1": empacotar("ESCOLHIDO: principal + resultado do cestão 1 (confirmado pela EVCOMX)", F_PRINCIPAL + ["carga_alta_c1"]),
          "sem_cestao1_reserva": empacotar("PRINCIPAL: média de 5 HGB, 120 features (histórico lag1)", F_PRINCIPAL),
          "como_usar": ("p = média de m.predict_proba(X[features])[:,1] para m em modelos; alarmar se p > limiar. "
                        "No cestão 2, carga_alta_c1 = resultado do cestão 1 da mesma corrida; no cestão 1, 0."),
          "versao_sklearn": sklearn.__version__}
joblib.dump(pacote, AQUI / "modelo_minutos.joblib")
for k in ["com_cestao1", "sem_cestao1_reserva"]:
    print(k, "limiares:", round(pacote[k]["limiar_minutos_TP0"], 4), round(pacote[k]["limiar_minutos_TP1"], 4))
