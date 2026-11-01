# %% [markdown]
# # Análise de Modelos e Métricas — Predição de Carga Alta (EVCOMX)
#
# Segue a **Aula 06 do bootcamp** (Análise de Modelos e Métricas de Desempenho) e o Guia de
# Modelagem da EVCOMX (§5), e continua `selecao_algoritmos/`. A seleção escolheu o modelo. Aqui
# medimos **o que ele entrega**, com as métricas certas, e traduzimos os erros para a operação.
#
# **Modelo analisado: o oficial**, a média de 5 Gradient Boosting com 121 features (120 +
# o resultado do cestão 1). Ao lado dele: a reserva (sem o cestão 1), a versão com histórico "lag2",
# a **RF de 6 features (o antigo plano A, guardado em `historico_rf6/`)** e a regra do histórico.
#
# **Esta etapa não escolhe nada.** Lê as previsões da seleção (`previsoes_dev_teste.csv`): no dev,
# as do walk-forward mensal (sempre fora da amostra); no teste, as dos modelos treinados no dev.
#
# | Passo da aula (slide 37) | Seção |
# |---|---|
# | 1. Qual erro custa mais caro? | 1 |
# | 2. Métrica principal escolhida antes de testar | 2 |
# | 3. Baselines na mesma métrica | 3 |
# | Tabela única (Guia §5.5) | 4 |
# | Curvas ROC e precisão-recall | 5 |
# | 4–5. Matriz de confusão e tradução para a operação | 6 |
# | 6. O ponto de corte pelo custo | 7 |
# | A chance é chance de verdade? (calibração e faixas de risco) | 8 |
# | O desempenho se mantém no tempo? | 9 |
# | Minutos: o modelo pega as paradas que mais doem? | 10 |
# | A diferença é real ou sorte? (bootstrap por dia) | 11 |
# | Diagnóstico de erros (Guia §5.6): aprendizado, segmentos, FN e FP | 12 |
# | E as métricas de regressão? | 13 |
# | Conclusão: é bom o suficiente para uso? | 14 |
#
# **Rode antes** a feature selection e a seleção de algoritmos. Leva ~3 min.

# %%
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, f1_score, fbeta_score,
                             matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "comum").is_dir())
sys.path.insert(0, str(RAIZ))
from comum.protocolo import (ARQ_COMPLETO, C_FP, MESES_CUSTO, NOME_OFICIAL, NOME_REGRA, NOME_RESERVA, NOME_RF6,
                             carregar, familia, ler_features, melhor_limiar, minutos, prever)

PASTA = RAIZ / "analise_modelos"
(PASTA / "graficos").mkdir(exist_ok=True)
SEL = RAIZ / "selecao_algoritmos"
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6, "axes.axisbelow": True})
AZUL, VERMELHO, CINZA, LARANJA, VERDE = "#2b6cb0", "#c0392b", "#7f8c8d", "#dd8a2e", "#4a8c4a"
LAG2 = "5 HGB, histórico sem a corrida anterior (lag2)"
ROT = {NOME_OFICIAL: "OFICIAL: 5 HGB + cestão 1", NOME_RESERVA: "Reserva: 5 HGB sem cestão 1", LAG2: "5 HGB, histórico lag2",
       NOME_RF6: "RF 6 features (antigo)", NOME_REGRA: "Regra do histórico"}
COR = {NOME_OFICIAL: AZUL, NOME_RESERVA: VERDE, LAG2: CINZA, NOME_RF6: LARANJA, NOME_REGRA: VERMELHO}

# %% [markdown]
# ## 0. Ler os arquivos das etapas anteriores

# %%
FEAT = ler_features()
D, DEV, TESTE = carregar(ARQ_COMPLETO)
PREV = pd.read_csv(SEL / "previsoes_dev_teste.csv")
RES_SEL = pd.read_csv(SEL / "resultado_teste.csv", index_col=0)
MODELOS = list(RES_SEL.index)
chave = ["cd_corrida", "nu_carregamento"]


def tabela(modelo, parte):
    t = PREV[(PREV.modelo == modelo) & (PREV.parte == parte)].drop(columns=["min_parada"])
    base = DEV if parte != "teste" else TESTE
    return t.merge(base, on=chave + ["carga_alta"], suffixes=("", "_d"), how="left")


WF = {m: tabela(m, "walk-forward (dev)") for m in MODELOS}
TE = {m: tabela(m, "teste") for m in MODELOS}
LIM = {m: RES_SEL.loc[m, "limiar"] for m in MODELOS}
LIM1 = {m: RES_SEL.loc[m, "limiar (TP=1)"] for m in MODELOS}
for m in MODELOS:
    assert len(TE[m]) == len(TESTE) and TE[m].min_parada.notna().all()

# conferência: as chances do teste batem com o oficial re-treinado aqui?
p_aqui = prever(familia(FEAT["features_oficial"]), DEV, TESTE)
p_sel = TE[NOME_OFICIAL].set_index(chave).loc[TESTE.set_index(chave).index, "p"].values
assert np.allclose(p_aqui, p_sel, atol=1e-12), "as previsões da seleção não batem com o modelo oficial"
print("Conferência: as chances do teste são idênticas às da seleção.")
SEM_TESTE = TESTE.min_parada.sum()
print(f"Teste: {len(TESTE)} cestões, {int(TESTE.carga_alta.sum())} cargas altas, {SEM_TESTE:.0f} min de parada")

# %% [markdown]
# ## 1. Qual erro custa mais caro? (slides 5, 7 e 9)
#
# - **Carga alta não detectada (FN):** o forno para pela duração real da parada.
# - **Alarme falso (FP):** 1 min para reajustar a montagem (EVCOMX, out/2026).
# Ou seja, **um FN custa em média ~4 vezes um FP**, e os FN não são todos iguais.

# %%
par = D[D.carga_alta == 1].min_parada
custo = pd.DataFrame({"erro": ["FN: carga alta sem alarme", "FP: alarme falso"],
                      "custo médio (min)": [par.mean(), C_FP], "mediana (min)": [par.median(), C_FP],
                      "p90 (min)": [par.quantile(0.9), C_FP]})
print(custo.round(2).to_string(index=False))
print(f"Paradas por 30 dias (ano todo): {par.sum() / ((D.t_ini.max() - D.t_ini.min()).days / 30):.0f} min")

# %% [markdown]
# ## 2. A métrica principal, escolhida antes de testar (slides 18, 35 e 36)
#
# Declarado antes de abrir o teste (`otimizacao_minutos/finalistas_dev_pre_registro.csv`):
# - **minutos perdidos** com o limiar vindo do dev: é a métrica de negócio;
# - **PR-AUC**: a métrica principal do grupo para a ordem dos cestões (não depende do limiar);
# - ROC-AUC, recall, precisão, F1, F2, MCC e Brier como apoio. **A acurácia engana**: quem sempre
#   diz "normal" acerta 85% no teste e perde 251 min.
#
# ## 3. Baselines na mesma métrica (slides 3 e 36)
#
# A régua mínima é **"sem modelo"** (todas as paradas acontecem). A régua que aperta é a **regra do
# histórico**: alarmar quando mais da metade das 5 corridas anteriores deu carga alta.

# %%
def metricas(t, lim, c_tp=0.0):
    y = t.carga_alta.values.astype(bool); p = t.p.values; a = p > lim
    return {"PR-AUC": average_precision_score(y, p), "ROC-AUC": roc_auc_score(y, p),
            "recall": recall_score(y, a), "precisão": precision_score(y, a, zero_division=0),
            "F1": f1_score(y, a), "F2": fbeta_score(y, a, beta=2), "MCC": matthews_corrcoef(y, a),
            "Brier": brier_score_loss(y, np.clip(p, 0, 1)), "acurácia": (a == y).mean(),
            "alarmes/100": 100 * a.mean(), "minutos": minutos(y, t.min_parada, a, C_FP, c_tp),
            "sem modelo": t.min_parada.sum()}


def metricas_wf(t, m):
    """No dev, o limiar de cada mês vem só dos meses anteriores (como seria na operação)."""
    out = []
    for mth in MESES_CUSTO:
        g = t[t.mes == mth]; pas = t[t.mes < mth]
        out.append(g.assign(alarme=g.p > melhor_limiar(pas.carga_alta, pas.p, pas.min_parada)))
    g = pd.concat(out)
    y = g.carga_alta.values.astype(bool); a = g.alarme.values
    return {"PR-AUC": average_precision_score(t.carga_alta, t.p), "ROC-AUC": roc_auc_score(t.carga_alta, t.p),
            "recall": recall_score(y, a), "precisão": precision_score(y, a, zero_division=0), "F1": f1_score(y, a),
            "F2": fbeta_score(y, a, beta=2), "MCC": matthews_corrcoef(y, a),
            "Brier": brier_score_loss(t.carga_alta, np.clip(t.p, 0, 1)), "acurácia": (a == y).mean(),
            "alarmes/100": 100 * a.mean(), "minutos": minutos(y, g.min_parada, a), "sem modelo": g.min_parada.sum()}


linhas = []
for m in MODELOS:
    linhas.append({"modelo": m, "parte": "dev (walk-forward, jan–jun/26)", **metricas_wf(WF[m], m)})
    linhas.append({"modelo": m, "parte": "teste (jun–ago/26)", **metricas(TE[m], LIM[m])})
for parte, base in [("dev (walk-forward, jan–jun/26)", DEV[DEV.mes.isin(MESES_CUSTO)]), ("teste (jun–ago/26)", TESTE)]:
    y = base.carga_alta
    linhas.append({"modelo": "Sem modelo (nunca alarmar)", "parte": parte, "PR-AUC": y.mean(), "ROC-AUC": 0.5, "recall": 0,
                   "precisão": 0, "F1": 0, "F2": 0, "MCC": 0, "Brier": brier_score_loss(y, np.full(len(y), y.mean())),
                   "acurácia": 1 - y.mean(), "alarmes/100": 0, "minutos": base.min_parada.sum(),
                   "sem modelo": base.min_parada.sum()})
TAB = pd.DataFrame(linhas)
TAB["economia %"] = 100 * (1 - TAB["minutos"] / TAB["sem modelo"])

# %% [markdown]
# ## 4. Tabela única de comparação (Guia de Modelagem §5.5)
#
# Modelos e baselines lado a lado, nas mesmas métricas, no dev (walk-forward) e no teste.

# %%
for parte, arq in [("dev (walk-forward, jan–jun/26)", "tabela_unica_dev.csv"), ("teste (jun–ago/26)", "tabela_unica_teste.csv")]:
    t = TAB[TAB.parte == parte].set_index("modelo").drop(columns="parte")
    t.to_csv(PASTA / arq)
    print(f"\n== {parte}"); print(t.round(3).to_string())
# conferência com a seleção
t = TAB[TAB.parte == "teste (jun–ago/26)"].set_index("modelo")
assert np.allclose(t.loc[MODELOS, "minutos"].values, RES_SEL.loc[MODELOS, "min perdidos teste"].values)

# %% [markdown]
# ## 5. Curvas ROC e precisão-recall (slides 20 a 23)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
y = TESTE.carga_alta.values
for m in MODELOS:
    t = TE[m]
    fpr, tpr, _ = roc_curve(t.carga_alta, t.p); pr, rc, _ = precision_recall_curve(t.carga_alta, t.p)
    lw = 2.4 if m == NOME_OFICIAL else 1.2
    axes[0].plot(fpr, tpr, color=COR[m], lw=lw, label=f"{ROT[m]} ({roc_auc_score(t.carga_alta, t.p):.2f})")
    axes[1].plot(rc, pr, color=COR[m], lw=lw, label=f"{ROT[m]} ({average_precision_score(t.carga_alta, t.p):.2f})")
axes[0].plot([0, 1], [0, 1], ls=":", color=CINZA); axes[1].axhline(y.mean(), ls=":", color=CINZA, label="chute")
axes[0].set(xlabel="alarmes falsos / normais (FPR)", ylabel="recall"); axes[0].set_title("ROC no teste (AUC)", loc="left")
axes[1].set(xlabel="recall", ylabel="precisão"); axes[1].set_title("Precisão × recall no teste (PR-AUC)", loc="left")
axes[1].axvspan(0.6, 0.8, color=AZUL, alpha=0.06)
axes[1].text(0.61, 0.92, "faixa onde cai o\nlimiar de menor custo", fontsize=8, color=AZUL)
for ax in axes:
    ax.legend(fontsize=7.5)
plt.tight_layout(); plt.savefig(PASTA / "graficos/01_curvas_roc_e_precisao_recall.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 6. Matriz de confusão e tradução para a operação (slides 8 a 10, 34 e 37)

# %%
t = TE[NOME_OFICIAL]; a = t.p > LIM[NOME_OFICIAL]; yb = t.carga_alta.astype(bool)
cm = confusion_matrix(yb, a)
dias = (TESTE.t_ini.max() - TESTE.t_ini.min()).days
trad = pd.DataFrame([{"modelo": ROT[m], "a cada 100 cestões: alarmes": 100 * (TE[m].p > LIM[m]).mean(),
                      "…certos": 100 * ((TE[m].p > LIM[m]) & TE[m].carga_alta.astype(bool)).mean(),
                      "…falsos": 100 * ((TE[m].p > LIM[m]) & ~TE[m].carga_alta.astype(bool)).mean(),
                      "cargas altas perdidas": 100 * (~(TE[m].p > LIM[m]) & TE[m].carga_alta.astype(bool)).mean(),
                      "minutos perdidos / 30 dias": minutos(TE[m].carga_alta, TE[m].min_parada, TE[m].p > LIM[m]) * 30 / dias,
                      "alarmes / dia": (TE[m].p > LIM[m]).sum() / dias} for m in MODELOS])
trad.loc[len(trad)] = {"modelo": "Sem modelo", "a cada 100 cestões: alarmes": 0, "…certos": 0, "…falsos": 0,
                       "cargas altas perdidas": 100 * TESTE.carga_alta.mean(),
                       "minutos perdidos / 30 dias": SEM_TESTE * 30 / dias, "alarmes / dia": 0}
trad.to_csv(PASTA / "teste_traduzido.csv", index=False)
print(trad.round(1).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
axes[0].imshow(cm, cmap="Blues"); axes[0].grid(False)
for (i, j), v in np.ndenumerate(cm):
    axes[0].text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
axes[0].set_xticks([0, 1], ["previsto: normal", "previsto: carga alta"]); axes[0].set_yticks([0, 1], ["real: normal", "real: carga alta"])
axes[0].set_title(f"Oficial no teste, limiar {LIM[NOME_OFICIAL]:.2f} (do dev)", loc="left")
r = trad.iloc[0]
axes[1].axis("off")
axes[1].text(0, 0.95, "A cada 100 cestões, o modelo oficial:", fontsize=12, weight="bold", va="top")
txt = (f"• dispara {r['a cada 100 cestões: alarmes']:.0f} alarmes: {r['…certos']:.0f} certos e {r['…falsos']:.0f} falsos\n"
       f"• deixa passar {r['cargas altas perdidas']:.0f} das {100 * TESTE.carga_alta.mean():.0f} cargas altas\n\n"
       f"A cada 30 dias:\n• {trad.iloc[-1]['minutos perdidos / 30 dias']:.0f} min de parada sem modelo\n"
       f"• {r['minutos perdidos / 30 dias']:.0f} min com o modelo oficial "
       f"({100 * (1 - r['minutos perdidos / 30 dias'] / trad.iloc[-1]['minutos perdidos / 30 dias']):.0f}% a menos)\n"
       f"• {trad.set_index('modelo').loc[ROT[NOME_RF6], 'minutos perdidos / 30 dias']:.0f} min com a RF antiga\n"
       f"• ~{r['alarmes / dia']:.1f} alarmes por dia para o pátio atender")
axes[1].text(0, 0.80, txt, fontsize=10.5, va="top", linespacing=1.6)
plt.tight_layout(); plt.savefig(PASTA / "graficos/02_teste_traduzido.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 7. O ponto de corte pelo custo do erro (slides 16 e 34)
#
# O limiar foi escolhido **no dev**. Aqui verificamos, no teste, se ele caiu perto do melhor
# possível (que só se saberia olhando o teste) e o que acontece nos dois cenários de custo.

# %%
grade = np.round(np.arange(0.02, 0.81, 0.01), 2)
cust = pd.DataFrame({"limiar": grade})
for m in [NOME_OFICIAL, NOME_RF6]:
    t = TE[m]
    cust[f"{ROT[m]} TP=0"] = [minutos(t.carga_alta, t.min_parada, t.p > g) for g in grade]
    cust[f"{ROT[m]} TP=1"] = [minutos(t.carga_alta, t.min_parada, t.p > g, C_FP, 1.0) for g in grade]
cust.to_csv(PASTA / "custo_por_limiar_teste.csv", index=False)
t = TE[NOME_OFICIAL]
otimo = grade[np.argmin(cust[f"{ROT[NOME_OFICIAL]} TP=0"])]
print(f"Oficial: limiar do dev {LIM[NOME_OFICIAL]:.3f} → {minutos(t.carga_alta, t.min_parada, t.p > LIM[NOME_OFICIAL]):.0f} min; "
      f"melhor limiar 'olhando o teste' {otimo:.2f} → {cust[f'{ROT[NOME_OFICIAL]} TP=0'].min():.0f} min")
fig, ax = plt.subplots(figsize=(8.5, 3.9))
for m, ls in [(NOME_OFICIAL, "-"), (NOME_RF6, "--")]:
    ax.plot(grade, cust[f"{ROT[m]} TP=0"], color=COR[m], ls=ls, label=f"{ROT[m]} (TP=0)")
    ax.plot(grade, cust[f"{ROT[m]} TP=1"], color=COR[m], ls=ls, alpha=0.45, label=f"{ROT[m]} (TP=1)")
ax.axvline(LIM[NOME_OFICIAL], color=AZUL, ls=":", lw=1.3); ax.axhline(SEM_TESTE, color=CINZA, ls=":", lw=1)
ax.text(LIM[NOME_OFICIAL] + 0.01, SEM_TESTE * 0.97, f"limiar do dev ({LIM[NOME_OFICIAL]:.2f})", color=AZUL, fontsize=8, va="top")
ax.set(xlabel="limiar de alarme", ylabel="minutos perdidos no teste")
ax.set_title("Custo no teste conforme o limiar (a RF antiga tinha chances infladas: limiar ~0,46)", loc="left")
ax.legend(fontsize=7.5, ncol=2)
plt.tight_layout(); plt.savefig(PASTA / "graficos/03_custo_conforme_limiar.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 8. A "chance" do modelo é chance de verdade? (calibração e faixas de risco)
#
# O oficial **não usa pesos de classe**, então a chance não vem inflada como na RF antiga (que
# precisava de calibração de Platt). Conferimos com a curva de confiabilidade e com **faixas de
# risco**: em cada faixa de chance prevista, qual foi a taxa real de carga alta?

# %%
bins = [0, 0.05, 0.10, 0.20, 0.40, 1.0]
rot_b = ["< 5%", "5–10%", "10–20%", "20–40%", "≥ 40%"]
faixas = []
for parte, dados in [("dev (walk-forward)", WF[NOME_OFICIAL]), ("teste", TE[NOME_OFICIAL])]:
    f = dados.assign(faixa=pd.cut(dados.p, bins, labels=rot_b, right=False))
    gq = f.groupby("faixa", observed=False).agg(cestoes=("p", "size"), chance_media=("p", "mean"),
                                                taxa_real=("carga_alta", "mean"), min_parada=("min_parada", "sum"))
    gq["% dos cestões"] = 100 * gq.cestoes / len(f); gq["parte"] = parte
    faixas.append(gq.reset_index())
faixas = pd.concat(faixas)
faixas.to_csv(PASTA / "faixas_de_risco.csv", index=False)
print(faixas.round(3).to_string(index=False))
cal = pd.DataFrame([{"modelo": ROT[m], "parte": parte, "chance média": d.p.mean(), "taxa real": d.carga_alta.mean(),
                     "Brier": brier_score_loss(d.carga_alta, np.clip(d.p, 0, 1))}
                    for m in [NOME_OFICIAL, NOME_RESERVA, NOME_RF6] for parte, d in [("dev WF", WF[m]), ("teste", TE[m])]])
cal.to_csv(PASTA / "calibracao.csv", index=False)
print(cal.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for m in [NOME_OFICIAL, NOME_RF6]:
    d = pd.concat([WF[m], TE[m]])
    q = pd.qcut(d.p, 10, duplicates="drop")
    gg = d.groupby(q, observed=True).agg(p=("p", "mean"), y=("carga_alta", "mean"))
    axes[0].plot(gg.p, gg.y, marker="o", color=COR[m], label=ROT[m])
axes[0].plot([0, 0.8], [0, 0.8], ls=":", color=CINZA, label="calibração perfeita")
axes[0].set(xlabel="chance prevista (média do decil)", ylabel="taxa real de carga alta")
axes[0].set_title("Confiabilidade (dev WF + teste): o oficial fica perto da diagonal", loc="left"); axes[0].legend(fontsize=8)
ft = faixas[faixas.parte == "teste"]
axes[1].bar(ft.faixa.astype(str), 100 * ft.taxa_real, color=AZUL)
for i, (v, n) in enumerate(zip(ft.taxa_real, ft["% dos cestões"])):
    axes[1].text(i, 100 * v + 1, f"{100 * v:.0f}%\n({n:.0f}% dos cestões)", ha="center", fontsize=8)
axes[1].set(xlabel="faixa de chance prevista", ylabel="% com carga alta (real)")
axes[1].set_title("Faixas de risco no teste: taxa real em cada faixa", loc="left")
plt.tight_layout(); plt.savefig(PASTA / "graficos/04_calibracao_e_faixas_de_risco.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 9. O desempenho se mantém no tempo?
#
# Mês a mês no dev (walk-forward, limiar do passado) e no teste. A mudança de regime de maio/26
# derrubava a RF antiga; o walk-forward re-treina todo mês.

# %%
per = []
for m in MODELOS:
    for parte, dados in [("dev", WF[m]), ("teste", TE[m])]:
        for mth, gq in dados.groupby("mes"):
            if parte == "dev" and mth not in MESES_CUSTO:
                continue
            pas = WF[m][WF[m].mes < mth] if parte == "dev" else None
            lim = melhor_limiar(pas.carga_alta, pas.p, pas.min_parada) if parte == "dev" else LIM[m]
            if gq.carga_alta.sum() == 0:
                continue
            per.append({"modelo": m, "parte": parte, "mes": mth, "cestoes": len(gq), "cargas_altas": int(gq.carga_alta.sum()),
                        "PR-AUC": average_precision_score(gq.carga_alta, gq.p),
                        "minutos": minutos(gq.carga_alta, gq.min_parada, gq.p > lim), "sem modelo": gq.min_parada.sum()})
per = pd.DataFrame(per); per["economia %"] = 100 * (1 - per.minutos / per["sem modelo"])
MESES_PT = {"01": "jan", "02": "fev", "03": "mar", "04": "abr", "05": "mai", "06": "jun", "07": "jul", "08": "ago"}
per["periodo"] = per.mes.str[5:].map(MESES_PT) + np.where(per.parte == "teste", " (teste)", "") + \
    np.where(per.cestoes < 30, f"\n{''}", "") + np.where(per.cestoes < 30, per.cestoes.astype(str) + " cestões", "")
per = per.sort_values(["parte", "mes"]).reset_index(drop=True)
per.to_csv(PASTA / "metricas_por_periodo.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for m in [NOME_OFICIAL, NOME_RF6, NOME_REGRA]:
    q = per[per.modelo == m]
    axes[0].plot(q.periodo, q["PR-AUC"], marker="o", color=COR[m], label=ROT[m])
    axes[1].plot(q.periodo, q["economia %"], marker="o", color=COR[m], label=ROT[m])
n_dev = (per[per.modelo == NOME_OFICIAL].parte == "dev").sum()
for ax in axes:
    ax.axvspan(n_dev - 0.5, len(per[per.modelo == NOME_OFICIAL]) - 0.5, color=CINZA, alpha=0.1)
    ax.tick_params(axis="x", rotation=45); ax.legend(fontsize=8)
axes[0].set_title("PR-AUC por mês (dev: walk-forward; sombreado: teste)", loc="left")
axes[1].set_title("Economia de minutos por mês (%)", loc="left"); axes[1].axhline(0, color="k", lw=0.8)
plt.tight_layout(); plt.savefig(PASTA / "graficos/05_estabilidade_no_tempo.png", dpi=130, bbox_inches="tight"); plt.show()
print(per[per.modelo == NOME_OFICIAL].round(3).to_string(index=False))

# %% [markdown]
# ## 10. Minutos: o modelo pega as paradas que mais doem?
#
# Recall contado em **eventos** (quantas cargas altas) e em **minutos** (quanto da parada total).
# Se o modelo pegasse só as paradas curtas, o recall em minutos seria bem menor.

# %%
rec = []
for m in MODELOS:
    for parte, dados, lim in [("teste", TE[m], LIM[m])]:
        a = dados.p > lim; ca = dados.carga_alta == 1
        rec.append({"modelo": ROT[m], "recall em eventos": (a & ca).sum() / ca.sum(),
                    "recall em minutos": dados.min_parada[a & ca].sum() / dados.min_parada[ca].sum(),
                    "duração média das pegas (min)": dados.min_parada[a & ca].mean(),
                    "duração média das perdidas (min)": dados.min_parada[~a & ca].mean()})
rec = pd.DataFrame(rec); rec.to_csv(PASTA / "recall_em_minutos.csv", index=False)
print(rec.round(3).to_string(index=False))
ca = pd.concat([WF[NOME_OFICIAL], TE[NOME_OFICIAL]]); ca = ca[ca.carga_alta == 1]
rho = spearmanr(ca.p, ca.min_parada)
print(f"Entre as cargas altas, a chance prevista quase não se relaciona com a duração: Spearman {rho.statistic:+.2f}")

# %% [markdown]
# ## 11. A diferença é real ou sorte? (slide 36: "ignorar a variação")
#
# Bootstrap **por dia** no teste (sorteamos 2.000 vezes os 40 dias, com reposição, mantendo os
# cestões de cada dia juntos), comparando cada modelo com a RF antiga e com a regra do histórico.
# IC 90%. Uma diferença é "real" quando o intervalo não cruza o zero.

# %%
dia = TESTE.set_index(chave)["t_ini"].dt.tz_convert("America/Sao_Paulo").dt.date
P = {m: TE[m].set_index(chave).loc[dia.index] for m in MODELOS}
dias_u = dia.unique(); idx = {d_: np.where(dia.values == d_)[0] for d_ in dias_u}
rng = np.random.default_rng(0)
yv = P[NOME_OFICIAL].carga_alta.values; dur = P[NOME_OFICIAL].min_parada.values
amostras = [s for s in (np.concatenate([idx[d_] for d_ in rng.choice(dias_u, len(dias_u))]) for _ in range(2000)) if yv[s].sum()]
boot = []
for ref in [NOME_RF6, NOME_REGRA]:
    for m in MODELOS:
        if m == ref:
            continue
        dpr, dmin = [], []
        for s in amostras:
            dpr.append(average_precision_score(yv[s], P[m].p.values[s]) - average_precision_score(yv[s], P[ref].p.values[s]))
            dmin.append(minutos(yv[s], dur[s], P[m].p.values[s] > LIM[m]) - minutos(yv[s], dur[s], P[ref].p.values[s] > LIM[ref]))
        boot.append({"modelo": ROT[m], "contra": ROT[ref], "dif PR-AUC": np.mean(dpr), "IC90 PR-AUC": np.round(np.percentile(dpr, [5, 95]), 3),
                     "dif minutos": np.mean(dmin), "IC90 minutos": np.round(np.percentile(dmin, [5, 95]), 1),
                     "% sorteios em que ganha nos minutos": 100 * np.mean(np.array(dmin) < 0)})
boot = pd.DataFrame(boot); boot.to_csv(PASTA / "bootstrap_teste.csv", index=False)
print(boot.round(3).to_string(index=False))

# %% [markdown]
# ## 12. Diagnóstico de erros (Guia de Modelagem §5.6)
#
# ### Curva de aprendizado
#
# Mais dados ajudam? Treinamos o oficial com os últimos 2, 4, 6 e 8 meses do **treino** (sempre
# terminando em abr/26) e medimos na **validação** (abr–jun/26). Sem olhar o teste.

# %%
TREINO = DEV[DEV.conjunto == "treino"]; VALID = DEV[DEV.conjunto == "validacao"]
fim = TREINO.t_ini.max(); apr = []
for meses in [2, 4, 6, 8, 99]:
    tr = TREINO[TREINO.t_ini > fim - pd.Timedelta(days=30.5 * meses)]
    for nome, cols in [(NOME_OFICIAL, FEAT["features_oficial"])]:
        p = prever(familia(cols), tr, VALID)
        apr.append({"meses de treino": "todos (8,4)" if meses == 99 else meses, "cestoes": len(tr),
                    "PR-AUC validacao": average_precision_score(VALID.carga_alta, p),
                    "minutos validacao (limiar do dev)": minutos(VALID.carga_alta, VALID.min_parada, p > LIM[NOME_OFICIAL])})
apr = pd.DataFrame(apr); apr.to_csv(PASTA / "curva_de_aprendizado.csv", index=False)
print(apr.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(apr.cestoes, apr["PR-AUC validacao"], marker="o", color=AZUL)
for _, r in apr.iterrows():
    ax.annotate(f"{r['meses de treino']} meses", (r.cestoes, r["PR-AUC validacao"]), fontsize=8, xytext=(3, 4), textcoords="offset points")
ax.axhline(VALID.carga_alta.mean(), ls=":", color=CINZA, label="chute")
ax.set(xlabel="cestões de treino (os mais recentes antes da validação)", ylabel="PR-AUC na validação")
ax.set_title("Curva de aprendizado do oficial", loc="left"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(PASTA / "graficos/06_curva_de_aprendizado.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ### Desempenho por segmento
#
# Cestão 1 × cestão 2, com × sem gusa líquida, histórico do forno alto × baixo e densidade em
# terços. Dev (walk-forward, jan–jun) + teste juntos, para ter cargas altas suficientes em cada
# segmento; o limiar é o de cada parte (dev: do passado; teste: do dev).

# %%
def com_alarme(m):
    out = []
    for mth in MESES_CUSTO:
        g = WF[m][WF[m].mes == mth]; pas = WF[m][WF[m].mes < mth]
        out.append(g.assign(alarme=g.p > melhor_limiar(pas.carga_alta, pas.p, pas.min_parada)))
    out.append(TE[m].assign(alarme=TE[m].p > LIM[m]))
    return pd.concat(out, ignore_index=True)


AL = {m: com_alarme(m) for m in [NOME_OFICIAL, NOME_RF6]}
SEG = {"cestão": lambda d_: np.where(d_.eh_cestao_2 == 1, "cestão 2", "cestão 1"),
       "gusa líquida": lambda d_: np.where(d_.tem_gusl == 1, "com gusa", "sem gusa"),
       "histórico (5 corridas)": lambda d_: np.where(d_.taxa_ca_ult5.fillna(0) > 0.4, "histórico alto (> 40%)", "histórico baixo"),
       "densidade": lambda d_: pd.qcut(d_.densidade_media, 3, labels=["densidade baixa", "densidade média", "densidade alta"]).astype(str)}
seg = []
for m, dd in AL.items():
    for nome_s, f in SEG.items():
        for val, gq in dd.groupby(f(dd)):
            y_ = gq.carga_alta.astype(bool); a_ = gq.alarme.astype(bool)
            seg.append({"modelo": ROT[m], "segmento": nome_s, "valor": val, "cestoes": len(gq), "taxa real": y_.mean(),
                        "PR-AUC": average_precision_score(y_, gq.p) if 0 < y_.sum() < len(y_) else np.nan,
                        "recall": (a_ & y_).sum() / max(y_.sum(), 1), "precisão": (a_ & y_).sum() / max(a_.sum(), 1),
                        "alarmes/100": 100 * a_.mean(), "minutos": minutos(y_, gq.min_parada, a_),
                        "sem modelo": gq.min_parada.sum()})
seg = pd.DataFrame(seg); seg["economia %"] = 100 * (1 - seg.minutos / seg["sem modelo"])
seg.to_csv(PASTA / "desempenho_por_segmento.csv", index=False)
print(seg[seg.modelo == ROT[NOME_OFICIAL]].round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(11, 4))
so = seg.pivot_table(index=["segmento", "valor"], columns="modelo", values="economia %", sort=False)
xs = np.arange(len(so))
ax.bar(xs - 0.2, so[ROT[NOME_OFICIAL]], 0.4, color=AZUL, label=ROT[NOME_OFICIAL])
ax.bar(xs + 0.2, so[ROT[NOME_RF6]], 0.4, color=LARANJA, label=ROT[NOME_RF6])
ax.set_xticks(xs, [v for _, v in so.index], rotation=30, ha="right"); ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("economia de minutos (%)"); ax.legend(fontsize=8)
ax.set_title("Economia por segmento (dev jan–jun + teste)", loc="left")
plt.tight_layout(); plt.savefig(PASTA / "graficos/07_desempenho_por_segmento.png", dpi=130, bbox_inches="tight"); plt.show()

# %% [markdown]
# ### Falsos negativos e falsos positivos: o que têm em comum?
#
# Comparamos as medianas de algumas features nas quatro células da matriz (dev jan–jun + teste) e
# listamos os casos para revisar com o pátio: as cargas altas perdidas mais longas e os alarmes falsos
# mais "certos".

# %%
dd = AL[NOME_OFICIAL].copy()
dd["celula"] = np.select([dd.alarme & (dd.carga_alta == 1), dd.alarme & (dd.carga_alta == 0),
                          ~dd.alarme & (dd.carga_alta == 1)], ["TP (alarme certo)", "FP (alarme falso)", "FN (perdida)"],
                         "TN (normal)")
COLS_ERRO = ["p", "taxa_ca_ult5", "h_taxa_24h", "gusl_t", "densidade_media", "prop_peso_pesado", "peso_t_total",
             "volume_m3_total", "n_ciclos", "eh_cestao_2", "carga_alta_c1", "min_parada"]
erros = dd.groupby("celula")[COLS_ERRO].median().T
erros.loc["n_cestoes"] = dd.celula.value_counts()
erros.to_csv(PASTA / "analise_erros_fn_fp.csv")
print(erros.round(3).to_string())
casos = pd.concat([dd[dd.celula == "FN (perdida)"].nlargest(10, "min_parada"),
                   dd[dd.celula == "FP (alarme falso)"].nlargest(10, "p")])
casos[["celula", "cd_corrida", "nu_carregamento", "mes", "p", "min_parada", "taxa_ca_ult5", "gusl_t", "densidade_media",
       "prop_peso_pesado", "carga_alta_c1"]].to_csv(PASTA / "casos_para_especialista.csv", index=False)

# %% [markdown]
# ## 13. E as métricas de regressão? (slides 24 a 29)
#
# Não se aplicam: o alvo é sim/não. A única grandeza contínua é a **duração da parada**, e ela não é
# prevista (seção 10: entre as cargas altas, a chance quase não se relaciona com a duração; na
# otimização, nenhuma feature passou de |Spearman| 0,1 com ela). Ela entra como peso na régua de minutos.
#
# ## 14. Conclusão: o desempenho é bom o suficiente para uso operacional? (Guia FE/FS §11.2)

# %%
tt = TAB[TAB.parte == "teste (jun–ago/26)"].set_index("modelo")
of, rf, rg = tt.loc[NOME_OFICIAL], tt.loc[NOME_RF6], tt.loc[NOME_REGRA]
b_rf = boot[(boot.modelo == ROT[NOME_OFICIAL]) & (boot.contra == ROT[NOME_RF6])].iloc[0]
b_rg = boot[(boot.modelo == ROT[NOME_OFICIAL]) & (boot.contra == ROT[NOME_REGRA])].iloc[0]
print(f"""Teste (24/jun–4/ago/26, {int(TESTE.carga_alta.sum())} cargas altas, {SEM_TESTE:.0f} min sem modelo):
- OFICIAL: PR-AUC {of['PR-AUC']:.3f}, {of['minutos']:.0f} min ({of['economia %']:.0f}% a menos), recall {of['recall']:.0%}, precisão {of['precisão']:.0%}, {of['alarmes/100']:.0f} alarmes/100.
- Contra a RF antiga ({rf['PR-AUC']:.3f}, {rf['minutos']:.0f} min): dif. PR-AUC {b_rf['dif PR-AUC']:+.3f} (IC90 {b_rf['IC90 PR-AUC']}), dif. minutos {b_rf['dif minutos']:+.0f} (IC90 {b_rf['IC90 minutos']}).
- Contra a regra do histórico ({rg['PR-AUC']:.3f}, {rg['minutos']:.0f} min): dif. PR-AUC {b_rg['dif PR-AUC']:+.3f} (IC90 {b_rg['IC90 PR-AUC']}), dif. minutos {b_rg['dif minutos']:+.0f} (IC90 {b_rg['IC90 minutos']}).
- Se o alarme certo também custar 1 min (TP=1): economia {100 * (1 - RES_SEL.loc[NOME_OFICIAL, 'min perdidos teste (TP=1)'] / SEM_TESTE):.0f}%.""")
