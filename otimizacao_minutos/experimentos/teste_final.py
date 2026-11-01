"""Conferência ÚNICA no teste (jun–ago/2026) dos finalistas declarados em finalistas.py.

Protocolo (igual ao do grupo):
  - cada modelo é treinado com TODO o dev (treino + validação) e prevê o teste;
  - o limiar de alarme é o de menor minutos perdidos nas previsões walk-forward do dev
    (nov/25–jun/26), ou seja, nada do teste entra na escolha;
  - minutos: FN = duração real da parada; FP = 1 min; TP = 0 (cenário principal) ou 1 (conservador).
Extra: re-treino mensal dentro do teste (jul e ago treinam também com os meses de teste anteriores).
"""
import warnings; warnings.filterwarnings("ignore")
from finalistas import *

TESTE = D[D.conjunto == "teste"].sort_values("t_ini").copy()
TESTE["min_parada"] = TESTE["min_parada"].fillna(0)
DEVc = DEV.copy(); DEVc["min_parada"] = DEVc["min_parada"].fillna(0)


def prever(membros, tr, te):
    ps = []
    for fab, cols in membros:
        m = fab(); m.fit(tr[cols], tr["carga_alta"]); ps.append(m.predict_proba(te[cols])[:, 1])
    return np.mean(ps, axis=0)


def oof_dev(membros):
    oof = pd.Series(np.nan, index=DEVc.index)
    for mth in MESES_WF:
        te = DEVc[DEVc.mes == mth]; tr = DEVc[DEVc.t_ini < te.t_ini.min()]
        oof[te.index] = prever(membros, tr, te)
    w = DEVc[DEVc.mes.isin(MESES_WF)].copy(); w["p"] = oof[w.index]
    return w


def metricas(y, p, dur, lim, c_tp):
    a = p > lim; y = np.asarray(y).astype(bool)
    return {"alarmes_100": 100 * a.mean(), "recall": (a & y).sum() / y.sum(),
            "precisao": (a & y).sum() / max(a.sum(), 1),
            "minutos": minutos(y, dur, a, 1.0, c_tp)}


linhas, previsoes = [], {}
regra = {"Regra: só o histórico (taxa_ca_ult5)": None}
for nome, membros in list(FINALISTAS.items()) + list(regra.items()):
    if membros is None:
        w = DEVc[DEVc.mes.isin(MESES_WF)].copy(); w["p"] = w["taxa_ca_ult5"].fillna(0)
        p = TESTE["taxa_ca_ult5"].fillna(0).values
    else:
        w = oof_dev(membros)
        p = prever(membros, DEVc, TESTE)
    previsoes[nome] = p
    lin = {"modelo": nome, "PR-AUC teste": average_precision_score(TESTE.carga_alta, p),
           "ROC-AUC teste": roc_auc_score(TESTE.carga_alta, p),
           "PR-AUC dev WF": average_precision_score(w.carga_alta, w.p)}
    for ctp, suf in [(0.0, ""), (1.0, " (TP=1)")]:
        lim = melhor_limiar(w.carga_alta, w.p, w.min_parada, 1.0, ctp)
        mt = metricas(TESTE.carga_alta, p, TESTE.min_parada.values, lim, ctp)
        lin[f"limiar{suf}"] = lim
        lin[f"min perdidos{suf}"] = mt["minutos"]
        if suf == "":
            lin.update({"alarmes/100": mt["alarmes_100"], "recall": mt["recall"], "precisão": mt["precisao"]})
    linhas.append(lin)
    print(nome, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in lin.items() if k != "modelo"})

res = pd.DataFrame(linhas)
sem = TESTE.min_parada.sum()
res["min sem modelo"] = sem
res["economia %"] = 100 * (1 - res["min perdidos"] / sem)
res["economia % (TP=1)"] = 100 * (1 - res["min perdidos (TP=1)"] / sem)
res.to_csv("resultado_teste_final.csv", index=False)

# ---- bootstrap por dia: diferença contra o RF do grupo
ref = "Referência: RF 6 features (plano A do grupo)"
TESTE["dia"] = TESTE.t_ini.dt.tz_convert("America/Sao_Paulo").dt.date
dias = TESTE.dia.unique(); idx = {d: np.where(TESTE.dia.values == d)[0] for d in dias}
rng = np.random.default_rng(0)
lim_de = dict(zip(res.modelo, res["limiar"]))
boot = []
y = TESTE.carga_alta.values; dur = TESTE.min_parada.values
for nome in res.modelo:
    dpr, dmin, prs = [], [], []
    for _ in range(2000):
        s = np.concatenate([idx[d] for d in rng.choice(dias, len(dias))])
        if y[s].sum() == 0: continue
        pa, pb = previsoes[ref][s], previsoes[nome][s]
        prs.append(average_precision_score(y[s], pb))
        dpr.append(prs[-1] - average_precision_score(y[s], pa))
        dmin.append(minutos(y[s], dur[s], pb > lim_de[nome]) - minutos(y[s], dur[s], pa > lim_de[ref]))
    boot.append({"modelo": nome, "PR-AUC IC90": np.round(np.percentile(prs, [5, 95]), 3),
                 "dif PR-AUC vs RF grupo": np.mean(dpr), "IC90 dif PR": np.round(np.percentile(dpr, [5, 95]), 3),
                 "dif minutos vs RF grupo": np.mean(dmin), "IC90 dif min": np.round(np.percentile(dmin, [5, 95]), 1)})
bt = pd.DataFrame(boot); bt.to_csv("bootstrap_teste_final.csv", index=False)

# ---- re-treino mensal no teste (cenário de uso com re-treino todo mês)
mens = []
TESTE["mes_r"] = np.where(TESTE.t_ini < pd.Timestamp("2026-07-01", tz="America/Sao_Paulo"), "jun", np.where(TESTE.t_ini < pd.Timestamp("2026-08-01", tz="America/Sao_Paulo"), "jul", "ago"))
TUDO = pd.concat([DEVc, TESTE.drop(columns=["dia", "mes_r"])])
for i_m, (nome, membros) in enumerate(FINALISTAS.items()):
    w = oof_dev(membros); pt = pd.Series(np.nan, index=TESTE.index)
    for mr in ["jun", "jul", "ago"]:
        te = TESTE[TESTE.mes_r == mr]
        tr = TUDO[TUDO.t_ini < te.t_ini.min()]
        pt[te.index] = prever(membros, tr, te)
        # limiar: passado (OOF do dev + meses de teste já previstos)
        pas_y = np.r_[w.carga_alta.values, TESTE.loc[TESTE.t_ini < te.t_ini.min(), "carga_alta"].values]
        pas_p = np.r_[w.p.values, pt[TESTE.t_ini < te.t_ini.min()].values]
        pas_d = np.r_[w.min_parada.values, TESTE.loc[TESTE.t_ini < te.t_ini.min(), "min_parada"].values]
        TESTE.loc[te.index, f"lim_{i_m}"] = melhor_limiar(pas_y, pas_p, pas_d)
    a = pt.values > TESTE[f"lim_{i_m}"].values
    mens.append({"modelo": nome, "PR-AUC (re-treino mensal)": average_precision_score(TESTE.carga_alta, pt),
                 "min perdidos (re-treino mensal)": minutos(TESTE.carga_alta, TESTE.min_parada, a),
                 "alarmes/100": 100 * a.mean()})
mn = pd.DataFrame(mens); mn.to_csv("teste_retreino_mensal.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res.round(3).to_string()); print(bt.round(3).to_string()); print(mn.round(3).to_string())
print("min sem modelo no teste:", round(sem, 1), "| cargas altas:", int(TESTE.carga_alta.sum()), "| cestões:", len(TESTE))
