"""Régua única para todos os experimentos.

Dados de desenvolvimento (dev) = treino + validação. O TESTE NÃO É USADO AQUI.

Duas provas:
  V   : treina no treino, prevê a validação (a mesma prova das etapas anteriores do grupo)
  WF  : walk-forward mensal dentro do dev. Para cada mês M (nov/2025 a jun/2026) treina com tudo
        que começou antes de M e prevê M. As previsões dos meses são empilhadas (OOF).

Minutos perdidos (pedido da EVCOMX / Guima):
  FN  -> duração real da parada daquele cestão
  FP  -> 1 minuto (reajustar a montagem)
  TP  -> C_TP minutos (0 no cenário principal; 1 no cenário conservador, em que o reajuste
         também custa 1 min quando o alarme está certo)
  Sem modelo = soma de todas as paradas.

Limiar honesto no WF: o limiar do mês M é o que minimiza os minutos nas previsões OOF dos meses
anteriores a M (nunca olha o próprio mês). Os minutos do WF são contados de jan/2026 a jun/2026.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

AQUI = Path(__file__).resolve().parent
D = pd.read_pickle(AQUI / "dados_exp.pkl")
DEV = D[D.conjunto != "teste"].sort_values("t_ini").reset_index(drop=True)
MESES_WF = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
MESES_CUSTO = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]


def minutos(y, dur, alarme, c_fp=1.0, c_tp=0.0):
    y = np.asarray(y).astype(bool); a = np.asarray(alarme).astype(bool); dur = np.asarray(dur)
    return float(dur[y & ~a].sum() + c_fp * (a & ~y).sum() + c_tp * (a & y).sum())


def melhor_limiar(y, p, dur, c_fp=1.0, c_tp=0.0):
    """Limiar que minimiza os minutos (varre os valores únicos de p)."""
    p = np.asarray(p); y = np.asarray(y).astype(bool); dur = np.asarray(dur)
    ordem = np.argsort(-p)
    ps, ys, ds = p[ordem], y[ordem], dur[ordem]
    # alarmar os k primeiros: custo = soma dur dos CA não alarmados + c_fp*FP + c_tp*TP
    tot = dur[y].sum()
    cum_dur_alarmados = np.concatenate([[0], np.cumsum(np.where(ys, ds, 0))])
    cum_fp = np.concatenate([[0], np.cumsum(~ys)])
    cum_tp = np.concatenate([[0], np.cumsum(ys)])
    custo = tot - cum_dur_alarmados + c_fp * cum_fp + c_tp * cum_tp
    # só cortes entre valores distintos de p
    validos = np.r_[True, ps[1:] != ps[:-1], True]
    custo_v = np.where(validos, custo, np.inf)
    k = int(np.argmin(custo_v))
    if k == 0:
        return np.inf
    if k == len(ps):
        return -np.inf
    return (ps[k - 1] + ps[k]) / 2


def _fit_predict(fabrica, cols, tr, te, peso_fn=None):
    m = fabrica()
    X, y = tr[cols], tr["carga_alta"].values
    kw = {}
    if peso_fn is not None:
        kw["sample_weight"] = peso_fn(tr)
    try:
        m.fit(X, y, **kw)
    except TypeError:
        m.fit(X, y)
    return m.predict_proba(te[cols])[:, 1]


def avaliar(fabrica, cols, nome="", peso_fn=None, janela_meses=None, c_tp=0.0, verbose=True, dev=None):
    dev = DEV if dev is None else dev
    out = {"nome": nome, "n_feat": len(cols)}
    # --- prova V
    tr = dev[dev.conjunto == "treino"]; va = dev[dev.conjunto == "validacao"]
    if janela_meses:
        tr = tr[tr.t_ini >= tr.t_ini.max() - pd.Timedelta(days=30 * janela_meses)]
    pv = _fit_predict(fabrica, cols, tr, va, peso_fn)
    out["V_prauc"] = average_precision_score(va.carga_alta, pv)
    out["V_roc"] = roc_auc_score(va.carga_alta, pv)
    # --- prova WF
    oof = pd.Series(np.nan, index=dev.index)
    for m in MESES_WF:
        te = dev[dev.mes == m]
        tr = dev[dev.t_ini < te.t_ini.min()]
        if janela_meses:
            tr = tr[tr.t_ini >= te.t_ini.min() - pd.Timedelta(days=30 * janela_meses)]
        oof.loc[te.index] = _fit_predict(fabrica, cols, tr, te, peso_fn)
    w = dev[dev.mes.isin(MESES_WF)].copy(); w["p"] = oof[w.index]
    out["WF_prauc"] = average_precision_score(w.carga_alta, w.p)
    out["WF_prauc_mes"] = np.mean([average_precision_score(g.carga_alta, g.p) for _, g in w.groupby("mes")])
    rec = w[w.mes.isin(["2026-04", "2026-05", "2026-06"])]
    out["WF_prauc_abr_jun"] = average_precision_score(rec.carga_alta, rec.p)
    # --- minutos no WF com limiar aprendido só no passado
    for ctp, suf in [(0.0, ""), (1.0, "_tp1")]:
        tot_mod, tot_sem, alarmes = 0.0, 0.0, 0
        for m in MESES_CUSTO:
            passado = w[w.t_ini < w[w.mes == m].t_ini.min()]
            lim = melhor_limiar(passado.carga_alta, passado.p, passado.min_parada.fillna(0), 1.0, ctp)
            g = w[w.mes == m]
            a = g.p.values > lim
            tot_mod += minutos(g.carga_alta, g.min_parada.fillna(0), a, 1.0, ctp)
            tot_sem += g.min_parada.fillna(0).sum()
            alarmes += a.sum()
        n = w.mes.isin(MESES_CUSTO).sum()
        out[f"WF_min{suf}"] = tot_mod
        out[f"WF_economia{suf}_%"] = 100 * (1 - tot_mod / tot_sem)
        if suf == "":
            out["WF_alarmes_100"] = 100 * alarmes / n
            out["WF_min_sem_modelo"] = tot_sem
    # minutos ótimos (limiar escolhido olhando o próprio conjunto: só referência, otimista)
    out["V_min_oraculo_limiar"] = minutos(va.carga_alta, va.min_parada.fillna(0),
                                          pv > melhor_limiar(va.carga_alta, pv, va.min_parada.fillna(0)))
    if verbose:
        print(f"{nome:55s} V={out['V_prauc']:.3f}  WF={out['WF_prauc']:.3f} (mês {out['WF_prauc_mes']:.3f}, abr-jun {out['WF_prauc_abr_jun']:.3f})  "
              f"min={out['WF_min']:.0f} ({out['WF_economia_%']:.1f}%)  tp1={out['WF_economia_tp1_%']:.1f}%  alarmes/100={out['WF_alarmes_100']:.0f}")
    out["_oof"] = oof
    out["_pv"] = pv
    return out


FINAL6 = ["densidade_media", "prop_peso_pesado", "peso_por_ciclo_t", "taxa_ca_ult5", "taxa_ca_ult10", "prop_PACB"]


# ---------------------------------------------------------------- utilidades de comparação
def tabela_oof(o, dev=None):
    dev = DEV if dev is None else dev
    w = dev[dev.mes.isin(MESES_WF)][["cd_corrida", "mes", "t_ini", "carga_alta", "min_parada"]].copy()
    w["min_parada"] = w["min_parada"].fillna(0)
    w["p"] = o["_oof"][w.index]
    return w


def politica_limiar(w, politica="custo", c_tp=0.0, meses=MESES_CUSTO):
    """Devolve vetor de alarmes para os meses de custo, decidindo cada mês só com o passado."""
    from sklearn.linear_model import LogisticRegression
    alarme = pd.Series(False, index=w.index)
    for m in meses:
        g = w[w.mes == m]
        pas = w[w.t_ini < g.t_ini.min()]
        if politica == "custo":
            lim = melhor_limiar(pas.carga_alta, pas.p, pas.min_parada, 1.0, c_tp)
            alarme[g.index] = g.p > lim
        elif politica.startswith("platt"):
            # calibra no passado (últimos N meses se 'platt3') e alarma se p*D >= custo do alarme
            if politica == "platt3":
                pas = pas[pas.t_ini >= g.t_ini.min() - pd.Timedelta(days=92)]
            lr = LogisticRegression(C=1e6).fit(np.log(np.clip(pas.p, 1e-4, 1 - 1e-4) / (1 - np.clip(pas.p, 1e-4, 1 - 1e-4))).values.reshape(-1, 1), pas.carga_alta)
            z = np.log(np.clip(g.p, 1e-4, 1 - 1e-4) / (1 - np.clip(g.p, 1e-4, 1 - 1e-4))).values.reshape(-1, 1)
            pc = lr.predict_proba(z)[:, 1]
            dmed = pas.loc[pas.carga_alta == 1, "min_parada"].mean()
            # custo de alarmar: (1-p)*1 + p*c_tp ; custo de não alarmar: p*D
            alarme[g.index] = pc * dmed > (1 - pc) * 1.0 + pc * c_tp
    return alarme


def minutos_politica(w, alarme, c_tp=0.0, meses=MESES_CUSTO):
    g = w[w.mes.isin(meses)]
    a = alarme[g.index].values
    return minutos(g.carga_alta, g.min_parada, a, 1.0, c_tp), g.min_parada.sum(), a.mean()


def bootstrap_dif(wa, wb, n=2000, seed=0, c_tp=0.0, alarme_a=None, alarme_b=None, meses=MESES_CUSTO):
    """Bootstrap em blocos de dia (corridas do mesmo dia juntas). Diferença B - A."""
    rng = np.random.default_rng(seed)
    g = wa[wa.mes.isin(meses)].copy()
    g["dia"] = g.t_ini.dt.tz_convert("America/Sao_Paulo").dt.date
    g["pa"] = wa.loc[g.index, "p"]; g["pb"] = wb.loc[g.index, "p"]
    if alarme_a is not None:
        g["ca"] = np.where(g.carga_alta == 1, g.min_parada * ~alarme_a[g.index] + c_tp * alarme_a[g.index], alarme_a[g.index] * 1.0)
        g["cb"] = np.where(g.carga_alta == 1, g.min_parada * ~alarme_b[g.index] + c_tp * alarme_b[g.index], alarme_b[g.index] * 1.0)
    dias = g.dia.unique(); idx = {d: np.where(g.dia.values == d)[0] for d in dias}
    dpr, dmin = [], []
    for _ in range(n):
        s = np.concatenate([idx[d] for d in rng.choice(dias, len(dias))])
        h = g.iloc[s]
        if h.carga_alta.sum() == 0:
            continue
        dpr.append(average_precision_score(h.carga_alta, h.pb) - average_precision_score(h.carga_alta, h.pa))
        if alarme_a is not None:
            dmin.append(h.cb.sum() - h.ca.sum())
    r = {"dif_prauc": np.mean(dpr), "ic90_prauc": np.percentile(dpr, [5, 95]), "P(B>A)": np.mean(np.array(dpr) > 0)}
    if dmin:
        r.update({"dif_min": np.mean(dmin), "ic90_min": np.percentile(dmin, [5, 95]), "P(B<A min)": np.mean(np.array(dmin) < 0)})
    return r
