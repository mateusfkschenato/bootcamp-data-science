"""Protocolo comum do modelo oficial (usado por todas as etapas de feature selection em diante).

Tudo o que precisa ser IGUAL entre as pastas fica aqui, num lugar só:
- a régua de negócio da EVCOMX: minutos de forno perdidos (FN = duração real da parada,
  FP = 1 min de reajuste; TP = 0 min, ou 1 min no cenário conservador "TP=1");
- o limiar de menor minutos (varre todos os cortes possíveis);
- o walk-forward mensal no dev (treino + validação): para cada mês de nov/25 a jun/26, treina
  com tudo o que veio antes e prevê o mês; o limiar de cada mês vem só dos meses anteriores;
- a receita do MODELO OFICIAL: média das probabilidades de 5 HistGradientBoosting com
  hiperparâmetros levemente diferentes, sem pesos de classe, 120 features + carga_alta_c1;
- a receita da referência antiga (plano A do grupo): Random Forest com 6 features.

Para importar de dentro de uma pasta de etapa:
    import sys; from pathlib import Path
    RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "comum").is_dir())
    sys.path.insert(0, str(RAIZ)); from comum.protocolo import *
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

RAIZ = Path(__file__).resolve().parents[1]
ARQ_DATASET = RAIZ / "feature_selection" / "dataset_modelagem_fs.csv"
ARQ_COMPLETO = RAIZ / "feature_selection" / "dataset_features_completo.csv"
ARQ_FEATURES = RAIZ / "feature_selection" / "features_finais.json"
ARQ_MODELO = RAIZ / "selecao_algoritmos" / "modelo_final.joblib"

MESES_WF = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
MESES_CUSTO = MESES_WF[2:]          # minutos contados de jan a jun (nov/dez só "aquecem" o limiar)
C_FP = 1.0                          # alarme falso = 1 min (resposta do Guima, out/2026)

# Referência antiga: as 6 features da votação da feature selection (rodada RF, out/2026)
FINAL6 = ["densidade_media", "prop_peso_pesado", "peso_por_ciclo_t", "taxa_ca_ult5", "taxa_ca_ult10", "prop_PACB"]

# As 5 variações de hiperparâmetro do modelo oficial (o resto fica no padrão de HGB() abaixo)
VARIANTES = {"l2=5": dict(l2=5.0), "lr.05": dict(lr=0.05, l2=5.0), "leaves16": dict(leaves=16, l2=5.0),
             "leaf20": dict(leaf=20, l2=5.0), "it150": dict(it=150, l2=5.0)}

NOME_OFICIAL = "OFICIAL: 5 HGB, 120 features + resultado do cestão 1"
NOME_RESERVA = "Reserva: 5 HGB, 120 features (sem o cestão 1)"
NOME_RF6 = "Antigo: RF 6 features (plano A)"
NOME_REGRA = "Regra: só o histórico (taxa_ca_ult5)"


# ---------------------------------------------------------------- dados
def ler_features():
    """Listas de features do arquivo da feature selection."""
    f = json.load(open(ARQ_FEATURES, encoding="utf-8"))
    return f


def carregar(arquivo=ARQ_DATASET):
    """Lê o dataset de modelagem e devolve (D, DEV, TESTE), na mesma ordem usada desde a otimização."""
    D = pd.read_csv(arquivo)
    for c in ["t", "t_ini", "t_fim"]:
        if c in D:
            D[c] = pd.to_datetime(D[c], utc=True, format="mixed")
    D["min_parada"] = D["min_parada"].fillna(0)
    DEV = D[D.conjunto != "teste"].sort_values("t_ini").reset_index(drop=True)
    TESTE = D[D.conjunto == "teste"].sort_values("t_ini").reset_index(drop=True)
    return D, DEV, TESTE


# ---------------------------------------------------------------- régua de negócio
def minutos(y, dur, alarme, c_fp=C_FP, c_tp=0.0):
    """Minutos perdidos: parada inteira de cada carga alta sem alarme + c_fp por alarme falso + c_tp por alarme certo."""
    y = np.asarray(y).astype(bool); a = np.asarray(alarme).astype(bool); dur = np.asarray(dur)
    return float(dur[y & ~a].sum() + c_fp * (a & ~y).sum() + c_tp * (a & y).sum())


def melhor_limiar(y, p, dur, c_fp=C_FP, c_tp=0.0):
    """Limiar que minimiza os minutos perdidos (varre todos os cortes possíveis)."""
    p = np.asarray(p); y = np.asarray(y).astype(bool); dur = np.asarray(dur)
    o = np.argsort(-p); ps, ys, ds = p[o], y[o], dur[o]
    custo = (dur[y].sum() - np.r_[0, np.cumsum(np.where(ys, ds, 0))]
             + c_fp * np.r_[0, np.cumsum(~ys)] + c_tp * np.r_[0, np.cumsum(ys)])
    custo = np.where(np.r_[True, ps[1:] != ps[:-1], True], custo, np.inf)
    k = int(np.argmin(custo))
    return np.inf if k == 0 else (-np.inf if k == len(ps) else (ps[k - 1] + ps[k]) / 2)


# ---------------------------------------------------------------- modelos
def HGB(lr=0.03, it=300, leaves=8, leaf=40, l2=1.0, seed=42):
    """Fábrica de um HistGradientBoosting (padrão do projeto; as variantes mudam só um ou dois valores)."""
    return lambda: HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves,
                                                  min_samples_leaf=leaf, l2_regularization=l2, random_state=seed)


def RF6():
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample",
                                                random_state=42, n_jobs=-1))


def familia(cols, seed=42):
    """Os 5 membros do comitê: (fábrica, colunas)."""
    return [(HGB(seed=seed, **kw), cols) for kw in VARIANTES.values()]


def membros_oficial():
    return familia(ler_features()["features_oficial"])


def treinar(membros, tr):
    out = []
    for fab, cols in membros:
        m = fab(); m.fit(tr[cols], tr["carga_alta"]); out.append((m, cols))
    return out


def prever_treinados(treinados, te):
    return np.mean([m.predict_proba(te[cols])[:, 1] for m, cols in treinados], axis=0)


def prever(membros, tr, te):
    """Treina cada membro em `tr` e devolve a média das probabilidades em `te`."""
    return prever_treinados(treinar(membros, tr), te)


# ---------------------------------------------------------------- walk-forward
def oof_dev(membros, DEV, meses=MESES_WF):
    """Walk-forward mensal no dev: previsões fora da amostra de nov/25 a jun/26."""
    oof = pd.Series(np.nan, index=DEV.index)
    for mth in meses:
        te = DEV[DEV.mes == mth]
        oof[te.index] = prever(membros, DEV[DEV.t_ini < te.t_ini.min()], te)
    w = DEV[DEV.mes.isin(meses)].copy(); w["p"] = oof[w.index]
    return w


def minutos_wf(w, c_tp=0.0, meses=MESES_CUSTO):
    """Minutos de jan a jun no dev, com o limiar de cada mês vindo só dos meses anteriores."""
    tot, alarmes = 0.0, 0
    for mth in meses:
        g = w[w.mes == mth]; pas = w[w.t_ini < g.t_ini.min()]
        a = g.p.values > melhor_limiar(pas.carga_alta, pas.p, pas.min_parada, C_FP, c_tp)
        tot += minutos(g.carga_alta, g.min_parada, a, C_FP, c_tp); alarmes += a.sum()
    return tot, 100 * alarmes / w.mes.isin(meses).sum()


def resumo_wf(w):
    """PR-AUC do walk-forward e minutos de jan–jun (com e sem TP=1)."""
    from sklearn.metrics import average_precision_score
    m0, al = minutos_wf(w); m1, _ = minutos_wf(w, 1.0)
    sem = w[w.mes.isin(MESES_CUSTO)].min_parada.sum()
    return {"PR-AUC WF": average_precision_score(w.carga_alta, w.p), "min WF jan-jun": m0,
            "economia WF %": 100 * (1 - m0 / sem), "min WF (TP=1)": m1, "alarmes/100 WF": al}


# ---------------------------------------------------------------- uso
def prever_pacote(pacote, X, chave="com_cestao1"):
    """Chance de carga alta com o pacote salvo em selecao_algoritmos/modelo_final.joblib."""
    pk = pacote[chave]
    return np.mean([m.predict_proba(X[pk["features"]])[:, 1] for m in pk["modelos"]], axis=0)
