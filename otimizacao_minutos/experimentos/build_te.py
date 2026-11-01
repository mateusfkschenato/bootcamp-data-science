"""Risco recente por material e por baia ("target encoding" no tempo, sem vazamento).

Ideia física: a densidade da tabela é fixa, mas a sucata real de cada pilha/baia muda. Se os cestões
que usaram a baia 6 nos últimos dias deram carga alta, a pilha da baia 6 pode estar mais "fofa".

Para cada cestão da corrida N, olha só os cestões das corridas anteriores (mesma suposição do
taxa_ca_ult5: o resultado da corrida N-1 já é conhecido). Variante _lag2: só até a corrida N-2.

Para cada chave (material, baia, material x baia) guarda o histórico (em ordem) dos cestões que
a usaram, e calcula a taxa de carga alta nos últimos K usos, encolhida para a taxa global:
    te = (soma_ca + A * prior) / (n + A)
A feature do cestão é a média dessas taxas ponderada pelo volume de cada camada.
Saída: acrescenta colunas em exp/dados_exp.pkl
"""
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
AQUI = Path(__file__).resolve().parent
D = pd.read_pickle(AQUI / "dados_exp.pkl")
b = pd.read_csv(RAIZ / "EDA/base.csv")
b = b[b.nu_carregamento < 3].copy()
b["cd_baia"] = b["cd_baia"].astype(str)
b["mat_baia"] = b["cd_codigo_material"].astype(str) + "@" + b["cd_baia"]
chave = ["cd_corrida", "nu_carregamento"]
y_cestao = D.set_index(chave)["carga_alta"]
ordem_corr = D.drop_duplicates("cd_corrida").set_index("cd_corrida")["ordem"].sort_values()

camadas = {k: g for k, g in b.groupby(chave)}
A = float(__import__("os").environ.get("TE_A", 10.0))
K_LIST = [int(x) for x in __import__("os").environ.get("TE_K", "10,30").split(",")]

def construir(lag):
    hist = {tipo: defaultdict(lambda: deque(maxlen=max(K_LIST))) for tipo in ["mat", "baia", "mb"]}
    glob = deque(maxlen=200)
    pend = deque()                 # corridas esperando para entrar no histórico
    linhas = []
    for corr in ordem_corr.index:
        prior = np.mean(glob) if glob else 0.16
        for nc in [1.0, 2.0]:
            k = (corr, nc)
            if k not in camadas:
                continue
            g = camadas[k]
            vol = g["volume_m3"].values
            row = {"cd_corrida": corr, "nu_carregamento": nc}
            for tipo, col in [("mat", "cd_codigo_material"), ("baia", "cd_baia"), ("mb", "mat_baia")]:
                for K in K_LIST:
                    vals, mx = [], 0.0
                    for chave_v in g[col].values:
                        h = list(hist[tipo][chave_v])[-K:]
                        te = (sum(h) + A * prior) / (len(h) + A)
                        vals.append(te)
                    vals = np.array(vals)
                    row[f"te_{tipo}{K}"] = float((vals * vol).sum() / vol.sum())
                    row[f"te_{tipo}{K}_max"] = float(vals.max())
            row["te_prior200"] = prior
            linhas.append(row)
        # a corrida atual entra no histórico depois de `lag` corridas
        resultado = []
        for nc in [1.0, 2.0]:
            k = (corr, nc)
            if k in camadas:
                g = camadas[k]
                resultado.append((y_cestao[k], set(g["cd_codigo_material"]), set(g["cd_baia"]), set(g["mat_baia"])))
        pend.append(resultado)
        while len(pend) >= lag:
            for yv, mats, baias, mbs in pend.popleft():
                glob.append(yv)
                for m in mats: hist["mat"][m].append(yv)
                for bb in baias: hist["baia"][bb].append(yv)
                for x in mbs: hist["mb"][x].append(yv)
    return pd.DataFrame(linhas)

t1 = construir(lag=1)
t2 = construir(lag=2)
t2 = t2.rename(columns={c: c + "_lag2" for c in t2.columns if c.startswith("te_")})
D = D.drop(columns=[c for c in D.columns if c.startswith("te_")], errors="ignore")
D = D.merge(t1, on=chave, how="left").merge(t2, on=chave, how="left")
D.to_pickle(AQUI / "dados_exp.pkl")
print(D.shape)
dev = D[D.conjunto != "teste"]
from sklearn.metrics import average_precision_score
for c in [c for c in D.columns if c.startswith("te_")]:
    v = dev[["carga_alta", c]].dropna()
    print(f"{c:22s} PR-AUC sozinha (dev): {average_precision_score(v.carga_alta, v[c]):.3f}")
print("taxa_ca_ult5 sozinha:", round(average_precision_score(dev.carga_alta, dev.taxa_ca_ult5.fillna(0)), 3))
