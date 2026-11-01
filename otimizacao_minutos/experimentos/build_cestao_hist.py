"""Histórico no nível do CESTÃO: para o cestão 2 inclui o cestão 1 da mesma corrida (confirmado
pela EVCOMX); para o cestão 1 vai até o cestão 2 da corrida anterior (mesma suposição do histórico lag1)."""
import numpy as np, pandas as pd
D=pd.read_pickle("dados_exp.pkl")
D=D.drop(columns=[c for c in D.columns if c.startswith("hc_")],errors="ignore")
s=D.sort_values(["ordem","nu_carregamento"]).reset_index()
y=s["carga_alta"].astype(float); yp=y.shift(1)
f=pd.DataFrame({"index":s["index"]})
f["hc_ult1"]=yp
f["hc_ult2"]=y.shift(2)
for n in [2,4,6,10,20]: f[f"hc_taxa{n}"]=yp.rolling(n,min_periods=1).mean()
for hl in [1,3,8]: f[f"hc_ewm{hl}"]=yp.ewm(halflife=hl,min_periods=1).mean()
mp=s["min_parada"].fillna(0)
f["hc_min_parada4"]=mp.shift(1).rolling(4,min_periods=1).sum()
# mesmo cestão (1 ou 2) nas corridas anteriores
for c in [1.0,2.0]:
    m=s.nu_carregamento==c
    yy=y[m].shift(1)
    f.loc[m,"hc_mesmo_taxa5"]=yy.rolling(5,min_periods=1).mean().values
    f.loc[m,"hc_mesmo_ewm3"]=yy.ewm(halflife=3,min_periods=1).mean().values
f=f.set_index("index")
D=D.join(f)
D["hc_c1_x_hist"]=D["carga_alta_c1"]*D["h_taxa5"].fillna(0)
D.to_pickle("dados_exp.pkl")
dev=D[D.conjunto!="teste"]
from sklearn.metrics import average_precision_score as ap
for c in [c for c in D.columns if c.startswith("hc_")]: print(c, round(ap(dev.carga_alta, dev[c].fillna(0)),3))
