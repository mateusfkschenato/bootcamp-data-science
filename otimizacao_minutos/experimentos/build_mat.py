"""Peso (t) por material e features do topo do cestão. Acrescenta em dados_exp.pkl."""
from pathlib import Path
import numpy as np, pandas as pd
RAIZ=Path(__file__).resolve().parents[1]; AQUI=Path(__file__).resolve().parent
D=pd.read_pickle(AQUI/"dados_exp.pkl")
b=pd.read_csv(RAIZ/"EDA/base.csv"); b=b[b.nu_carregamento<3].copy()
chave=["cd_corrida","nu_carregamento"]
D=D.drop(columns=[c for c in D.columns if c.startswith(("pt_","topo3_","topo5_","dens_topo","vol_topo"))],errors="ignore")
treino_corr=set(D.loc[D.conjunto=="treino","cd_corrida"])
freq=b[b.cd_corrida.isin(treino_corr)].groupby("cd_codigo_material").apply(lambda g: g[chave].drop_duplicates().shape[0])
n_ces=D[D.conjunto=="treino"].shape[0]
comuns=freq[freq/n_ces>=0.03].index
b["mat"]=np.where(b.cd_codigo_material.isin(comuns),b.cd_codigo_material,"outros")
pt=b.pivot_table(index=chave,columns="mat",values="peso_t",aggfunc="sum",fill_value=0).add_prefix("pt_")
b=b.sort_values(chave+["nu_camada"]); g=b.groupby(chave)
b["ordem_desc"]=g.cumcount(ascending=False)  # 0 = última camada (topo)
top=pd.DataFrame(index=pt.index)
for k in [3,5]:
    t=b[b.ordem_desc<k].groupby(chave)
    top[f"vol_topo{k}_m3"]=t.volume_m3.sum(); top[f"dens_topo{k}"]=t.peso_t.sum()/t.volume_m3.sum()
    top[f"topo{k}_leve_m3"]=b[(b.ordem_desc<k)&(b.tp_material=="Leve")].groupby(chave).volume_m3.sum().reindex(top.index).fillna(0)
D=D.merge(pt.reset_index(),on=chave,how="left").merge(top.reset_index(),on=chave,how="left")
D.to_pickle(AQUI/"dados_exp.pkl"); print(D.shape, list(pt.columns))
