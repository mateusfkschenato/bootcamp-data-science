from modelos import *
S2=S+["h_ewm2","h_ewm5","h_ult1","h_taxa_24h","eh_cestao_2","tempo_montagem_min","rendimento_medio"]
T_TE=[c for c in TODAS if not c.startswith("te_")]
def w_dur(tr): return np.where(tr.carga_alta==1, tr.min_parada.fillna(0).clip(lower=0.5), 1.0)
def w_dur_sqrt(tr): return np.where(tr.carga_alta==1, np.sqrt(tr.min_parada.fillna(0).clip(lower=0.5))*2, 1.0)
def w_const4(tr): return np.where(tr.carga_alta==1, 4.0, 1.0)
for nm,fn in [("dur",w_dur),("sqrtdur",w_dur_sqrt),("x4",w_const4)]:
    rodar(f"HGB todas-TE w{nm}", HGB(l2=5.0), T_TE, peso_fn=fn)
    rodar(f"HGB S2 w{nm}", HGB(l2=5.0), S2, peso_fn=fn)
