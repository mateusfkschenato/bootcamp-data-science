from modelos import *
S2=S+["h_ewm2","h_ewm5","h_ult1","h_taxa_24h","eh_cestao_2","tempo_montagem_min","rendimento_medio"]
T_TE=[c for c in TODAS if not c.startswith("te_")]
for tag,kw in [("lr.05",dict(lr=0.05,l2=5.0)),("leaves16",dict(leaves=16,l2=5.0)),("leaf20",dict(leaf=20,l2=5.0)),("it150",dict(it=150,l2=5.0))]:
    a=rodar(f"HGB S2 {tag}", HGB(**kw), S2); b=rodar(f"HGB todas-TE {tag}", HGB(**kw), T_TE)
    ensemble(f"ENS {tag}", [f"HGB S2 {tag}", f"HGB todas-TE {tag}"])
