from modelos import *
S2=S+["h_ewm2","h_ewm5","h_ult1","h_taxa_24h","eh_cestao_2","tempo_montagem_min","rendimento_medio"]
T_TE=[c for c in TODAS if not c.startswith("te_")]
rodar("HGB S2+c1", HGB(l2=5.0), S2+["carga_alta_c1"])
rodar("HGB todas-TE+c1", HGB(l2=5.0), T_TE+["carga_alta_c1"])
ensemble("ENS (S2+c1)+(todas-TE+c1)", ["HGB S2+c1","HGB todas-TE+c1"])
# histórico 'seguro' (lag2) para medir dependência do timing
S2L=[c.replace("h_taxa3","h_taxa3_lag2").replace("h_taxa5","h_taxa5_lag2").replace("h_taxa10","h_taxa10_lag2").replace("h_taxa20","h_taxa20_lag2").replace("h_ewm2","h_ewm2_lag2").replace("h_ewm5","h_ewm5_lag2") for c in S2 if c not in ("h_ult1","h_taxa_24h")]
rodar("HGB S2 lag2", HGB(l2=5.0), S2L)
TL=[c for c in T_TE if not (c.startswith("h_") or c.startswith("taxa_ca"))]+[c for c in DEV.columns if c.endswith("_lag2") and c.startswith("h_")]
rodar("HGB todas-TE lag2", HGB(l2=5.0), TL)
ensemble("ENS lag2", ["HGB S2 lag2","HGB todas-TE lag2"])
