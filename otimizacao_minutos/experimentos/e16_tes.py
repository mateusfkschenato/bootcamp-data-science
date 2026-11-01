from modelos import *
print(len(TODAS))
S2=S+["h_ewm2","h_ewm5","h_ult1","h_taxa_24h","eh_cestao_2","tempo_montagem_min","rendimento_medio"]
T_TE=[c for c in TODAS if not c.startswith("te_")]
rodar("HGB S2+tes5", HGB(l2=5.0), S2+["tes_mat5","tes_mb5","tes_mb5_max"])
rodar("HGB todas-TE+tes", HGB(l2=5.0), T_TE+["tes_mat3","tes_mat5","tes_mb5","tes_mb5_max","tes_baia5"])
ensemble("ENS tes", ["HGB S2+tes5","HGB todas-TE+tes"])
