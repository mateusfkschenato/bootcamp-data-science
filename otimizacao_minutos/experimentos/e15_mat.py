from modelos import *
print(len(TODAS))
PT=[c for c in MAT if c.startswith("pt_")]; TOP=[c for c in MAT if not c.startswith("pt_")]
LR=lambda C: (lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=5000)))
H=["h_taxa3","h_taxa5","h_taxa10","h_taxa20","h_ewm2","h_ewm5"]
rodar("LR pt", LR(0.1), PT)
rodar("LR pt+eh2+H", LR(0.1), PT+["eh_cestao_2"]+H)
rodar("LR pt+eh2+H+gusl", LR(0.1), PT+["eh_cestao_2","gusl_t"]+H)
S2=S+["h_ewm2","h_ewm5","h_ult1","h_taxa_24h","eh_cestao_2","tempo_montagem_min","rendimento_medio"]
rodar("HGB S2+TOP", HGB(l2=5.0), S2+TOP)
T_TE=[c for c in TODAS if not c.startswith("te_")]
rodar("HGB todas-TE+MAT", HGB(l2=5.0), T_TE+MAT)
