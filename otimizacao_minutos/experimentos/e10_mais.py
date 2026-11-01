from modelos import *
# monotonia física: + histórico => + risco; + pesado/densidade => - risco
def mono_for(cols):
    m={"densidade_media":-1,"prop_peso_pesado":-1,"h_taxa3":1,"h_taxa5":1,"h_taxa10":1,"h_taxa20":1,"h_ewm2":1,"h_ewm5":1,"h_ewm15":1,"taxa_ca_ult5":1,"taxa_ca_ult10":1,"taxa_ca_ult20":1,"h_taxa_24h":1}
    return [m.get(c,0) for c in cols]
rodar("HGB todas l2=5 mono", HGB(l2=5.0, mono=mono_for(TODAS)), TODAS)
rodar("HGB S mono", HGB(l2=5.0, mono=mono_for(S)), S)
S2=S+["h_ewm2","h_ewm5","h_ult1","h_taxa_24h","eh_cestao_2","tempo_montagem_min","rendimento_medio"]
rodar("RF S2", RF(), S2)
rodar("RF S2 sem cw", RF(cw=None), S2)
rodar("HGB S2 l2=5", HGB(l2=5.0), S2)
LRf=lambda C: (lambda: make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=4, degree=2), StandardScaler(), LogisticRegression(C=C, max_iter=5000)))
rodar("LRspl S2 C.1", LRf(0.1), S2)
rodar("LR todas L2 C.02", lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.02, max_iter=5000)), TODAS)
