import warnings; warnings.filterwarnings("ignore")
from harness import *
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import pickle
excl={"cd_corrida","nu_carregamento","conjunto","inicio","carga_alta","min_parada","t","t_ini","t_fim","mes","ordem","carga_alta_c1"}
todas=[c for c in DEV.columns if c not in excl and not c.endswith("_lag2") and DEV[c].dtype!=object]
REC = ["densidade_media", "prop_peso_pesado", "peso_por_ciclo_t", "prop_PACB"]
S = REC+["h_taxa3","h_taxa5","h_taxa10","h_taxa20","gusl_t"]
rf=lambda leaf=5: (lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, class_weight="balanced_subsample", random_state=42, n_jobs=-1)))
hgb=lambda: HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1.0, random_state=42)
cands={"RF6 (grupo)":(rf(),FINAL6),"RF S":(rf(),S),"HGB todas":(hgb,todas)}
W={}
for k,(f,c) in cands.items():
    W[k]=tabela_oof(avaliar(f,c,k))
pickle.dump(W,open("oof_e06.pkl","wb"))
for k,w in W.items():
    for pol in ["custo","platt","platt3"]:
        for ctp in [0.0,1.0]:
            a=politica_limiar(w,pol,ctp); mm,sem,ar=minutos_politica(w,a,ctp)
            print(f"{k:12s} {pol:7s} c_tp={ctp:.0f}: {mm:6.0f} min de {sem:.0f} ({100*(1-mm/sem):5.1f}% economia) alarmes={100*ar:4.1f}/100")
base=W["RF6 (grupo)"]
for k in ["RF S","HGB todas"]:
    print(k,"vs RF6:",bootstrap_dif(base,W[k],alarme_a=politica_limiar(base,"platt"),alarme_b=politica_limiar(W[k],"platt")))
