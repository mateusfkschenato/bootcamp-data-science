import warnings; warnings.filterwarnings("ignore")
from harness import *
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
imp=lambda: SimpleImputer(strategy="median")
REC = ["densidade_media", "prop_peso_pesado", "peso_por_ciclo_t", "prop_PACB"]
HIST = ["h_taxa3","h_taxa5","h_taxa10","h_taxa20","h_ewm2","h_ewm5","h_ewm15","h_ult1","h_desde_ultima"]
sets={"6":FINAL6, "rec+hist9":REC+HIST, "rec+hist9+te":REC+HIST+["te_mb10","te_mat10"],"rec+ult3-20+gusl":REC+["h_taxa3","h_taxa5","h_taxa10","h_taxa20","gusl_t"]}
modelos={
 "RF leaf5": lambda: make_pipeline(imp(), RandomForestClassifier(n_estimators=500, min_samples_leaf=5, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
 "RF leaf20": lambda: make_pipeline(imp(), RandomForestClassifier(n_estimators=500, min_samples_leaf=20, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
 "ET leaf10": lambda: make_pipeline(imp(), ExtraTreesClassifier(n_estimators=500, min_samples_leaf=10, class_weight="balanced_subsample", random_state=42, n_jobs=-1)),
 "HGB lento": lambda: HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1.0, random_state=42),
 "LR splines": lambda: make_pipeline(imp(), SplineTransformer(n_knots=4, degree=2), StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
}
rows=[]
for sn,cols in sets.items():
    for mn,f in modelos.items():
        o=avaliar(f,cols,f"{mn} | {sn}"); rows.append({k:v for k,v in o.items() if not k.startswith('_')})
pd.DataFrame(rows).to_csv("r03_algos.csv",index=False)
