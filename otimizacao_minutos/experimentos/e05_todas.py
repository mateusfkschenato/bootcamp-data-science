import warnings; warnings.filterwarnings("ignore")
from harness import *
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
excl={"cd_corrida","nu_carregamento","conjunto","inicio","carga_alta","min_parada","t","t_ini","t_fim","mes","ordem","carga_alta_c1"}
todas=[c for c in DEV.columns if c not in excl and not c.endswith("_lag2") and DEV[c].dtype!=object]
print(len(todas))
hgb=lambda: HistGradientBoostingClassifier(learning_rate=0.03, max_iter=300, max_leaf_nodes=8, min_samples_leaf=40, l2_regularization=1.0, random_state=42)
rf=lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=500, min_samples_leaf=5, max_features=0.2, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
avaliar(hgb,todas,"HGB todas")
avaliar(rf,todas,"RF todas")
# importância por permutação: treino até mar/2026, avalia abr-jun (dev)
tr=DEV[DEV.t_ini<pd.Timestamp("2026-03-01",tz="UTC")]; te=DEV[DEV.t_ini>=pd.Timestamp("2026-03-01",tz="UTC")]
m=rf().fit(tr[todas],tr.carga_alta)
pi=permutation_importance(m,te[todas],te.carga_alta,scoring="average_precision",n_repeats=10,random_state=0,n_jobs=-1)
imp=pd.Series(pi.importances_mean,index=todas).sort_values(ascending=False)
print(imp.head(30).round(4))
imp.to_csv("r05_perm.csv")
