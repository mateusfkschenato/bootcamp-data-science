import warnings; warnings.filterwarnings("ignore")
from harness import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
print("sanity h_taxa5 == taxa_ca_ult5:", np.nanmax(np.abs(DEV.h_taxa5-DEV.taxa_ca_ult5)))
rf = lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
lr = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
class Regra:
    def __init__(s,col): s.col=col
    def fit(s,X,y,**k): return s
    def predict_proba(s,X): v=X[s.col].fillna(0).values; return np.c_[1-v,v]
res=[avaliar(lambda: Regra("taxa_ca_ult5"), ["taxa_ca_ult5"], "regra histórico ult5"),
     avaliar(rf, FINAL6, "RF plano A (6) [grupo]"),
     avaliar(lr, FINAL6, "LogReg (6)")]
