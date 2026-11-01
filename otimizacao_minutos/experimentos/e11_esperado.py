from modelos import *
from sklearn.ensemble import HistGradientBoostingRegressor
class Esperado:
    """score = P(carga alta) * E[duração | carga alta, x]. 'min_parada' entra só no fit (alvo do regressor)."""
    def __init__(self, clf, reg, cols_reg=None, shrink=0.5, cols_clf=None):
        self.clf=clf; self.reg=reg; self.cols_reg=cols_reg; self.shrink=shrink; self.cols_clf=cols_clf
    def fit(self, X, y, sample_weight=None):
        dur=X["min_parada"].values; X=X.drop(columns="min_parada")
        self.cc=self.cols_clf or list(X.columns); self.clf.fit(X[self.cc],y)
        pos=y==1; cr=self.cols_reg or list(X.columns)
        self.reg.fit(X.loc[pos,cr], np.log1p(dur[pos]))
        self.mu=np.log1p(dur[pos]).mean(); self.cr=cr; return self
    def predict_proba(self, X):
        X=X.drop(columns="min_parada")
        p=self.clf.predict_proba(X[self.cc])[:,1]
        ld=self.reg.predict(X[self.cr]); ld=self.mu+self.shrink*(ld-self.mu)
        s=p*np.expm1(ld)/4.0
        s=np.clip(s,0,0.999)
        return np.c_[1-s,s]
regH=lambda: HistGradientBoostingRegressor(learning_rate=0.03,max_iter=200,max_leaf_nodes=4,min_samples_leaf=40,l2_regularization=10)
T_TE=[c for c in TODAS if not c.startswith("te_")]
S2=S+["h_ewm2","h_ewm5","h_ult1","h_taxa_24h","eh_cestao_2","tempo_montagem_min","rendimento_medio"]
ALL=S2 + [c for c in T_TE if c not in S2] + ["min_parada"]
for sh in [0.5,1.0]:
    rodar(f"ESP2 HGB S2 x regH todas sh{sh}", lambda sh=sh: Esperado(HGB(l2=5.0)(), regH(), T_TE, sh, S2), ALL)
    rodar(f"ESP2 HGB todas-TE x regH sh{sh}", lambda sh=sh: Esperado(HGB(l2=5.0)(), regH(), T_TE, sh, T_TE), ALL)
    rodar(f"ESP2 RF6 x regH sh{sh}", lambda sh=sh: Esperado(RF()(), regH(), T_TE, sh, FINAL6), list(dict.fromkeys(ALL[:-1]+FINAL6))+["min_parada"])
