import warnings; warnings.filterwarnings("ignore")
from harness import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
REC = ["densidade_media", "prop_peso_pesado", "peso_por_ciclo_t", "prop_PACB"]
S = REC+["h_taxa3","h_taxa5","h_taxa10","h_taxa20","gusl_t"]
def rf(cw="balanced_subsample", leaf=5):
    return lambda: Pipeline([("imp",SimpleImputer(strategy="median")),("m",RandomForestClassifier(n_estimators=500, min_samples_leaf=leaf, class_weight=cw, random_state=42, n_jobs=-1))])
class W:  # wrapper p/ passar sample_weight ao passo final
    def __init__(s,f,fn): s.f=f; s.fn=fn
    def __call__(s): 
        m=s.f(); fn=s.fn
        class M:
            def fit(self,X,y,sample_weight=None):
                m.fit(X,y,m__sample_weight=sample_weight); return self
            def predict_proba(self,X): return m.predict_proba(X)
        return M()
def recencia(meia_vida_dias):
    def f(tr):
        idade=(tr.t_ini.max()-tr.t_ini).dt.total_seconds()/86400
        return 0.5**(idade/meia_vida_dias)
    return f
def duracao(tr):
    w=np.where(tr.carga_alta==1, tr.min_parada.fillna(0)/tr.loc[tr.carga_alta==1,"min_parada"].mean(), 1.0)
    return w
rows=[]
def run(*a,**k):
    o=avaliar(*a,**k); rows.append({kk:v for kk,v in o.items() if not kk.startswith('_')})
run(rf(),S,"base")
for hl in [30,60,120]:
    run(W(rf(),recencia(hl)),S,f"recência meia-vida {hl}d",peso_fn=recencia(hl))
for jm in [3,6]:
    run(rf(),S,f"janela {jm} meses",janela_meses=jm)
run(W(rf(),duracao),S,"positivos pesados pela duração",peso_fn=duracao)
run(W(rf(cw=None),duracao),S,"duração sem class_weight",peso_fn=duracao)
run(rf(cw=None),S,"sem class_weight")
pd.DataFrame(rows).to_csv("r04_pesos.csv",index=False)
