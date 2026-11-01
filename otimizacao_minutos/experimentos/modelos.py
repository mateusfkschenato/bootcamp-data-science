import warnings; warnings.filterwarnings("ignore")
import pickle, os
from harness import *
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.linear_model import LogisticRegression
EXCL={"cd_corrida","nu_carregamento","conjunto","inicio","carga_alta","min_parada","t","t_ini","t_fim","mes","ordem","carga_alta_c1"}
NOVO_MAT=("pt_","topo3_","topo5_","dens_topo","vol_topo","tes_","hc_")
TODAS=[c for c in DEV.columns if c not in EXCL and not c.endswith("_lag2") and DEV[c].dtype!=object and not c.startswith(NOVO_MAT)]
MAT=[c for c in DEV.columns if c.startswith(NOVO_MAT)]
REC=["densidade_media","prop_peso_pesado","peso_por_ciclo_t","prop_PACB"]
S=REC+["h_taxa3","h_taxa5","h_taxa10","h_taxa20","gusl_t"]
def RF(leaf=5, cw="balanced_subsample", mf="sqrt", n=500, seed=42):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=n, min_samples_leaf=leaf, max_features=mf, class_weight=cw, random_state=seed, n_jobs=-1))
def HGB(lr=0.03, it=300, leaves=8, leaf=40, l2=1.0, seed=42, mono=None, cw=None, maxf=1.0):
    return lambda: HistGradientBoostingClassifier(learning_rate=lr, max_iter=it, max_leaf_nodes=leaves, min_samples_leaf=leaf, l2_regularization=l2, random_state=seed, monotonic_cst=mono, class_weight=cw, max_features=maxf)
CACHE="cache_oof.pkl"
def cache():
    return pickle.load(open(CACHE,"rb")) if os.path.exists(CACHE) else {}
def rodar(nome, fab, cols, **kw):
    c=cache()
    if nome in c: 
        o=c[nome]; print(f"[cache] {nome:50s} V={o['V_prauc']:.3f} WF={o['WF_prauc']:.3f} min={o['WF_min']:.0f} ({o['WF_economia_%']:.1f}%)"); return o
    o=avaliar(fab, cols, nome, **kw)
    o["cols"]=cols
    c=cache(); c[nome]=o; pickle.dump(c,open(CACHE,"wb"))
    return o

def ensemble(nome, membros, pesos=None, modo="prob"):
    c=cache(); pesos=pesos or [1]*len(membros)
    def comb(key):
        arr=[c[m][key] for m in membros]
        if modo=="logit":
            arr=[np.log(np.clip(a,1e-4,1-1e-4)/(1-np.clip(a,1e-4,1-1e-4))) for a in arr]
        v=sum(w*np.asarray(a) for w,a in zip(pesos,arr))/sum(pesos)
        if modo=="logit": v=1/(1+np.exp(-v))
        return v
    oof=pd.Series(comb("_oof"),index=c[membros[0]]["_oof"].index); pv=comb("_pv")
    class Fixo:
        pass
    # recalcula as métricas com as previsões combinadas
    o={"nome":nome,"n_feat":None}
    va=DEV[DEV.conjunto=="validacao"]
    o["V_prauc"]=average_precision_score(va.carga_alta,pv); o["V_roc"]=roc_auc_score(va.carga_alta,pv)
    w=tabela_oof({"_oof":oof})
    o["WF_prauc"]=average_precision_score(w.carga_alta,w.p)
    o["WF_prauc_mes"]=np.mean([average_precision_score(g.carga_alta,g.p) for _,g in w.groupby("mes")])
    rec=w[w.mes.isin(["2026-04","2026-05","2026-06"])]; o["WF_prauc_abr_jun"]=average_precision_score(rec.carga_alta,rec.p)
    for ctp,suf in [(0.0,""),(1.0,"_tp1")]:
        a=politica_limiar(w,"custo",ctp); mm,sem,ar=minutos_politica(w,a,ctp)
        o[f"WF_min{suf}"]=mm; o[f"WF_economia{suf}_%"]=100*(1-mm/sem)
        if suf=="": o["WF_alarmes_100"]=100*ar; o["WF_min_sem_modelo"]=sem
    o["_oof"]=oof; o["_pv"]=pv; o["membros"]=membros
    print(f"{nome:55s} V={o['V_prauc']:.3f}  WF={o['WF_prauc']:.3f} (mês {o['WF_prauc_mes']:.3f}, abr-jun {o['WF_prauc_abr_jun']:.3f})  min={o['WF_min']:.0f} ({o['WF_economia_%']:.1f}%)  tp1={o['WF_economia_tp1_%']:.1f}%  alarmes/100={o['WF_alarmes_100']:.0f}")
    c=cache(); c[nome]=o; pickle.dump(c,open(CACHE,"wb"))
    return o
