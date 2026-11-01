"""Segunda olhada no teste (31/out, depois da confirmação do cestão 1). Candidato escolhido SÓ no dev:
'ENS FAM c1+hc + FAM pequeno' (maior PR-AUC no walk-forward sem piorar os minutos). Compara com o
escolhido anterior (família c1)."""
import warnings; warnings.filterwarnings("ignore")
from modelos import *
from sklearn.metrics import *
T_TE=[c for c in TODAS if not c.startswith("te_")]; BASE=T_TE+["carga_alta_c1"]; HC=[c for c in DEV.columns if c.startswith("hc_")]
S2=S+["h_ewm2","h_ewm5","h_ult1","h_taxa_24h","eh_cestao_2","tempo_montagem_min","rendimento_medio"]
small=S2+["carga_alta_c1","hc_mesmo_ewm3","hc_ewm3"]
VAR=[dict(l2=5.0),dict(lr=0.05,l2=5.0),dict(leaves=16,l2=5.0),dict(leaf=20,l2=5.0),dict(it=150,l2=5.0)]
CAND={"novo: 5 HGB grandes (c1+hc) + 5 HGB pequenos":([(kw,BASE+HC) for kw in VAR]+[(kw,small) for kw in VAR],"ENS FAM c1+hc + FAM pequeno"),
      "anterior: 5 HGB (c1)":([(kw,BASE) for kw in VAR],"ENS família todas-TE c1")}
TE=D[D.conjunto=="teste"].copy(); TE["min_parada"]=TE.min_parada.fillna(0); y=TE.carga_alta.values
c=cache(); out={}
for nome,(membros,chave) in CAND.items():
    p=np.mean([HGB(**kw)().fit(DEV[cols],DEV.carga_alta).predict_proba(TE[cols])[:,1] for kw,cols in membros],axis=0)
    w=tabela_oof(c[chave]); lim=melhor_limiar(w.carga_alta,w.p,w.min_parada)
    a=(p>lim).astype(int); tn,fp,fn,tp=confusion_matrix(y,a).ravel()
    out[nome]=p
    print(f"{nome}: limiar {lim:.3f} | PR-AUC {average_precision_score(y,p):.3f} ROC {roc_auc_score(y,p):.3f} | F1 {f1_score(y,a):.3f} F2 {fbeta_score(y,a,beta=2):.3f} MCC {matthews_corrcoef(y,a):.3f} | "
          f"recall {tp/(tp+fn):.2f} precisão {tp/(tp+fp):.3f} | TN {tn} FP {fp} FN {fn} TP {tp} | min {minutos(y,TE.min_parada,a):.1f} (TP=1: {minutos(y,TE.min_parada,a,1,1):.1f}) de {TE.min_parada.sum():.0f}")
# bootstrap por dia: novo - anterior
TE["dia"]=TE.t_ini.dt.tz_convert("America/Sao_Paulo").dt.date; dias=TE.dia.unique(); idx={d:np.where(TE.dia.values==d)[0] for d in dias}
rng=np.random.default_rng(0); k=list(out); dd=[]
for _ in range(2000):
    s=np.concatenate([idx[d] for d in rng.choice(dias,len(dias))])
    if y[s].sum()==0: continue
    dd.append(average_precision_score(y[s],out[k[0]][s])-average_precision_score(y[s],out[k[1]][s]))
print("dif PR-AUC novo-anterior:",round(np.mean(dd),3),"IC90",np.round(np.percentile(dd,[5,95]),3))
