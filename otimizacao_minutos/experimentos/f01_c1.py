from modelos import *
T_TE=[c for c in TODAS if not c.startswith("te_")]
BASE=T_TE+["carga_alta_c1"]
VAR={"l2=5":dict(l2=5.0),"lr.05":dict(lr=0.05,l2=5.0),"leaves16":dict(leaves=16,l2=5.0),"leaf20":dict(leaf=20,l2=5.0),"it150":dict(it=150,l2=5.0)}
def fam(tag, cols, extra=None):
    nomes=[]
    for v,kw in VAR.items():
        kw=dict(kw, **(extra or {})); n=f"[{tag}] {v}"; rodar(n, HGB(**kw), cols); nomes.append(n)
    return ensemble(f"FAM {tag}", nomes)
# ablações por grupo (sobre a base com c1)
G={"ritmo":["gap_corrida_min","gap_c2ant_c1_min"],
   "tempo":["tempo_montagem_min","intervalo_medio_min","hora","turno_A","turno_B","dia_semana","fim_de_semana"],
   "materiais":[c for c in T_TE if c.startswith("prop_") and c not in ["prop_leve","prop_vol_leve","prop_peso_misto","prop_vol_misto","prop_peso_pesado","prop_vol_pesado","prop_leve_topo","prop_leve_base"]]+["num_materiais","num_baias","hhi_materiais"],
   "rel":[c for c in T_TE if c.startswith("rel")],
   "gusa":["gusl_t","tem_gusl","carga3_solida_t"]}
import sys
if sys.argv[1]=="abl":
    for g,cols in G.items():
        fam(f"c1 sem {g}", [c for c in BASE if c not in cols])
    fam("c1 sem ritmo,tempo,materiais", [c for c in BASE if c not in G["ritmo"]+G["tempo"]+G["materiais"]])
HC=[c for c in DEV.columns if c.startswith("hc_")]
if sys.argv[1]=="hc":
    fam("c1 + hc", BASE+HC)
    fam("c1 + hc_mesmo", BASE+["hc_mesmo_taxa5","hc_mesmo_ewm3"])
    fam("c1 + hc_mesmo + hc_ewm", BASE+["hc_mesmo_taxa5","hc_mesmo_ewm3","hc_ewm3","hc_ewm8","hc_taxa10"])
class Separado:
    """Um modelo para o cestão 1 e outro para o cestão 2."""
    def __init__(self, fab): self.fab=fab
    def fit(self, X, y, sample_weight=None):
        self.m={}
        for c in [0,1]:
            s=(X["eh_cestao_2"]==c).values; self.m[c]=self.fab().fit(X[s],y[s])
        return self
    def predict_proba(self, X):
        p=np.zeros(len(X))
        for c in [0,1]:
            s=(X["eh_cestao_2"]==c).values
            if s.any(): p[s]=self.m[c].predict_proba(X[s])[:,1]
        return np.c_[1-p,p]
S2=S+["h_ewm2","h_ewm5","h_ult1","h_taxa_24h","eh_cestao_2","tempo_montagem_min","rendimento_medio"]
if sys.argv[1]=="sep":
    nomes=[]
    for v,kw in VAR.items():
        n=f"[c1 separado] {v}"; rodar(n, (lambda kw=kw: Separado(HGB(**kw))), BASE+HC); nomes.append(n)
    ensemble("FAM c1 separado c1/c2 (+hc)", nomes)
    # modelos pequenos com c1 (diversidade)
    small=S2+["carga_alta_c1","hc_mesmo_ewm3","hc_ewm3"]
    fam("pequeno S2+c1+hc", small)
    rodar("RF pequeno S2+c1+hc", RF(cw=None, leaf=5), small)
    rodar("RF grande c1+hc mf.2", RF(cw=None, mf=0.2), BASE+HC)
    ensemble("ENS FAM c1+hc + FAM pequeno", ["FAM c1 + hc","FAM pequeno S2+c1+hc"])
    ensemble("ENS FAM c1+hc + RF grande", ["FAM c1 + hc","RF grande c1+hc mf.2"])
    ensemble("ENS FAM c1+hc + separado", ["FAM c1 + hc","FAM c1 separado c1/c2 (+hc)"])
    ensemble("ENS FAM c1 + FAM c1+hc", ["FAM c1 sem ritmo","FAM c1 + hc"])
if sys.argv[1]=="ens":
    ensemble("ENS3 FAM c1+hc + pequeno + RF grande", ["FAM c1 + hc","FAM pequeno S2+c1+hc","RF grande c1+hc mf.2"])
    ensemble("ENS4 + separado", ["FAM c1 + hc","FAM pequeno S2+c1+hc","RF grande c1+hc mf.2","FAM c1 separado c1/c2 (+hc)"])
    ensemble("ENS FAM c1 (sem hc) + pequeno", ["ENS família todas-TE c1","FAM pequeno S2+c1+hc"])
    ensemble("ENS c1+hc + pequeno + RF pequeno", ["FAM c1 + hc","FAM pequeno S2+c1+hc","RF pequeno S2+c1+hc"])
    c=cache(); base=tabela_oof(c["ENS família todas-TE c1"]); ab=politica_limiar(base,"custo")
    for n in ["ENS FAM c1+hc + FAM pequeno","ENS3 FAM c1+hc + pequeno + RF grande","ENS4 + separado","ENS FAM c1 (sem hc) + pequeno","FAM c1 + hc"]:
        w=tabela_oof(c[n]); a=politica_limiar(w,"custo")
        r=bootstrap_dif(base,w,n=1000,alarme_a=ab,alarme_b=a)
        print(f"{n:42s} dPR={r['dif_prauc']:+.3f} IC90{np.round(r['ic90_prauc'],3)} P={r['P(B>A)']:.2f} | dMin={r['dif_min']:+.0f} IC90{np.round(r['ic90_min'],0)}")
