from modelos import *
ORIG=[c for c in pd.read_csv("../feature_selection/dataset_features_completo.csv",nrows=1).columns if c not in EXCL]
H=[c for c in TODAS if c.startswith("h_")]; TE=[c for c in TODAS if c.startswith("te_")]; REL=[c for c in TODAS if c.startswith("rel")]
OUTRAS=["gusl_t","tem_gusl","carga3_solida_t","gap_corrida_min","gap_c2ant_c1_min"]
print(len(ORIG),len(H),len(TE),len(REL))
rodar("HGB orig84", HGB(l2=5.0), ORIG)
rodar("HGB orig84+H", HGB(l2=5.0), ORIG+H)
rodar("HGB orig84+H+outras", HGB(l2=5.0), ORIG+H+OUTRAS)
rodar("HGB todas-REL", HGB(l2=5.0), [c for c in TODAS if c not in REL])
rodar("HGB todas-TE", HGB(l2=5.0), [c for c in TODAS if c not in TE])
rodar("HGB todas-REL-TE", HGB(l2=5.0), [c for c in TODAS if c not in TE+REL])
