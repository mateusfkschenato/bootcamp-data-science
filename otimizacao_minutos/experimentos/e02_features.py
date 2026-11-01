import warnings; warnings.filterwarnings("ignore")
from harness import *
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
rf = lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
REC = ["densidade_media", "prop_peso_pesado", "peso_por_ciclo_t", "prop_PACB"]
sets = {
 "6 finais": FINAL6,
 "rec + ult5,10 + ult1": REC+["taxa_ca_ult5","taxa_ca_ult10","h_ult1"],
 "rec + ewm2,ewm5": REC+["h_ewm2","h_ewm5"],
 "rec + ewm2,5,15": REC+["h_ewm2","h_ewm5","h_ewm15"],
 "rec + ult3,5,10,20": REC+["h_taxa3","h_taxa5","h_taxa10","h_taxa20"],
 "rec + taxa24h,8h": REC+["h_taxa_24h","h_taxa_8h"],
 "rec + ult5,10 + 24h": REC+["taxa_ca_ult5","taxa_ca_ult10","h_taxa_24h"],
 "rec + ult5,10 + desde_ultima": REC+["taxa_ca_ult5","taxa_ca_ult10","h_desde_ultima"],
 "rec + ult5,10 + min_parada10": REC+["taxa_ca_ult5","taxa_ca_ult10","h_min_parada10"],
 "6 + gusl_t": FINAL6+["gusl_t"],
 "6 + gap_corrida": FINAL6+["gap_corrida_min"],
 "6 + eh_cestao_2": FINAL6+["eh_cestao_2"],
 "6 + rel_dens,rel_pes": FINAL6+["rel_densidade_media","rel_prop_peso_pesado"],
 "rel2 versões no lugar": ["rel2_densidade_media","rel2_prop_peso_pesado","rel2_peso_por_ciclo_t","prop_PACB","taxa_ca_ult5","taxa_ca_ult10"],
 "6 + carga_alta_c1 (se disponível)": FINAL6+["carga_alta_c1"],
 "6 lag2 (histórico seguro)": REC+["h_taxa5_lag2","h_taxa10_lag2"],
 "6 + volume, peso": FINAL6+["volume_m3_total","peso_t_total"],
}
rows=[]
for k,v in sets.items():
    o=avaliar(rf,v,k); rows.append({kk:vv for kk,vv in o.items() if not kk.startswith("_")})
pd.DataFrame(rows).to_csv("r02_features.csv",index=False)
