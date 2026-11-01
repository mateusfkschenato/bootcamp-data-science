"""Finalistas declarados ANTES de abrir o teste (31/out/2026)."""
from modelos import *
S2=S+["h_ewm2","h_ewm5","h_ult1","h_taxa_24h","eh_cestao_2","tempo_montagem_min","rendimento_medio"]
T_TE=[c for c in TODAS if not c.startswith("te_")]
HIST_LAG1=[c for c in T_TE if c.startswith("h_") or c.startswith("taxa_ca_")]
LAG2=[c for c in DEV.columns if c.startswith("h_") and c.endswith("_lag2")]
T_TE_LAG2=[c for c in T_TE if c not in HIST_LAG1]+LAG2
VARIANTES={"l2=5":dict(l2=5.0),"lr.05":dict(lr=0.05,l2=5.0),"leaves16":dict(leaves=16,l2=5.0),"leaf20":dict(leaf=20,l2=5.0),"it150":dict(it=150,l2=5.0)}
def familia(cols):
    return [(HGB(**kw), cols) for kw in VARIANTES.values()]
FINALISTAS={
  # nome: lista de (fabrica, colunas) cuja média de probabilidade é o modelo
  "PRINCIPAL: média de 5 HGB, 120 features (histórico lag1)": familia(T_TE),
  "Equilíbrio: 5 HGB 120 feat + 5 HGB 16 feat": familia(T_TE)+familia(S2),
  "PRINCIPAL + resultado do cestão 1 (se a EVCOMX confirmar)": familia(T_TE+["carga_alta_c1"]),
  "PRINCIPAL com histórico seguro (lag2)": familia(T_TE_LAG2),
  "Referência: RF 6 features (plano A do grupo)": [(RF(), FINAL6)],
}
if __name__=="__main__":
    print(len(T_TE), len(T_TE_LAG2), LAG2)
    for k,(fab,cols) in [("c1",(None,T_TE+["carga_alta_c1"])),("lag2",(None,T_TE_LAG2))]:
        nomes=[]
        for v,kw in VARIANTES.items():
            nm=f"HGB todas-TE {k} {v}"; rodar(nm, HGB(**kw), cols); nomes.append(nm)
        ensemble(f"ENS família todas-TE {k}", nomes)
