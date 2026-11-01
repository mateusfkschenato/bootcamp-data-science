from modelos import *
def por_mes(nomes, pol="custo", ctp=0.0):
    c=cache(); rows=[]
    for n in nomes:
        w=tabela_oof(c[n]); a=politica_limiar(w,pol,ctp)
        for m in MESES_CUSTO:
            g=w[w.mes==m]; rows.append({"modelo":n,"mes":m,"min":minutos(g.carga_alta,g.min_parada,a[g.index],1,ctp),"sem":g.min_parada.sum(),"prauc":average_precision_score(g.carga_alta,g.p),"alarm":a[g.index].mean()})
    t=pd.DataFrame(rows)
    print(t.pivot(index="mes",columns="modelo",values="min").round(0).assign(sem=t.groupby("mes")["sem"].first().round(0)))
    print(t.pivot(index="mes",columns="modelo",values="prauc").round(3))
    print(t.pivot(index="mes",columns="modelo",values="alarm").round(2))
    return t
if __name__=="__main__":
    import sys; por_mes(sys.argv[1:])
