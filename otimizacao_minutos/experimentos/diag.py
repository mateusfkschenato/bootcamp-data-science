from modelos import *
def diag(nomes):
    c=cache(); rows=[]
    for n in nomes:
        w=tabela_oof(c[n]); g=w[w.mes.isin(MESES_CUSTO)]
        orc=sum(minutos(h.carga_alta,h.min_parada,h.p>melhor_limiar(h.carga_alta,h.p,h.min_parada)) for _,h in g.groupby("mes"))
        orc_pool=minutos(g.carga_alta,g.min_parada,g.p>melhor_limiar(g.carga_alta,g.p,g.min_parada))
        # minutos com taxa fixa de alarme por mês (top x% de cada mês, só diagnóstico)
        fx={}
        for fr in [0.15,0.25,0.35]:
            fx[fr]=sum(minutos(h.carga_alta,h.min_parada,h.p>=h.p.quantile(1-fr)) for _,h in g.groupby("mes"))
        sw=np.where(g.carga_alta==1,g.min_parada,1)
        a=politica_limiar(w,"custo"); real,sem,_=minutos_politica(w,a)
        ap=politica_limiar(w,"platt"); realp,_,_=minutos_politica(w,ap)
        rows.append({"modelo":n,"PR jan-jun":average_precision_score(g.carga_alta,g.p),"AP_min":average_precision_score(g.carga_alta,g.p,sample_weight=sw),
                     "min real(custo)":real,"min real(platt)":realp,"min oráculo/mês":orc,"min oráculo pool":orc_pool,"top15%":fx[0.15],"top25%":fx[0.25],"top35%":fx[0.35],"sem":sem})
    t=pd.DataFrame(rows).set_index("modelo").round(3); pd.set_option("display.width",250); pd.set_option("display.max_columns",20)
    print(t); return t
if __name__=="__main__":
    import sys; diag(sys.argv[1:] or list(cache().keys()))
