from modelos import *
c=cache()
base=tabela_oof(c["RF6 grupo"]); ab=politica_limiar(base,"custo")
for n in ["HGB S2 l2=5","HGB todas-TE","ENS HGB S2 + HGB todas-TE","ENS HGB S2 + HGB todas-TE + HGB todas mono","ENS (S2+c1)+(todas-TE+c1)","ENS lag2"]:
    w=tabela_oof(c[n]); a=politica_limiar(w,"custo")
    r=bootstrap_dif(base,w,n=1500,alarme_a=ab,alarme_b=a)
    rr=bootstrap_dif(base,w,n=1500,alarme_a=ab,alarme_b=a,meses=["2026-04","2026-05","2026-06"])
    print(f"{n:45s} jan-jun: dPR={r['dif_prauc']:+.3f} IC90{np.round(r['ic90_prauc'],3)}  dMin={r['dif_min']:+.0f} IC90{np.round(r['ic90_min'],0)} P(menos min)={r['P(B<A min)']:.2f} | abr-jun: dPR={rr['dif_prauc']:+.3f} dMin={rr['dif_min']:+.0f} IC90{np.round(rr['ic90_min'],0)}")
