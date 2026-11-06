# %% [markdown]
# # E se a operação mudasse? Três cenários "fora da caixa" (31/out/2026)
#
# Não mexe no modelo. Usa o modelo escolhido (5 HGB + resultado do cestão 1) e os dados para
# estimar o impacto, em minutos, de três mudanças baratas na regra de negócio:
#
# 1. **Pausa preventiva antes do cestão 2.** Quando o modelo alarma o cestão 2, em vez de (ou além
#    de) reajustar a montagem, o forno espera 1–2 min (ou acomoda a carga) antes de carregá-lo.
#    Evidência nos dados (experimento natural): quando o cestão 1 parou por carga alta, mesmo que
#    por menos de 2 min, o cestão 2 da mesma corrida deu carga alta ~45% menos (18,7% → ~10%),
#    efeito que se mantém dentro de faixas de densidade e de histórico.
# 2. **Regra pós-carga-alta.** Depois de uma corrida com carga alta, a seguinte é montada mais densa.
#    Evidência: depois de uma carga alta, a corrida seguinte dá carga alta 54% das vezes (19% se a
#    anterior não deu), e o pátio hoje NÃO reage (a corrida seguinte vem até um pouco menos densa).
#    Entre as corridas pós-carga-alta, as do terço mais denso dão carga alta 23%, contra 67%.
# 3. **Teto de alarmes por dia** (capacidade do pátio).
#
# **Cuidado:** os cenários 1 e 2 usam associações observadas, não experimentos controlados. O
# número é uma estimativa do potencial, para justificar um TESTE piloto na usina, não uma promessa.

# %%
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
warnings.filterwarnings("ignore")
AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parents[1]
D = pd.read_csv(RAIZ / "src" / "feature_selection" / "dataset_features_completo.csv")
D["t_ini"] = pd.to_datetime(D.t_ini, utc=True, format="mixed"); D["min_parada"] = D.min_parada.fillna(0)
D = D.sort_values("t_ini").reset_index(drop=True)
F = json.load(open(RAIZ / "src" / "feature_selection" / "features_finais.json", encoding="utf-8"))["features_oficial"]
VAR = [dict(l2=5.0), dict(lr=0.05, l2=5.0), dict(leaves=16, l2=5.0), dict(leaf=20, l2=5.0), dict(it=150, l2=5.0)]
H = lambda lr=0.03, it=300, leaves=8, leaf=40, l2=1.0: HistGradientBoostingClassifier(
    learning_rate=lr, max_iter=it, max_leaf_nodes=leaves, min_samples_leaf=leaf, l2_regularization=l2, random_state=42)
prever = lambda tr, te: np.mean([H(**v).fit(tr[F], tr.carga_alta).predict_proba(te[F])[:, 1] for v in VAR], axis=0)

# previsões honestas: walk-forward mensal no dev (nov/25–jun/26) e teste com modelo treinado no dev
DEV = D[D.conjunto != "teste"]; TE = D[D.conjunto == "teste"]
D["p"] = np.nan
for m in ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]:
    te = DEV[DEV.mes == m]; D.loc[te.index, "p"] = prever(DEV[DEV.t_ini < te.t_ini.min()], te)
D.loc[TE.index, "p"] = prever(DEV, TE)
P = D[D.p.notna()].copy()                       # nov/25 a ago/26
DIAS = (P.t_ini.max() - P.t_ini.min()).days
POR30 = 30 / DIAS
DMED = DEV.loc[DEV.carga_alta == 1, "min_parada"].mean()
print(f"Período simulado: {P.t_ini.min():%d/%m/%Y} a {P.t_ini.max():%d/%m/%Y} ({DIAS} dias, {len(P)} cestões, "
      f"{int(P.carga_alta.sum())} cargas altas, {P.min_parada.sum():.0f} min). Parada média (dev): {DMED:.2f} min")


def melhor_limiar(score, beneficio, custo_acao):
    """Limiar que maximiza (benefício - custo) somados; benefício e custo por cestão alarmado."""
    o = np.argsort(-score); ganho = np.r_[0, np.cumsum(beneficio[o] - custo_acao[o])]
    k = int(np.argmax(ganho)); return np.inf if k == 0 else score[o][k - 1] - 1e-12

# %% [markdown]
# ## Cenário 1: pausa preventiva antes do cestão 2
#
# Política atual: alarme → reajuste da montagem (1 min, supõe-se que evita a parada inteira).
# Alternativa para o cestão 2: alarme → pausa de `c` min que reduz a chance de carga alta em `r`.
# Para os cestões 2 cujo cestão 1 já parou, a pausa já aconteceu e não se aplica.
# Limiar de cada política escolhido nos meses do dev (nov–jun) e aplicado a todo o período.

# %%
def simula(politica, c_pausa=1.0, r=0.45, c_tp=0.0, base=P):
    """Minutos esperados no período. politica: 'sem', 'reajuste', 'pausa_c2', 'reajuste_c1+pausa_c2'."""
    y = base.carga_alta.values.astype(bool); dur = base.min_parada.values; p = base.p.values
    c2 = (base.eh_cestao_2 == 1).values & (base.carga_alta_c1 == 0).values
    dev = (base.conjunto != "teste").values
    perda = dur.copy() * y                       # sem modelo
    if politica == "sem":
        return perda.sum(), 0
    usa_pausa = c2 if politica in ("pausa_c2", "reajuste_c1+pausa_c2") else np.zeros(len(y), bool)
    usa_reaj = ~usa_pausa if politica != "pausa_c2" else np.zeros(len(y), bool)
    # benefício esperado de agir em cada cestão, usado só para escolher o limiar no dev (com p previsto)
    benef = np.where(usa_reaj, p * DMED, np.where(usa_pausa, p * r * DMED, 0))
    custo = np.where(usa_reaj, (1 - p) * 1.0 + p * c_tp, np.where(usa_pausa, c_pausa, np.inf))
    lim = {}
    for nome, mask in [("reaj", usa_reaj), ("pausa", usa_pausa)]:
        m = mask & dev
        lim[nome] = melhor_limiar(p[m], benef[m], custo[m]) if m.any() else np.inf
    a_reaj = usa_reaj & (p > lim["reaj"]); a_pausa = usa_pausa & (p > lim["pausa"])
    # minutos realizados (pausa: valor esperado, porque o efeito é probabilístico)
    perda = np.where(a_reaj & y, c_tp, perda) + (a_reaj & ~y) * 1.0
    perda = np.where(a_pausa & y, (1 - r) * dur, perda) + a_pausa * c_pausa
    return perda.sum(), (a_reaj | a_pausa).sum()


linhas = []
sem, _ = simula("sem")
for nome, kw in [("Sem modelo", dict(politica="sem")),
                 ("Atual: alarme → reajuste (1 min, evita 100%)", dict(politica="reajuste")),
                 ("Atual, se o alarme certo também custar 1 min", dict(politica="reajuste", c_tp=1.0)),
                 ("Cestão 2: pausa de 1 min (reduz 45%); cestão 1: reajuste", dict(politica="reajuste_c1+pausa_c2", c_pausa=1, r=0.45)),
                 ("Cestão 2: pausa de 2 min (reduz 45%); cestão 1: reajuste", dict(politica="reajuste_c1+pausa_c2", c_pausa=2, r=0.45)),
                 ("Cestão 2: pausa de 1 min (reduz 30%, pessimista)", dict(politica="reajuste_c1+pausa_c2", c_pausa=1, r=0.30)),
                 ("Cestão 2: pausa de 1 min (reduz 60%, otimista)", dict(politica="reajuste_c1+pausa_c2", c_pausa=1, r=0.60)),
                 ("Só a pausa no cestão 2 (sem reajuste em lugar nenhum)", dict(politica="pausa_c2", c_pausa=1, r=0.45))]:
    mm, al = simula(**kw)
    linhas.append({"cenário": nome, "min perdidos / 30 dias": mm * POR30, "economia vs sem modelo %": 100 * (1 - mm / sem),
                   "ações / dia": al / DIAS})
c1tab = pd.DataFrame(linhas)
print(c1tab.round(1).to_string(index=False))

# robustez do efeito da pausa (o experimento natural), por estrato
c1 = D[D.nu_carregamento == 1].set_index("cd_corrida")
c2 = D[D.nu_carregamento == 2].copy()
c2["c1_parou"] = c2.cd_corrida.map(c1.carga_alta).fillna(0).astype(int)
c2["min_c1"] = c2.cd_corrida.map(c1.min_parada).fillna(0)
c2["densidade"] = pd.qcut(c2.densidade_media, 3, labels=["leve", "média", "densa"])
c2["historico"] = pd.cut(c2.h_taxa5.fillna(0), [-.01, .1, .3, 1], labels=["baixo", "médio", "alto"])
efeito = pd.concat([
    c2.groupby("c1_parou").carga_alta.agg(["mean", "size"]).assign(estrato="todos"),
    c2.groupby(["densidade", "c1_parou"]).carga_alta.agg(["mean", "size"]).reset_index(level=0).rename(columns={"densidade": "estrato"}),
    c2.groupby(["historico", "c1_parou"]).carga_alta.agg(["mean", "size"]).reset_index(level=0).rename(columns={"historico": "estrato"}),
]).reset_index()
efeito.to_csv(AQUI / "cenario1_experimento_natural.csv", index=False)
print(efeito.round(3).to_string(index=False))
print("Por duração da parada do cestão 1:")
print(c2.groupby(pd.cut(c2.min_c1, [-.1, 0, 2, 4, 100], labels=["sem parada", "≤2 min", "2–4 min", ">4 min"])).carga_alta.agg(["mean", "size"]).round(3))

# %% [markdown]
# ## Cenário 2: depois de uma carga alta, montar a corrida seguinte mais densa
#
# Estimativa por estratificação: para as corridas que vêm logo depois de uma carga alta e estão
# abaixo do limiar de densidade, supõe-se que passem a ter a taxa de carga alta das corridas
# pós-carga-alta que já estão acima do limiar. Os minutos evitados usam a parada média.

# %%
cr = (D.groupby("cd_corrida").agg(t=("t_ini", "min"), ca=("carga_alta", "max"), n_ca=("carga_alta", "sum"),
                                  minutos=("min_parada", "sum"), dens=("densidade_media", "mean"))
      .sort_values("t"))
cr["ca_anterior"] = cr.ca.shift(1)
dias_total = (cr.t.max() - cr.t.min()).days
print(cr.groupby("ca_anterior")[["ca", "n_ca", "dens"]].mean().round(3))
lin2 = []
for q in [0.50, 0.67, 0.80]:
    lim = cr.dens.quantile(q)
    pos = cr[cr.ca_anterior == 1]
    abaixo, acima = pos[pos.dens < lim], pos[pos.dens >= lim]
    taxa_cf = acima.n_ca.mean()                          # cargas altas por corrida acima do limiar
    evitadas = (abaixo.n_ca.mean() - taxa_cf) * len(abaixo)
    # versão conservadora: compara abaixo × acima DENTRO do mesmo mês (controla a sucata/preço da época)
    pos = pos.assign(mes=pos.index.map(D.drop_duplicates("cd_corrida").set_index("cd_corrida").mes), acima=pos.dens >= lim)
    mm = pos.groupby(["mes", "acima"]).n_ca.mean().unstack().dropna()
    w = pos[~pos.acima].groupby("mes").size().reindex(mm.index)
    dif_mes = ((mm[False] - mm[True]) * w).sum() / w.sum()
    evit_mes = dif_mes * len(abaixo)
    lin2.append({"densidade mínima pós-carga-alta (t/m³)": round(lim, 3), "percentil": q,
                 "corridas que mudariam / dia": len(abaixo) / dias_total,
                 "cargas altas por corrida: abaixo → acima": f"{abaixo.n_ca.mean():.2f} → {taxa_cf:.2f}",
                 "cargas altas evitadas / 30 dias": evitadas * 30 / dias_total,
                 "minutos evitados / 30 dias": evitadas * DMED * 30 / dias_total,
                 "% dos minutos de parada": 100 * evitadas * DMED / cr.minutos.sum(),
                 "minutos evitados / 30 dias (dentro do mês)": evit_mes * DMED * 30 / dias_total,
                 "% dos minutos (dentro do mês)": 100 * evit_mes * DMED / cr.minutos.sum()})
c2tab = pd.DataFrame(lin2); print(c2tab.round(2).to_string(index=False))

# %% [markdown]
# ## Cenário 3: teto de alarmes por dia (capacidade do pátio)

# %%
P["dia"] = P.t_ini.dt.tz_convert("America/Sao_Paulo").dt.date
lim_atual = melhor_limiar(P.loc[P.conjunto != "teste", "p"].values,
                          (P.loc[P.conjunto != "teste", "p"] * DMED).values,
                          (1 - P.loc[P.conjunto != "teste", "p"]).values)
lin3 = []
for teto in [1, 2, 3, 4, 6, 99]:
    a = np.zeros(len(P), bool)
    for _, g in P.groupby("dia"):          # no dia, alarma os mais arriscados acima do limiar, até o teto
        cand = g[g.p > lim_atual].sort_values("t_ini")
        # operação real: decide cestão a cestão; aproximação: primeiros `teto` cestões acima do limiar no dia
        a[P.index.get_indexer(cand.index[:teto])] = True
    y = P.carga_alta.values.astype(bool)
    mm = (P.min_parada.values * (y & ~a)).sum() + (a & ~y).sum()
    lin3.append({"teto de alarmes por dia": "sem teto" if teto == 99 else teto, "alarmes / dia": a.sum() / DIAS,
                 "recall": (a & y).sum() / y.sum(), "min perdidos / 30 dias": mm * POR30,
                 "economia vs sem modelo %": 100 * (1 - mm / P.min_parada.sum())})
c3tab = pd.DataFrame(lin3); print(c3tab.round(2).to_string(index=False))

c1tab.to_csv(AQUI / "cenario1_pausa_cestao2.csv", index=False)
c2tab.to_csv(AQUI / "cenario2_regra_pos_carga_alta.csv", index=False)
c3tab.to_csv(AQUI / "cenario3_teto_alarmes.csv", index=False)
