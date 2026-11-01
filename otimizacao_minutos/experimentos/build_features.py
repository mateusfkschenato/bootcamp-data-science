"""Monta a tabela de experimentos: as 84 candidatas da feature selection + features novas.

Tudo que é novo usa só informação anterior ao cestão (histórico com shift) ou do próprio cestão.
Saída: exp/dados_exp.pkl
"""
from pathlib import Path
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

d = pd.read_csv(RAIZ / "feature_selection/dataset_features_completo.csv")
b = pd.read_csv(RAIZ / "EDA/base.csv")
r = pd.read_csv(RAIZ / "original_dataset/VariavelResposta.csv")
chave = ["cd_corrida", "nu_carregamento"]

# ---- alvo e duração da parada (duração NÃO é feature; só para medir minutos)
alvo = pd.concat([
    r.assign(nu_carregamento=1.0, seg=r.qt_segundos_duracao_parada_carregamento_1),
    r.assign(nu_carregamento=2.0, seg=r.qt_segundos_duracao_parada_carregamento_2)])[chave + ["seg"]]
d = d.merge(alvo, on=chave, how="left")
d["min_parada"] = d["seg"] / 60
d = d.drop(columns="seg")
d["t"] = pd.to_datetime(d["inicio"], utc=True)

# ---- tempos de montagem por cestão
b["t"] = pd.to_datetime(b["dt_hora_consumo"], utc=True, format="mixed")
tc = b[b.nu_carregamento < 3].groupby(chave)["t"].agg(t_ini="min", t_fim="max").reset_index()
d = d.merge(tc, on=chave, how="left")

# ---- gusa líquido (carregamento 3) da corrida
gu = b[(b.nu_carregamento == 3) & (b.cd_codigo_material == "GUSL")].groupby("cd_corrida")["peso_t"].sum()
d["gusl_t"] = d["cd_corrida"].map(gu).fillna(0)
d["tem_gusl"] = (d["gusl_t"] > 0).astype(int)
outro3 = b[(b.nu_carregamento == 3) & (b.cd_codigo_material != "GUSL")].groupby("cd_corrida")["peso_t"].sum()
d["carga3_solida_t"] = d["cd_corrida"].map(outro3).fillna(0)

# ---- ordem das corridas e histórico
corr = (d[d.nu_carregamento == 1][["cd_corrida", "t_ini", "t_fim"]]
        .rename(columns={"t_ini": "c1_ini", "t_fim": "c1_fim"}))
c2 = d[d.nu_carregamento == 2][["cd_corrida", "t_fim"]].rename(columns={"t_fim": "c2_fim"})
corr = corr.merge(c2, on="cd_corrida", how="left")
corr = corr.merge(r[["cd_corrida", "is_carga_alta", "is_carga_alta_carregamento_1",
                     "is_carga_alta_carregamento_2", "qt_segundos_parada_carga_alta"]], on="cd_corrida")
corr = corr.sort_values("c1_ini").reset_index(drop=True)
corr["ordem"] = np.arange(len(corr))

y = corr["is_carga_alta"].astype(float)
yprev = y.shift(1)          # corrida anterior (o que o grupo já usa)
for lag_nome, yy in [("", yprev), ("_lag2", y.shift(2))]:   # _lag2 = versão "segura": pula a corrida anterior
    for n in [3, 5, 10, 20, 50]:
        corr[f"h_taxa{n}{lag_nome}"] = yy.rolling(n, min_periods=1).mean()
    for hl in [2, 5, 15]:
        corr[f"h_ewm{hl}{lag_nome}"] = yy.ewm(halflife=hl, min_periods=1).mean()
corr["h_ult1"] = yprev
corr["h_ult2_c2"] = corr["is_carga_alta_carregamento_2"].shift(1)  # cestão 2 da corrida anterior
corr["h_ult1_c1"] = corr["is_carga_alta_carregamento_1"].shift(1)
# corridas desde a última carga alta
cnt, ultimo = [], None
for i, v in enumerate(y):
    cnt.append(np.nan if ultimo is None else i - ultimo)
    if v == 1:
        ultimo = i
corr["h_desde_ultima"] = pd.Series(cnt).clip(upper=50)
corr["h_min_parada10"] = (corr["qt_segundos_parada_carga_alta"] / 60).shift(1).rolling(10, min_periods=1).sum()

# janela de tempo: taxa de carga alta nas corridas que COMEÇARAM nas 24 h / 8 h antes (excluindo a atual)
ts = corr["c1_ini"].values.astype("datetime64[s]").astype(np.int64)
yy = y.values
for horas in [8, 24]:
    w = horas * 3600
    taxa, nn = [], []
    j0 = 0
    for i in range(len(corr)):
        while ts[j0] < ts[i] - w:
            j0 += 1
        seg = yy[j0:i]
        taxa.append(seg.mean() if len(seg) else np.nan)
        nn.append(len(seg))
    corr[f"h_taxa_{horas}h"] = taxa
    corr[f"h_n_corridas_{horas}h"] = nn

# ritmo do forno
corr["gap_corrida_min"] = (corr["c1_ini"] - corr["c1_ini"].shift(1)).dt.total_seconds() / 60
corr["gap_c2ant_c1_min"] = (corr["c1_ini"] - corr["c2_fim"].shift(1)).dt.total_seconds() / 60

hcols = [c for c in corr.columns if c.startswith("h_")] + ["gap_corrida_min", "gap_c2ant_c1_min", "ordem"]
d = d.merge(corr[["cd_corrida"] + hcols], on="cd_corrida", how="left")

# ---- composição relativa ao "regime" recente (média móvel das features dos 100 cestões anteriores)
d = d.sort_values(["t_ini"]).reset_index(drop=True)
for c in ["densidade_media", "prop_peso_pesado", "peso_por_ciclo_t", "peso_t_total", "volume_m3_total",
          "prop_leve", "n_ciclos"]:
    m = d[c].shift(1).rolling(100, min_periods=20).mean()
    d[f"rel_{c}"] = d[c] - m
    # posição relativa por cestão (1 ou 2), pois os dois têm perfis diferentes
    m2 = d.groupby("nu_carregamento")[c].transform(lambda s: s.shift(1).rolling(60, min_periods=15).mean())
    d[f"rel2_{c}"] = d[c] - m2

d["mes"] = d["t"].dt.tz_convert("America/Sao_Paulo").dt.strftime("%Y-%m")
d = d.copy(); d.to_pickle(OUT / "dados_exp.pkl")
print(d.shape)
print(d[[c for c in d.columns if c.startswith(("h_", "gap", "rel_", "gusl"))]].describe().T.round(3).iloc[:, :4])
