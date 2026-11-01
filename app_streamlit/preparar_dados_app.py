"""
Gera as tabelas pequenas que o app lê (pasta dados_app/), a partir do que as
etapas anteriores do projeto já produziram. Rodar de novo sempre que o modelo
ou as features mudarem:

    python app_streamlit/preparar_dados_app.py

Lê:
  EDA/base.csv                                (camadas limpas, corte de 2.000 kg)
  otimizacao_minutos/dataset_minutos.csv      (as 120 features + resultado do cestão 1)
  otimizacao_minutos/features_finais.json     (lista de features do modelo)
  otimizacao_minutos/modelo_minutos.joblib    (limiares de minutos)
  feature_selection/dataset_modelagem_fs.csv  (referência do treino para o painel)
  original_dataset/VariavelResposta.csv       (gusa e duração das paradas)
Escreve:
  dados_app/materiais_app.csv   propriedades e camada típica de cada material
  dados_app/momentos.csv        "momento do forno": o contexto de cada corrida real
  dados_app/config_app.json     limiares, referência, presets de momento, estimativas
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

APP = Path(__file__).resolve().parent
RAIZ = APP.parent
SAIDA = APP / "dados_app"
SAIDA.mkdir(exist_ok=True)

base = pd.read_csv(RAIZ / "EDA" / "base.csv", dtype={"cd_codigo_material": str})
dm = pd.read_csv(RAIZ / "otimizacao_minutos" / "dataset_minutos.csv")
feats = json.loads((RAIZ / "otimizacao_minutos" / "features_finais.json").read_text(encoding="utf-8"))
F = feats["features_principal"]
pacote = joblib.load(RAIZ / "otimizacao_minutos" / "modelo_minutos.joblib")
fs = pd.read_csv(RAIZ / "feature_selection" / "dataset_modelagem_fs.csv")
resposta = pd.read_csv(RAIZ / "original_dataset" / "VariavelResposta.csv")

# ---------------------------------------------------------------- materiais
cam = base[base["nu_carregamento"].isin([1, 2])].copy()
n_cestoes = cam[["cd_corrida", "nu_carregamento"]].drop_duplicates().shape[0]
mat = (cam.groupby("cd_codigo_material")
       .agg(descricao=("ds_descricao_material_x", "first"), classe=("tp_material", "first"),
            densidade=("Densidade t/m3", "first"), rendimento=("Rendimento Metálico %", "first"),
            energia_kwh_t=("Energia Elétrica", "first"), peso_tipico_t=("peso_t", "median"),
            peso_p10_t=("peso_t", lambda s: s.quantile(0.10)),
            peso_p90_t=("peso_t", lambda s: s.quantile(0.90)),
            n_camadas=("peso_t", "size")))
mat["pct_cestoes"] = (cam.drop_duplicates(["cd_corrida", "nu_carregamento", "cd_codigo_material"])
                      .groupby("cd_codigo_material").size() / n_cestoes)
mat["descricao"] = mat["descricao"].str.upper()
# materiais com coluna prop_<código> no modelo; os demais entram em prop_outros
mat["coluna_propria"] = [f"prop_{c}" in F for c in mat.index]
mat = mat.reset_index().rename(columns={"cd_codigo_material": "codigo"})
ordem_classe = {"Leve": 0, "Misto": 1, "Pesado": 2}
mat = mat.sort_values(["classe", "n_camadas"], key=lambda s: s.map(ordem_classe) if s.name == "classe" else -s)
mat.round(4).to_csv(SAIDA / "materiais_app.csv", index=False)
print(f"materiais_app.csv: {len(mat)} materiais")

# ---------------------------------------------------------------- momentos do forno
# Tudo o que o modelo usa e que NÃO vem da receita: histórico, ritmo, horário e as
# médias móveis dos cestões anteriores (rel_X = X − média  ⇒  média = X − rel_X).
HIST = [c for c in F if c.startswith(("h_", "taxa_ca_ult", "gap_"))]
TEMPO = ["hora", "turno_A", "turno_B", "dia_semana", "fim_de_semana"]
REL = [c[4:] for c in F if c.startswith("rel_")]
dm["t_ini"] = pd.to_datetime(dm["t_ini"], utc=True, format="mixed")
completas = dm.groupby("cd_corrida")["nu_carregamento"].nunique() == 2
dm = dm[dm["cd_corrida"].map(completas)]
linhas = []
for cid, g in dm.groupby("cd_corrida"):
    g = g.set_index("nu_carregamento")
    if g[[f"rel_{c}" for c in REL] + [f"rel2_{c}" for c in REL]].isna().any().any():
        continue  # começo da série: ainda não há 100 cestões anteriores
    c1, c2 = g.loc[1.0], g.loc[2.0]
    lin = {"cd_corrida": cid, "conjunto": c1["conjunto"],
           "inicio": c1["t_ini"].tz_convert("America/Sao_Paulo").strftime("%Y-%m-%d %H:%M")}
    lin.update({c: c1[c] for c in HIST})
    for k, row in [(1, c1), (2, c2)]:
        lin.update({f"c{k}_{c}": row[c] for c in TEMPO})
        lin.update({f"c{k}_m_{c}": row[c] - row[f"rel_{c}"] for c in REL})
        lin.update({f"c{k}_m2_{c}": row[c] - row[f"rel2_{c}"] for c in REL})
    linhas.append(lin)
mom = pd.DataFrame(linhas).sort_values("inicio").reset_index(drop=True)
mom.to_csv(SAIDA / "momentos.csv", index=False)
print(f"momentos.csv: {len(mom)} momentos ({mom.inicio.min()} a {mom.inicio.max()})")

# presets, sempre do período mais recente (teste: jun–ago/2026)
rec = mom[mom.conjunto == "teste"]
med10 = rec["h_taxa10"].median()
presets = {
    "Forno típico": rec.loc[(rec.h_ult1 == 0)].assign(d=lambda x: (x.h_taxa10 - med10).abs())
                     .sort_values(["d", "inicio"], ascending=[True, False]).iloc[0]["cd_corrida"],
    "Forno calmo": rec[(rec.h_taxa10 == 0) & (rec.h_ult1 == 0)].sort_values(["h_desde_ultima", "inicio"],
                                                                           ascending=False).iloc[0]["cd_corrida"],
    "Forno numa sequência de cargas altas": rec[(rec.h_ult1 == 1)].sort_values(["h_taxa5", "h_taxa10", "inicio"],
                                                                              ascending=False).iloc[0]["cd_corrida"],
}
for k, v in presets.items():
    r = mom.set_index("cd_corrida").loc[v]
    print(f"preset {k}: {v} {r.inicio} | ult5 {r.h_taxa5:.1f} ult10 {r.h_taxa10:.1f} ult1 {r.h_ult1}")

# ---------------------------------------------------------------- execução da montagem (estimada)
# num_baias = num_materiais em 3.497 dos 3.501 cestões; o tempo sai do intervalo típico entre camadas.
intervalo = dm.groupby("num_camadas")["intervalo_medio_min"].median()
intervalo_por_n = {int(k): float(v) for k, v in intervalo.items()}

# ---------------------------------------------------------------- referência do painel (treino)
treino = fs[fs["conjunto"] == "treino"]
referencia = {c: {q: float(treino[c].quantile(p)) for q, p in
                  [("p10", .10), ("p25", .25), ("p50", .50), ("p75", .75), ("p90", .90)]}
              for c in ["densidade_media", "prop_peso_pesado", "peso_por_ciclo_t", "prop_PACB"]}
por_cestao = cam.groupby(["cd_corrida", "nu_carregamento"]).agg(peso=("peso_t", "sum"), vol=("volume_m3", "sum"))
referencia["peso_t_total"] = {"p10": float(por_cestao.peso.quantile(.1)), "p90": float(por_cestao.peso.quantile(.9))}

# ---------------------------------------------------------------- gusa e paradas
gusl = base[base["cd_codigo_material"] == "GUSL"].groupby("cd_corrida")["peso_t"].sum()
resposta["tem_gusl"] = resposta["cd_corrida"].isin(gusl.index)
taxa_gusl = resposta.groupby("tem_gusl")["is_carga_alta"].mean()
dur = pd.concat([resposta.loc[resposta.is_carga_alta_carregamento_1 == 1, "qt_segundos_duracao_parada_carregamento_1"],
                 resposta.loc[resposta.is_carga_alta_carregamento_2 == 1, "qt_segundos_duracao_parada_carregamento_2"]]) / 60

config = {
    "modelo": {"arquivo": "otimizacao_minutos/modelo_minutos.joblib",
               "descricao": "média de 5 HistGradientBoosting, 120 features + resultado do cestão 1",
               "n_features": len(F)},
    "limiares": {
        "principal": {"Menos minutos perdidos": pacote["com_cestao1"]["limiar_minutos_TP0"],
                      "Alarme certo também custa 1 min": pacote["com_cestao1"]["limiar_minutos_TP1"]},
        "reserva": {"Menos minutos perdidos": pacote["sem_cestao1_reserva"]["limiar_minutos_TP0"],
                    "Alarme certo também custa 1 min": pacote["sem_cestao1_reserva"]["limiar_minutos_TP1"]},
    },
    "parada_media_min": float(dur.mean()), "parada_mediana_min": float(dur.median()),
    "referencia": referencia,
    "presets_momento": presets,
    "intervalo_por_n_camadas": intervalo_por_n,
    "gusa": {"t_p10": float(gusl.quantile(.1)), "t_mediana": float(gusl.median()), "t_p90": float(gusl.quantile(.9)),
             "pct_corridas_com_gusa": float(resposta["tem_gusl"].mean()),
             "taxa_ca_com_gusa": float(taxa_gusl.get(True, np.nan)),
             "taxa_ca_sem_gusa": float(taxa_gusl.get(False, np.nan))},
}
(SAIDA / "config_app.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
print("config_app.json salvo")
