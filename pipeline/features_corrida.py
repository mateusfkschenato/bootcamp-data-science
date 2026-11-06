"""Monta as 121 features de corridas novas, com o histórico do forno de verdade.

O app usa um "momento do forno" copiado de uma corrida real. Aqui o histórico é calculado:
as corridas novas entram depois das corridas conhecidas, e histórico, ritmo e regime recente
(`h_*`, `taxa_ca_ult*`, `gap_*`, `rel_*`, `rel2_*`) são refeitos sobre a série completa.

As regras são as mesmas de src/EDA/transforma_dataset.py (limpeza) e de
src/feature_selection/feature_selection.py (seções 2 e 4). pipeline/test_previsao.py confere,
linha a linha, que o resultado bate com src/feature_selection/dataset_features_completo.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from comum.receita import Camada, motor, tabela_materiais  # noqa: E402

CHAVE = ["cd_corrida", "nu_carregamento"]
FUSO = "America/Sao_Paulo"
PESO_MINIMO_CAMADA_KG = 2000
COLUNAS_CAMADAS = ["cd_corrida", "nu_carregamento", "nu_camada", "cd_codigo_material", "qt_peso_carregamento"]
COLUNAS_RESULTADO = ["is_carga_alta", "is_carga_alta_carregamento_1", "is_carga_alta_carregamento_2",
                     "qt_segundos_duracao_parada_carregamento_1", "qt_segundos_duracao_parada_carregamento_2",
                     "qt_segundos_parada_carga_alta"]
REGIME = ["densidade_media", "prop_peso_pesado", "peso_por_ciclo_t", "peso_t_total", "volume_m3_total",
          "prop_leve", "n_ciclos"]
CALCULADAS = ("h_", "taxa_ca_ult", "gap_", "rel_", "rel2_")


# ---------------------------------------------------------------- leitura
def carregar_historico() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(cestões conhecidos, resultados conhecidos, tabela de materiais)."""
    import json
    hist = pd.read_csv(RAIZ / "src" / "feature_selection" / "dataset_features_completo.csv")
    for c in ["t_ini", "t_fim"]:
        hist[c] = pd.to_datetime(hist[c], utc=True, format="mixed")
    resposta = pd.read_csv(RAIZ / "original_dataset" / "VariavelResposta.csv")
    base = pd.read_csv(RAIZ / "src" / "EDA" / "base.csv", dtype={"cd_codigo_material": str})
    feats = json.loads((RAIZ / "src" / "feature_selection" / "features_finais.json").read_text(encoding="utf-8"))
    return hist, resposta, tabela_materiais(base, feats["features_oficial"])


def limpar_camadas(cru: pd.DataFrame) -> pd.DataFrame:
    """Mesma limpeza do src/EDA/transforma_dataset.py, para o formato do infos_camadas.csv."""
    faltam = [c for c in COLUNAS_CAMADAS + ["dt_hora_consumo"] if c not in cru.columns]
    if faltam:
        raise ValueError(f"Faltam colunas no arquivo de camadas: {', '.join(faltam)}")
    b = cru.copy()
    b["cd_corrida"] = b["cd_corrida"].astype(str)
    b["cd_codigo_material"] = b["cd_codigo_material"].astype(str)
    b["dt_hora_consumo"] = pd.to_datetime(b["dt_hora_consumo"], utc=True, format="ISO8601")
    if "cd_baia" not in b:
        b["cd_baia"] = np.nan
    # a EDA guarda a baia sem a casa decimal: 12.3 e 12.0 contam como a baia 12
    b["cd_baia"] = pd.to_numeric(b["cd_baia"], errors="coerce").map(lambda v: str(int(v)), na_action="ignore")
    b = b.drop_duplicates(keep="first")
    # hora que faltou num cestão: usa a mediana dos cestões da mesma corrida (como na EDA)
    cestao = b["nu_carregamento"].isin([1, 2])
    mediana = b[cestao & b["dt_hora_consumo"].notna()].groupby("cd_corrida")["dt_hora_consumo"].median()
    sem_hora = cestao & b["dt_hora_consumo"].isna()
    b.loc[sem_hora, "dt_hora_consumo"] = b.loc[sem_hora, "cd_corrida"].map(mediana)
    # corte de 2.000 kg (Guima, 21/out/2026): remove a camada, não o cestão
    b = b[b["qt_peso_carregamento"] >= PESO_MINIMO_CAMADA_KG].reset_index(drop=True)
    b["peso_t"] = b["qt_peso_carregamento"] / 1000
    return b


def normalizar_resultados(res: pd.DataFrame | None, corridas=()) -> pd.DataFrame:
    """Resultados no formato do VariavelResposta.csv. Vazio = ainda não se sabe."""
    if res is None or res.empty:
        res = pd.DataFrame({"cd_corrida": list(corridas)})
    res = res.copy()
    res["cd_corrida"] = res["cd_corrida"].astype(str)
    res = res.drop_duplicates("cd_corrida", keep="last")
    for c in COLUNAS_RESULTADO:
        res[c] = pd.to_numeric(res[c], errors="coerce") if c in res else np.nan
    c1, c2 = res["is_carga_alta_carregamento_1"], res["is_carga_alta_carregamento_2"]
    # a corrida deu carga alta se algum cestão deu; só fica "não sei" se ainda falta um cestão
    deduzido = pd.Series(np.select([(c1 == 1) | (c2 == 1), c1.notna() & c2.notna()], [1.0, 0.0], np.nan), index=res.index)
    res["is_carga_alta"] = res["is_carga_alta"].fillna(deduzido)
    return res[["cd_corrida", *COLUNAS_RESULTADO]]


# ---------------------------------------------------------------- receita
def linhas_novas(camadas: pd.DataFrame, mats: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por cestão novo, com tudo o que não depende do histórico."""
    cam = camadas[camadas.nu_carregamento.isin([1, 2])].sort_values(CHAVE + ["nu_camada"])
    linhas, receitas, materiais = [], {}, {}
    for (cid, k), g in cam.groupby(CHAVE):
        lista = [Camada(c, p) for c, p in zip(g.cd_codigo_material, g.peso_t)]
        f = motor.features_receita(lista, mats)
        f["num_camadas"] = g["nu_camada"].nunique()        # igual ao dataset: nunique, não len
        # baia vazia conta como uma baia a mais, como na EDA; sem nenhuma baia, estima como o app
        tem_baia = g["cd_baia"].notna().any()
        f["num_baias"] = g["cd_baia"].fillna("SEM_BAIA").nunique() if tem_baia else f["num_materiais"]
        ini, fim = g["dt_hora_consumo"].min(), g["dt_hora_consumo"].max()
        local = ini.tz_convert(FUSO)
        f["tempo_montagem_min"] = (fim - ini).total_seconds() / 60
        f["intervalo_medio_min"] = f["tempo_montagem_min"] / (f["num_camadas"] - 1) if f["num_camadas"] > 1 else np.nan
        f.update({"cd_corrida": cid, "nu_carregamento": float(k), "t_ini": ini, "t_fim": fim, "t": ini,
                  "hora": local.hour, "turno_A": int(6 <= local.hour <= 13), "turno_B": int(14 <= local.hour <= 21),
                  "dia_semana": local.dayofweek, "fim_de_semana": int(local.dayofweek >= 5),
                  "eh_cestao_2": int(k == 2), "mes": local.strftime("%Y-%m")})
        receitas[(cid, k)], materiais[(cid, k)] = f, frozenset(g.cd_codigo_material)
        linhas.append(f)
    for f in linhas:
        cid, k = f["cd_corrida"], f["nu_carregamento"]
        eh2 = k == 2
        c1 = receitas.get((cid, 1.0)) or receitas.get((cid, 1))
        f["volume_c1_m3"] = (c1["volume_m3_total"] if c1 else np.nan) if eh2 else 0.0
        f["peso_c1_t"] = (c1["peso_t_total"] if c1 else np.nan) if eh2 else 0.0
        f["vol_acumulado_m3"] = f["volume_m3_total"] + (0 if pd.isna(f["volume_c1_m3"]) else f["volume_c1_m3"])
        f["delta_peso_12_t"] = f["peso_t_total"] - f["peso_c1_t"] if eh2 else 0.0
        mc1 = materiais.get((cid, 1.0)) or materiais.get((cid, 1))
        f["mudou_composicao"] = int(mc1 != materiais[(cid, k)]) if eh2 else 0
    novo = pd.DataFrame(linhas)
    gusa = camadas[(camadas.nu_carregamento == 3) & (camadas.cd_codigo_material == "GUSL")].groupby("cd_corrida")["peso_t"].sum()
    outro = camadas[(camadas.nu_carregamento == 3) & (camadas.cd_codigo_material != "GUSL")].groupby("cd_corrida")["peso_t"].sum()
    novo["gusl_t"] = novo["cd_corrida"].map(gusa).fillna(0)
    novo["tem_gusl"] = (novo["gusl_t"] > 0).astype(int)
    novo["carga3_solida_t"] = novo["cd_corrida"].map(outro).fillna(0)
    return novo


# ---------------------------------------------------------------- histórico do forno
def _taxa_janela(ts: np.ndarray, y: np.ndarray, horas: int) -> tuple[list, list]:
    w_s, taxa, nn, j0 = horas * 3600, [], [], 0
    for i in range(len(ts)):
        while ts[j0] < ts[i] - w_s:
            j0 += 1
        seg = y[j0:i]
        conhecidos = seg[~np.isnan(seg)]
        taxa.append(conhecidos.mean() if len(conhecidos) else np.nan)
        nn.append(len(seg))
    return taxa, nn


def com_historico(d: pd.DataFrame, resposta: pd.DataFrame) -> pd.DataFrame:
    """Refaz as features de histórico, ritmo e regime sobre toda a série (feature_selection.py, seção 4)."""
    d = d.drop(columns=[c for c in d.columns if c.startswith(CALCULADAS)] + ["ordem"], errors="ignore").copy()

    inicio = d.groupby("cd_corrida")["t_ini"].min()
    hist = resposta[["cd_corrida", "is_carga_alta"]].assign(t=lambda x: x.cd_corrida.map(inicio)).sort_values("t")
    for n in [5, 10, 20]:
        hist[f"taxa_ca_ult{n}"] = hist["is_carga_alta"].shift(1).rolling(n, min_periods=1).mean()
        d[f"taxa_ca_ult{n}"] = d["cd_corrida"].map(hist.set_index("cd_corrida")[f"taxa_ca_ult{n}"])

    corr = (d[d.nu_carregamento == 1][["cd_corrida", "t_ini", "t_fim"]]
            .rename(columns={"t_ini": "c1_ini", "t_fim": "c1_fim"}))
    c2 = d[d.nu_carregamento == 2][["cd_corrida", "t_fim"]].rename(columns={"t_fim": "c2_fim"})
    corr = corr.merge(c2, on="cd_corrida", how="left").merge(resposta, on="cd_corrida", how="left")
    corr = corr.sort_values("c1_ini").reset_index(drop=True)
    corr["ordem"] = np.arange(len(corr))

    y = corr["is_carga_alta"].astype(float)
    ant = y.shift(1)
    for n in [3, 5, 10, 20, 50]:
        corr[f"h_taxa{n}"] = ant.rolling(n, min_periods=1).mean()
    for hl in [2, 5, 15]:
        corr[f"h_ewm{hl}"] = ant.ewm(halflife=hl, min_periods=1).mean()
    corr["h_ult1"] = ant
    corr["h_ult2_c2"] = corr["is_carga_alta_carregamento_2"].shift(1)
    corr["h_ult1_c1"] = corr["is_carga_alta_carregamento_1"].shift(1)
    cnt, ultimo = [], None
    for i, v in enumerate(y):
        cnt.append(np.nan if ultimo is None else i - ultimo)
        if v == 1:
            ultimo = i
    corr["h_desde_ultima"] = pd.Series(cnt).clip(upper=50)
    corr["h_min_parada10"] = (corr["qt_segundos_parada_carga_alta"] / 60).shift(1).rolling(10, min_periods=1).sum()

    ts = corr["c1_ini"].values.astype("datetime64[s]").astype(np.int64)
    for horas in [8, 24]:
        corr[f"h_taxa_{horas}h"], corr[f"h_n_corridas_{horas}h"] = _taxa_janela(ts, y.values, horas)

    corr["gap_corrida_min"] = (corr["c1_ini"] - corr["c1_ini"].shift(1)).dt.total_seconds() / 60
    corr["gap_c2ant_c1_min"] = (corr["c1_ini"] - corr["c2_fim"].shift(1)).dt.total_seconds() / 60

    hcols = [c for c in corr.columns if c.startswith("h_")] + ["gap_corrida_min", "gap_c2ant_c1_min", "ordem"]
    d = d.merge(corr[["cd_corrida"] + hcols], on="cd_corrida", how="left")

    d = d.sort_values(["t_ini"]).reset_index(drop=True)
    for c in REGIME:
        m = d[c].shift(1).rolling(100, min_periods=20).mean()
        d[f"rel_{c}"] = d[c] - m
        m2 = d.groupby("nu_carregamento")[c].transform(lambda s: s.shift(1).rolling(60, min_periods=15).mean())
        d[f"rel2_{c}"] = d[c] - m2
    return d


def resultado_cestao_1(d: pd.DataFrame, resposta: pd.DataFrame) -> pd.Series:
    """0 no cestão 1; no cestão 2, o resultado do cestão 1 (NaN = ainda não se sabe)."""
    r1 = resposta.set_index("cd_corrida")["is_carga_alta_carregamento_1"]
    return pd.Series(np.where(d.nu_carregamento == 2, d.cd_corrida.map(r1), 0.0), index=d.index)
