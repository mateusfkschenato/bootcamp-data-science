"""Recalcular as features da receita de cestões a partir de camadas novas (camadas alteradas).

Usado na verificação (erro de balança, corte de peso) e na interpretação (contrafactuais).
A conta das features da receita é a do app (`app_streamlit/motor.py`, função `features_receita`),
conferida contra o dataset em `app_streamlit/test_motor.py` e na verificação (seção 5c).

O que muda junto com a receita:
- as features da própria receita (massa, volume, densidade, materiais, ordem e ciclos);
- as `rel_`/`rel2_` (valor − média móvel do passado): a média do passado fica, o valor muda;
- no cestão 2, as features do cestão 1 (volume, peso, volume acumulado, diferença de peso).
O que não muda: histórico, ritmo, gusa, tempo/turno e o resultado do cestão 1.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "app_streamlit"))
import motor  # noqa: E402

CHAVE = ["cd_corrida", "nu_carregamento"]
Camada = motor.Camada


def tabela_materiais(base, features):
    """Classe, densidade, rendimento e energia de cada material, tirados do src/EDA/base.csv."""
    mats = (base.drop_duplicates("cd_codigo_material").set_index("cd_codigo_material")
            .rename(columns={"tp_material": "classe", "Densidade t/m3": "densidade", "Rendimento Metálico %": "rendimento",
                             "Energia Elétrica": "energia_kwh_t"})[["classe", "densidade", "rendimento", "energia_kwh_t"]])
    mats["coluna_propria"] = [f"prop_{c}" in features for c in mats.index]
    return mats


def camadas_da_base(base, corridas=None):
    """{(corrida, carregamento): [(material, peso_t), ...]} na ordem das camadas."""
    b = base[base.nu_carregamento.isin([1, 2])]
    if corridas is not None:
        b = b[b.cd_corrida.isin(corridas)]
    b = b.sort_values(CHAVE + ["nu_camada"])
    return {k: list(zip(g.cd_codigo_material, g.peso_t)) for k, g in b.groupby(CHAVE)}


def colunas_receita(mats, features):
    return [c for c in motor.features_receita([Camada(mats.index[0], 5.0)], mats) if c in features and c != "num_camadas"]


def recalcular(tab, camadas_por_cestao, mats, features):
    """Devolve uma cópia de `tab` com as features da receita refeitas a partir das camadas dadas.

    camadas_por_cestao: {(corrida, carregamento): [Camada, ...]}"""
    rec = colunas_receita(mats, features)
    rel = [c for c in features if c.startswith(("rel_", "rel2_"))]
    out = tab.copy().set_index(CHAVE)
    novas = {k: motor.features_receita(v, mats) for k, v in camadas_por_cestao.items() if k in out.index}
    nv = pd.DataFrame.from_dict(novas, orient="index")
    nv.index = pd.MultiIndex.from_tuples(nv.index, names=CHAVE)
    nv = nv.reindex(out.index)
    velho = out[rec].copy()
    tem = nv[rec[0]].notna().values
    out.loc[tem, rec] = nv.loc[tem, rec].values
    for c in rel:
        base_c = c.split("_", 1)[1]
        out[c] = out[c] + (out[base_c] - velho[base_c])
    eh2 = out.index.get_level_values(1) == 2
    c1v = out.xs(1.0, level=1)
    corr2 = out.index.get_level_values(0)[eh2]
    out.loc[eh2, "volume_c1_m3"] = c1v.reindex(corr2)["volume_m3_total"].values
    out.loc[eh2, "peso_c1_t"] = c1v.reindex(corr2)["peso_t_total"].values
    out["vol_acumulado_m3"] = out["volume_m3_total"] + out["volume_c1_m3"]
    out.loc[eh2, "delta_peso_12_t"] = out.loc[eh2, "peso_t_total"].values - out.loc[eh2, "peso_c1_t"].values
    return out.reset_index()
