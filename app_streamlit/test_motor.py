"""
Confere que o app monta as features exatamente como o projeto.
Rodar da raiz do repositório:  python -m pytest app_streamlit/test_motor.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motor  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
EXECUCAO = ["num_baias", "tempo_montagem_min", "intervalo_medio_min"]


def _dados():
    mats, cfg, mom, pac = motor.carregar_materiais(), motor.carregar_config(), motor.carregar_momentos(), motor.carregar_modelo()
    dm = pd.read_csv(RAIZ / "feature_selection" / "dataset_features_completo.csv").set_index(["cd_corrida", "nu_carregamento"])
    base = pd.read_csv(RAIZ / "EDA" / "base.csv", dtype={"cd_codigo_material": str})
    cam = base[base.nu_carregamento.isin([1, 2])].sort_values(["cd_corrida", "nu_carregamento", "nu_camada"])
    return mats, cfg, mom, pac, dm, cam


def test_120_features_iguais_ao_dataset_de_modelagem():
    """Para 400 corridas reais: receita + momento da própria corrida = linha do feature_selection/dataset_features_completo.csv."""
    mats, cfg, mom, pac, dm, cam = _dados()
    F = pac["com_cestao1"]["features"]
    ids = mom.sample(400, random_state=0).index
    cam = cam[cam.cd_corrida.isin(ids)]
    camadas = {k: [motor.Camada(c, p) for c, p in zip(g.cd_codigo_material, g.peso_t)]
               for k, g in cam.groupby(["cd_corrida", "nu_carregamento"])}
    app, real = [], []
    for cid in ids:
        cs = [camadas[(cid, 1.0)], camadas[(cid, 2.0)]]
        rec = [motor.features_receita(c, mats) for c in cs]
        cod = [{x.codigo for x in c} for c in cs]
        for k in (1, 2):
            row = dm.loc[(cid, float(k))]
            lin = motor.linha_cestao(k, rec, cod, mom.loc[cid], row["gusl_t"], int(row["carga_alta_c1"]),
                                     {c: row[c] for c in EXECUCAO})
            lin["carga3_solida_t"] = row["carga3_solida_t"]   # o app só tem gusa líquida (0 de sólido)
            app.append(lin)
            real.append(row)
    A = pd.DataFrame(app)[F]
    R = pd.DataFrame(real)[F].reset_index(drop=True)
    # num_camadas: nos dados, algumas camadas repetem o número (nunique); no app cada camada conta
    cols = [c for c in F if c != "num_camadas"]
    assert (A[cols].isna() == R[cols].isna()).all().all()
    np.testing.assert_allclose(A[cols].fillna(0).values.astype(float), R[cols].fillna(0).values.astype(float), atol=1e-9)


def test_validacao_recusa_valores_absurdos():
    mats = motor.carregar_materiais()
    assert motor.validar_camada("SHRE", 1.0, mats)          # abaixo do corte de 2 t
    assert motor.validar_camada("SHRE", 50.0, mats)         # camada impossível
    assert motor.validar_camada("XXXX", 5.0, mats)          # material que não existe
    assert motor.validar_camada("SHRE", 5.0, mats) is None
    cheio = [motor.Camada("ESPS", 14.0)] * 3                 # ~105 m³
    assert "Não cabe" in motor.validar_insercao(cheio, motor.Camada("ESPS", 5.0), mats)


def test_previsao_cestao_1_principal_cestao_2_reserva():
    mats, cfg, mom, pac = motor.carregar_materiais(), motor.carregar_config(), motor.carregar_momentos(), motor.carregar_modelo()
    m = mom.loc[cfg["presets_momento"]["Forno típico"]]
    c1 = [motor.Camada("SHRE", 5), motor.Camada("GUSS", 6), motor.Camada("OXIC", 4)]
    c2 = [motor.Camada("SHRE", 4), motor.Camada("MISM", 4)]
    modo = list(cfg["limiares"]["principal"])[0]
    r = motor.prever_corrida([c1, c2], m, 34.3, None, modo, pac, mats, cfg)
    assert r["cestoes"][0]["modelo"] == "principal" and r["cestoes"][1]["modelo"] == "reserva"
    assert 0 <= r["cestoes"][1]["chance"] <= 1


def _alarme():
    mats, cfg, mom, pac = motor.carregar_materiais(), motor.carregar_config(), motor.carregar_momentos(), motor.carregar_modelo()
    m = mom.loc[cfg["presets_momento"]["Forno típico"]]
    modo = list(cfg["limiares"]["principal"])[0]
    c1 = [motor.Camada(c, p) for c, p in [("SHRE", 5), ("OXIC", 4.5), ("SHRE", 5), ("MISM", 4.5), ("OXIC", 4),
                                          ("SHRE", 5), ("ESPS", 4), ("MISM", 4)]]
    return c1, m, modo, pac, mats, cfg


def test_recombinacao_respeita_as_restricoes():
    c1, m, modo, pac, mats, cfg = _alarme()
    assert motor.prever_corrida([c1], m, 35, None, modo, pac, mats, cfg)["cestoes"][0]["alarme"]
    r = motor.recombinar_cestao([c1], 1, m, 35, None, modo, pac, mats, cfg)
    assert r["status"] == "ok" and 1 <= len(r["mudancas"]) <= motor.MAX_MUDANCAS
    novo = [motor.Camada(*c) for c in r["camadas"]]
    f = motor.features_receita(novo, mats)
    assert abs(f["peso_t_total"] - sum(c.peso_t for c in c1)) < 1e-9       # peso total igual
    assert f["volume_m3_total"] <= motor.CAPACIDADE_M3 and f["violacao_regra"] == 0
    assert all(f[c] <= cfg["limites_receita"][c] for c in motor.LIMITADAS)
    p = motor.prever_corrida([novo], m, 35, None, modo, pac, mats, cfg)["cestoes"][0]
    np.testing.assert_allclose(p["chance"], r["chance_depois"])            # a chance mostrada é a da receita aplicada
    assert not p["alarme"]


def test_recombinacao_nao_faz_o_cestao_2_passar_a_alarmar():
    c1, m, modo, pac, mats, cfg = _alarme()
    c2 = [motor.Camada("GUSS", 5), motor.Camada("OXIC", 4), motor.Camada("GUSS", 5), motor.Camada("SHRE", 4)]
    c2 = [c2[3], c2[0], c2[1], c2[2], motor.Camada("SHRE", 4)]
    antes = motor.prever_corrida([c1, c2], m, 35, None, modo, pac, mats, cfg)["cestoes"]
    r = motor.recombinar_cestao([c1, c2], 1, m, 35, None, modo, pac, mats, cfg)
    if r["status"] != "sem_solucao":
        novo = [motor.Camada(*c) for c in r["camadas"]]
        depois = motor.prever_corrida([novo, c2], m, 35, None, modo, pac, mats, cfg)["cestoes"]
        assert antes[1]["alarme"] or not depois[1]["alarme"]


def test_resultado_do_cestao_1_muda_o_modelo_do_cestao_2():
    c1, m, modo, pac, mats, cfg = _alarme()
    c2 = [motor.Camada("SHRE", 4), motor.Camada("MISM", 4)]
    r = motor.prever_corrida([c1, c2], m, 35, 1, modo, pac, mats, cfg)["cestoes"][1]
    assert r["modelo"] == "principal"


def test_exemplos_dao_o_resultado_prometido():
    mats, cfg, mom, pac = motor.carregar_materiais(), motor.carregar_config(), motor.carregar_momentos(), motor.carregar_modelo()
    modo = list(cfg["limiares"]["principal"])[0]
    for e in motor.carregar_exemplos():
        cs = [[motor.Camada(**c) for c in cam] for cam in e["cestoes"]]
        for cam in cs:
            assert motor.resumo_cestao(cam, mats)["volume_m3_total"] <= motor.CAPACIDADE_M3
        r = motor.prever_corrida(cs, mom.loc[cfg["presets_momento"][e["momento"]]], e["gusa_t"], None, modo, pac, mats, cfg)
        assert r["alarme_corrida"] == e["alarme"], e["nome"]
