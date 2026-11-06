"""Confere a previsão de corridas novas contra o dataset de modelagem.

    python -m pytest pipeline/test_previsao.py -q
"""
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from pipeline import features_corrida as fc, prever as pv  # noqa: E402

N_REPLAY = 300
TOL_CHANCE = 0.02


@pytest.fixture(scope="module")
def dados():
    hist, resposta, mats = fc.carregar_historico()
    pacote = joblib.load(RAIZ / "src" / "selecao_algoritmos" / "modelo_final.joblib")
    cru = pd.read_csv(RAIZ / "original_dataset" / "infos_camadas.csv", na_values=["null"], dtype={"cd_codigo_material": str})
    ordem = hist[hist.nu_carregamento == 1].sort_values("t_ini").cd_corrida.tolist()
    return hist, resposta, mats, pacote, cru, ordem


def cortar(dados, novas):
    """Finge que as corridas `novas` ainda não aconteceram."""
    hist, resposta, _, _, cru, _ = dados
    return (hist[~hist.cd_corrida.isin(novas)], resposta[~resposta.cd_corrida.isin(novas)],
            cru[cru.cd_corrida.isin(novas)], resposta[resposta.cd_corrida.isin(novas)])


def compara(a, b, colunas):
    x, y = a[colunas].apply(pd.to_numeric).astype(float).values, b[colunas].astype(float).values
    assert (np.isnan(x) == np.isnan(y)).all()
    np.testing.assert_allclose(np.nan_to_num(x), np.nan_to_num(y), atol=1e-9)


def test_historico_refeito_igual_ao_dataset(dados):
    hist, resposta, _, pacote, _, _ = dados
    res = fc.normalizar_resultados(resposta)
    d = fc.com_historico(hist, res)
    d["carga_alta_c1"] = fc.resultado_cestao_1(d, res)
    F = pacote["com_cestao1"]["features"]
    compara(d.set_index(fc.CHAVE).sort_index(), hist.set_index(fc.CHAVE).sort_index(), F)


def test_corridas_novas_dao_as_mesmas_features_e_chances(dados):
    """As últimas 300 corridas viram "novas": 121 features e chances iguais às do dataset."""
    hist, _, mats, pacote, _, ordem = dados
    h, r, cam, res = cortar(dados, ordem[-N_REPLAY:])
    saida, alvos = pv.prever_lote(cam, res, pacote, h, r, mats, todos=True)
    F = pacote["com_cestao1"]["features"]
    a = alvos.set_index(fc.CHAVE).sort_index()
    b = hist.set_index(fc.CHAVE).loc[a.index]
    compara(a, b, F)
    assert (saida.modelo == "principal").all()
    esperado = np.mean([m.predict_proba(b[F])[:, 1] for m in pacote["com_cestao1"]["modelos"]], axis=0)
    obtido = saida.assign(nu_carregamento=saida.cestao.astype(float)).set_index(fc.CHAVE).loc[a.index]
    # Somas feitas em outra ordem mudam o último dígito do float. Nestas corridas, que o modelo final viu no
    # treino, um valor pode cair exatamente num corte de árvore e mudar a chance em até ~1,5 ponto.
    assert np.abs(obtido.chance.values - esperado).max() < TOL_CHANCE
    limiar = pacote["com_cestao1"]["limiar_minutos_TP0"]
    assert (obtido.alarme.values == (esperado > limiar)).all()


def test_resultado_da_propria_corrida_nao_entra_na_previsao(dados):
    """Sem vazamento: esconder o resultado da corrida prevista não muda a chance."""
    _, _, mats, pacote, _, ordem = dados
    cid = ordem[-1]
    h, r, cam, res = cortar(dados, [cid])
    com, _ = pv.prever_lote(cam, res, pacote, h, r, mats, todos=True)
    res_c1 = res.copy()
    res_c1[["is_carga_alta", "is_carga_alta_carregamento_2", "qt_segundos_duracao_parada_carregamento_2",
            "qt_segundos_parada_carga_alta"]] = np.nan
    sem, _ = pv.prever_lote(cam, res_c1, pacote, h, r, mats)
    assert sem.cestao.tolist() == [2] and sem.modelo.tolist() == ["principal"]
    assert sem.chance.iloc[0] == com[com.cestao == 2].chance.iloc[0]


def test_sem_o_cestao_1_usa_a_reserva(dados):
    _, _, mats, pacote, _, ordem = dados
    h, r, cam, _ = cortar(dados, [ordem[-1]])
    saida, _ = pv.prever_lote(cam, None, pacote, h, r, mats)
    assert saida.modelo.tolist() == ["principal", "reserva"]
    assert "modelo reserva" in saida.avisos.iloc[1]


def test_material_sem_cadastro_bloqueia_a_corrida(dados):
    _, _, mats, pacote, _, ordem = dados
    h, r, cam, _ = cortar(dados, ordem[-2:])
    cam = cam.copy()
    primeira = cam.cd_corrida == ordem[-2]
    cam.loc[primeira & cam.nu_carregamento.isin([1, 2]), "cd_codigo_material"] = "NAOEXISTE"
    saida, _ = pv.prever_lote(cam, None, pacote, h, r, mats)
    bloqueada = saida[saida.cd_corrida == ordem[-2]]
    assert bloqueada.chance.isna().all() and "BLOQUEADA" in bloqueada.avisos.iloc[0]
    assert saida[saida.cd_corrida == ordem[-1]].chance.notna().all()


def test_coluna_faltando_da_erro_claro():
    with pytest.raises(ValueError, match="Faltam colunas"):
        fc.limpar_camadas(pd.DataFrame({"cd_corrida": ["X"]}))


def test_exemplo_roda_pela_linha_de_comando():
    r = subprocess.run([sys.executable, str(RAIZ / "pipeline" / "prever.py"), "--entrada", "previsao/exemplo",
                        "--saida", "previsao/saida/teste_exemplo.csv"], capture_output=True, text=True, cwd=RAIZ)
    arq = RAIZ / "previsao" / "saida" / "teste_exemplo.csv"
    assert r.returncode == 0, r.stdout + r.stderr
    saida = pd.read_csv(arq)
    arq.unlink()
    assert saida.chance.between(0, 1).all()
    assert set(saida.modelo) == {"principal", "reserva"}
