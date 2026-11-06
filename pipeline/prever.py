"""Previsão de corridas novas: chance de carga alta e alarme por cestão.

    python pipeline/prever.py                       # lê previsao/entrada/, grava previsao/saida/previsoes.csv
    python pipeline/prever.py --entrada previsao/exemplo
    python pipeline/prever.py --conservador         # limiar em que o alarme certo também custa 1 min

Entrada (pasta --entrada):
    camadas.csv     no formato do infos_camadas.csv, com TODAS as corridas depois do fim do histórico
                    (carregamentos 1 e 2 e, se houver, o 3 com a gusa líquida)
    resultados.csv  opcional, no formato do VariavelResposta.csv: o que já se sabe dessas corridas

Prevê todo cestão de camadas.csv cujo resultado ainda está vazio. Corridas com resultado alimentam o
histórico do forno. Por isso os arquivos são acumulativos: cada corrida nova entra na linha seguinte.

Cestão 1: modelo principal. Cestão 2: modelo principal com o resultado do cestão 1; sem esse
resultado, o modelo reserva. A coluna `avisos` diz quando a previsão não é confiável.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from pipeline import features_corrida as fc  # noqa: E402

FAIXAS = [(0.05, "abaixo de 5%"), (0.10, "5% a 10%"), (0.20, "10% a 20%"), (0.40, "20% a 40%"), (np.inf, "acima de 40%")]
HIST_MAX_HORAS = 24          # sem corrida conhecida há mais que isso, o histórico não descreve o forno de agora
N_RECENTES = 5


def faixa(p: float) -> str:
    return next(nome for corte, nome in FAIXAS if p < corte)


def avisos_entrada(camadas: pd.DataFrame, mats: pd.DataFrame) -> dict:
    """Problemas por corrida. Material desconhecido impede a previsão; o resto só avisa."""
    av: dict[str, list[str]] = {}
    desconhecidos = camadas[camadas.nu_carregamento.isin([1, 2]) & ~camadas.cd_codigo_material.isin(mats.index)]
    for cid, g in desconhecidos.groupby("cd_corrida"):
        av.setdefault(cid, []).append("BLOQUEADA: material sem cadastro (" + ", ".join(sorted(set(g.cd_codigo_material))) + ")")
    cam = camadas[camadas.nu_carregamento.isin([1, 2])]
    for (cid, k), peso in cam.groupby(fc.CHAVE)["peso_t"]:
        if peso.max() > fc.motor.PESO_MAX_CAMADA_T:
            av.setdefault(cid, []).append(f"cestão {k:.0f} com camada acima de {fc.motor.PESO_MAX_CAMADA_T:.0f} t")
        if peso.sum() > fc.motor.MAX_PESO_CESTAO_T:
            av.setdefault(cid, []).append(f"cestão {k:.0f} acima de {fc.motor.MAX_PESO_CESTAO_T:.0f} t")
    tem_3 = set(camadas[camadas.nu_carregamento == 3].cd_corrida)
    for cid in set(cam.cd_corrida) - tem_3:
        av.setdefault(cid, []).append("sem gusa informada: o modelo trata como corrida sem gusa")
    return av


def avisos_historico(d: pd.DataFrame, resposta: pd.DataFrame) -> pd.Series:
    """Olha as corridas anteriores a cada cestão: o histórico está completo e recente?"""
    corr = d[d.nu_carregamento == 1].sort_values("t_ini")[["cd_corrida", "t_ini"]]
    corr["y"] = corr.cd_corrida.map(resposta.set_index("cd_corrida")["is_carga_alta"])
    corr["sem_resultado"] = corr.y.isna().astype(int).shift(1).rolling(N_RECENTES, min_periods=1).sum()
    conhecido = corr.t_ini.where(corr.y.notna()).ffill().shift(1)
    corr["horas"] = (corr.t_ini - conhecido).dt.total_seconds() / 3600
    texto = {}
    for r in corr.itertuples():
        partes = []
        if r.sem_resultado > 0:
            partes.append(f"{r.sem_resultado:.0f} das últimas {N_RECENTES} corridas sem resultado")
        if pd.isna(r.horas) or r.horas > HIST_MAX_HORAS:
            partes.append(f"histórico desatualizado: última corrida conhecida há {r.horas:.0f} h")
        texto[r.cd_corrida] = partes
    return d.cd_corrida.map(texto)


def prever_lote(camadas_cruas: pd.DataFrame, resultados: pd.DataFrame | None, pacote: dict, hist: pd.DataFrame,
                resposta: pd.DataFrame, mats: pd.DataFrame, conservador: bool = False, todos: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devolve (previsões, tabela de features dos cestões novos). `todos` prevê também os cestões já com resultado."""
    camadas = fc.limpar_camadas(camadas_cruas)
    repetidas = set(camadas.cd_corrida) & set(hist.cd_corrida)
    if repetidas:
        print(f"Ignorando {len(repetidas)} corridas que já estão no histórico.")
        camadas = camadas[~camadas.cd_corrida.isin(repetidas)]
    if camadas.empty:
        raise ValueError("Nenhuma corrida nova no arquivo de camadas.")
    if camadas[camadas.nu_carregamento.isin([1, 2])].dt_hora_consumo.isna().any():
        raise ValueError("Há camadas de cestão sem dt_hora_consumo em corridas sem nenhuma hora registrada.")

    av = avisos_entrada(camadas, mats)
    bloqueadas = {c for c, t in av.items() if any(x.startswith("BLOQUEADA") for x in t)}
    validas = camadas[~camadas.cd_corrida.isin(bloqueadas)]
    novos = fc.linhas_novas(validas, mats)
    novas_corr = list(dict.fromkeys(novos.cd_corrida))

    res_novos = fc.normalizar_resultados(resultados, novas_corr)
    res_novos = res_novos[res_novos.cd_corrida.isin(novas_corr)]
    faltam = [c for c in novas_corr if c not in set(res_novos.cd_corrida)]
    res_novos = pd.concat([res_novos, fc.normalizar_resultados(None, faltam)]) if faltam else res_novos
    res_total = pd.concat([resposta[["cd_corrida", *fc.COLUNAS_RESULTADO]], res_novos], ignore_index=True)

    d = fc.com_historico(pd.concat([hist, novos], ignore_index=True), res_total)
    d["carga_alta_c1"] = fc.resultado_cestao_1(d, res_total)
    av_hist = avisos_historico(d, res_total)

    r = res_total.set_index("cd_corrida")
    alvo = pd.Series(np.where(d.nu_carregamento == 1, d.cd_corrida.map(r["is_carga_alta_carregamento_1"]),
                              d.cd_corrida.map(r["is_carga_alta_carregamento_2"])), index=d.index)
    prever = d.cd_corrida.isin(novas_corr) & (todos | alvo.isna())
    alvos = d[prever].copy()
    if alvos.empty:
        raise ValueError("Todos os cestões novos já têm resultado: não há o que prever.")

    principal, reserva = pacote["com_cestao1"], pacote["sem_cestao1_reserva"]
    usa_reserva = alvos.carga_alta_c1.isna().values
    chance = np.full(len(alvos), np.nan)
    limiar = np.full(len(alvos), np.nan)
    chave_limiar = "limiar_minutos_TP1" if conservador else "limiar_minutos_TP0"
    for parte, mascara in [(principal, ~usa_reserva), (reserva, usa_reserva)]:
        if mascara.any():
            X = alvos.loc[mascara, parte["features"]]
            chance[mascara] = np.mean([m.predict_proba(X)[:, 1] for m in parte["modelos"]], axis=0)
            limiar[mascara] = parte[chave_limiar]

    avisos = []
    for i, (idx, lin) in enumerate(alvos.iterrows()):
        t = list(av.get(lin.cd_corrida, [])) + list(av_hist.loc[idx] or [])
        if lin.nu_carregamento == 2 and pd.isna(lin.peso_c1_t):
            t.append("cestão 2 sem as camadas do cestão 1")
        if usa_reserva[i]:
            t.append("modelo reserva: resultado do cestão 1 não informado")
        avisos.append("; ".join(t))
    saida = pd.DataFrame({"cd_corrida": alvos.cd_corrida.values, "cestao": alvos.nu_carregamento.astype(int).values,
                          "inicio": alvos.t_ini.dt.tz_convert(fc.FUSO).dt.strftime("%Y-%m-%d %H:%M").values,
                          "chance": chance.round(4), "faixa_de_risco": [faixa(p) for p in chance],
                          "limiar": limiar.round(4), "alarme": chance > limiar,
                          "modelo": np.where(usa_reserva, "reserva", "principal"), "avisos": avisos})
    if bloqueadas:
        fora = pd.DataFrame([{"cd_corrida": c, "avisos": "; ".join(av[c])} for c in sorted(bloqueadas)])
        saida = pd.concat([saida, fora], ignore_index=True)
    return saida, alvos


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entrada", default="previsao/entrada", help="pasta com camadas.csv e resultados.csv")
    p.add_argument("--saida", default="previsao/saida/previsoes.csv", help="arquivo de saída")
    p.add_argument("--conservador", action="store_true", help="usa o limiar em que o alarme certo também custa 1 min")
    a = p.parse_args()

    entrada = RAIZ / a.entrada
    if not (entrada / "camadas.csv").exists():
        print(f"ERRO: não achei {entrada / 'camadas.csv'}.")
        return 2
    camadas = pd.read_csv(entrada / "camadas.csv", na_values=["null"], dtype={"cd_codigo_material": str, "cd_corrida": str})
    arq_res = entrada / "resultados.csv"
    resultados = pd.read_csv(arq_res, na_values=["null"], dtype={"cd_corrida": str}) if arq_res.exists() else None

    pacote = joblib.load(RAIZ / "src" / "selecao_algoritmos" / "modelo_final.joblib")
    hist, resposta, mats = fc.carregar_historico()
    try:
        saida, _ = prever_lote(camadas, resultados, pacote, hist, resposta, mats, a.conservador)
    except ValueError as erro:
        print(f"ERRO: {erro}")
        return 2

    destino = RAIZ / a.saida
    destino.parent.mkdir(parents=True, exist_ok=True)
    saida.to_csv(destino, index=False, encoding="utf-8-sig")
    com_chance = saida[saida.chance.notna()]
    print(f"{len(com_chance)} cestões previstos, {int(com_chance.alarme.sum())} alarmes, "
          f"{(saida.avisos != '').sum()} com aviso. Saída: {destino.relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
