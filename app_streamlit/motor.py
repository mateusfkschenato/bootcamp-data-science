"""
Motor do simulador: transforma a receita da corrida (camadas de cada cestão + gusa
líquida) e o "momento do forno" nas 120 features do modelo escolhido do projeto
(selecao_algoritmos/modelo_final.joblib: o modelo oficial, média de 5 HistGradientBoosting) e
devolve a chance de carga alta de cada cestão.

De onde vem cada grupo de features:
  receita do cestão (67)   calculadas aqui, com as mesmas regras de
                           feature_selection/feature_selection.py
  cestão 1 → cestão 2 (6)  eh_cestao_2, volume_c1_m3, peso_c1_t, vol_acumulado_m3,
                           delta_peso_12_t, mudou_composicao
  gusa líquida (3)         gusl_t, tem_gusl, carga3_solida_t
  momento do forno (41)    histórico, ritmo e horário copiados de uma corrida real
                           (dados_app/momentos.csv); rel_X = X da receita − média
                           móvel dos cestões anteriores naquele momento
  execução (3)             num_baias = num_materiais (vale em 3.497 de 3.501 cestões);
                           tempo de montagem = intervalo típico × (camadas − 1)
  carga_alta_c1            0 no cestão 1; no cestão 2, o resultado do cestão 1
                           (se ainda não se sabe, usa o modelo reserva)

Este arquivo não depende do Streamlit: dá para testar com pytest ou usar num notebook.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PASTA_APP = Path(__file__).resolve().parent
RAIZ = PASTA_APP.parent
CAMINHO_MODELO = RAIZ / "selecao_algoritmos" / "modelo_final.joblib"
PASTA_DADOS = PASTA_APP / "dados_app"

CAPACIDADE_M3 = 78.0          # volume do cestão (EVCOMX)
PESO_MIN_CAMADA_T = 2.0       # corte de 2.000 kg (Guima, 21/out/2026)
PESO_MAX_CAMADA_T = 15.0      # acima disso não aparece nos dados (p99 = 8 t)
MAX_CESTOES = 2
MAX_CAMADAS = 20              # nos dados: até 18 camadas, 78 t e 106 m³ estimados
MAX_PESO_CESTAO_T = 80.0
MAX_VOLUME_CESTAO_M3 = 110.0

HIST_PREFIXOS = ("h_", "taxa_ca_ult", "gap_")
TEMPO = ["hora", "turno_A", "turno_B", "dia_semana", "fim_de_semana"]


# ====================================================================== carga
def carregar_materiais() -> pd.DataFrame:
    return pd.read_csv(PASTA_DADOS / "materiais_app.csv", dtype={"codigo": str}).set_index("codigo")


def carregar_config() -> dict:
    return json.loads((PASTA_DADOS / "config_app.json").read_text(encoding="utf-8"))


def carregar_momentos() -> pd.DataFrame:
    return pd.read_csv(PASTA_DADOS / "momentos.csv").set_index("cd_corrida")


def carregar_modelo(caminho: Path = CAMINHO_MODELO) -> dict:
    # Só abrir arquivos de modelo confiáveis: joblib pode executar código ao carregar.
    return joblib.load(caminho)


@dataclass
class Camada:
    codigo: str
    peso_t: float


# ====================================================================== validação
def validar_camada(codigo: str, peso_t: float, materiais: pd.DataFrame) -> str | None:
    """Devolve uma mensagem de erro, ou None se a camada é válida."""
    if codigo not in materiais.index:
        return f"Material {codigo} não existe na tabela de materiais."
    if peso_t is None or pd.isna(peso_t):
        return "Informe o peso da camada."
    if peso_t < PESO_MIN_CAMADA_T:
        return (f"Camadas abaixo de {PESO_MIN_CAMADA_T:.0f} t não entram: a EVCOMX trata "
                "como gancho que não pegou o material.")
    if peso_t > PESO_MAX_CAMADA_T:
        return (f"{peso_t:.1f} t numa camada só é fora do que existe nos dados "
                f"(a maior camada típica tem até ~8 t). Máximo aceito: {PESO_MAX_CAMADA_T:.0f} t.")
    return None


def validar_insercao(camadas: list[Camada], nova: Camada, materiais: pd.DataFrame) -> str | None:
    """Confere a camada e o cestão como fica depois dela. None = pode inserir."""
    erro = validar_camada(nova.codigo, nova.peso_t, materiais)
    if erro:
        return erro
    if len(camadas) + 1 > MAX_CAMADAS:
        return f"O cestão já tem {len(camadas)} camadas. Nos dados, nenhum passa de 18."
    depois = resumo_cestao(camadas + [nova], materiais)
    if depois["peso_t_total"] > MAX_PESO_CESTAO_T:
        return (f"Com essa camada o cestão iria a {depois['peso_t_total']:.1f} t. "
                f"O cestão leva de 60 a 70 t (máximo aceito aqui: {MAX_PESO_CESTAO_T:.0f} t).")
    if depois["volume_m3_total"] > MAX_VOLUME_CESTAO_M3:
        return (f"Não cabe: o volume estimado iria a {depois['volume_m3_total']:.0f} m³ "
                f"(boca do cestão: {CAPACIDADE_M3:.0f} m³; máximo aceito aqui: {MAX_VOLUME_CESTAO_M3:.0f} m³).")
    return None


# ====================================================================== features da receita
def _tabela(camadas: list[Camada], materiais: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame([{"codigo": c.codigo, "peso_t": float(c.peso_t)} for c in camadas])
    m = materiais.loc[df["codigo"]]
    df["classe"] = m["classe"].values
    df["volume_m3"] = df["peso_t"] / m["densidade"].values
    df["rendimento"] = m["rendimento"].values
    df["energia_kwh_t"] = m["energia_kwh_t"].values
    n = len(df)
    df["pos_rel"] = np.arange(n) / max(n - 1, 1)          # 0 = fundo, 1 = topo
    return df


def features_receita(camadas: list[Camada], materiais: pd.DataFrame) -> dict:
    """As features que dependem só das camadas do cestão (feature_selection.py, seções 2.1–2.3 e 2.5)."""
    df = _tabela(camadas, materiais)
    f = {}
    peso, vol = df["peso_t"].sum(), df["volume_m3"].sum()
    f["peso_t_total"], f["volume_m3_total"] = peso, vol
    f["ocupacao_cestao"] = vol / CAPACIDADE_M3
    f["densidade_media"] = peso / vol
    f["razao_vol_peso"] = vol / peso
    f["peso_max_camada_t"] = df["peso_t"].max()
    f["peso_medio_camada_t"] = df["peso_t"].mean()
    for classe, k in [("Leve", "leve"), ("Misto", "misto"), ("Pesado", "pesado")]:
        m = df["classe"] == classe
        f[f"peso_{k}"] = df.loc[m, "peso_t"].sum()
        f[f"vol_{k}_m3"] = df.loc[m, "volume_m3"].sum()
        f[f"prop_peso_{k}"] = f[f"peso_{k}"] / peso
        f[f"prop_vol_{k}"] = f[f"vol_{k}_m3"] / vol
    f["prop_leve"] = f.pop("prop_peso_leve")
    f["rendimento_medio"] = (df["rendimento"] * df["peso_t"]).sum() / peso
    f["energia_estimada_mwh"] = (df["energia_kwh_t"] * df["peso_t"]).sum() / 1000
    f["energia_por_ton"] = f["energia_estimada_mwh"] * 1000 / peso

    f["num_camadas"] = len(df)
    f["num_materiais"] = df["codigo"].nunique()
    prop = df.groupby("codigo")["peso_t"].sum() / peso
    f["hhi_materiais"] = float((prop ** 2).sum())
    proprias = materiais.index[materiais["coluna_propria"]]
    for cod in proprias:
        f[f"prop_{cod}"] = float(prop.get(cod, 0.0))
    f["prop_outros"] = float(prop[~prop.index.isin(proprias)].sum())

    # blocos (camadas seguidas da mesma classe) e ciclos (= blocos de pesado)
    df["id_bloco"] = (df["classe"] != df["classe"].shift()).cumsum()
    blocos = df.groupby("id_bloco").agg(classe=("classe", "first"), volume_m3=("volume_m3", "sum"))
    f["n_blocos"] = len(blocos)
    for classe, k in [("Leve", "leve"), ("Misto", "misto"), ("Pesado", "pesado")]:
        bc = blocos[blocos["classe"] == classe]
        f[f"n_blocos_{k}"] = len(bc)
        f[f"max_bloco_{k}_m3"] = bc["volume_m3"].max() if len(bc) else 0.0
    f["n_ciclos"] = f.pop("n_blocos_pesado")
    seq = "".join(blocos["classe"].map({"Leve": "L", "Pesado": "P", "Misto": "M"}))
    f["n_ciclos_LPM"] = sum(1 for i in range(len(seq)) if seq.startswith("LPM", i))

    primeira, ultima = df["classe"].iloc[0], df["classe"].iloc[-1]
    for classe, k in [("Leve", "leve"), ("Misto", "misto"), ("Pesado", "pesado")]:
        f[f"fundo_{k}"] = int(primeira == classe)
        f[f"topo_{k}"] = int(ultima == classe)
    f["violacao_regra"] = int(primeira == "Pesado" or ultima == "Pesado")

    topo30, base30 = df["pos_rel"] >= 0.7, df["pos_rel"] <= 0.3
    leve, pesado = df["classe"] == "Leve", df["classe"] == "Pesado"
    f["peso_topo_30pct_t"] = df.loc[topo30, "peso_t"].sum()
    f["peso_base_30pct_t"] = df.loc[base30, "peso_t"].sum()
    f["peso_leve_topo_t"] = df.loc[topo30 & leve, "peso_t"].sum()
    f["prop_leve_topo"] = f["peso_leve_topo_t"] / f["peso_topo_30pct_t"] if f["peso_topo_30pct_t"] else np.nan
    f["prop_leve_base"] = df.loc[base30 & leve, "peso_t"].sum() / f["peso_base_30pct_t"] if f["peso_base_30pct_t"] else np.nan

    def pos_ponderada(m):
        d = df[m]
        return float((d["pos_rel"] * d["peso_t"]).sum() / d["peso_t"].sum()) if len(d) else np.nan
    f["pos_media_pesado"] = pos_ponderada(pesado)
    f["pos_media_leve"] = pos_ponderada(leve)

    f["vol_x_prop_leve"] = vol * f["prop_leve"]
    f["peso_x_hhi"] = peso * f["hhi_materiais"]
    f["peso_por_ciclo_t"] = peso / max(f["n_ciclos"], 1)
    return f


def execucao_estimada(n_camadas: int, num_materiais: int, config: dict) -> dict:
    """Número de baias e tempo de montagem, que não fazem parte da receita."""
    tab = {int(k): v for k, v in config["intervalo_por_n_camadas"].items()}
    intervalo = tab.get(n_camadas, tab[min(tab, key=lambda k: abs(k - n_camadas))])
    tempo = intervalo * (n_camadas - 1)
    return {"num_baias": num_materiais, "tempo_montagem_min": tempo,
            "intervalo_medio_min": tempo / (n_camadas - 1) if n_camadas > 1 else np.nan}


def resumo_cestao(camadas: list[Camada], materiais: pd.DataFrame) -> dict:
    """Indicadores físicos para o painel (e para validar o cestão)."""
    if not camadas:
        return {"n_camadas": 0, "peso_t_total": 0.0, "volume_m3_total": 0.0, "ocupacao": 0.0,
                "densidade_media": np.nan, "prop_peso_pesado": np.nan, "n_ciclos": 0,
                "peso_por_ciclo_t": np.nan, "prop_PACB": np.nan}
    f = features_receita(camadas, materiais)
    return {"n_camadas": f["num_camadas"], "peso_t_total": f["peso_t_total"],
            "volume_m3_total": f["volume_m3_total"], "ocupacao": f["ocupacao_cestao"],
            "densidade_media": f["densidade_media"], "prop_peso_pesado": f["prop_peso_pesado"],
            "n_ciclos": f["n_ciclos"], "peso_por_ciclo_t": f["peso_por_ciclo_t"],
            "prop_PACB": f.get("prop_PACB", 0.0)}


# ====================================================================== linha completa do modelo
def linha_cestao(k: int, receitas: list[dict], codigos: list[set], momento: pd.Series,
                 gusa_t: float | None, carga_alta_c1: int, execucao: dict) -> dict:
    """As 120 features + carga_alta_c1 de um cestão (k = 1 ou 2).

    receitas[i] = features_receita do cestão i+1; codigos[i] = materiais do cestão i+1."""
    f = dict(receitas[k - 1])
    f.update(execucao)
    f.update({c: momento[f"c{k}_{c}"] for c in TEMPO})
    f.update({c: momento[c] for c in momento.index if c.startswith(HIST_PREFIXOS)})

    eh2 = k == 2
    c1 = receitas[0]
    f["eh_cestao_2"] = int(eh2)
    f["volume_c1_m3"] = c1["volume_m3_total"] if eh2 else 0.0
    f["peso_c1_t"] = c1["peso_t_total"] if eh2 else 0.0
    f["vol_acumulado_m3"] = f["volume_m3_total"] + f["volume_c1_m3"]
    f["delta_peso_12_t"] = f["peso_t_total"] - c1["peso_t_total"] if eh2 else 0.0
    f["mudou_composicao"] = int(codigos[0] != codigos[1]) if eh2 else 0

    f["gusl_t"] = float(gusa_t or 0.0)
    f["tem_gusl"] = int(f["gusl_t"] > 0)
    f["carga3_solida_t"] = 0.0

    for nome in [c[len(f"c{k}_m_"):] for c in momento.index if c.startswith(f"c{k}_m_")]:
        f[f"rel_{nome}"] = f[nome] - momento[f"c{k}_m_{nome}"]
        f[f"rel2_{nome}"] = f[nome] - momento[f"c{k}_m2_{nome}"]
    f["carga_alta_c1"] = carga_alta_c1 if eh2 else 0
    return f


def chance(parte: dict, linhas: list[dict]) -> np.ndarray:
    """Média das probabilidades dos 5 modelos (como_usar do pacote)."""
    X = pd.DataFrame(linhas)[parte["features"]]
    return np.mean([m.predict_proba(X)[:, 1] for m in parte["modelos"]], axis=0)


def prever_corrida(cestoes: list[list[Camada]], momento: pd.Series, gusa_t: float | None,
                   resultado_c1: int | None, modo: str, pacote: dict, materiais: pd.DataFrame,
                   config: dict) -> dict:
    """Chance e alarme de cada cestão. resultado_c1: 0, 1 ou None (ainda não se sabe).

    Cestão 1: modelo principal com carga_alta_c1 = 0 (como foi treinado).
    Cestão 2: modelo principal com o resultado do cestão 1; se não se sabe, o modelo reserva.
    A corrida alarma se qualquer cestão passar do limiar de minutos."""
    receitas = [features_receita(c, materiais) for c in cestoes]
    codigos = [{x.codigo for x in c} for c in cestoes]
    if len(codigos) == 1:
        codigos.append(set())
    resultados = []
    for k in range(1, len(cestoes) + 1):
        ex = execucao_estimada(receitas[k - 1]["num_camadas"], receitas[k - 1]["num_materiais"], config)
        base = lambda c1: linha_cestao(k, receitas, codigos, momento, gusa_t, c1, ex)
        r = {"cestao": k, "x": base(0)}
        if k == 1 or resultado_c1 is not None:
            parte, nome = pacote["com_cestao1"], "principal"
            p = float(chance(parte, [base(0 if k == 1 else resultado_c1)])[0])
        else:
            parte, nome = pacote["sem_cestao1_reserva"], "reserva"
            p = float(chance(parte, [base(0)])[0])
        limiar = config["limiares"][nome][modo]
        r.update({"chance": p, "limiar": limiar, "alarme": p > limiar, "modelo": nome})
        resultados.append(r)
    return {"cestoes": resultados, "alarme_corrida": any(r["alarme"] for r in resultados)}


# ====================================================================== leitura das features
# Sentido no treino (feature_selection/tabela_features_finais.csv): + = valor maior, mais risco.
SENTIDO = {"densidade_media": -1, "prop_peso_pesado": -1, "peso_por_ciclo_t": +1, "prop_PACB": +1}


def ler_feature(nome: str, valor: float, referencia: dict) -> str:
    """'mais risco', 'menos risco' ou 'normal', comparando com o miolo (p25–p75) do treino."""
    ref = referencia[nome]
    if ref["p25"] <= valor <= ref["p75"]:
        return "normal"
    acima = valor > ref["p75"]
    return "mais risco" if (acima and SENTIDO[nome] > 0) or (not acima and SENTIDO[nome] < 0) else "menos risco"


def resumo_momento(momento: pd.Series) -> dict:
    """O que o momento escolhido diz, em português, para mostrar na tela."""
    return {
        "inicio": momento["inicio"],
        "turno": "A (06–14h)" if momento["c1_turno_A"] else ("B (14–22h)" if momento["c1_turno_B"] else "C (22–06h)"),
        "ult1": momento["h_ult1"], "ult5": momento["h_taxa5"], "ult10": momento["h_taxa10"],
        "ult50": momento["h_taxa50"], "taxa_24h": momento["h_taxa_24h"], "n_24h": momento["h_n_corridas_24h"],
        "desde_ultima": momento["h_desde_ultima"],
    }


# ====================================================================== o que mudar
PESADOS_COMUNS = ["GUSS", "194032", "RECA", "SUGU", "118441", "FOFO"]


def sugerir_troca(cestoes: list[list[Camada]], k: int, momento: pd.Series, gusa_t: float | None,
                  resultado_c1: int | None, modo: str, pacote: dict, materiais: pd.DataFrame,
                  config: dict) -> dict | None:
    """Testa trocar UMA camada não pesada do cestão k por um material pesado comum, com o
    mesmo peso, e devolve a troca que mais baixa a chance desse cestão. É uma pergunta ao
    modelo, não uma regra de operação."""
    antes = prever_corrida(cestoes, momento, gusa_t, resultado_c1, modo, pacote, materiais, config)
    alvo = antes["cestoes"][k - 1]
    parte = pacote["com_cestao1"] if alvo["modelo"] == "principal" else pacote["sem_cestao1_reserva"]
    c1_val = 0 if k == 1 else (resultado_c1 or 0)
    candidatos, linhas = [], []
    for j, cam in enumerate(cestoes[k - 1]):
        if materiais.loc[cam.codigo, "classe"] == "Pesado":
            continue
        for p in PESADOS_COMUNS:
            novo = [list(c) for c in cestoes]
            novo[k - 1][j] = Camada(p, cam.peso_t)
            receitas = [features_receita(c, materiais) for c in novo]
            codigos = [{x.codigo for x in c} for c in novo] + ([set()] if len(novo) == 1 else [])
            ex = execucao_estimada(receitas[k - 1]["num_camadas"], receitas[k - 1]["num_materiais"], config)
            linhas.append(linha_cestao(k, receitas, codigos, momento, gusa_t, c1_val, ex))
            candidatos.append((j, p))
    if not linhas:
        return None
    ch = chance(parte, linhas)
    i = int(ch.argmin())
    j, p = candidatos[i]
    return {"camada": j, "de": cestoes[k - 1][j].codigo, "para": p, "peso_t": cestoes[k - 1][j].peso_t,
            "chance_antes": alvo["chance"], "chance_depois": float(ch[i]), "limiar": alvo["limiar"]}


# ====================================================================== recombinar o cestão
MAX_MUDANCAS = 3              # no máximo 3 mudanças na receita
PCT_MIN_MATERIAL = 0.05       # só sugere material presente em pelo menos 5% dos cestões
FAIXA_PESO_MATERIAL = (0.8, 1.2)   # o peso da camada precisa caber na faixa típica do material novo (p10–p90 ± 20%)
N_MOVIMENTOS = 30             # da 2ª mudança em diante, combina só os 30 movimentos mais promissores
LARGURA_BUSCA = 3             # receitas parciais guardadas a cada passo
GANHO_MINIMO = 0.02           # abaixo de 2 pontos, a mudança não vale a sugestão
LIMITADAS = ["densidade_media", "prop_peso_pesado", "num_materiais", "n_ciclos", "max_bloco_pesado_m3"]


def _movimentos(camadas: list[Camada], materiais: pd.DataFrame) -> list[tuple]:
    """Mudanças de um passo: trocar uma camada por um material mais denso, com o mesmo peso,
    ou inverter duas camadas vizinhas de classes diferentes."""
    comuns = materiais[materiais["pct_cestoes"] >= PCT_MIN_MATERIAL]
    mov = []
    for j, c in enumerate(camadas):
        dens = materiais.loc[c.codigo, "densidade"]
        for cod, m in comuns.iterrows():
            lo, hi = m["peso_p10_t"] * FAIXA_PESO_MATERIAL[0], m["peso_p90_t"] * FAIXA_PESO_MATERIAL[1]
            if m["densidade"] > dens and lo <= c.peso_t <= hi:
                mov.append(("material", j, cod))
    for j in range(len(camadas) - 1):
        if materiais.loc[camadas[j].codigo, "classe"] != materiais.loc[camadas[j + 1].codigo, "classe"]:
            mov.append(("ordem", j, None))
    return mov


def _aplicar(camadas: list[Camada], mov: tuple, materiais: pd.DataFrame) -> list[Camada] | None:
    """Receita depois do movimento; None se ele não fizer sentido nesta receita."""
    tipo, j, cod = mov
    novo = list(camadas)
    if tipo == "ordem":
        if materiais.loc[novo[j].codigo, "classe"] == materiais.loc[novo[j + 1].codigo, "classe"]:
            return None
        novo[j], novo[j + 1] = novo[j + 1], novo[j]
        return novo
    if materiais.loc[cod, "densidade"] <= materiais.loc[novo[j].codigo, "densidade"]:
        return None
    m = materiais.loc[cod]
    lo, hi = m["peso_p10_t"] * FAIXA_PESO_MATERIAL[0], m["peso_p90_t"] * FAIXA_PESO_MATERIAL[1]
    if not lo <= novo[j].peso_t <= hi:
        return None
    novo[j] = Camada(cod, novo[j].peso_t)
    return novo


def _texto_mudanca(camadas: list[Camada], mov: tuple, materiais: pd.DataFrame) -> str:
    """Frase da mudança, com as posições contadas a partir do fundo (camada 1)."""
    tipo, j, cod = mov
    if tipo == "ordem":
        return (f"Inverter as camadas {j + 1} e {j + 2} "
                f"({camadas[j].codigo} e {camadas[j + 1].codigo})")
    peso = f"{camadas[j].peso_t:.1f}".replace(".", ",")
    return (f"Trocar a camada {j + 1}: {camadas[j].codigo} por {cod} "
            f"({materiais.loc[cod, 'descricao'].title()}), {peso} t")


def _dentro_dos_limites(f: dict, original: dict, limites: dict) -> bool:
    """Receita realista: não cria pesado no fundo/topo e não passa do p99 dos cestões reais.
    Se a receita original já estava fora de um limite, a nova só não pode piorar."""
    if f["violacao_regra"] > original["violacao_regra"]:
        return False
    return all(f[c] <= max(limites[c], original[c]) for c in LIMITADAS)


def recombinar_cestao(cestoes: list[list[Camada]], k: int, momento: pd.Series, gusa_t: float | None,
                      resultado_c1: int | None, modo: str, pacote: dict, materiais: pd.DataFrame,
                      config: dict, max_mudancas: int = MAX_MUDANCAS) -> dict | None:
    """Procura a menor recombinação do cestão k que tire a corrida do alarme nesse cestão.

    Mudanças permitidas: material mais denso com o mesmo peso e inversão de camadas vizinhas.
    O peso total nunca muda. A receita final precisa caber na boca do cestão, respeitar os
    limites reais (config["limites_receita"]) e não pode fazer o outro cestão passar a alarmar.
    É uma pergunta feita ao modelo, não uma regra de operação.

    status: "ok" (fica abaixo do limiar), "parcial" (baixa, mas continua acima) ou
    "sem_solucao" (nenhuma mudança realista baixa pelo menos GANHO_MINIMO)."""
    antes = prever_corrida(cestoes, momento, gusa_t, resultado_c1, modo, pacote, materiais, config)
    alvo = antes["cestoes"][k - 1]
    original = cestoes[k - 1]
    if not original:
        return None
    parte = pacote["com_cestao1"] if alvo["modelo"] == "principal" else pacote["sem_cestao1_reserva"]
    c1_val = 0 if k == 1 else (resultado_c1 or 0)
    limites = config["limites_receita"]
    receitas = [features_receita(c, materiais) for c in cestoes]
    f_orig = receitas[k - 1]

    def linha(camadas):
        rec = list(receitas)
        rec[k - 1] = features_receita(camadas, materiais)
        cods = [{x.codigo for x in (camadas if i == k - 1 else c)} for i, c in enumerate(cestoes)]
        if len(cods) == 1:
            cods.append(set())
        ex = execucao_estimada(rec[k - 1]["num_camadas"], rec[k - 1]["num_materiais"], config)
        return rec[k - 1], linha_cestao(k, rec, cods, momento, gusa_t, c1_val, ex)

    def outro_cestao_piora(camadas):
        """Mudar o cestão 1 mexe nas features do cestão 2: ele não pode passar a alarmar."""
        if k != 1 or len(cestoes) < 2:
            return False, None
        novo = [camadas, cestoes[1]]
        r = prever_corrida(novo, momento, gusa_t, resultado_c1, modo, pacote, materiais, config)["cestoes"][1]
        a = antes["cestoes"][1]
        info = {"cestao": 2, "antes": a["chance"], "depois": r["chance"], "alarme": r["alarme"]}
        return (r["alarme"] and not a["alarme"]) or r["chance"] > a["chance"] + GANHO_MINIMO, info

    fila = [(original, [])]
    vistos = {tuple((c.codigo, c.peso_t) for c in original)}
    movimentos = None
    melhor = None                       # melhor receita válida que não chega ao limiar
    for passo in range(1, max_mudancas + 1):
        filhos, linhas = [], []
        for camadas, hist in fila:
            for mov in (movimentos if movimentos is not None else _movimentos(camadas, materiais)):
                novo = _aplicar(camadas, mov, materiais)
                if novo is None:
                    continue
                chave = tuple((c.codigo, c.peso_t) for c in novo)
                if chave in vistos:
                    continue
                vistos.add(chave)
                f, lin = linha(novo)
                if _dentro_dos_limites(f, f_orig, limites):
                    filhos.append((novo, hist + [(camadas, mov)], f))
                    linhas.append(lin)
        if not filhos:
            break
        ch = chance(parte, linhas)
        ordem = np.argsort(ch, kind="stable")
        if movimentos is None:
            movimentos = list(dict.fromkeys(filhos[i][1][-1][1] for i in ordem[:N_MOVIMENTOS]))
        for i in ordem:
            novo, hist, f = filhos[i]
            cabe = f["volume_m3_total"] <= CAPACIDADE_M3
            piora, info = outro_cestao_piora(novo) if cabe else (False, None)
            if not cabe or piora:
                continue
            if melhor is None or ch[i] < melhor["chance"]:
                melhor = {"camadas": novo, "hist": hist, "chance": float(ch[i]), "outro": info}
            break                       # em ordem de chance: a 1ª válida é a melhor deste passo
        if melhor is not None and melhor["chance"] <= alvo["limiar"]:
            break
        fila = [(filhos[i][0], filhos[i][1]) for i in ordem[:LARGURA_BUSCA]]

    if melhor is None or alvo["chance"] - melhor["chance"] < GANHO_MINIMO:
        status = "sem_solucao"
    else:
        status = "ok" if melhor["chance"] <= alvo["limiar"] else "parcial"
    res = {"status": status, "cestao": k, "chance_antes": alvo["chance"], "limiar": alvo["limiar"],
           "modelo": alvo["modelo"], "antes": resumo_cestao(original, materiais),
           "excede_volume": f_orig["volume_m3_total"] > CAPACIDADE_M3}
    if status != "sem_solucao":
        res.update({"chance_depois": melhor["chance"], "outro": melhor["outro"],
                    "mudancas": [_texto_mudanca(c, m, materiais) for c, m in melhor["hist"]],
                    "camadas": [(c.codigo, c.peso_t) for c in melhor["camadas"]],
                    "depois": resumo_cestao(melhor["camadas"], materiais)})
    return res
