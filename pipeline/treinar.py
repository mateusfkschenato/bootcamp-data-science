"""Pipeline de treino: do dado bruto até o modelo, com um comando só.

    python pipeline/treinar.py                  # roda tudo, em ordem
    python pipeline/treinar.py --listar         # mostra as etapas
    python pipeline/treinar.py --de selecao     # recomeça de uma etapa
    python pipeline/treinar.py --ate selecao    # só até o modelo sair
    python pipeline/treinar.py --apenas analise verificacao

Cada etapa é o script da própria pasta, sem alteração, rodado no diretório que ele espera.
O pipeline para na primeira etapa que falhar. O log de cada uma fica em pipeline/logs/.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
BRUTOS = RAIZ / "original_dataset"
LOGS = RAIZ / "pipeline" / "logs"
ARQUIVOS_BRUTOS = ["infos_camadas.csv", "infos_materiais.csv", "dicionario_tipo_material.csv", "VariavelResposta.csv"]


@dataclass
class Etapa:
    nome: str
    descricao: str
    comando: list[str] | None        # None = função interna (só a etapa "dados")
    pasta: str = "."
    minutos: str = "<1"


ETAPAS = [
    Etapa("dados", "confere os 4 arquivos brutos", None),
    Etapa("limpeza", "limpa os dados e gera EDA/base.csv", ["EDA/transforma_dataset.py"], "EDA"),
    Etapa("eda", "gráficos da análise exploratória", ["EDA/EDA.py"], "EDA", "1"),
    Etapa("features", "121 features e dataset de modelagem", ["feature_selection/feature_selection.py"], ".", "20"),
    Etapa("selecao", "walk-forward, limiares, teste e modelo_final.joblib", ["selecao_algoritmos/selecao_algoritmos.py"], ".", "8"),
    Etapa("analise", "métricas, calibração e segmentos", ["analise_modelos/analise_modelos.py"], ".", "1"),
    Etapa("verificacao", "conferências automáticas, drift e robustez", ["verificacao_validacao/verificacao_validacao.py"], ".", "11"),
    Etapa("interpretacao", "SHAP, contrafactuais e explicações", ["interpretacao_modelos/interpretacao_modelos.py"], ".", "1"),
    Etapa("app", "tabelas do simulador (dados_app/)", ["app_streamlit/preparar_dados_app.py"], ".", "<1"),
    Etapa("conferencia", "compara o modelo com a referência guardada", ["pipeline/conferir_modelo.py"], ".", "<1"),
    Etapa("testes", "testes do app e da previsão", ["-m", "pytest", "app_streamlit/test_motor.py", "pipeline/test_previsao.py", "-q"], ".", "1"),
]


def etapa_dados(log) -> None:
    """Confere se os 4 arquivos brutos estão em original_dataset/ antes de começar."""
    faltam = [a for a in ARQUIVOS_BRUTOS if not (BRUTOS / a).exists()]
    if faltam:
        raise FileNotFoundError(f"Faltam arquivos em original_dataset/: {', '.join(faltam)}")
    log.write("Os 4 arquivos brutos estão em original_dataset/.\n")


def rodar(etapa: Etapa) -> float:
    ini = time.time()
    arq_log = LOGS / f"{etapa.nome}.log"
    with open(arq_log, "w", encoding="utf-8") as log:
        if etapa.comando is None:
            etapa_dados(log)
            return time.time() - ini
        cmd = [sys.executable]
        for parte in etapa.comando:
            eh_script = parte.endswith(".py") and not parte.startswith("-")
            cmd.append(str(RAIZ / parte) if eh_script else parte)
        env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(RAIZ)}
        r = subprocess.run(cmd, cwd=RAIZ / etapa.pasta, env=env, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        fim = arq_log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        print("\n".join("    " + l for l in fim))
        raise RuntimeError(f"A etapa '{etapa.nome}' falhou. Log completo: {arq_log.relative_to(RAIZ)}")
    return time.time() - ini


def escolher(args) -> list[Etapa]:
    nomes = [e.nome for e in ETAPAS]
    for n in [args.de, args.ate, *(args.apenas or [])]:
        if n and n not in nomes:
            raise SystemExit(f"Etapa desconhecida: {n}. Opções: {', '.join(nomes)}")
    if args.apenas:
        return [e for e in ETAPAS if e.nome in args.apenas]
    i = nomes.index(args.de) if args.de else 0
    j = nomes.index(args.ate) if args.ate else len(nomes) - 1
    return ETAPAS[i:j + 1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--listar", action="store_true", help="mostra as etapas e sai")
    p.add_argument("--de", help="começa nesta etapa")
    p.add_argument("--ate", help="termina nesta etapa")
    p.add_argument("--apenas", nargs="+", help="roda só estas etapas")
    args = p.parse_args()
    if args.listar:
        for e in ETAPAS:
            print(f"{e.nome:14s} ~{e.minutos:>3s} min  {e.descricao}")
        return 0

    LOGS.mkdir(exist_ok=True)
    etapas = escolher(args)
    resumo = {"inicio": datetime.now().isoformat(timespec="seconds"), "etapas": [], "status": "ok"}
    for k, e in enumerate(etapas, 1):
        print(f"[{k}/{len(etapas)}] {e.nome}: {e.descricao}", flush=True)
        try:
            seg = rodar(e)
        except (RuntimeError, FileNotFoundError) as erro:
            print(f"ERRO: {erro}", flush=True)
            resumo["status"] = f"falhou em {e.nome}"
            resumo["etapas"].append({"etapa": e.nome, "status": "falhou"})
            (LOGS / "ultima_execucao.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
            return 1
        print(f"      ok em {seg:.0f} s", flush=True)
        resumo["etapas"].append({"etapa": e.nome, "status": "ok", "segundos": round(seg, 1)})
    (LOGS / "ultima_execucao.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Pipeline concluído.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
