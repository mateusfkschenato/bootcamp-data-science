"""Compara o modelo que acabou de sair com os números guardados em pipeline/referencia.json.

Com os mesmos dados e as mesmas versões, tudo deve bater. Se os dados mudaram (lote novo), a
diferença é esperada: revise os resultados e atualize a referência com --atualizar.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from comum.protocolo import NOME_OFICIAL, carregar, prever_pacote  # noqa: E402

ARQ = RAIZ / "pipeline" / "referencia.json"
TOLERANCIA = 1e-9


def medir() -> dict:
    pac = joblib.load(RAIZ / "selecao_algoritmos" / "modelo_final.joblib")
    D, _, TESTE = carregar()
    res = pd.read_csv(RAIZ / "selecao_algoritmos" / "resultado_teste.csv", index_col=0).loc[NOME_OFICIAL]
    chances = prever_pacote(pac, D)
    return {"cestoes": int(len(D)), "cargas_altas": int(D.carga_alta.sum()), "cestoes_teste": int(len(TESTE)),
            "minutos_teste": float(res["min perdidos teste"]), "soma_das_chances": float(np.sum(chances)),
            "limiar_principal_TP0": float(pac["com_cestao1"]["limiar_minutos_TP0"]),
            "limiar_principal_TP1": float(pac["com_cestao1"]["limiar_minutos_TP1"]),
            "limiar_reserva_TP0": float(pac["sem_cestao1_reserva"]["limiar_minutos_TP0"])}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--atualizar", action="store_true", help="grava os números atuais como referência")
    a = p.parse_args()
    atual = medir()
    if a.atualizar:
        ARQ.write_text(json.dumps(atual, indent=1), encoding="utf-8")
        print("Referência atualizada.")
        return 0
    ref = json.loads(ARQ.read_text(encoding="utf-8"))
    dif = {k: (ref[k], atual[k]) for k in ref if abs(ref[k] - atual[k]) > TOLERANCIA}
    for k, (r, v) in dif.items():
        print(f"DIFERENTE {k}: referência {r}, agora {v}")
    if dif:
        print("O modelo não bate com a referência. Dados novos? Revise e rode com --atualizar.")
        return 1
    print(f"Modelo igual à referência: {atual['minutos_teste']:.1f} min no teste, limiar {atual['limiar_principal_TP0']:.4f}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
