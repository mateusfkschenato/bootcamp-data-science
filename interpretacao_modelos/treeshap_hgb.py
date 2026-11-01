"""TreeSHAP exato (path-dependent, Lundberg et al. 2018) para HistGradientBoostingClassifier.

Por que um arquivo próprio: o pacote `shap` não pôde ser instalado no ambiente onde o projeto foi
rodado. A conta é a mesma do `shap.TreeExplainer(modelo)` no modo padrão ("tree_path_dependent"),
na escala do modelo (log-odds). Se o `shap` estiver instalado, `conferir_com_shap` compara as duas.

A ideia, folha a folha: num caminho da raiz até uma folha com d features distintas, cada feature j
tem
- z_j: a fração dos cestões de treino que seguiu esse caminho nas divisões por j ("não sei o valor");
- o_j: 1 se o cestão explicado segue o caminho em todas as divisões por j, senão 0 ("sei o valor").
A contribuição da folha para a feature i é
    valor_folha × (o_i − z_i) × Σ_{S ⊆ outras} |S|!(d−|S|−1)!/d! × Π_{j∈S} o_j × Π_{j∉S} z_j,
que é o valor de Shapley exato do "jogo" em que saber uma feature = seguir o caminho dela e não
saber = dividir pela proporção do treino. Somando as folhas e as árvores, vale sempre
    referência + Σ_i SHAP_i = saída do modelo (log-odds), cestão a cestão.
"""
from math import factorial

import numpy as np


def _caminhos(nodes):
    """Lista de folhas: (valor, [(feature, limiar, vai_esquerda, nan_esquerda, fração), ...])."""
    out = []

    def desce(i, passos):
        n = nodes[i]
        if n["is_leaf"]:
            out.append((float(n["value"]), passos))
            return
        f, t, ml = int(n["feature_idx"]), float(n["num_threshold"]), bool(n["missing_go_to_left"])
        for filho, esq in [(int(n["left"]), True), (int(n["right"]), False)]:
            desce(filho, passos + [(f, t, esq, ml, nodes[filho]["count"] / n["count"])])

    desce(0, [])
    return out


def _segue(X, f, t, esq, ml):
    x = X[:, f]
    nan = np.isnan(x)
    vai_esq = np.where(nan, ml, x <= t)
    return vai_esq if esq else ~vai_esq


def _pesos(d):
    return np.array([factorial(s) * factorial(d - s - 1) / factorial(d) for s in range(d)])


def shap_hgb(modelo, X):
    """SHAP (log-odds) de um HistGradientBoostingClassifier. Devolve (phi [n, p], referência)."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    phi = np.zeros((n, p))
    ref = float(np.ravel(modelo._baseline_prediction)[0])
    for (pred,) in modelo._predictors:
        nodes = pred.nodes
        tot = nodes[0]["count"]
        for valor, passos in _caminhos(nodes):
            if not passos:
                ref += valor
                continue
            feats = sorted({s[0] for s in passos})
            d = len(feats)
            z = np.ones(d)
            o = np.ones((n, d), dtype=bool)
            for f, t, esq, ml, frac in passos:
                k = feats.index(f)
                z[k] *= frac
                o[:, k] &= _segue(X, f, t, esq, ml)
            ref += valor * np.prod(z)          # E[f] = Σ folhas valor × Π frações
            o = o.astype(float)
            w = _pesos(d)
            for k, fi in enumerate(feats):
                # polinômio Π_{j≠k} (z_j + o_j·u): coeficiente de u^s = Σ_{|S|=s} Π o Π z
                coef = np.zeros((n, d))
                coef[:, 0] = 1.0
                grau = 0
                for j in range(d):
                    if j == k:
                        continue
                    novo = coef * z[j]
                    novo[:, 1:grau + 2] += coef[:, :grau + 1] * o[:, [j]]
                    coef = novo; grau += 1
                g = coef[:, :d] @ w
                phi[:, fi] += valor * (o[:, k] - z[k]) * g
    return phi, ref


def shap_comite(modelos, X):
    """SHAP médio dos membros do comitê (log-odds médio) e a referência média."""
    res = [shap_hgb(m, X) for m in modelos]
    return np.mean([r[0] for r in res], axis=0), float(np.mean([r[1] for r in res]))


def conferir_com_shap(modelo, X, phi):
    """Compara com o pacote shap, se instalado. Devolve a maior diferença, ou None sem o pacote."""
    try:
        import shap  # noqa: F401
    except ImportError:
        return None
    sv = shap.TreeExplainer(modelo).shap_values(np.asarray(X, dtype=float))
    sv = sv[1] if isinstance(sv, list) else sv
    return float(np.max(np.abs(np.asarray(sv) - phi)))
