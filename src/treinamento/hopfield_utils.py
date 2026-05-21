import numpy as np


def sorti(x, mode="ascend"):
    """Retorna os índices que ordenam x (equivalente a sorti.m).

    mode: "ascend" (padrão) ou "descend".
    """
    x = np.asarray(x).ravel()
    idx = np.argsort(x)
    if str(mode).lower().startswith("desc"):
        idx = idx[::-1]
    return idx.copy()


def princomp_(W):
    """PCA sem centralização via SVD (equivalente a princomp_.m).

    Equivale a pca(W, 'Centered', false, 'Algorithm', 'svd') do MATLAB.
    Retorna a matriz de loadings (d x d) com componentes nas colunas,
    em ordem decrescente de variância explicada.
    """
    W = np.asarray(W, dtype=np.float64)
    _, _, Vt = np.linalg.svd(W, full_matrices=False)
    return Vt.T


def closervects(W, Wi, k, distance="euclidean"):
    """Índices dos k vetores em W mais próximos de Wi (equivalente a closervects.m).

    Wi pode ser:
      - vetor(es) com mesma dimensão das linhas de W → toma a média;
      - vetor de índices em W → busca em torno da média de W[indices].

    distance: "euclidean" (padrão) ou número para Lk-norm.
    """
    W = np.asarray(W, dtype=np.float64)
    Wi = np.asarray(Wi, dtype=np.float64)
    if Wi.ndim == 1:
        Wi = Wi[None, :]
    _, mm = W.shape
    _, q = Wi.shape

    if q == mm:
        query = Wi.mean(axis=0)
    else:
        idx = Wi.ravel().astype(int) - 1
        query = W[idx].mean(axis=0)

    if isinstance(distance, str) and distance.lower() == "euclidean":
        diff = W - query[None, :]
        u = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    else:
        kp = float(distance)
        diff = np.abs(W - query[None, :])
        u = np.sum(diff ** kp, axis=1)
        if kp >= 1:
            u = u ** (1.0 / kp)

    ii = np.argsort(u)[:k]
    return int(ii[0]) if k == 1 else ii


def contaocorr(v, ordby_max=True):
    """Conta ocorrências de cada valor distinto em v (equivalente a contaocorr.m).

    Retorna matriz [valor, contagem]:
      - ordenada por contagem decrescente se ordby_max=True (padrão);
      - ordenada por valor crescente se ordby_max=False.
    """
    v = np.asarray(v).ravel()
    vals, counts = np.unique(v, return_counts=True)
    order = np.argsort(-counts, kind="stable") if ordby_max else np.argsort(vals)
    return np.column_stack([vals[order], counts[order]])


def mat2celllines(M):
    """Converte as linhas de M em lista de vetores 1D (equivalente a mat2celllines.m)."""
    M = np.asarray(M)
    return [M[i] for i in range(M.shape[0])]


def wsort(W, return_perm=False, rng=None):
    """Embaralha aleatoriamente as linhas de W (equivalente a wsort.m).

    Equivale a [~, id] = sort(rand(N,1)); W = W(id,:) do MATLAB.
    Não confundir com ordenação por conteúdo.
    """
    W = np.asarray(W)
    if rng is None:
        rng = np.random.default_rng()
    perm = rng.permutation(W.shape[0])
    out = W[perm]
    return (out, perm) if return_perm else out


def indexa(X, xinds):
    """Indexação estilo MATLAB X(xinds) (equivalente a indexa.m).

    Aceita também a string 'SECOND' para retornar o segundo elemento.
    """
    X = np.asarray(X)
    if isinstance(xinds, str):
        if xinds.upper() == "SECOND":
            return X.ravel()[1] if X.size > 1 else np.array([])
        raise NotImplementedError(f"indexa: modo '{xinds}' não implementado.")
    inds = np.asarray(
        list(xinds) if hasattr(xinds, "__iter__") else [xinds], dtype=int
    ).ravel() - 1
    return X[inds]
