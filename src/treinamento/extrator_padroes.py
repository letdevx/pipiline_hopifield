"""Módulo de Extração de Padrões Biológicos Representativos por Subcluster.

Utiliza estratégias de agrupamento no espaço latente SWeeP para descobrir
protótipos e centróides celulares armazenáveis na memória Hopfield.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from .estrategias_clusterizacao import (
    EstrategiaClusterizacao,
    EstrategiaKMeansDinamico,
    EstrategiaKMeansFixo,
)
from .hopfield_utils import closervects


def calcular_nc_estratificado(
    labels: NDArray[Any] | Sequence[int],
    classes: Sequence[int],
    total_padroes: int = 180,
    min_nc: int = 12,
    max_nc: int = 40,
    power: float = 0.5,
) -> dict[int, int]:
    """Calcula a alocação estratificada de subclusters proporcional à raiz quadrada da abundância celular.

    Equilibra a representatividade biológica na memória Hopfield:
    - Previne a superfragmentação e ruído em classes raras (ex.: Endoteliais, OPCs).
    - Assegura diversidade suficiente para capturar sublinhagens em classes abundantes (ex.: Inibitórios, Excitatórios).

    Parameters
    ----------
    labels : NDArray | Sequence[int]
        Vetor de classes de cada célula no dataset.
    classes : Sequence[int]
        Lista ordenada das classes canônicas a alocar.
    total_padroes : int, default=180
        Meta total aproximada de padrões na memória associativa.
    min_nc : int, default=12
        Cota mínima de subclusters por classe.
    max_nc : int, default=40
        Cota máxima de subclusters por classe.
    power : float, default=0.5
        Expoente de amortecimento da abundância (0.5 = raiz quadrada, padrão para scRNA-seq).

    Returns
    -------
    dict[int, int]
        Dicionário mapeando classe -> número de subclusters recomendados.
    """
    labels_arr: NDArray[np.int_] = np.asarray(labels, dtype=int)
    classes_list: list[int] = list(classes)

    contagens: dict[int, int] = {c: int(np.sum(labels_arr == c)) for c in classes_list}
    valid_classes: list[int] = [c for c in classes_list if contagens[c] > 0]
    if not valid_classes:
        return {c: min_nc for c in classes_list}

    # Pesos amortecidos pela raiz quadrada
    pesos: dict[int, float] = {c: float(contagens[c] ** power) for c in valid_classes}
    soma_pesos: float = sum(pesos.values())
    if soma_pesos == 0.0:
        soma_pesos = 1.0

    nc_map: dict[int, int] = {}
    for c in valid_classes:
        frac = pesos[c] / soma_pesos
        aloc = round(total_padroes * frac)
        aloc_clamped = max(min_nc, min(max_nc, aloc))
        # Não pode ter mais clusters do que células disponíveis
        aloc_clamped = min(aloc_clamped, contagens[c])
        nc_map[c] = max(1, aloc_clamped)

    for c in classes_list:
        if c not in nc_map:
            nc_map[c] = min_nc

    return nc_map


class ExtratorPadroesSubcluster:
    """Extrai perfis binários representativos por subcluster biológico (perf180 / protótipos).

    Utiliza uma Estratégia de Clusterização injetada (KMeans Dinâmico, HDBSCAN, etc.)
    ou alocação estratificada por classe no espaço SWeeP e seleciona o consenso binário
    dos k vizinhos mais próximos de cada centróide como protótipo.

    Parameters
    ----------
    W0 : NDArray | sp.spmatrix
        Matriz de expressão binarizada (n_células × n_genes).
    labels : NDArray | Sequence[int]
        Vetor com classes/rótulos biológicos de cada célula.
    classes : Sequence[int] | None, optional
        Lista de classes a processar (padrão: [1, 3, 4, 5, 6, 7, 2]).
    estrategia : EstrategiaClusterizacao | None, optional
        Instância concreta de estratégia de clusterização a aplicar.
    seed : int, default=42
        Semente para reprodutibilidade.
    k : int, default=5
        Número de vizinhos locais para consenso majoritário ao redor do centróide.
    nc : int | dict[int, int] | None, optional
        Número fixo de centróides global ou dicionário de alocação estratificada por classe.

    Attributes
    ----------
    W0 : NDArray[np.float32] | sp.csr_matrix
        Matriz base de expressão.
    labels : NDArray[np.int_]
        Rótulos celulares.
    classes : list[int]
        Classes selecionadas.
    estrategia : EstrategiaClusterizacao
        Estratégia de particionamento ativa.
    nc_map : dict[int, int] | None
        Mapeamento de subclusters por classe se configurado.
    seed : int
        Semente pseudoaleatória.
    k : int
        Contagem k de vizinhos para consenso.
    padroes : NDArray[np.float32] | None
        Matriz de protótipos extraídos.
    meta : list[tuple[int, int]] | None
        Metadados `(classe, idx_global)` de cada padrão.
    """

    def __init__(
        self,
        W0: NDArray[Any] | sp.spmatrix,
        labels: NDArray[Any] | Sequence[int],
        classes: Sequence[int] | None = None,
        estrategia: EstrategiaClusterizacao | None = None,
        seed: int = 42,
        k: int = 5,
        nc: int | dict[int, int] | None = None,
    ) -> None:
        if sp.issparse(W0):
            self.W0: NDArray[np.float32] | sp.csr_matrix = sp.csr_matrix(W0)
        else:
            self.W0 = np.asarray(W0, dtype=np.float32)
        self.labels: NDArray[np.int_] = np.asarray(labels, dtype=int)
        self.classes: list[int] = (
            list(classes) if classes is not None else [1, 3, 4, 5, 6, 7, 2]
        )

        self.nc_map: dict[int, int] | None = None
        if isinstance(nc, dict):
            self.nc_map = {int(k_cls): int(v_nc) for k_cls, v_nc in nc.items()}
            self.nc: int = round(float(np.mean(list(self.nc_map.values()))))
        elif isinstance(nc, (int, np.integer)):
            self.nc = int(nc)
            self.nc_map = {c: self.nc for c in self.classes}
        else:
            self.nc = 10

        # Fallback para retrocompatibilidade
        if estrategia is None:
            if self.nc_map is not None:
                self.estrategia: EstrategiaClusterizacao = EstrategiaKMeansFixo(
                    n_clusters=self.nc, seed=seed
                )
            else:
                self.estrategia = EstrategiaKMeansDinamico(k_range=[10], seed=seed)
        else:
            self.estrategia = estrategia

        self.seed: int = int(seed)
        self.k: int = int(k)
        self.padroes: NDArray[np.float32] | None = None
        self.meta: list[tuple[int, int]] | None = None

    def extrair(
        self, Wswp: NDArray[Any] | Sequence[Sequence[float]]
    ) -> ExtratorPadroesSubcluster:
        """Extrai padrões representativos de subclusters projetados no espaço SWeeP.

        Parameters
        ----------
        Wswp : NDArray | Sequence
            Matriz de projeção de baixa dimensão (SWeeP) das células.

        Returns
        -------
        ExtratorPadroesSubcluster
            A própria instância com os padrões e metadados gerados.
        """
        Wswp_arr: NDArray[np.float32] = np.asarray(Wswp, dtype=np.float32)
        classes_validas: list[int] = [
            c for c in self.classes if (self.labels == c).any()
        ]
        padroes_list: list[NDArray[np.float32]] = []
        meta_list: list[tuple[int, int]] = []

        for cls in classes_validas:
            ids_cls: NDArray[np.intp] = np.where(self.labels == cls)[0]
            Wswp_cls: NDArray[np.float32] = Wswp_arr[ids_cls]

            # Seleciona centróides respeitando nc_map da classe se configurado
            centroides: NDArray[np.float32]
            if self.nc_map is not None and cls in self.nc_map:
                nc_cls = self.nc_map[cls]
                print(
                    f"[ExtratorPadroesSubcluster] Classe {cls}: n={len(ids_cls)} células, "
                    f"alocando nc={nc_cls} subclusters estratificados (k={self.k})..."
                )
                centroides = EstrategiaKMeansFixo(
                    n_clusters=nc_cls, seed=self.seed
                ).clusterizar(Wswp_cls)
            else:
                print(
                    f"[ExtratorPadroesSubcluster] Classe {cls}: n={len(ids_cls)} "
                    f"Aplicando estratégia de clusterização..."
                )
                centroides = self.estrategia.clusterizar(Wswp_cls)

            for centroide in centroides:
                if self.k <= 1:
                    idx_local_res = closervects(Wswp_cls, centroide, k=1)
                    idx_local: int = (
                        int(idx_local_res)
                        if isinstance(idx_local_res, (int, np.integer))
                        else int(idx_local_res[0])
                    )
                    idx_global: int = int(ids_cls[idx_local])
                    row = self.W0[idx_global]
                    row_dense: NDArray[np.float32]
                    if sp.issparse(row):
                        row_dense = (
                            sp.csr_matrix(row).toarray().ravel().astype(np.float32)
                        )
                    else:
                        row_dense = np.asarray(row, dtype=np.float32).ravel()
                    padroes_list.append(row_dense)
                    meta_list.append((cls, idx_global))
                else:
                    k_efetivo = min(self.k, len(Wswp_cls))
                    idxs_res = closervects(Wswp_cls, centroide, k=k_efetivo)
                    idxs: NDArray[np.intp] = np.asarray(idxs_res, dtype=np.intp).ravel()
                    idxs_global: NDArray[np.intp] = ids_cls[idxs]
                    sub = self.W0[idxs_global]
                    sub_dense: NDArray[np.float32]
                    if sp.issparse(sub):
                        sub_dense = sp.csr_matrix(sub).toarray().astype(np.float32)
                    else:
                        sub_dense = np.asarray(sub, dtype=np.float32)
                    # Consenso por voto majoritário dos k vizinhos mais próximos
                    padrao: NDArray[np.float32] = (
                        sub_dense.mean(axis=0) >= 0.5
                    ).astype(np.float32)
                    padroes_list.append(padrao)
                    meta_list.append((cls, int(ids_cls[idxs[0]])))

        self.padroes = np.vstack(padroes_list).astype(np.float32)
        self.meta = meta_list
        print(
            f"[ExtratorPadroesSubcluster] Extração concluída: "
            f"{self.padroes.shape[0]} padrões ({len(classes_validas)} classes, amostragem k={self.k})",
            flush=True,
        )
        return self

    def __repr__(self) -> str:
        """Representação textual do extrator de padrões."""
        padroes = self.padroes.shape if self.padroes is not None else "não extraídos"
        return (
            f"ExtratorPadroesSubcluster(\n"
            f"  W0         = {self.W0.shape}\n"
            f"  labels     = {self.labels.shape}\n"
            f"  classes    = {self.classes}\n"
            f"  estrategia = {self.estrategia.__class__.__name__}\n"
            f"  k          = {self.k}\n"
            f"  seed       = {self.seed}\n"
            f"  padroes    = {padroes}\n"
            f")"
        )
