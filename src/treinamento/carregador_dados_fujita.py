"""Módulo de Carga e Pré-Processamento de Datasets scRNA-seq e Rótulos.

Fornece utilitários para ler matrizes de expressão, listas de genes canônicos
e anotações de metadados celulares com checagem de tipos estrita.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence, Union

import anndata as ad
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import scipy.sparse as sp

PathType = Union[str, os.PathLike[str]]


def carregar_labels(path_ou_array: Union[PathType, NDArray[Any], Sequence[int], pd.Series, None]) -> NDArray[np.int_] | None:
    """Carrega array 1D de rótulos de tipo celular a partir de arquivo (.txt, .csv, .tsv) ou vetor em memória.

    Suporta arquivos com ou sem cabeçalho e diferentes delimitadores.

    Parameters
    ----------
    path_ou_array : str | os.PathLike[str] | NDArray | Sequence[int] | pd.Series | None
        Caminho do arquivo ou vetor de rótulos.

    Returns
    -------
    NDArray[np.int_] | None
        Array unidimensional com rótulos numéricos inteiros, ou None se nulo.
    """
    if path_ou_array is None:
        return None
    if isinstance(path_ou_array, (np.ndarray, list, tuple, pd.Series)):
        return np.asarray(path_ou_array, dtype=int).ravel()

    path_str: str = str(path_ou_array)
    labels: NDArray[np.int_]
    try:
        labels = np.loadtxt(path_str, dtype=int).ravel()
    except Exception:
        try:
            labels = np.loadtxt(path_str, dtype=int, skiprows=1).ravel()
        except Exception:
            df = pd.read_csv(path_str, header=None)
            try:
                labels = df.iloc[:, 0].to_numpy(dtype=int).ravel()
            except ValueError:
                labels = pd.read_csv(path_str).iloc[:, 0].to_numpy(dtype=int).ravel()
    return labels


class CarregadorDados:
    """Carregador unificado de dados scRNA-seq para análise com rede Hopfield.

    Parameters
    ----------
    path_matriz : str | os.PathLike[str]
        Caminho para o arquivo contendo a matriz de expressão (.npy, .h5ad, .csv).
    path_genes : str | os.PathLike[str] | Sequence[str] | pd.DataFrame | None, optional
        Arquivo ou estrutura contendo nomes/identificadores dos genes.
    path_labels : str | os.PathLike[str] | Sequence[int] | None, optional
        Arquivo ou vetor de anotações celulares / classes.
    path_sweep : str | os.PathLike[str] | None, optional
        Caminho para arquivo CSV contendo projeção SWeeP pré-calculada.
    n_genes : int | None, optional
        Número de genes a serem retidos.

    Attributes
    ----------
    path_matriz : str
        Caminho da matriz.
    path_genes : str | Sequence[str] | pd.DataFrame | None
        Referência aos genes.
    path_labels : str | Sequence[int] | None
        Referência aos rótulos.
    path_sweep : str | None
        Caminho SWeeP.
    n_genes : int | None
        Contagem de genes.
    X : NDArray[np.float32] | sp.spmatrix | None
        Matriz de expressão carregada.
    W0 : NDArray[np.float32] | sp.spmatrix | None
        Matriz de trabalho selecionada.
    ids_top : NDArray[np.int_] | None
        Índices das colunas de genes ativas.
    genes : pd.DataFrame | None
        DataFrame com nomes de genes.
    labels : NDArray[np.int_] | None
        Array de rótulos celulares.
    Wswp : NDArray[np.float32] | None
        Matriz de projeções SWeeP.
    """

    def __init__(
        self,
        path_matriz: PathType,
        path_genes: Union[PathType, Sequence[str], pd.DataFrame, None] = None,
        path_labels: Union[PathType, Sequence[int], NDArray[Any], None] = None,
        path_sweep: PathType | None = None,
        n_genes: int | None = None,
    ) -> None:
        self.path_matriz: str = str(path_matriz)
        self.path_genes: Union[PathType, Sequence[str], pd.DataFrame, None] = path_genes
        self.path_labels: Union[PathType, Sequence[int], NDArray[Any], None] = path_labels
        self.path_sweep: str | None = str(path_sweep) if path_sweep is not None else None
        self.n_genes: int | None = n_genes
        self.X: Union[NDArray[np.float32], sp.spmatrix, None] = None
        self.W0: Union[NDArray[np.float32], sp.spmatrix, None] = None
        self.ids_top: NDArray[np.int_] | None = None
        self.genes: pd.DataFrame | None = None
        self.labels: NDArray[np.int_] | None = None
        self.Wswp: NDArray[np.float32] | None = None

    def carregar(self) -> CarregadorDados:
        """Carrega todos os arquivos de entrada e metadados associados.

        Returns
        -------
        CarregadorDados
            A própria instância com os atributos preenchidos.
        """
        self._carregar_matriz()
        self._selecionar_top_genes()
        if self.path_genes is not None:
            self._carregar_genes()
        if self.path_labels is not None:
            self._carregar_labels()
        if self.path_sweep is not None:
            self._carregar_sweep()
        assert self.X is not None
        print(f"[{self.__class__.__name__}] Carregamento concluído: "
              f"{self.X.shape[0]} células, {self.n_genes} genes selecionados")
        return self

    def _carregar_matriz(self) -> None:
        print(f"[{self.__class__.__name__}] Carregando matriz: {self.path_matriz}")
        if self.path_matriz.endswith(".npy"):
            self.X = np.load(self.path_matriz, mmap_mode="r")
        elif self.path_matriz.endswith(".h5ad"):
            adata: ad.AnnData = ad.read_h5ad(self.path_matriz)
            self.X = adata.X
        else:
            self.X = pd.read_csv(self.path_matriz).to_numpy(dtype=np.float32)
        assert self.X is not None
        print(f"[{self.__class__.__name__}] Matriz carregada: {self.X.shape}")

    def _selecionar_top_genes(self) -> None:
        assert self.X is not None
        self.ids_top = np.arange(int(self.X.shape[1]), dtype=np.intp)
        self.W0 = self.X
        self.n_genes = int(self.X.shape[1])
        print(f"[{self.__class__.__name__}] W0 shape: {self.W0.shape} ({self.n_genes} genes)")

    def _carregar_genes(self) -> None:
        print(f"[{self.__class__.__name__}] Carregando genes: {self.path_genes}")
        if isinstance(self.path_genes, pd.DataFrame):
            self.genes = self.path_genes
        elif isinstance(self.path_genes, (list, np.ndarray, tuple)):
            self.genes = pd.DataFrame({"gene": list(self.path_genes)})
        elif self.path_genes is not None:
            self.genes = pd.read_csv(str(self.path_genes))
        if self.genes is not None:
            print(f"[{self.__class__.__name__}] {len(self.genes)} genes carregados")

    def _carregar_labels(self) -> None:
        print(f"[{self.__class__.__name__}] Carregando rótulos: {self.path_labels}")
        self.labels = carregar_labels(self.path_labels)
        if self.labels is not None:
            tipos = np.unique(self.labels)
            print(f"[{self.__class__.__name__}] Rótulos shape: {self.labels.shape}, tipos: {tipos}")

    def _carregar_sweep(self) -> None:
        if self.path_sweep is not None and os.path.exists(self.path_sweep):
            print(f"[{self.__class__.__name__}] Carregando SWeeP pré-computado: {self.path_sweep}")
            self.Wswp = pd.read_csv(self.path_sweep).to_numpy(dtype=np.float32)
            print(f"[{self.__class__.__name__}] Wswp shape: {self.Wswp.shape}")

    def __repr__(self) -> str:
        """Representação textual do carregador de dados."""
        x_shape = self.X.shape if self.X is not None else "não carregado"
        w0_shape = self.W0.shape if self.W0 is not None else "não gerado"
        wswp = self.Wswp.shape if self.Wswp is not None else "não carregado"
        labels = self.labels.shape if self.labels is not None else "não carregados"
        return (
            f"{self.__class__.__name__}(\n"
            f"  path_matriz  = {self.path_matriz}\n"
            f"  path_genes   = {self.path_genes}\n"
            f"  path_labels  = {self.path_labels}\n"
            f"  path_sweep   = {self.path_sweep}\n"
            f"  X            = {x_shape}\n"
            f"  W0           = {w0_shape}\n"
            f"  labels       = {labels}\n"
            f"  Wswp         = {wswp}\n"
            f")"
        )


# Alias para retrocompatibilidade
CarregadorDadosFujita = CarregadorDados
