"""Módulo de Binarização de Matrizes de Expressão scRNA-Seq.

Converte dados contínuos de contagem/expressão em estados discretos {0, 1},
preservando padrões esparsos ou densos no formato AnnData (.h5ad).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import anndata as ad
import numpy as np
import scipy.sparse as sp


PathType = Union[str, os.PathLike[str]]


class Binarizador:
    """Responsável pelas etapas de pré-processamento e binarização da matriz de expressão.

    Parameters
    ----------
    path_h5ad : str | os.PathLike[str]
        Caminho para o arquivo .h5ad de entrada contendo a matriz de expressão gênica.
    out_dir : str | os.PathLike[str] | None, optional
        Diretório geral de saída do pipeline. Se None, utiliza o diretório 'outputs'.
    out_dir_binarizada : str | os.PathLike[str] | None, optional
        Pasta específica onde o .h5ad binarizado será salvo. Se None, usa out_dir.

    Attributes
    ----------
    path_h5ad : str
        Caminho normalizado do arquivo de entrada.
    out_dir : str
        Diretório base de saída.
    out_dir_binarizada : str
        Diretório específico de destino dos arquivos binarizados.
    path_binarizada : str | None
        Caminho completo do arquivo binarizado gerado, preenchido após .binarizar().
    """

    def __init__(
        self,
        path_h5ad: PathType,
        out_dir: PathType | None = None,
        out_dir_binarizada: PathType | None = None,
    ) -> None:
        self.path_h5ad: str = str(path_h5ad)
        self.out_dir: str = str(out_dir) if out_dir is not None else os.path.join(os.getcwd(), "outputs")
        self.out_dir_binarizada: str = str(out_dir_binarizada) if out_dir_binarizada is not None else self.out_dir
        self.path_binarizada: str | None = None

    def binarizar(self, nome_arquivo: str = "matrizBinarizadaM.h5ad") -> Binarizador:
        """Binariza a matriz de expressão e salva como .h5ad no diretório de saída.

        Valores > 0 viram 1, zeros permanecem 0 (dtype int8).

        Parameters
        ----------
        nome_arquivo : str, default="matrizBinarizadaM.h5ad"
            Nome do arquivo .h5ad resultante.

        Returns
        -------
        Binarizador
            A própria instância para encadeamento fluente de chamadas.
        """
        nome_entrada: str = os.path.splitext(os.path.basename(self.path_h5ad))[0]
        pasta_saida: str = os.path.join(self.out_dir_binarizada, nome_entrada)
        self.path_binarizada = os.path.join(pasta_saida, nome_arquivo)

        if os.path.exists(self.path_binarizada):
            print(f"[Binarizador] Arquivo já existe, pulando: {self.path_binarizada}")
            return self

        print("[Binarizador] Carregando arquivo h5ad...")
        adata: ad.AnnData = ad.read_h5ad(self.path_h5ad)
        print(f"[Binarizador] Shape da matriz: {adata.shape}")

        print("[Binarizador] Binarizando a matriz...")
        if sp.issparse(adata.X):
            X_csr: sp.csr_matrix = sp.csr_matrix(adata.X)
            X_csr.data = np.where(X_csr.data > 0, 1, 0)
            adata.X = X_csr.astype(np.int8)
        else:
            assert isinstance(adata.X, np.ndarray)
            adata.X = np.where(adata.X > 0, 1, 0).astype(np.int8)

        os.makedirs(pasta_saida, exist_ok=True)

        print("[Binarizador] Salvando arquivo h5ad binarizado...")
        adata.write_h5ad(self.path_binarizada)
        print(f"[Binarizador] Arquivo salvo: {self.path_binarizada}")
        return self

    def carregar_binarizada(self) -> ad.AnnData:
        """Carrega a matriz binarizada já gerada como AnnData.

        Returns
        -------
        ad.AnnData
            Objeto AnnData carregado da matriz binarizada.

        Raises
        ------
        RuntimeError
            Se o método .binarizar() ainda não tiver sido executado com sucesso.
        """
        if self.path_binarizada is None:
            raise RuntimeError("Execute .binarizar() antes de carregar.")
        print(f"[Binarizador] Carregando: {self.path_binarizada}")
        return ad.read_h5ad(self.path_binarizada)

    def __repr__(self) -> str:
        """Representação textual do objeto Binarizador."""
        binarizada: str = self.path_binarizada or "ainda não gerada"
        return (
            f"Binarizador(\n"
            f"  path_h5ad          = {self.path_h5ad}\n"
            f"  out_dir            = {self.out_dir}\n"
            f"  out_dir_binarizada = {self.out_dir_binarizada}\n"
            f"  path_binarizada    = {binarizada}\n"
            f")"
        )
