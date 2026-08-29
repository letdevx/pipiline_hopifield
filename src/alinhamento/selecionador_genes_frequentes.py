"""Módulo de Seleção de Genes Altamente Frequentes (scRNA-Seq).

Identifica os N genes com maior índice de expressão/detecção (frequência)
a partir de matrizes binarizadas no formato AnnData (.h5ad) ou CSV em streaming.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import polars as pl

PathType = Union[str, os.PathLike[str]]


class SelecionadorGenesFrequentes:
    """Calcula os N genes mais frequentes a partir de um arquivo .h5ad ou CSV/TXT binarizado.

    Frequência = soma da coluna (equivalente a células com valor > 0 em dados binarizados).

    Parameters
    ----------
    path_txt : str | os.PathLike[str] | None, optional
        Caminho para o arquivo CSV/TXT denso alinhado.
    path_h5ad : str | os.PathLike[str] | None, optional
        Caminho para a matriz de expressão em formato AnnData (.h5ad).
    n : int, default=5000
        Número de genes mais frequentes a selecionar.

    Attributes
    ----------
    path_txt : str | None
        Caminho do arquivo texto de entrada.
    path_h5ad : str | None
        Caminho do arquivo .h5ad.
    n : int
        Quantidade de genes alvo.
    df_resultado : pl.DataFrame | None
        DataFrame com as colunas ['gene', 'frequencia'] dos top-N genes ordenados.
    """

    def __init__(
        self,
        path_txt: PathType | None = None,
        path_h5ad: PathType | None = None,
        n: int = 5000,
    ) -> None:
        self.path_txt: str | None = str(path_txt) if path_txt is not None else None
        if path_h5ad is not None:
            self.path_h5ad: str | None = str(path_h5ad)
        elif path_txt is not None and str(path_txt).endswith((".txt", ".csv")):
            self.path_h5ad = os.path.splitext(str(path_txt))[0] + ".h5ad"
        else:
            self.path_h5ad = None
        self.n: int = int(n)
        self.df_resultado: pl.DataFrame | None = None

    def calcular(self, out_csv: PathType | None = None) -> SelecionadorGenesFrequentes:
        """Calcula o vetor de frequências por gene e extrai o ranking top-N.

        Parameters
        ----------
        out_csv : str | os.PathLike[str] | None, optional
            Caminho do CSV de cache para reaproveitamento rápido se existir.

        Returns
        -------
        SelecionadorGenesFrequentes
            A própria instância com `df_resultado` preenchido.
        """
        if out_csv is not None and os.path.exists(str(out_csv)):
            print(f"[SelecionadorGenesFrequentes] Arquivo já existe, pulando: {out_csv}")
            self.df_resultado = pl.read_csv(str(out_csv))
            return self

        # 1. Tenta carregar diretamente via .h5ad (rápido e OOM-Safe)
        path_h5ad_alvo: str | None = (
            self.path_h5ad if (self.path_h5ad and os.path.exists(self.path_h5ad)) else None
        )
        if path_h5ad_alvo is None and self.path_txt:
            candidato_h5ad: str = os.path.splitext(self.path_txt)[0] + ".h5ad"
            if os.path.exists(candidato_h5ad):
                path_h5ad_alvo = candidato_h5ad

        gene_names: list[str]
        somas: NDArray[np.int64]
        total_genes: int

        if path_h5ad_alvo and os.path.exists(path_h5ad_alvo):
            print(f"[SelecionadorGenesFrequentes] Arquivo .h5ad detectado: {path_h5ad_alvo}")
            print("  Calculando frequências diretamente via AnnData esparso (OOM-Safe & ultrarrápido)...")
            import anndata as ad
            import scipy.sparse as sp

            adata: ad.AnnData = ad.read_h5ad(path_h5ad_alvo)
            gene_names = list(adata.var_names)
            total_genes = len(gene_names)

            if sp.issparse(adata.X):
                X_csr: sp.csr_matrix = sp.csr_matrix(adata.X)
                somas = np.asarray((X_csr > 0).sum(axis=0)).ravel().astype(np.int64)
            else:
                assert isinstance(adata.X, np.ndarray)
                somas = np.asarray((adata.X > 0).sum(axis=0)).ravel().astype(np.int64)

            del adata
            gc.collect()
        else:
            path_txt_alvo: str | None = self.path_txt or self.path_h5ad
            if path_txt_alvo is None or not os.path.exists(path_txt_alvo):
                raise FileNotFoundError(f"[SelecionadorGenesFrequentes] Nenhum arquivo de entrada encontrado: {path_txt_alvo}")

            print(f"[SelecionadorGenesFrequentes] Lendo arquivo texto: {path_txt_alvo}")
            with open(path_txt_alvo, encoding="utf-8") as fh:
                gene_names = fh.readline().strip().split(",")
            total_genes = len(gene_names)

            print(f"  Calculando frequências para {total_genes} genes (streaming por chunks em RAM otimizada)...")
            somas = np.zeros(total_genes, dtype=np.int64)

            n_celulas: int = 0
            for chunk in pd.read_csv(path_txt_alvo, chunksize=250, dtype=np.float32, header=0, engine="c"):
                somas += (chunk.values > 0).sum(axis=0).astype(np.int64)
                n_celulas += len(chunk)
                del chunk
                gc.collect()
                if n_celulas % 5000 < 250:
                    print(f"  {n_celulas} células processadas...")

        df_frequencias: pl.DataFrame = pl.DataFrame({"gene": gene_names, "frequencia": somas})

        n_real: int = min(self.n, total_genes)
        self.df_resultado = (
            df_frequencias
            .sort("frequencia", descending=True)
            .head(n_real)
        )

        print(f"[SelecionadorGenesFrequentes] Concluído. Top {n_real} genes selecionados.")
        return self

    def salvar(self, out_csv: PathType) -> SelecionadorGenesFrequentes:
        """Salva a tabela de genes frequentes selecionados em disco.

        Parameters
        ----------
        out_csv : str | os.PathLike[str]
            Caminho do arquivo CSV de destino.

        Returns
        -------
        SelecionadorGenesFrequentes
            A própria instância.

        Raises
        ------
        RuntimeError
            Se o cálculo ainda não tiver sido executado.
        """
        if self.df_resultado is None:
            raise RuntimeError("Execute .calcular() antes de salvar.")
        out_csv_str: str = str(out_csv)
        os.makedirs(os.path.dirname(os.path.abspath(out_csv_str)), exist_ok=True)
        self.df_resultado.write_csv(out_csv_str)
        print(f"[SelecionadorGenesFrequentes] Salvo em: {out_csv_str}")
        return self

    def filtrar_matriz(self, in_csv: PathType, out_csv: PathType) -> SelecionadorGenesFrequentes:
        """Salva nova matriz contendo apenas as colunas dos top N genes selecionados.

        Parameters
        ----------
        in_csv : str | os.PathLike[str]
            Arquivo CSV/TXT de entrada completo.
        out_csv : str | os.PathLike[str]
            Arquivo de destino (.csv ou .npy).

        Returns
        -------
        SelecionadorGenesFrequentes
            A própria instância.
        """
        if self.df_resultado is None:
            raise RuntimeError("Execute .calcular() antes de filtrar.")

        in_csv_str: str = str(in_csv)
        out_csv_str: str = str(out_csv)

        with open(in_csv_str, encoding="utf-8") as fh:
            header: list[str] = fh.readline().strip("\n").strip("\r").split(",")

        coluna_celulas: str = header[0]
        lista_genes: list[str] = self.df_resultado["gene"].to_list()
        colunas_validas: list[str] = [coluna_celulas] + [c for c in lista_genes if c in header]

        os.makedirs(os.path.dirname(os.path.abspath(out_csv_str)), exist_ok=True)
        if os.path.exists(out_csv_str):
            os.remove(out_csv_str)

        if out_csv_str.endswith(".npy"):
            # Para exportação binária rápida (.npy), lê via polars/pandas e salva numpy float32
            df_filtered: pl.DataFrame = pl.read_csv(in_csv_str, columns=colunas_validas)
            # Remove a primeira coluna se for identificador não-numérico ou mantém se já for genes
            arr: NDArray[np.float32]
            if df_filtered.columns[0] == coluna_celulas and not df_filtered.dtypes[0].is_numeric():
                arr = df_filtered.select(colunas_validas[1:]).to_numpy().astype(np.float32)
            else:
                arr = df_filtered.to_numpy().astype(np.float32)
            np.save(out_csv_str, arr)
            print(f"[SelecionadorGenesFrequentes] Matriz filtrada salva em binário: {out_csv_str} ({arr.shape})")
        else:
            pl.scan_csv(in_csv_str).select(colunas_validas).sink_csv(out_csv_str)
            print(f"[SelecionadorGenesFrequentes] Matriz filtrada salva em: {out_csv_str} ({len(colunas_validas)} colunas)")
        return self

    def __repr__(self) -> str:
        """Representação textual do selecionador de genes frequentes."""
        n: str = str(len(self.df_resultado)) if self.df_resultado is not None else "não calculado"
        return (
            f"SelecionadorGenesFrequentes(\n"
            f"  path_txt     = {self.path_txt}\n"
            f"  n            = {self.n}\n"
            f"  df_resultado = {n} genes\n"
            f")"
        )
