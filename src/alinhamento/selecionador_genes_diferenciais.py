"""Módulo de Seleção de Genes Diferenciais por Qui-Quadrado (Chi2).

Mede o Ganho de Informação Biológica entre a presença/ausência binária do gene
e os tipos celulares anotados, isolando marcadores informativos e descartando
genes constitutivos (housekeeping) sem poder de discriminação celular.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Sequence, Union

import anndata as ad
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import polars as pl
import scipy.sparse as sp
from sklearn.feature_selection import chi2

PathType = Union[str, os.PathLike[str]]
LabelsType = Union[PathType, Sequence[int], NDArray[np.int_]]


class SelecionadorGenesDiferenciais:
    """Calcula os N genes com maior valor de Chi-Square (qui-quadrado) em relação aos rótulos celulares.

    Parameters
    ----------
    path_input : str | os.PathLike[str]
        Caminho da matriz binária de entrada (AnnData .h5ad, CSV ou TXT).
    path_labels : str | os.PathLike[str] | Sequence[int] | NDArray
        Vetor ou arquivo contendo os rótulos de tipo celular (ex: clo_ref).
    n : int, default=5000
        Número de genes marcadores a selecionar.

    Attributes
    ----------
    path_input : str
        Caminho do arquivo de entrada.
    path_labels : LabelsType
        Rótulos das células.
    n : int
        Quantidade de genes alvo.
    df_resultado : pl.DataFrame | None
        Tabela com genes e pontuações Chi2 ordenadas.
    """

    def __init__(
        self,
        path_input: PathType,
        path_labels: LabelsType,
        n: int = 5000,
    ) -> None:
        self.path_input: str = str(path_input)
        self.path_labels: LabelsType = path_labels
        self.n: int = int(n)
        self.df_resultado: pl.DataFrame | None = None

    def calcular(
        self,
        path_h5ad: str = "",
        classes_validas: Sequence[int] | None = None,
        out_csv: PathType | None = None,
    ) -> SelecionadorGenesDiferenciais:
        """Executa o cálculo do teste qui-quadrado gene a gene contra os rótulos.

        Parameters
        ----------
        path_h5ad : str, default=""
            Caminho alternativo AnnData se path_input for texto.
        classes_validas : Sequence[int] | None, optional
            Subconjunto de classes para inclusão no cálculo (ex: descarta classe 0).
        out_csv : str | os.PathLike[str] | None, optional
            Caminho para salvar o CSV com a pontuação imediatamente.

        Returns
        -------
        SelecionadorGenesDiferenciais
            A própria instância.
        """
        print(f"[SelecionadorGenesDiferenciais] Iniciando seleção dos Top {self.n} genes marcadores...")

        # 1. Carregamento dos rótulos
        clo: NDArray[np.int_]
        if isinstance(self.path_labels, (str, os.PathLike)):
            clo = np.loadtxt(str(self.path_labels), dtype=int)
        elif isinstance(self.path_labels, np.ndarray):
            clo = self.path_labels.astype(int)
        else:
            clo = np.array(list(self.path_labels), dtype=int)

        mask_valid: NDArray[np.bool_]
        if classes_validas is not None:
            mask_valid = np.isin(clo, list(classes_validas))
            print(f"  Filtrando classes válidas {list(classes_validas)}: {mask_valid.sum()} de {len(clo)} células.")
        else:
            mask_valid = np.ones(len(clo), dtype=bool)

        # 2. Carregamento da Matriz de Expressão Binária
        gene_names: list[str]
        X: NDArray[np.float32]

        if os.path.exists(path_h5ad) or self.path_input.endswith(".h5ad"):
            target_h5ad: str = path_h5ad if os.path.exists(path_h5ad) else self.path_input
            print(f"  Utilizando matriz AnnData para carregamento otimizado: {target_h5ad}")
            adata: ad.AnnData = ad.read_h5ad(target_h5ad)
            gene_names = list(adata.var_names)
            mat_slice = adata.X[mask_valid]
            if sp.issparse(mat_slice):
                X = sp.csr_matrix(mat_slice).toarray().astype(np.float32)
            else:
                X = np.asarray(mat_slice, dtype=np.float32)
            del adata
            gc.collect()
        else:
            # Leitura de CSV/TXT binarizado
            df: pd.DataFrame = pd.read_csv(self.path_input, dtype=np.float32)
            gene_names = list(df.columns)
            X = np.asarray(df.values[mask_valid], dtype=np.float32)
            del df
            gc.collect()

        y: NDArray[np.int_] = clo[mask_valid]

        print(f"  Executando Chi-Square em {X.shape[1]} genes para {X.shape[0]} células...")
        scores, _ = chi2(X, y)
        scores_arr: NDArray[np.float64] = np.asarray(np.nan_to_num(scores, nan=0.0), dtype=np.float64)

        n_real: int = min(self.n, len(gene_names))
        idx_top: NDArray[np.intp] = np.argsort(scores_arr)[-n_real:][::-1]

        top_genes: list[str] = [gene_names[i] for i in idx_top]
        top_scores: list[float] = [float(scores_arr[i]) for i in idx_top]

        self.df_resultado = pl.DataFrame({
            "gene": top_genes,
            "chi2_score": top_scores,
        })

        print(f"[SelecionadorGenesDiferenciais] Concluído. Top {n_real} genes discriminativos selecionados por Chi2.")
        print(f"  Maior escore Chi2: {top_scores[0]:.2f} (Gene: {top_genes[0]})")
        print(f"  Menor escore Chi2 do Top: {top_scores[-1]:.2f} (Gene: {top_genes[-1]})")

        if out_csv:
            self.salvar(out_csv)

        return self

    def salvar(self, out_csv: PathType) -> SelecionadorGenesDiferenciais:
        """Persiste os resultados em formato CSV.

        Parameters
        ----------
        out_csv : str | os.PathLike[str]
            Caminho do CSV de destino.

        Returns
        -------
        SelecionadorGenesDiferenciais
            A própria instância.
        """
        if self.df_resultado is None:
            raise RuntimeError("Execute .calcular() antes de salvar.")
        out_csv_str: str = str(out_csv)
        os.makedirs(os.path.dirname(os.path.abspath(out_csv_str)), exist_ok=True)
        self.df_resultado.write_csv(out_csv_str)
        print(f"[SelecionadorGenesDiferenciais] Salvo em: {out_csv_str}")
        return self

    def filtrar_matriz(self, in_csv_or_npy: PathType, out_csv_or_npy: PathType) -> SelecionadorGenesDiferenciais:
        """Salva nova matriz contendo apenas as colunas dos genes selecionados por Chi2.

        Parameters
        ----------
        in_csv_or_npy : str | os.PathLike[str]
            Matriz de entrada (.csv, .txt ou .npy).
        out_csv_or_npy : str | os.PathLike[str]
            Matriz filtrada de saída (.csv ou .npy).

        Returns
        -------
        SelecionadorGenesDiferenciais
            A própria instância.
        """
        if self.df_resultado is None:
            raise RuntimeError("Execute .calcular() antes de filtrar.")

        lista_genes: list[str] = self.df_resultado["gene"].to_list()
        in_str: str = str(in_csv_or_npy)
        out_str: str = str(out_csv_or_npy)
        os.makedirs(os.path.dirname(os.path.abspath(out_str)), exist_ok=True)

        if in_str.endswith(".npy"):
            arr: NDArray[Any] = np.load(in_str)
            np.save(out_str, arr)
            print(f"[SelecionadorGenesDiferenciais] Matriz salva: {out_str}")
        else:
            with open(in_str, encoding="utf-8") as fh:
                header: list[str] = fh.readline().strip("\n").strip("\r").split(",")

            coluna_celulas: str = header[0]
            colunas_validas: list[str] = [coluna_celulas] + [c for c in lista_genes if c in header]

            if out_str.endswith(".npy"):
                df_filtered: pl.DataFrame = pl.scan_csv(in_str).select(colunas_validas).collect()
                arr_f32: NDArray[np.float32]
                if df_filtered.columns[0] == coluna_celulas and not df_filtered.dtypes[0].is_numeric():
                    arr_f32 = df_filtered.select(colunas_validas[1:]).to_numpy().astype(np.float32)
                else:
                    arr_f32 = df_filtered.to_numpy().astype(np.float32)
                np.save(out_str, arr_f32)
                print(f"[SelecionadorGenesDiferenciais] Matriz filtrada salva em binário (lazy): {out_str} ({arr_f32.shape})")
            else:
                pl.scan_csv(in_str).select(colunas_validas).sink_csv(out_str)
                print(f"[SelecionadorGenesDiferenciais] Matriz filtrada salva em: {out_str}")
        return self

    def __repr__(self) -> str:
        """Representação textual do selecionador de genes diferenciais."""
        n: str = str(len(self.df_resultado)) if self.df_resultado is not None else "não calculado"
        return f"SelecionadorGenesDiferenciais(input={self.path_input}, n={self.n}, resultado={n} genes)"
