"""Módulo de Filtro e Geração de Subconjuntos de Treinamento e Teste.

Filtra matrizes alinhadas (.txt, .h5ad, .npy) preservando apenas os genes
selecionados (ex: Top 5000) e injeta sentinelas quando aplicável.
"""

from __future__ import annotations

import gc
import os

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from numpy.typing import NDArray

PathType = str | os.PathLike[str]


class GeradorConjuntoTreinamento:
    """Filtra arquivos alinhados para reter apenas os genes do conjunto de treino/teste selecionado.

    Usa Polars streaming (sink_csv) ou manipulação direta de AnnData e NumPy
    para suportar arquivos massivos de forma OOM-Safe.

    Parameters
    ----------
    path_top_genes_csv : str | os.PathLike[str]
        Caminho para o arquivo CSV com a lista de genes prioritários (coluna 'gene').
    out_dir : str | os.PathLike[str]
        Diretório base de saída.
    chunk : int, default=3000
        Tamanho de lote para processamento.

    Attributes
    ----------
    path_top_genes_csv : str
        Caminho dos genes prioritários.
    out_dir : str
        Diretório de saída.
    chunk : int
        Tamanho de lote.
    genes_selecionados : Set[str]
        Conjunto de identificadores de genes retidos.
    path_saida : str | None
        Caminho do último arquivo gerado.
    """

    def __init__(
        self, path_top_genes_csv: PathType, out_dir: PathType, chunk: int = 3000
    ) -> None:
        self.path_top_genes_csv: str = str(path_top_genes_csv)
        self.out_dir: str = str(out_dir)
        self.chunk: int = int(chunk)
        self.genes_selecionados: set[str] = set()
        self.path_saida: str | None = None
        self._carregar_genes()

    def _carregar_genes(self) -> None:
        df: pl.DataFrame = pl.read_csv(self.path_top_genes_csv)
        col_nome: str = "gene" if "gene" in df.columns else df.columns[0]
        self.genes_selecionados = set(df[col_nome].to_list())
        print(
            f"[GeradorConjuntoTreinamento] {len(self.genes_selecionados)} genes carregados de: {self.path_top_genes_csv}"
        )

    def gerar(self, path_txt: PathType) -> GeradorConjuntoTreinamento:
        """Filtra arquivo CSV/TXT delimitado por vírgula em modo streaming via Polars.

        Parameters
        ----------
        path_txt : str | os.PathLike[str]
            Caminho do arquivo de texto alinhado completo.

        Returns
        -------
        GeradorConjuntoTreinamento
            A própria instância.
        """
        path_txt_str: str = str(path_txt)
        with open(path_txt_str, encoding="utf-8") as f:
            todos_genes: list[str] = f.readline().strip().split(",")

        genes_filtrados: list[str] = [
            g for g in todos_genes if g in self.genes_selecionados
        ]
        n: int = len(genes_filtrados)
        nome: str = os.path.splitext(os.path.basename(path_txt_str))[0]
        path_saida: str = os.path.join(self.out_dir, f"{nome}_top{n}.txt")
        os.makedirs(self.out_dir, exist_ok=True)

        if os.path.exists(path_saida):
            print(
                f"[GeradorConjuntoTreinamento] Arquivo já existe, pulando: {path_saida}"
            )
            self.path_saida = path_saida
            return self

        path_tmp: str = path_saida + ".tmp"
        if os.path.exists(path_tmp):
            os.remove(path_tmp)

        print(f"\n[GeradorConjuntoTreinamento] Processando: {path_txt_str}")
        print(
            f"  Genes encontrados no arquivo: {n} de {len(self.genes_selecionados)} selecionados"
        )
        print("  Escrevendo via Polars streaming...")

        (
            pl.scan_csv(path_txt_str, infer_schema_length=1)
            .select(genes_filtrados)
            .sink_csv(path_tmp)
        )

        os.rename(path_tmp, path_saida)
        print(f"[GeradorConjuntoTreinamento] Salvo: {path_saida}  ({n} genes)")
        self.path_saida = path_saida
        return self

    def gerar_de_h5ad(
        self,
        path_h5ad: PathType,
        is_mathys: bool = False,
        fill_value: float = 0.5,
        exportar_npy: bool = True,
        exportar_h5ad: bool = True,
    ) -> GeradorConjuntoTreinamento:
        """Filtra AnnData .h5ad para os top genes e salva diretamente em .npy e .h5ad de forma OOM-Safe.

        Parameters
        ----------
        path_h5ad : str | os.PathLike[str]
            Caminho do arquivo .h5ad de entrada.
        is_mathys : bool, default=False
            Se True, aplica injeção de valor sentinela nos genes ausentes anotados.
        fill_value : float, default=0.5
            Valor sentinela a ser injetado em colunas ausentes.
        exportar_npy : bool, default=True
            Se True, gera arquivo binário NumPy (.npy).
        exportar_h5ad : bool, default=True
            Se True, gera arquivo compactado AnnData (.h5ad).

        Returns
        -------
        GeradorConjuntoTreinamento
            A própria instância.
        """
        path_h5ad_str: str = str(path_h5ad)
        adata: ad.AnnData = ad.read_h5ad(path_h5ad_str)
        todos_genes: list[str] = list(adata.var_names)
        genes_filtrados: list[str] = [
            g for g in todos_genes if g in self.genes_selecionados
        ]
        n: int = len(genes_filtrados)
        nome: str = os.path.splitext(os.path.basename(path_h5ad_str))[0]

        path_npy: str = os.path.join(self.out_dir, f"{nome}_top{n}.npy")
        path_saida_h5ad: str = os.path.join(self.out_dir, f"{nome}_top{n}.h5ad")
        os.makedirs(self.out_dir, exist_ok=True)

        if os.path.exists(path_npy) and (
            not exportar_h5ad or os.path.exists(path_saida_h5ad)
        ):
            print(
                f"[GeradorConjuntoTreinamento] Arquivos já existem, pulando: {nome}_top{n}"
            )
            self.path_saida = path_npy
            return self

        print(f"\n[GeradorConjuntoTreinamento] Processando .h5ad: {path_h5ad_str}")
        print(
            f"  Genes selecionados encontrados: {n} de {len(self.genes_selecionados)}"
        )

        gene_to_idx: dict[str, int] = {g: i for i, g in enumerate(todos_genes)}
        col_indices: list[int] = [gene_to_idx[g] for g in genes_filtrados]

        X_sub = adata.X[:, col_indices]
        X_dense: NDArray[np.float32]
        if sp.issparse(X_sub):
            X_dense = sp.csr_matrix(X_sub).toarray().astype(np.float32)
        else:
            X_dense = np.asarray(X_sub, dtype=np.float32)

        # Injeção de sentinela se for Mathys e houver anotação de presença
        if is_mathys and "presente_no_dataset" in adata.var:
            presente_sub: NDArray[np.bool_] = np.asarray(
                adata.var["presente_no_dataset"].to_numpy()
            )[col_indices].astype(bool)
            ausentes_mask: NDArray[np.bool_] = ~presente_sub
            n_ausentes: int = int(np.sum(ausentes_mask))
            if n_ausentes > 0:
                print(
                    f"  Injetando sentinela {fill_value} em {n_ausentes} genes ausentes no Mathys..."
                )
                X_dense[:, ausentes_mask] = fill_value

        if exportar_npy:
            np.save(path_npy, X_dense)
            print(
                f"[GeradorConjuntoTreinamento] Salvo .npy: {path_npy} ({X_dense.shape})"
            )

        if exportar_h5ad:
            var_df: pd.DataFrame = pd.DataFrame(
                index=pd.Index(genes_filtrados, name="ensembl_id")
            )
            assert isinstance(adata.obs, pd.DataFrame)
            obs_df: pd.DataFrame = adata.obs.copy()
            adata_sub: ad.AnnData = ad.AnnData(
                X=sp.csr_matrix(X_dense), obs=obs_df, var=var_df
            )
            adata_sub.write_h5ad(path_saida_h5ad, compression="gzip")
            print(f"[GeradorConjuntoTreinamento] Salvo .h5ad: {path_saida_h5ad}")

        del adata, X_sub, X_dense
        gc.collect()

        self.path_saida = path_npy
        return self

    def __repr__(self) -> str:
        """Representação textual do gerador de conjunto de treinamento."""
        n = len(self.genes_selecionados) if self.genes_selecionados else "não carregado"
        return (
            f"GeradorConjuntoTreinamento(\n"
            f"  path_top_genes_csv = {self.path_top_genes_csv}\n"
            f"  out_dir            = {self.out_dir}\n"
            f"  genes_selecionados = {n}\n"
            f")"
        )
