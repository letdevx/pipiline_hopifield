"""Módulo de alinhamento esparso e OOM-Safe para matrizes scRNA-seq."""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Mapping, Sequence, Set, Union

import anndata as ad
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import polars as pl
import scipy.sparse as sp

PathType = Union[str, os.PathLike[str]]


class AlinhadorEsparso:
    """Alinha dois arquivos .h5ad binarizados ao mesmo espaço gênico de referência (Fujita) de forma 100% esparsa.

    Projetado para ambientes com restrição de memória RAM (<= 16GB). Evita a materialização densa
    de colunas ausentes no espaço de genoma completo (~36k genes), mantendo a representação
    estritamente em matrizes esparsas CSR comprimidas e permitindo a injeção do valor sentinela (0.5)
    sob demanda ou nos subconjuntos filtrados (Top 5k / 11k).

    Parameters
    ----------
    path_binarizada_m : str | os.PathLike[str]
        Caminho do .h5ad binarizado do alvo (Mathys).
    path_binarizada_f : str | os.PathLike[str]
        Caminho do .h5ad binarizado da referência (Fujita).
    out_dir : str | os.PathLike[str]
        Diretório base de saída.
    map_f : Mapping[str, str]
        Dicionário {Gene Symbol: Ensembl ID} da referência.
    map_m : Mapping[str, str]
        Dicionário {Gene Symbol: Ensembl ID} do alvo.
    gene_alvo_idx : Mapping[str, int]
        Índices canônicos dos genes.
    genes_ordenados : Sequence[str]
        Lista ordenada de Ensembl IDs do espaço gênico unificado.

    Attributes
    ----------
    path_binarizada_m : str
        Caminho do arquivo binarizado alvo.
    path_binarizada_f : str
        Caminho do arquivo binarizado referência.
    out_dir : str
        Diretório de saída.
    map_f : Mapping[str, str]
        Mapa de features da referência.
    map_m : Mapping[str, str]
        Mapa de features do alvo.
    gene_alvo_idx : Mapping[str, int]
        Índices canônicos.
    genes_ordenados : list[str]
        Lista canônica de genes.
    path_f_alinhado : str
        Caminho do arquivo alinhado de referência.
    path_m_alinhado : str
        Caminho do arquivo alinhado do alvo.
    """

    def __init__(
        self,
        path_binarizada_m: PathType,
        path_binarizada_f: PathType,
        out_dir: PathType,
        map_f: Mapping[str, str],
        map_m: Mapping[str, str],
        gene_alvo_idx: Mapping[str, int],
        genes_ordenados: Sequence[str],
    ) -> None:
        self.path_binarizada_m: str = str(path_binarizada_m)
        self.path_binarizada_f: str = str(path_binarizada_f)
        self.out_dir: str = str(out_dir)
        self.map_f: Mapping[str, str] = map_f
        self.map_m: Mapping[str, str] = map_m
        self.gene_alvo_idx: Mapping[str, int] = gene_alvo_idx
        self.genes_ordenados: list[str] = list(genes_ordenados)

        nome_f: str = "adataF_binarizado_alinhado"
        pasta_f: str = os.path.join(self.out_dir, nome_f)
        self.path_f_alinhado: str = os.path.join(pasta_f, f"{nome_f}.h5ad")

        nome_m: str = "adataM_binarizado_alinhado"
        pasta_m: str = os.path.join(self.out_dir, nome_m)
        self.path_m_alinhado: str = os.path.join(pasta_m, f"{nome_m}.h5ad")

    def _projetar_esparso(
        self,
        adata: ad.AnnData,
        ensembl_map: Mapping[str, str],
        dataset_name: str = "Dataset",
    ) -> ad.AnnData:
        """Projeta matriz AnnData para o espaço canônico de genes_ordenados usando projeção CSR."""
        n_celulas: int = adata.n_obs
        n_genes_alvo: int = len(self.genes_ordenados)

        old_idx: list[int] = []
        new_idx: list[int] = []
        present_new_cols: Set[int] = set()

        for old_i, gene_name in enumerate(adata.var_names):
            eid: str = ensembl_map.get(gene_name, gene_name)
            if eid in self.gene_alvo_idx:
                new_col: int = self.gene_alvo_idx[eid]
                old_idx.append(old_i)
                new_idx.append(new_col)
                present_new_cols.add(new_col)

        print(f"[{dataset_name}] {len(old_idx)} genes mapeados para o espaço canônico ({n_genes_alvo} genes totais).")

        # Matriz de projeção P (n_vars_original -> n_genes_alvo)
        P_data: NDArray[np.float32] = np.ones(len(old_idx), dtype=np.float32)
        P: sp.csr_matrix = sp.csr_matrix(
            (P_data, (old_idx, new_idx)), shape=(adata.n_vars, n_genes_alvo), dtype=np.float32
        )

        print(f"[{dataset_name}] Projetando matriz esparsa...")
        X_novo: sp.csr_matrix
        if sp.issparse(adata.X):
            X_csr: sp.csr_matrix = sp.csr_matrix(adata.X)
            X_novo = sp.csr_matrix(X_csr.dot(P))
        else:
            assert isinstance(adata.X, np.ndarray)
            X_novo = sp.csr_matrix(adata.X).dot(P)

        if not sp.isspmatrix_csr(X_novo):
            X_novo = X_novo.tocsr()

        del P
        gc.collect()

        # Anota metadados de presença por gene
        presente_arr: NDArray[np.bool_] = np.zeros(n_genes_alvo, dtype=bool)
        if len(present_new_cols) > 0:
            presente_arr[list(present_new_cols)] = True

        var_novo: pd.DataFrame = pd.DataFrame(
            {
                "presente_no_dataset": presente_arr,
            },
            index=pd.Index(self.genes_ordenados, name="ensembl_id"),
        )
        assert isinstance(adata.obs, pd.DataFrame)
        obs_df: pd.DataFrame = adata.obs.copy()
        return ad.AnnData(X=X_novo, obs=obs_df, var=var_novo)

    def alinhar(self, forcar: bool = False) -> AlinhadorEsparso:
        """Executa o alinhamento esparso para ambos os datasets (Fujita e Mathys).

        Parameters
        ----------
        forcar : bool, default=False
            Se True, recalcula mesmo se os arquivos de destino já existirem.

        Returns
        -------
        AlinhadorEsparso
            A própria instância.
        """
        pasta_f: str = os.path.dirname(self.path_f_alinhado)
        pasta_m: str = os.path.dirname(self.path_m_alinhado)

        # 1. Alinhamento Fujita (Referência)
        if os.path.exists(self.path_f_alinhado) and not forcar:
            print(f"[AlinhadorEsparso] Fujita já alinhado, pulando: {self.path_f_alinhado}")
        else:
            print("[AlinhadorEsparso] Carregando Fujita binarizado...")
            adataf: ad.AnnData = ad.read_h5ad(self.path_binarizada_f)
            print(f"  Shape original Fujita: {adataf.shape}")
            adataf_alinhado: ad.AnnData = self._projetar_esparso(adataf, self.map_f, dataset_name="Fujita")
            del adataf
            gc.collect()

            os.makedirs(pasta_f, exist_ok=True)
            print("  Salvando Fujita alinhado (.h5ad CSR)...")
            adataf_alinhado.write_h5ad(self.path_f_alinhado, compression="gzip")
            print(f"  Salvo em {self.path_f_alinhado} (shape: {adataf_alinhado.shape})  [OK]\n")
            del adataf_alinhado
            gc.collect()

        # 2. Alinhamento Mathys (Alvo)
        if os.path.exists(self.path_m_alinhado) and not forcar:
            print(f"[AlinhadorEsparso] Mathys já alinhado, pulando: {self.path_m_alinhado}")
        else:
            print("[AlinhadorEsparso] Carregando Mathys binarizado...")
            adatam: ad.AnnData = ad.read_h5ad(self.path_binarizada_m)
            print(f"  Shape original Mathys: {adatam.shape}")
            adatam_alinhado: ad.AnnData = self._projetar_esparso(adatam, self.map_m, dataset_name="Mathys")
            del adatam
            gc.collect()

            os.makedirs(pasta_m, exist_ok=True)
            print("  Salvando Mathys alinhado (.h5ad CSR)...")
            adatam_alinhado.write_h5ad(self.path_m_alinhado, compression="gzip")
            print(f"  Salvo em {self.path_m_alinhado} (shape: {adatam_alinhado.shape})  [OK]\n")
            del adatam_alinhado
            gc.collect()

        print("[AlinhadorEsparso] Alinhamento genômico canônico concluído com sucesso.")
        return self

    def gerar_tracking(self, ids_so_f: Set[str], map_f: Mapping[str, str]) -> pl.DataFrame:
        """Gera relatório de genes exclusivos do Fujita (ausentes no Mathys).

        Parameters
        ----------
        ids_so_f : Set[str]
            Genes exclusivos da referência.
        map_f : Mapping[str, str]
            Mapeamento de features.

        Returns
        -------
        pl.DataFrame
            DataFrame Polars com as linhas de rastreamento.
        """
        if self.path_m_alinhado is None:
            raise RuntimeError("Execute .alinhar() antes de gerar o tracking.")

        out_tracking: str = os.path.join(self.out_dir, "tracking_genes_adicionados_mathys.csv")
        if os.path.exists(out_tracking):
            print(f"[AlinhadorEsparso] Tracking já existe, pulando: {out_tracking}")
            return pl.read_csv(out_tracking)

        inv_map_f: dict[str, str] = {v: k for k, v in map_f.items()}
        tracking_rows: list[dict[str, Union[str, int, float, bool]]] = []
        for eid in sorted(ids_so_f):
            gene_name: str = inv_map_f.get(eid, eid)
            col_idx: int | None = self.gene_alvo_idx.get(eid, self.gene_alvo_idx.get(gene_name, None))
            if col_idx is not None:
                tracking_rows.append({
                    "gene_name": gene_name,
                    "ensembl_id": eid,
                    "posicao_coluna": int(col_idx),
                    "valor_inserido": 0.5,
                    "presente_fujita": True,
                    "presente_mathys": False,
                })

        df_tracking: pl.DataFrame
        if not tracking_rows:
            schema: dict[str, Any] = {
                "gene_name": pl.Utf8,
                "ensembl_id": pl.Utf8,
                "posicao_coluna": pl.Int64,
                "valor_inserido": pl.Float64,
                "presente_fujita": pl.Boolean,
                "presente_mathys": pl.Boolean,
            }
            df_tracking = pl.DataFrame(schema=schema)
        else:
            df_tracking = pl.DataFrame(tracking_rows).sort("posicao_coluna")

        os.makedirs(os.path.dirname(os.path.abspath(out_tracking)), exist_ok=True)
        df_tracking.write_csv(out_tracking)
        print(f"[AlinhadorEsparso] Tracking salvo em: {out_tracking} ({len(df_tracking)} genes)")
        return df_tracking

    def obter_mascara_ausentes(self, path_mathys: PathType | None = None) -> NDArray[np.bool_]:
        """Retorna máscara booleana (True = gene ausente no Mathys) de forma eficiente sem carregar matriz X.

        Parameters
        ----------
        path_mathys : str | os.PathLike[str] | None, optional
            Caminho do arquivo alinhado Mathys.

        Returns
        -------
        NDArray[np.bool_]
            Vetor booleano 1D indexado por gene.
        """
        path_m: str = str(path_mathys) if path_mathys is not None else self.path_m_alinhado
        if not os.path.exists(path_m):
            raise FileNotFoundError(f"Arquivo alinhado do Mathys não encontrado: {path_m}")

        adata: ad.AnnData = ad.read_h5ad(path_m, backed="r")
        mask: NDArray[np.bool_]
        if "presente_no_dataset" in adata.var:
            mask = ~adata.var["presente_no_dataset"].to_numpy().astype(bool)
        else:
            out_tracking: str = os.path.join(self.out_dir, "tracking_genes_adicionados_mathys.csv")
            if os.path.exists(out_tracking):
                df_track: pl.DataFrame = pl.read_csv(out_tracking)
                col_indices: NDArray[np.int_] = df_track["posicao_coluna"].to_numpy()
                mask = np.zeros(adata.n_vars, dtype=bool)
                mask[col_indices] = True
            else:
                mask = np.zeros(adata.n_vars, dtype=bool)
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()
        return mask

    def extrair_subconjunto(
        self,
        lista_genes_ou_csv: Union[Sequence[str], PathType],
        out_dir: PathType | None = None,
        exportar_npy: bool = True,
        exportar_h5ad: bool = True,
        fill_value_mathys: float = 0.5,
    ) -> dict[str, str | None]:
        """Extrai um subconjunto ordenado de genes (ex: Top 5000) e injeta sentinela 0.5 no Mathys de forma OOM-Safe.

        Parameters
        ----------
        lista_genes_ou_csv : Sequence[str] | str | os.PathLike[str]
            Lista de identificadores de genes ou caminho para CSV com coluna 'gene'.
        out_dir : str | os.PathLike[str] | None, optional
            Diretório de destino dos arquivos extraídos.
        exportar_npy : bool, default=True
            Se True, exporta formato binário NumPy (.npy).
        exportar_h5ad : bool, default=True
            Se True, exporta formato AnnData (.h5ad).
        fill_value_mathys : float, default=0.5
            Valor sentinela a aplicar em posições ausentes do Mathys.

        Returns
        -------
        dict[str, str | None]
            Caminhos dos arquivos gerados.
        """
        genes_desejados: list[str]
        if isinstance(lista_genes_ou_csv, (str, os.PathLike)):
            df_g: pl.DataFrame = pl.read_csv(str(lista_genes_ou_csv))
            col_nome: str = "gene" if "gene" in df_g.columns else df_g.columns[0]
            genes_desejados = df_g[col_nome].to_list()
        else:
            genes_desejados = list(lista_genes_ou_csv)

        n_genes: int = len(genes_desejados)
        destino_dir: str = str(out_dir) if out_dir is not None else self.out_dir
        os.makedirs(destino_dir, exist_ok=True)

        print(f"\n[AlinhadorEsparso] Extraindo subconjunto de {n_genes} genes...")

        # Mapeia posições dos genes desejados no espaço canônico
        gene_to_pos: dict[str, int] = {g: i for i, g in enumerate(self.genes_ordenados)}
        col_indices_list: list[int] = []
        genes_encontrados: list[str] = []
        for g in genes_desejados:
            if g in gene_to_pos:
                col_indices_list.append(gene_to_pos[g])
                genes_encontrados.append(g)

        col_indices: NDArray[np.int32] = np.array(col_indices_list, dtype=np.int32)
        n_encontrados: int = len(col_indices)
        print(f"  {n_encontrados} de {n_genes} genes localizados no espaço canônico.")

        # --- 1. Extração Fujita ---
        path_f_npy: str = os.path.join(destino_dir, f"adataF_binarizado_alinhado_top{n_encontrados}.npy")
        path_f_h5ad: str = os.path.join(destino_dir, f"adataF_binarizado_alinhado_top{n_encontrados}.h5ad")

        print(f"  Processando Fujita (Top {n_encontrados})...")
        adata_f: ad.AnnData = ad.read_h5ad(self.path_f_alinhado)
        X_f_sub = adata_f.X[:, col_indices]
        X_f_dense: NDArray[np.float32]
        if sp.issparse(X_f_sub):
            X_f_dense = sp.csr_matrix(X_f_sub).toarray().astype(np.float32)
        else:
            X_f_dense = np.asarray(X_f_sub, dtype=np.float32)

        if exportar_npy:
            np.save(path_f_npy, X_f_dense)
            print(f"  Salvo Fujita .npy: {path_f_npy} ({X_f_dense.shape})")

        if exportar_h5ad:
            var_f: pd.DataFrame = pd.DataFrame(index=pd.Index(genes_encontrados, name="ensembl_id"))
            assert isinstance(adata_f.obs, pd.DataFrame)
            obs_f: pd.DataFrame = adata_f.obs.copy()
            adata_f_sub: ad.AnnData = ad.AnnData(X=sp.csr_matrix(X_f_dense), obs=obs_f, var=var_f)
            adata_f_sub.write_h5ad(path_f_h5ad, compression="gzip")
            print(f"  Salvo Fujita .h5ad: {path_f_h5ad}")

        del adata_f, X_f_sub, X_f_dense
        gc.collect()

        # --- 2. Extração Mathys (com Sentinela 0.5) ---
        path_m_npy: str = os.path.join(destino_dir, f"adataM_binarizado_alinhado_top{n_encontrados}.npy")
        path_m_h5ad: str = os.path.join(destino_dir, f"adataM_binarizado_alinhado_top{n_encontrados}.h5ad")

        print(f"  Processando Mathys (Top {n_encontrados} com sentinela={fill_value_mathys})...")
        adata_m: ad.AnnData = ad.read_h5ad(self.path_m_alinhado)
        presente_mathys_all: NDArray[np.bool_] = adata_m.var["presente_no_dataset"].to_numpy().astype(bool)
        presente_mathys_sub: NDArray[np.bool_] = presente_mathys_all[col_indices]

        X_m_sub = adata_m.X[:, col_indices]
        X_m_dense: NDArray[np.float32]
        if sp.issparse(X_m_sub):
            X_m_dense = sp.csr_matrix(X_m_sub).toarray().astype(np.float32)
        else:
            X_m_dense = np.asarray(X_m_sub, dtype=np.float32)

        # Injeta sentinela apenas nas colunas ausentes no Mathys
        ausentes_mask: NDArray[np.bool_] = ~presente_mathys_sub
        n_ausentes: int = int(np.sum(ausentes_mask))
        if n_ausentes > 0:
            print(f"  Injetando sentinela {fill_value_mathys} em {n_ausentes} colunas ausentes no Mathys...")
            X_m_dense[:, ausentes_mask] = fill_value_mathys

        if exportar_npy:
            np.save(path_m_npy, X_m_dense)
            print(f"  Salvo Mathys .npy: {path_m_npy} ({X_m_dense.shape})")

        if exportar_h5ad:
            var_m: pd.DataFrame = pd.DataFrame(
                {"presente_no_dataset": presente_mathys_sub},
                index=pd.Index(genes_encontrados, name="ensembl_id"),
            )
            assert isinstance(adata_m.obs, pd.DataFrame)
            obs_m: pd.DataFrame = adata_m.obs.copy()
            adata_m_sub: ad.AnnData = ad.AnnData(X=sp.csr_matrix(X_m_dense), obs=obs_m, var=var_m)
            adata_m_sub.write_h5ad(path_m_h5ad, compression="gzip")
            print(f"  Salvo Mathys .h5ad: {path_m_h5ad}")

        del adata_m, X_m_sub, X_m_dense
        gc.collect()

        return {
            "path_f_npy": path_f_npy if exportar_npy else None,
            "path_f_h5ad": path_f_h5ad if exportar_h5ad else None,
            "path_m_npy": path_m_npy if exportar_npy else None,
            "path_m_h5ad": path_m_h5ad if exportar_h5ad else None,
        }

    def salvar_como_txt(self, chunk: int = 500) -> AlinhadorEsparso:
        """Método de retrocompatibilidade com aviso sobre descontinuação de TXT de 36k genes."""
        print("[AlinhadorEsparso] AVISO: A geração de arquivos .txt com 36k colunas foi descontinuada "
              "para evitar estouro de memória (>20GB). Use extrair_subconjunto() para exportar .npy / .h5ad.")
        return self

    def __repr__(self) -> str:
        """Representação textual do alinhador esparso."""
        return (
            f"AlinhadorEsparso(\n"
            f"  path_binarizada_m = {self.path_binarizada_m}\n"
            f"  path_binarizada_f = {self.path_binarizada_f}\n"
            f"  out_dir           = {self.out_dir}\n"
            f"  path_f_alinhado   = {self.path_f_alinhado}\n"
            f"  path_m_alinhado   = {self.path_m_alinhado}\n"
            f")"
        )
