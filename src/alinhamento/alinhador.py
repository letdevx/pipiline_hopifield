"""Módulo de Alinhamento Dimensional e Injeção de Sentinela Neutra.

Alinha matrizes de expressão scRNA-seq ao espaço canônico comum e projeta
genes ausentes com o valor sentinela 0.5 (neutro na dinâmica Hopfield).
"""

from __future__ import annotations

import gc
import os
from collections.abc import Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.sparse as sp
from numpy.typing import NDArray

PathType = str | os.PathLike[str]


class Alinhador:
    """Alinha dois arquivos .h5ad binarizados ao mesmo espaço gênico de referência (Fujita).

    Parameters
    ----------
    path_binarizada_m : str | os.PathLike[str]
        Caminho do .h5ad binarizado do conjunto alvo (Mathys).
    path_binarizada_f : str | os.PathLike[str]
        Caminho do .h5ad binarizado do conjunto de referência (Fujita).
    out_dir : str | os.PathLike[str]
        Diretório base onde as matrizes alinhadas serão salvas.
    map_f : Mapping[str, str]
        Dicionário {Gene Symbol: Ensembl ID} da referência.
    map_m : Mapping[str, str]
        Dicionário {Gene Symbol: Ensembl ID} do alvo.
    gene_alvo_idx : Mapping[str, int]
        Dicionário mapeando Ensembl ID para a coluna canônica correspondente.
    genes_ordenados : Sequence[str]
        Lista ordenada de Ensembl IDs do espaço gênico unificado.

    Attributes
    ----------
    path_binarizada_m : str
        Caminho normalizado do arquivo alvo.
    path_binarizada_f : str
        Caminho normalizado do arquivo de referência.
    out_dir : str
        Diretório de saída.
    map_f : Mapping[str, str]
        Mapeamento genômico da referência.
    map_m : Mapping[str, str]
        Mapeamento genômico do alvo.
    gene_alvo_idx : Mapping[str, int]
        Índices canônicos dos genes.
    genes_ordenados : list[str]
        Genes canônicos ordenados.
    path_f_alinhado : str | None
        Caminho do .h5ad da referência gerado após alinhamento.
    path_m_alinhado : str | None
        Caminho do .h5ad do alvo gerado após alinhamento.
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
        self.path_f_alinhado: str | None = None
        self.path_m_alinhado: str | None = None

    def alinhar(self) -> Alinhador:
        """Executa a projeção linear e preenchimento de sentinelas nos datasets.

        Returns
        -------
        Alinhador
            A própria instância com os caminhos dos arquivos gerados.
        """
        nome_f: str = "adataF_binarizado_alinhado"
        pasta_f: str = os.path.join(self.out_dir, nome_f)
        self.path_f_alinhado = os.path.join(pasta_f, f"{nome_f}.h5ad")

        nome_m: str = "adataM_binarizado_alinhado"
        pasta_m: str = os.path.join(self.out_dir, nome_m)
        self.path_m_alinhado = os.path.join(pasta_m, f"{nome_m}.h5ad")

        if os.path.exists(self.path_f_alinhado):
            print(f"[Alinhador] Fujita já alinhado, pulando: {self.path_f_alinhado}")
        else:
            print("[Alinhador] Carregando Fujita binarizado...")
            adataf: ad.AnnData = ad.read_h5ad(self.path_binarizada_f)
            print(f"  shape original: {adataf.shape}")
            print("[Alinhador] Alinhando Fujita (fill=0.0)...")
            adataf_alinhado: ad.AnnData = self._alinhar_direto(
                adataf, self.map_f, fill_value=0.0
            )
            del adataf
            gc.collect()
            os.makedirs(pasta_f, exist_ok=True)
            adataf_alinhado.write_h5ad(self.path_f_alinhado)
            print(f"  shape final: {adataf_alinhado.shape}")
            del adataf_alinhado
            gc.collect()
            print(f"  salvo em {self.path_f_alinhado}  [OK]\n")

        if os.path.exists(self.path_m_alinhado):
            print(f"[Alinhador] Mathys já alinhado, pulando: {self.path_m_alinhado}")
        else:
            print("[Alinhador] Carregando Mathys binarizado...")
            adatam: ad.AnnData = ad.read_h5ad(self.path_binarizada_m)
            print(f"  shape original: {adatam.shape}")
            print("[Alinhador] Alinhando Mathys (genes ausentes → 0.5)...")
            adatam_alinhado: ad.AnnData = self._alinhar_direto(
                adatam, self.map_m, fill_value=0.5
            )
            del adatam
            gc.collect()
            os.makedirs(pasta_m, exist_ok=True)
            adatam_alinhado.write_h5ad(self.path_m_alinhado)
            print(f"  shape final: {adatam_alinhado.shape}")
            del adatam_alinhado
            gc.collect()
            print(f"  salvo em {self.path_m_alinhado}  [OK]")

        print("\n[Alinhador] Concluído.")
        return self

    def _alinhar_direto(
        self,
        adata: ad.AnnData,
        ensembl_map: Mapping[str, str],
        fill_value: float = 0.0,
    ) -> ad.AnnData:
        """Alinha um AnnData construindo a matriz de projeção esparsa."""
        n_celulas: int = adata.n_obs
        n_genes: int = len(self.genes_ordenados)

        old_idx: list[int] = []
        new_idx: list[int] = []
        present_new_cols: set[int] = set()
        for old_i, gene_name in enumerate(adata.var_names):
            eid: str = ensembl_map.get(gene_name, gene_name)
            if eid in self.gene_alvo_idx:
                new_col: int = self.gene_alvo_idx[eid]
                old_idx.append(old_i)
                new_idx.append(new_col)
                present_new_cols.add(new_col)

        # Matriz de projeção P (old_vars -> new_genes)
        P_data: NDArray[np.float32] = np.ones(len(old_idx), dtype=np.float32)
        P: sp.csr_matrix = sp.csr_matrix(
            (P_data, (old_idx, new_idx)), shape=(adata.n_vars, n_genes)
        )

        print("  Multiplicando matrizes (projeção)...")
        # Multiplicação é extremamente rápida e consome pouca memória
        X_novo: sp.csr_matrix
        if sp.issparse(adata.X):
            X_csr: sp.csr_matrix = sp.csr_matrix(adata.X)
            X_novo = sp.csr_matrix(X_csr.dot(P)).astype(np.float32)
        else:
            assert isinstance(adata.X, np.ndarray)
            X_novo = sp.csr_matrix(adata.X).dot(P).astype(np.float32)

        del P
        gc.collect()

        if fill_value != 0.0:
            missing_cols: NDArray[np.int32] = np.array(
                sorted(set(range(n_genes)) - present_new_cols), dtype=np.int32
            )
            if len(missing_cols) > 0:
                print(
                    f"  Preenchendo {len(missing_cols)} colunas ausentes com {fill_value}..."
                )
                print("  Mesclando colunas ausentes (otimizado para baixa memória)...")
                n_miss: int = len(missing_cols)
                n_novo: int = X_novo.nnz
                n_total: int = n_novo + n_celulas * n_miss

                final_indices: NDArray[np.int32] = np.empty(n_total, dtype=np.int32)
                final_data: NDArray[np.float32] = np.empty(n_total, dtype=np.float32)
                final_indptr: NDArray[np.int64] = np.empty(
                    n_celulas + 1, dtype=np.int64
                )
                final_indptr[0] = 0

                novo_indptr: NDArray[np.int32] = X_novo.indptr
                novo_indices: NDArray[np.int32] = X_novo.indices
                novo_data: NDArray[np.float32] = X_novo.data

                idx_ptr: int = 0
                fill_dat_arr: NDArray[np.float32] = np.full(
                    n_miss, fill_value, dtype=np.float32
                )

                for i in range(n_celulas):
                    start: int = int(novo_indptr[i])
                    end: int = int(novo_indptr[i + 1])
                    ex_idx: NDArray[np.int32] = novo_indices[start:end]
                    ex_dat: NDArray[np.float32] = novo_data[start:end]

                    if len(ex_idx) == 0:
                        row_idx = missing_cols
                        row_dat = fill_dat_arr
                    else:
                        row_idx = np.concatenate([ex_idx, missing_cols])
                        row_dat = np.concatenate([ex_dat, fill_dat_arr])
                        sort_mask = np.argsort(row_idx)
                        row_idx = row_idx[sort_mask]
                        row_dat = row_dat[sort_mask]

                    n_elem: int = len(row_idx)
                    end_ptr: int = idx_ptr + n_elem

                    final_indices[idx_ptr:end_ptr] = row_idx
                    final_data[idx_ptr:end_ptr] = row_dat

                    idx_ptr = end_ptr
                    final_indptr[i + 1] = idx_ptr

                X_novo = sp.csr_matrix(
                    (final_data, final_indices, final_indptr),
                    shape=(n_celulas, n_genes),
                )
                del final_data, final_indices, final_indptr
                gc.collect()
                print("  Preenchimento concluído.")

        # AnnData exige pd.DataFrame para .var
        var_novo: pd.DataFrame = pd.DataFrame(
            index=pd.Index(self.genes_ordenados, name="ensembl_id")
        )
        assert isinstance(adata.obs, pd.DataFrame)
        obs_df: pd.DataFrame = adata.obs.copy()
        return ad.AnnData(X=X_novo, obs=obs_df, var=var_novo)

    def salvar_como_txt(self, chunk: int = 500) -> Alinhador:
        """Salva os arquivos alinhados em formato TXT (CSV) dentro de suas respectivas pastas.

        Parameters
        ----------
        chunk : int, default=500
            Número de linhas por lote para gravação em streaming.

        Returns
        -------
        Alinhador
            A própria instância.
        """
        if self.path_f_alinhado is None or self.path_m_alinhado is None:
            raise RuntimeError("Execute .alinhar() antes de salvar como TXT.")

        for path_h5ad in (self.path_f_alinhado, self.path_m_alinhado):
            path_txt: str = os.path.splitext(path_h5ad)[0] + ".txt"
            path_tmp: str = path_txt + ".tmp"

            if os.path.exists(path_txt):
                print(f"[Alinhador] TXT já existe, pulando: {path_txt}")
                continue

            if os.path.exists(path_tmp):
                os.remove(path_tmp)

            print(f"[Alinhador] Salvando TXT: {path_txt}")
            adata: ad.AnnData = ad.read_h5ad(path_h5ad, backed="r")
            n_celulas: int = adata.n_obs
            gene_names: list[str] = list(adata.var_names)
            total: int = 0

            with open(
                path_tmp, "w", buffering=128 * 1024 * 1024, encoding="utf-8"
            ) as fout:
                fout.write(",".join(gene_names) + "\n")
                for start in range(0, n_celulas, chunk):
                    end: int = min(start + chunk, n_celulas)
                    X_slice = adata.X[start:end]
                    if sp.issparse(X_slice):
                        X_arr = sp.csr_matrix(X_slice).toarray().astype(np.float32)
                    else:
                        X_arr = np.asarray(X_slice, dtype=np.float32)
                    fout.write(
                        pl.from_numpy(np.asfortranarray(X_arr)).write_csv(
                            include_header=False
                        )
                    )
                    total += end - start
                    if total % (chunk * 5) == 0:
                        print(f"  {total} células processadas...")

            if hasattr(adata, "file") and adata.file is not None:
                adata.file.close()
            os.rename(path_tmp, path_txt)
            print(
                f"  Salvo: {path_txt}  ({total} células x {len(gene_names)} genes)  [Ok]"
            )
        return self

    def gerar_tracking(
        self, ids_so_f: set[str], map_f: Mapping[str, str]
    ) -> pl.DataFrame:
        """Gera a tabela de rastreamento de genes adicionados artificialmente ao alvo.

        Parameters
        ----------
        ids_so_f : Set[str]
            Conjunto de genes ausentes no alvo e presentes na referência.
        map_f : Mapping[str, str]
            Mapeamento genômico da referência.

        Returns
        -------
        pl.DataFrame
            DataFrame Polars com o relatório de tracking.
        """
        if self.path_m_alinhado is None:
            raise RuntimeError("Execute .alinhar() antes de gerar o tracking.")
        out_tracking: str = os.path.join(
            self.out_dir, "tracking_genes_adicionados_mathys.csv"
        )
        if os.path.exists(out_tracking):
            print(f"[Alinhador] Tracking já existe, pulando: {out_tracking}")
            return pl.read_csv(out_tracking)
        inv_map_f: dict[str, str] = {v: k for k, v in map_f.items()}
        tracking_rows: list[dict[str, str | int | float | bool]] = []
        for eid in sorted(ids_so_f):
            if eid in self.gene_alvo_idx:
                tracking_rows.append(
                    {
                        "gene_name": inv_map_f.get(eid, eid),
                        "ensembl_id": eid,
                        "posicao_coluna": self.gene_alvo_idx[eid],
                        "valor_inserido": 0.5,
                        "presente_fujita": True,
                        "presente_mathys": False,
                    }
                )
        df_tracking: pl.DataFrame = pl.DataFrame(tracking_rows).sort("posicao_coluna")
        os.makedirs(os.path.dirname(os.path.abspath(out_tracking)), exist_ok=True)
        df_tracking.write_csv(out_tracking)
        print(
            f"[Alinhador] Tracking salvo em: {out_tracking} ({len(df_tracking)} genes)"
        )
        return df_tracking

    def __repr__(self) -> str:
        """Representação textual do alinhador."""
        return (
            f"Alinhador(\n"
            f"  path_binarizada_m = {self.path_binarizada_m}\n"
            f"  path_binarizada_f = {self.path_binarizada_f}\n"
            f"  out_dir           = {self.out_dir}\n"
            f"  path_f_alinhado   = {self.path_f_alinhado or 'ainda não gerado'}\n"
            f"  path_m_alinhado   = {self.path_m_alinhado or 'ainda não gerado'}\n"
            f")"
        )
