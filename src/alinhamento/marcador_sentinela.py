"""Módulo de Inserção Física e Padronizada de Sentinelas (0.5) em Datasets Alvo.

Implementa o componente MarcadorSentinela para alinhar qualquer dataset de scRNA-seq
ao espaço canônico de referência e materializar fisicamente o valor sentinela 0.5
nas colunas de genes ausentes através de processamento em lotes (chunks) OOM-Safe.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Mapping, Sequence
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

from .exportador_mtx import ExportadorMTX

PathType = str | os.PathLike[str]


class MarcadorSentinela:
    """Insere fisicamente valores sentinela (0.5) nas colunas de genes ausentes de qualquer dataset alvo.

    Permite alinhar matrizes scRNA-seq provenientes de múltiplos estudos ou plataformas
    ao espaço gênico canônico definido pela referência, preenchendo as colunas de genes
    não mensurados com a sentinela neutra (0.5) de forma compatível com Modern Hopfield Networks
    e projeções dimensionais rSWeeP.

    Parameters
    ----------
    genes_referencia : Sequence[str] | PathType
        Vetor canônico ordenado de genes da referência (Ensembl IDs) ou caminho
        para arquivo CSV, TSV ou AnnData (.h5ad) contendo a lista de referência.
    map_features_alvo : Mapping[str, str] | None, optional
        Dicionário opcional mapeando identificadores do alvo para Ensembl IDs canônicos
        (ex.: {Gene Symbol: Ensembl ID}). Se None, assume correspondência direta.
    fill_value : float, default=0.5
        Valor sentinela neutro inserido nas coordenadas ausentes.
    chunk_size : int, default=5000
        Número de células processadas por bloco em memória durante a injeção física.

    Attributes
    ----------
    genes_canonica : list[str]
        Lista ordenada de identificadores gênicos da referência canônica.
    gene_to_idx : dict[str, int]
        Índice inverso {gene_id: posição_coluna}.
    fill_value : float
        Valor sentinela configurado.
    chunk_size : int
        Tamanho de lote para processamento OOM-Safe.
    map_features_alvo : Mapping[str, str]
        Mapeamento de features do alvo.
    mask_ausentes : NDArray[np.bool_] | None
        Máscara booleana 1D (True = gene ausente no alvo) após processamento.
    df_tracking : pd.DataFrame | None
        Tabela com posições e nomes dos genes que receberam sentinela.
    """

    def __init__(
        self,
        genes_referencia: Sequence[str] | PathType,
        map_features_alvo: Mapping[str, str] | None = None,
        fill_value: float = 0.5,
        chunk_size: int = 5000,
    ) -> None:
        self.genes_canonica: list[str] = self._carregar_genes_referencia(
            genes_referencia
        )
        self.gene_to_idx: dict[str, int] = {
            g: i for i, g in enumerate(self.genes_canonica)
        }
        self.map_features_alvo: Mapping[str, str] = (
            map_features_alvo if map_features_alvo is not None else {}
        )
        self.fill_value: float = float(fill_value)
        self.chunk_size: int = max(int(chunk_size), 1)

        self.mask_ausentes: NDArray[np.bool_] | None = None
        self.df_tracking: pd.DataFrame | None = None

    @staticmethod
    def _sanitizar_id(gene_id: str) -> str:
        """Remove sufixos de versão de Ensembl IDs mantendo identificadores padrão."""
        gid = str(gene_id).strip()
        if gid.startswith("ENSG") and "." in gid:
            return gid.split(".")[0]
        return gid

    def _carregar_genes_referencia(
        self, genes_referencia: Sequence[str] | PathType
    ) -> list[str]:
        """Carrega e valida o espaço gênico canônico da referência a partir de múltiplos formatos."""
        if isinstance(genes_referencia, (str, os.PathLike)):
            path_ref = str(genes_referencia)
            if not os.path.exists(path_ref):
                raise FileNotFoundError(
                    f"[MarcadorSentinela] Arquivo de referência não encontrado: {path_ref}"
                )

            if path_ref.endswith(".h5ad"):
                adata_ref = ad.read_h5ad(path_ref, backed="r")
                genes_list = [self._sanitizar_id(g) for g in adata_ref.var_names]
                if hasattr(adata_ref, "file") and adata_ref.file is not None:
                    adata_ref.file.close()
                return genes_list

            if path_ref.endswith((".csv", ".tsv", ".txt")):
                sep = "\t" if path_ref.endswith(".tsv") else ","
                df = pd.read_csv(path_ref, sep=sep)
                coluna = None
                for c in ["ensembl_id", "gene", "genes", "Ensembl_ID", "gene_id"]:
                    if c in df.columns:
                        coluna = c
                        break
                if coluna is None:
                    coluna = df.columns[0]
                return [self._sanitizar_id(g) for g in df[coluna].dropna()]

            raise ValueError(
                f"[MarcadorSentinela] Formato de arquivo de referência não suportado: {path_ref}"
            )

        # Sequência direta em memória
        return [self._sanitizar_id(g) for g in genes_referencia]

    def _resolver_alinhamento_colunas(
        self, var_names_alvo: Sequence[str]
    ) -> tuple[list[int], list[int], NDArray[np.bool_]]:
        """Mapeia índices de colunas do dataset alvo para as coordenadas canônicas da referência.

        Returns
        -------
        tuple[list[int], list[int], NDArray[np.bool_]]
            (old_indices, new_indices, mask_ausentes_1d)
        """
        n_canonica = len(self.genes_canonica)
        old_idx: list[int] = []
        new_idx: list[int] = []
        presentes_colunas_canonica: set[int] = set()

        for old_col, g_raw in enumerate(var_names_alvo):
            g_str = str(g_raw).strip()
            # 1. Tenta mapear via dicionário de features
            eid = self.map_features_alvo.get(g_str, g_str)
            eid_sanitizado = self._sanitizar_id(eid)

            target_pos = self.gene_to_idx.get(
                eid_sanitizado, self.gene_to_idx.get(eid, self.gene_to_idx.get(g_str))
            )

            if target_pos is not None:
                old_idx.append(old_col)
                new_idx.append(target_pos)
                presentes_colunas_canonica.add(target_pos)

        mask_ausentes = np.ones(n_canonica, dtype=bool)
        if presentes_colunas_canonica:
            mask_ausentes[list(presentes_colunas_canonica)] = False

        return old_idx, new_idx, mask_ausentes

    def gerar_tracking(
        self,
        mask_ausentes: NDArray[np.bool_],
        path_tracking: PathType | None = None,
        inv_map_features: Mapping[str, str] | None = None,
    ) -> pd.DataFrame:
        """Gera DataFrame de rastreamento dos genes ausentes que receberam a sentinela 0.5.

        Parameters
        ----------
        mask_ausentes : NDArray[np.bool_]
            Vetor booleano de genes ausentes.
        path_tracking : PathType | None, optional
            Caminho do arquivo CSV de destino.
        inv_map_features : Mapping[str, str] | None, optional
            Mapeamento inverso {Ensembl ID: Gene Symbol}.

        Returns
        -------
        pd.DataFrame
            Tabela de auditoria com posições e identificadores dos genes adicionados.
        """
        inv_map = inv_map_features or {}
        ausentes_idx = np.where(mask_ausentes)[0]

        rows: list[dict[str, Any]] = []
        for col_idx in ausentes_idx:
            eid = self.genes_canonica[col_idx]
            gene_name = inv_map.get(eid, eid)
            rows.append(
                {
                    "posicao_coluna": int(col_idx),
                    "ensembl_id": eid,
                    "gene_name": gene_name,
                    "valor_inserido": self.fill_value,
                    "presente_referencia": True,
                    "presente_alvo": False,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("posicao_coluna").reset_index(drop=True)

        self.df_tracking = df

        if path_tracking is not None:
            path_str = str(path_tracking)
            os.makedirs(os.path.dirname(os.path.abspath(path_str)), exist_ok=True)
            df.to_csv(path_str, index=False)
            print(
                f"[MarcadorSentinela] Tracking salvo em: {path_str} ({len(df)} genes ausentes anotados)"
            )

        return df

    def marcar_dataset(
        self,
        adata_alvo: ad.AnnData | PathType,
        path_saida_h5ad: PathType,
        path_tracking_csv: PathType | None = None,
        exportar_mtx: bool = False,
        path_saida_mtx: PathType | None = None,
        forcar: bool = False,
    ) -> ad.AnnData:
        """Executa a injeção física de sentinelas em lotes OOM-Safe e exporta o AnnData com CSR compactado.

        Parameters
        ----------
        adata_alvo : ad.AnnData | PathType
            Objeto AnnData em memória ou caminho para o arquivo .h5ad do alvo.
        path_saida_h5ad : PathType
            Caminho do arquivo .h5ad de saída.
        path_tracking_csv : PathType | None, optional
            Caminho do arquivo CSV de rastreamento. Se None, gera automaticamente.
        exportar_mtx : bool, default=False
            Se True, exporta também a pasta Matrix Market (.mtx) via ExportadorMTX.
        path_saida_mtx : PathType | None, optional
            Diretório de destino da exportação MTX (obrigatório se exportar_mtx=True).
        forcar : bool, default=False
            Se True, recalcula e regrava mesmo se o arquivo de destino já existir.

        Returns
        -------
        ad.AnnData
            Objeto AnnData alinhado com colunas de sentinela fisicamente preenchidas.
        """
        path_h5ad_str = str(path_saida_h5ad)
        mtx_ok = (
            True
            if not exportar_mtx
            else (
                path_saida_mtx is not None
                and os.path.exists(os.path.join(str(path_saida_mtx), "matrix.mtx"))
            )
        )

        # 1. Checagem de Idempotência
        if os.path.exists(path_h5ad_str) and mtx_ok and not forcar:
            print(
                f"[MarcadorSentinela] Arquivo com sentinelas já existente: {path_h5ad_str}. Carregando..."
            )
            adata_carregado = ad.read_h5ad(path_h5ad_str)
            if "ausente" in adata_carregado.var:
                self.mask_ausentes = adata_carregado.var["ausente"].to_numpy(dtype=bool)
            return adata_carregado

        # 2. Resolução do AnnData Alvo de Entrada
        deve_fechar = False
        adata: ad.AnnData
        if isinstance(adata_alvo, (str, os.PathLike)):
            path_in = str(adata_alvo)
            if not os.path.exists(path_in):
                raise FileNotFoundError(
                    f"[MarcadorSentinela] Dataset alvo não encontrado: {path_in}"
                )
            adata = ad.read_h5ad(path_in)
            deve_fechar = True
        else:
            adata = adata_alvo

        n_obs = int(adata.n_obs)
        n_canonica = len(self.genes_canonica)
        var_names_sanitizados = [self._sanitizar_id(g) for g in adata.var_names]

        # 3. Análise de Alinhamento e Mapeamento de Colunas
        ja_alinhado = (
            n_canonica == adata.n_vars and var_names_sanitizados == self.genes_canonica
        )

        old_idx: list[int]
        new_idx: list[int]
        mask_ausentes: NDArray[np.bool_]

        if ja_alinhado and "presente_no_dataset" in adata.var:
            mask_ausentes = ~adata.var["presente_no_dataset"].to_numpy(dtype=bool)
            old_idx = list(range(n_canonica))
            new_idx = list(range(n_canonica))
        else:
            old_idx, new_idx, mask_ausentes = self._resolver_alinhamento_colunas(
                list(adata.var_names)
            )

        self.mask_ausentes = mask_ausentes
        ausentes_cols_idx = np.where(mask_ausentes)[0]
        n_ausentes = len(ausentes_cols_idx)

        print("[MarcadorSentinela] Resumo do alinhamento:")
        print(f"  • Células no alvo         : {n_obs:,}")
        print(f"  • Genes canônicos (total) : {n_canonica:,}")
        print(f"  • Genes medidos no alvo   : {n_canonica - n_ausentes:,}")
        print(
            f"  • Genes ausentes (sent. {self.fill_value}) : {n_ausentes:,} "
            f"({(n_ausentes / n_canonica) * 100:.2f}%)"
        )

        # 4. Geração de Rastreamento (Tracking)
        inv_map_feat = {v: k for k, v in self.map_features_alvo.items()}
        if path_tracking_csv is None:
            pasta_h5ad = os.path.dirname(os.path.abspath(path_h5ad_str))
            path_tracking_csv = os.path.join(
                pasta_h5ad, "tracking_genes_adicionados.csv"
            )
        self.gerar_tracking(
            mask_ausentes=mask_ausentes,
            path_tracking=path_tracking_csv,
            inv_map_features=inv_map_feat,
        )

        # 5. Processamento e Injeção Física em Chunks OOM-Safe
        chunks_csr: list[sp.csr_matrix] = []
        is_sparse = sp.issparse(adata.X)

        # Projeção esparsa de alinhamento se ainda não alinhado
        P_proj: sp.csr_matrix | None = None
        if not ja_alinhado:
            p_data = np.ones(len(old_idx), dtype=np.float32)
            P_proj = sp.csr_matrix(
                (p_data, (old_idx, new_idx)),
                shape=(adata.n_vars, n_canonica),
                dtype=np.float32,
            )

        print(
            f"[MarcadorSentinela] Injetando sentinelas fisicamente em chunks de {self.chunk_size} células..."
        )
        for start in range(0, n_obs, self.chunk_size):
            end = min(start + self.chunk_size, n_obs)
            chunk_slice = adata.X[start:end]

            # 5.1 Alinha para o espaço canônico
            chunk_base: sp.csr_matrix
            if ja_alinhado:
                if is_sparse and hasattr(chunk_slice, "tocsr"):
                    chunk_base = chunk_slice.tocsr()
                else:
                    chunk_base = sp.csr_matrix(chunk_slice, dtype=np.float32)
            else:
                assert P_proj is not None
                if is_sparse and hasattr(chunk_slice, "dot"):
                    chunk_base = chunk_slice.dot(P_proj).tocsr()
                else:
                    chunk_base = (
                        sp.csr_matrix(chunk_slice, dtype=np.float32).dot(P_proj).tocsr()
                    )

            # 5.2 Injeta fisicamente o valor sentinela 0.5 nas colunas ausentes
            chunk_lil = chunk_base.tolil()
            if n_ausentes > 0:
                chunk_lil[:, ausentes_cols_idx] = self.fill_value

            chunks_csr.append(chunk_lil.tocsr())
            del chunk_lil, chunk_base, chunk_slice
            if (end % (self.chunk_size * 2)) == 0 or end == n_obs:
                gc.collect()

        print("[MarcadorSentinela] Consolidando matriz CSR alinhada...")
        X_final: sp.csr_matrix = sp.vstack(chunks_csr, format="csr")
        del chunks_csr, P_proj
        gc.collect()

        # 6. Montagem dos Metadados .var e .obs
        presente_arr = ~mask_ausentes
        val_sent_arr = np.where(mask_ausentes, self.fill_value, 0.0).astype(np.float32)

        var_df = pd.DataFrame(
            {
                "presente_no_dataset": presente_arr,
                "ausente": mask_ausentes,
                "valor_sentinela": val_sent_arr,
            },
            index=pd.Index(self.genes_canonica, name="ensembl_id"),
        )

        assert isinstance(adata.obs, pd.DataFrame)
        obs_df: pd.DataFrame = adata.obs.copy()

        adata_resultado = ad.AnnData(X=X_final, obs=obs_df, var=var_df)

        # 7. Serialização em Disco
        os.makedirs(os.path.dirname(os.path.abspath(path_h5ad_str)), exist_ok=True)
        print(
            f"[MarcadorSentinela] Salvando AnnData com sentinelas em: {path_h5ad_str}"
        )
        adata_resultado.write_h5ad(path_h5ad_str, compression="gzip")
        print(f"[MarcadorSentinela] Salvo com sucesso! Shape: {adata_resultado.shape}")

        # 8. Exportação Matrix Market Opcional
        if exportar_mtx:
            if path_saida_mtx is None:
                raise ValueError(
                    "[MarcadorSentinela] 'path_saida_mtx' é obrigatório quando 'exportar_mtx=True'."
                )
            print(
                f"[MarcadorSentinela] Exportando matriz MTX para rSWeeP em: {path_saida_mtx}..."
            )
            exportador = ExportadorMTX(out_dir=path_saida_mtx)
            barcodes_list = adata_resultado.obs_names.tolist()
            exportador.exportar(
                matriz=adata_resultado,
                genes=self.genes_canonica,
                genes_referencia=self.genes_canonica,
                map_features=self.map_features_alvo,
                barcodes=barcodes_list,
                nome_etapa="Alvo com Sentinela 0.5 (MarcadorSentinela)",
                forcar=forcar,
            )

        if deve_fechar and hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

        return adata_resultado

    def obter_mascara_ausentes(self) -> NDArray[np.bool_]:
        """Retorna a máscara booleana 1D de genes ausentes gerada no alinhamento.

        Returns
        -------
        NDArray[np.bool_]
            Vetor booleano onde True indica gene ausente preenchido com sentinela.

        Raises
        ------
        RuntimeError
            Se chamado antes de executar .marcar_dataset().
        """
        if self.mask_ausentes is None:
            raise RuntimeError(
                "[MarcadorSentinela] Execute .marcar_dataset() antes de obter a máscara de ausentes."
            )
        return self.mask_ausentes
