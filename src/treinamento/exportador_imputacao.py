"""Módulo de Exportação OOM-Safe de Imputação Cross-Dataset.

Converte e salva os dados recuperados pela Rede Hopfield Moderna em objetos
AnnData (.h5ad) comprimidos com gzip e camadas de auditoria (layers),
preservando metadados celulares (obs), gênicos (var), proveniência (uns)
e retrocompatibilidade em matriz binária NumPy (.npy).
"""

from __future__ import annotations

import datetime
import gc
import json
import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

try:
    from src.config import OUT_IMPUTACAO  # pyright: ignore[reportMissingImports]
except ImportError:
    try:
        from config import OUT_IMPUTACAO  # type: ignore[import-not-found]
    except ImportError:
        from ..config import OUT_IMPUTACAO  # type: ignore[import-not-found]

PathType = str | os.PathLike[str]
MatrixInput = NDArray[Any] | sp.spmatrix


class ExportadorImputacao:
    """Exportador OOM-Safe de matrizes imputadas cross-dataset.

    Monta estruturas AnnData completas com camadas de rastreamento de dados
    originais e máscaras booleanas de imputação, exportando arquivos .h5ad,
    .npy e relatórios JSON de auditoria.

    Parameters
    ----------
    out_dir : PathType | None, optional
        Diretório base onde os arquivos exportados serão gravados.
        Se None, utiliza `src.config.OUT_IMPUTACAO`.
    chunk_size : int, default=4096
        Número de células por lote durante o processamento em streaming
        para manter a pegada de memória RAM controlada.

    Attributes
    ----------
    out_dir : str
        Caminho resolvido do diretório de saída.
    chunk_size : int
        Tamanho de lote para processamento OOM-Safe.
    """

    def __init__(
        self,
        out_dir: PathType | None = None,
        chunk_size: int = 4096,
    ) -> None:
        self.out_dir: str = str(out_dir) if out_dir is not None else OUT_IMPUTACAO
        self.chunk_size: int = max(int(chunk_size), 128)
        os.makedirs(self.out_dir, exist_ok=True)

    def exportar(
        self,
        w_original: MatrixInput,
        w_recuperado: MatrixInput,
        genes_canonica: Sequence[str],
        map_features: Mapping[str, str] | None = None,
        adata_alvo_original: ad.AnnData | PathType | None = None,
        pred_classes: Sequence[int | str] | NDArray[Any] | None = None,
        classes_reais: Sequence[int | str] | NDArray[Any] | None = None,
        prototipos_idx: Sequence[int] | NDArray[Any] | None = None,
        info_modelo: Mapping[str, Any] | None = None,
        nome_modelo: str = "rede35",
        exportar_npy: bool = True,
        compressao: Literal["gzip", "lzf"] | None = "gzip",
        prefixo_nome: str | None = None,
        substituir_sentinela: bool = True,
        limiar_sentinela: float = 0.5,
    ) -> dict[str, Any]:
        """Executa a exportação completa da matriz imputada em .h5ad, .npy e .json.

        Parameters
        ----------
        w_original : MatrixInput
            Matriz de entrada pré-imputação contendo os valores sentinela (ex: 0.5).
        w_recuperado : MatrixInput
            Matriz de saída predita pela Rede de Hopfield (.retrieve()).
        genes_canonica : Sequence[str]
            Identificadores canônicos ordenados dos genes (Ensembl IDs).
        map_features : Mapping[str, str] | None, optional
            Dicionário {Ensembl ID: Gene Symbol} para anotação em var.
        adata_alvo_original : ad.AnnData | PathType | None, optional
            Objeto AnnData ou caminho para arquivo .h5ad de onde extrair
            metadados celulares reais (obs).
        pred_classes : Sequence[int | str] | NDArray | None, optional
            Classes celulares preditas pelo protótipo Hopfield associado.
        classes_reais : Sequence[int | str] | NDArray | None, optional
            Rótulos verdadeiros das células (ex: clo_m).
        prototipos_idx : Sequence[int] | NDArray | None, optional
            Índices dos protótipos de Hopfield associados a cada célula.
        info_modelo : Mapping[str, Any] | None, optional
            Metadados e hiperparâmetros da rede para registro em uns.
        nome_modelo : str, default="rede35"
            Identificador textual do modelo de inferência.
        exportar_npy : bool, default=True
            Se True, gera também o arquivo .npy retrocompatível.
        compressao : str, default="gzip"
            Algoritmo de compressão para gravação do arquivo AnnData.
        prefixo_nome : str | None, optional
            Prefixo customizado dos arquivos gerados.
        substituir_sentinela : bool, default=True
            Se True, substitui apenas os valores iguais a `limiar_sentinela`,
            mantendo a expressão original onde ela já existia.
        limiar_sentinela : float, default=0.5
            Valor indicativo de gene ausente na matriz de entrada.

        Returns
        -------
        dict[str, Any]
            Dicionário contendo os caminhos dos artefatos gerados e estatísticas de imputação.
        """
        n_celulas: int = w_original.shape[0]
        n_genes: int = w_original.shape[1]

        if len(genes_canonica) != n_genes:
            raise ValueError(
                f"[ExportadorImputacao] Inconsistência de dimensões: {len(genes_canonica)} genes "
                f"fornecidos para matriz com {n_genes} colunas."
            )

        if w_recuperado.shape != (n_celulas, n_genes):
            raise ValueError(
                f"[ExportadorImputacao] Formatos incompatíveis: w_original {w_original.shape} "
                f"vs w_recuperado {w_recuperado.shape}."
            )

        base_nome: str
        if prefixo_nome is not None and str(prefixo_nome).strip():
            base_nome = str(prefixo_nome).strip()
        else:
            base_nome = f"mathys_imputado_fujita_{nome_modelo}_{n_genes}genes"

        path_h5ad: str = os.path.join(self.out_dir, f"{base_nome}.h5ad")
        path_npy: str = os.path.join(self.out_dir, f"{base_nome}.npy")
        path_relatorio: str = os.path.join(self.out_dir, f"relatorio_{base_nome}.json")

        print(f"\n[ExportadorImputacao] Iniciando exportação para '{base_nome}'...")
        print(f"  Dimensões: {n_celulas} células x {n_genes} genes")
        print(f"  Modo streaming com lotes de {self.chunk_size} células")

        # Configuração de buffer OOM-Safe para NPY via memmap se requisitado
        memmap_npy: np.memmap[Any, np.dtype[np.float32]] | None = None
        if exportar_npy:
            memmap_npy = np.lib.format.open_memmap(
                path_npy,
                mode="w+",
                dtype=np.float32,
                shape=(n_celulas, n_genes),
            )

        csr_x_list: list[sp.csr_matrix] = []
        csr_orig_list: list[sp.csr_matrix] = []
        csr_mask_list: list[sp.csr_matrix] = []

        total_sentinelas: int = 0
        total_imputados_zero: int = 0
        total_imputados_um: int = 0
        celula_imputados_contagem = np.zeros(n_celulas, dtype=np.int32)
        gene_imputados_contagem = np.zeros(n_genes, dtype=np.int64)

        w_orig_csr: sp.csr_matrix | None = None
        w_orig_np: NDArray[Any] | None = None
        if sp.issparse(w_original):
            w_orig_csr = sp.csr_matrix(w_original)
        else:
            w_orig_np = np.asarray(w_original)

        w_rec_csr: sp.csr_matrix | None = None
        w_rec_np: NDArray[Any] | None = None
        if sp.issparse(w_recuperado):
            w_rec_csr = sp.csr_matrix(w_recuperado)
        else:
            w_rec_np = np.asarray(w_recuperado)

        for start in range(0, n_celulas, self.chunk_size):
            end: int = min(start + self.chunk_size, n_celulas)

            # Extração de chunk denso float32
            chunk_orig: NDArray[np.float32]
            if w_orig_csr is not None:
                chunk_orig = w_orig_csr[start:end].toarray().astype(np.float32)
            else:
                assert w_orig_np is not None
                chunk_orig = np.asarray(w_orig_np[start:end], dtype=np.float32)

            chunk_rec: NDArray[np.float32]
            if w_rec_csr is not None:
                chunk_rec = w_rec_csr[start:end].toarray().astype(np.float32)
            else:
                assert w_rec_np is not None
                chunk_rec = np.asarray(w_rec_np[start:end], dtype=np.float32)

            mask_sentinela: NDArray[np.bool_] = np.isclose(chunk_orig, limiar_sentinela)

            chunk_final: NDArray[np.float32]
            if substituir_sentinela:
                chunk_final = np.where(mask_sentinela, chunk_rec, chunk_orig).astype(
                    np.float32
                )
            else:
                chunk_final = chunk_rec.copy().astype(np.float32)

            # Coleta métricas de imputação do chunk
            n_sentinelas_chunk = int(np.sum(mask_sentinela))
            total_sentinelas += n_sentinelas_chunk

            val_imputados = chunk_final[mask_sentinela]
            total_imputados_zero += int(np.sum(val_imputados == 0.0))
            total_imputados_um += int(np.sum(val_imputados > 0.0))

            celula_imputados_contagem[start:end] = np.sum(mask_sentinela, axis=1)
            gene_imputados_contagem += np.sum(mask_sentinela, axis=0)

            # Grava no memmap .npy se ativo
            if memmap_npy is not None:
                memmap_npy[start:end] = chunk_final

            # Converte em matriz esparsa CSR para acumulação
            csr_x_list.append(sp.csr_matrix(chunk_final))
            csr_orig_list.append(sp.csr_matrix(chunk_orig))
            csr_mask_list.append(sp.csr_matrix(mask_sentinela.astype(np.float32)))

            del chunk_orig, chunk_rec, mask_sentinela, chunk_final
            gc.collect()

        if memmap_npy is not None:
            memmap_npy.flush()
            del memmap_npy
            gc.collect()
            print(f"  Salvo NPY (.npy): {path_npy}")

        # Concatenação esparsa vertical OOM-Safe
        print("  Concatenando camadas esparsas CSR...")
        X_csr: sp.csr_matrix = sp.csr_matrix(sp.vstack(csr_x_list, format="csr"))
        del csr_x_list
        gc.collect()

        orig_csr: sp.csr_matrix = sp.csr_matrix(sp.vstack(csr_orig_list, format="csr"))
        del csr_orig_list
        gc.collect()

        mask_csr: sp.csr_matrix = sp.csr_matrix(sp.vstack(csr_mask_list, format="csr"))
        del csr_mask_list
        gc.collect()

        # ---------------------------------------------------------------------
        # Construção dos Metadados Celulares (obs)
        # ---------------------------------------------------------------------
        obs_df: pd.DataFrame
        if adata_alvo_original is not None:
            raw_obs: Any
            if isinstance(adata_alvo_original, (str, os.PathLike)):
                adata_src = ad.read_h5ad(str(adata_alvo_original), backed="r")
                raw_obs = adata_src.obs.copy()
                if hasattr(adata_src, "file") and adata_src.file is not None:
                    adata_src.file.close()
                del adata_src
            else:
                raw_obs = adata_alvo_original.obs.copy()

            if isinstance(raw_obs, pd.DataFrame):
                obs_df = raw_obs.copy()
            else:
                obs_df = pd.DataFrame(index=[f"celula_{i}" for i in range(n_celulas)])

            if len(obs_df) != n_celulas:
                print(
                    f"  [Aviso] Tamanho de obs ({len(obs_df)}) diverge de n_celulas ({n_celulas}). "
                    f"Ajustando index padrão."
                )
                obs_df = pd.DataFrame(index=[f"celula_{i}" for i in range(n_celulas)])
        else:
            obs_df = pd.DataFrame(index=[f"celula_{i}" for i in range(n_celulas)])

        if classes_reais is not None:
            obs_df["tipo_celular_real"] = np.asarray(classes_reais)
        if pred_classes is not None:
            obs_df["tipo_predito_hopfield"] = np.asarray(pred_classes)
        if prototipos_idx is not None:
            obs_df["prototipo_hopfield_idx"] = np.asarray(prototipos_idx)

        obs_df["n_genes_imputados"] = celula_imputados_contagem
        obs_df["pct_genes_imputados"] = (
            celula_imputados_contagem / float(n_genes)
        ) * 100.0

        # ---------------------------------------------------------------------
        # Construção dos Metadados Gênicos (var)
        # ---------------------------------------------------------------------
        var_data: dict[str, Any] = {}
        if map_features is not None:
            var_data["gene_symbol"] = [map_features.get(g, g) for g in genes_canonica]
        else:
            var_data["gene_symbol"] = list(genes_canonica)

        # Flag booleana: se todas as células (ou maioria estrita) eram sentinelas
        var_data["gene_imputado"] = gene_imputados_contagem > 0
        var_data["n_celulas_imputadas"] = gene_imputados_contagem
        var_df: pd.DataFrame = pd.DataFrame(
            var_data, index=pd.Index(genes_canonica, name="ensembl_id")
        )

        # ---------------------------------------------------------------------
        # Metadados Estruturais e de Proveniência (uns)
        # ---------------------------------------------------------------------
        uns_dict: dict[str, Any] = {
            "modelo": str(nome_modelo),
            "tipo_imputacao": "Hopfield_CrossDataset",
            "dataset_referencia": "Fujita",
            "dataset_alvo": "Mathys",
            "data_execucao": datetime.datetime.now().isoformat(),
            "n_celulas": int(n_celulas),
            "n_genes": int(n_genes),
            "total_sentinelas_resolvidos": int(total_sentinelas),
        }
        if info_modelo is not None:
            uns_dict["parametros_modelo"] = {
                str(k): (v if isinstance(v, (int, float, str, bool, list)) else str(v))
                for k, v in info_modelo.items()
            }

        # ---------------------------------------------------------------------
        # Montagem do Objeto AnnData e Gravação em Disco
        # ---------------------------------------------------------------------
        print("  Montando AnnData e gravando .h5ad (Gzip)...")
        adata_imp = ad.AnnData(
            X=X_csr,
            obs=obs_df,
            var=var_df,
            uns=uns_dict,
        )
        adata_imp.layers["original"] = orig_csr
        adata_imp.layers["mascara_imputada"] = mask_csr

        adata_imp.write_h5ad(path_h5ad, compression=compressao)
        print(f"  Salvo AnnData (.h5ad): {path_h5ad}")

        del X_csr, orig_csr, mask_csr, adata_imp
        gc.collect()

        # ---------------------------------------------------------------------
        # Relatório Estatístico em JSON
        # ---------------------------------------------------------------------
        total_elementos: int = n_celulas * n_genes
        pct_imputado_global: float = (
            (float(total_sentinelas) / float(total_elementos)) * 100.0
            if total_elementos > 0
            else 0.0
        )

        dist_preditas: dict[str, int] = {}
        if pred_classes is not None:
            vals, counts = np.unique(np.asarray(pred_classes), return_counts=True)
            dist_preditas = {str(v): int(c) for v, c in zip(vals, counts, strict=True)}

        relatorio: dict[str, Any] = {
            "modelo": nome_modelo,
            "data_execucao": uns_dict["data_execucao"],
            "dimensoes": {
                "n_celulas": n_celulas,
                "n_genes": n_genes,
                "total_elementos": total_elementos,
            },
            "estatisticas_imputacao": {
                "total_sentinelas_resolvidos": total_sentinelas,
                "percentual_imputado_global": round(pct_imputado_global, 4),
                "valores_resolvidos_para_zero": total_imputados_zero,
                "valores_resolvidos_para_um": total_imputados_um,
                "proporcao_zeros": (
                    round(total_imputados_zero / total_sentinelas, 4)
                    if total_sentinelas > 0
                    else 0.0
                ),
                "proporcao_uns": (
                    round(total_imputados_um / total_sentinelas, 4)
                    if total_sentinelas > 0
                    else 0.0
                ),
                "genes_completamente_ausentes_no_alvo": int(
                    np.sum(gene_imputados_contagem == n_celulas)
                ),
                "genes_com_alguma_imputacao": int(np.sum(gene_imputados_contagem > 0)),
            },
            "distribuicao_predicoes_hopfield": dist_preditas,
            "arquivos_gerados": {
                "h5ad": path_h5ad,
                "npy": path_npy if exportar_npy else None,
                "relatorio_json": path_relatorio,
            },
        }

        with open(path_relatorio, "w", encoding="utf-8") as f_json:
            json.dump(relatorio, f_json, indent=2, ensure_ascii=False)
        print(f"  Salvo Relatório de Auditoria (.json): {path_relatorio}")
        print("[ExportadorImputacao] Exportação finalizada com sucesso. [Ok]\n")

        return relatorio
