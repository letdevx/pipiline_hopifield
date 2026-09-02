"""Módulo de Validação Biológica e Estatística de Imputação Cross-Dataset.

Audita a integridade, densidade de ativação e coerência biológica dos dados
imputados pela Rede Hopfield Moderna, avaliando marcadores canônicos de linhagens
celulares do tecido cerebral e métricas quantitativas globais.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

PathType = str | os.PathLike[str]

# Marcadores canônicos de referência para o tecido cerebral humano (classes 1 a 7)
MARCADORES_CANONICOS_CEREBRO: dict[int, list[str]] = {
    1: ["GFAP", "AQP4", "ALDH1L1", "SLC1A2"],  # Astrocytes
    2: ["CLDN5", "PECAM1", "FLT1", "VWF"],  # Endothelial
    3: ["SLC17A7", "CAMK2A", "NRGN", "SATB2"],  # Excitatory Neurons
    4: ["GAD1", "GAD2", "SLC32A1", "PVALB", "SST"],  # Inhibitory Neurons
    5: ["CSF1R", "CD74", "AIF1", "CX3CR1", "C1QA"],  # Microglia
    6: ["MBP", "MOG", "PLP1", "MAG"],  # Oligodendrocytes
    7: ["PDGFRA", "VCAN", "CSPG4", "SOX10"],  # OPCs
}

NOMES_CLASSES_CEREBRO: dict[int, str] = {
    1: "Astrocyte",
    2: "Endothelial / Outros",
    3: "Excitatory Neurons",
    4: "Inhibitory Neurons",
    5: "Microglia",
    6: "Oligodendrocytes",
    7: "OPCs",
}


def _is_sparse_like(mat: Any) -> bool:
    """Verifica se a matriz é esparsa SciPy ou dataset esparso backed do AnnData."""
    if mat is None:
        return False
    if sp.issparse(mat):
        return True
    return bool(
        hasattr(mat, "to_memory") and getattr(mat, "format", None) in ("csr", "csc")
    )


def _para_csr_matrix(mat: Any) -> sp.csr_matrix:
    """Converte transparentemente matrizes em memória ou datasets backed para sp.csr_matrix."""
    if hasattr(mat, "to_memory"):
        return sp.csr_matrix(mat.to_memory())
    if sp.issparse(mat):
        return sp.csr_matrix(mat)
    return sp.csr_matrix(mat)


def _para_array_denso(mat: Any) -> NDArray[Any]:
    """Converte representações densas (numpy, h5py Dataset, backed ArrayView) para NDArray."""
    if hasattr(mat, "to_memory"):
        return np.asarray(mat.to_memory().toarray())
    if sp.issparse(mat):
        return np.asarray(mat.toarray())
    if (
        hasattr(mat, "shape")
        and hasattr(mat, "__getitem__")
        and not isinstance(mat, np.ndarray)
    ):
        return np.asarray(mat[:])
    return np.asarray(mat)


def _extrair_coluna_1d(mat: Any, col: int) -> NDArray[Any]:
    """Extrai uma única coluna 1D de qualquer matriz (SciPy, NumPy, AnnData backed CSR/CSC ou Dataset)."""
    if mat is None:
        return np.array([])
    col_slice = mat[:, col]
    if hasattr(col_slice, "toarray"):
        return np.asarray(col_slice.toarray()).ravel()
    if hasattr(col_slice, "to_memory"):
        return np.asarray(col_slice.to_memory().toarray()).ravel()
    if sp.issparse(col_slice):
        return np.asarray(col_slice.toarray()).ravel()
    return np.asarray(col_slice).ravel()


class ValidadorImputacao:
    """Validador e auditor de qualidade e biologia da imputação cross-dataset.

    Audita em três níveis:
    1. Métricas quantitativas globais (total de sentinelas resolvidos, % 1s e 0s).
    2. Auditoria biológica celular com marcadores canônicos conhecidos.
    3. Exibição e exportação estruturada de relatórios de conformidade.
    """

    def __init__(self) -> None:
        pass

    def auditar_imputacao_global(
        self,
        adata: ad.AnnData,
        mask_ausentes: NDArray[np.bool_] | Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Calcula métricas quantitativas globais da imputação no objeto AnnData.

        Parameters
        ----------
        adata : ad.AnnData
            Objeto AnnData imputado contendo camadas 'mascara_imputada' e opcionalmente 'original'.
        mask_ausentes : NDArray[bool] | Sequence[int] | None, optional
            Máscara booleana das colunas de genes ausentes originalmente na plataforma alvo.

        Returns
        -------
        dict[str, Any]
            Dicionário com estatísticas quantitativas globais da imputação.
        """
        n_celulas = adata.n_obs
        n_genes = adata.n_vars
        total_coordenadas = n_celulas * n_genes

        mask_csr: sp.csr_matrix | None = None
        if "mascara_imputada" in adata.layers:
            mask_csr = _para_csr_matrix(adata.layers["mascara_imputada"])

        total_sentinelas: int = 0
        total_ativados: int = 0
        total_inativados: int = 0

        X_mat = adata.X
        is_sparse = _is_sparse_like(X_mat)

        if mask_csr is not None and mask_csr.nnz > 0:
            total_sentinelas = int(mask_csr.nnz)
            # Elementos onde mask_csr é 1.0
            if is_sparse:
                X_csr = _para_csr_matrix(X_mat)
                # Interseção esparsa de X com a máscara
                X_imputados = X_csr.multiply(mask_csr)
                total_ativados = int(np.sum(X_imputados.data > 0.0))
                total_inativados = total_sentinelas - total_ativados
            else:
                X_arr = _para_array_denso(X_mat)
                mask_dense = mask_csr.toarray().astype(bool)
                vals = X_arr[mask_dense]
                total_ativados = int(np.sum(vals > 0.0))
                total_inativados = total_sentinelas - total_ativados
        elif mask_ausentes is not None:
            mask_arr = np.asarray(mask_ausentes)
            if mask_arr.dtype == bool:
                cols_ausentes = np.where(mask_arr)[0]
            else:
                cols_ausentes = np.asarray(mask_arr, dtype=int)

            n_ausentes = len(cols_ausentes)
            total_sentinelas = n_celulas * n_ausentes

            if is_sparse:
                X_csr = _para_csr_matrix(X_mat)[:, cols_ausentes]
                total_ativados = int(np.sum(X_csr.data > 0.0))
                total_inativados = total_sentinelas - total_ativados
            else:
                X_arr = _para_array_denso(X_mat)[:, cols_ausentes]
                total_ativados = int(np.sum(X_arr > 0.0))
                total_inativados = int(np.sum(X_arr == 0.0))

        pct_ativados = (
            (total_ativados / total_sentinelas * 100.0) if total_sentinelas > 0 else 0.0
        )
        pct_inativados = (
            (total_inativados / total_sentinelas * 100.0)
            if total_sentinelas > 0
            else 0.0
        )
        pct_global_imputado = (
            (total_sentinelas / total_coordenadas * 100.0)
            if total_coordenadas > 0
            else 0.0
        )

        conf_media: float | None = None
        if "probabilidade_imputada" in adata.layers:
            prob_raw = adata.layers["probabilidade_imputada"]
            if _is_sparse_like(prob_raw):
                prob_mat = _para_csr_matrix(prob_raw)
                if mask_csr is not None and mask_csr.nnz > 0:
                    prob_imp = prob_mat.multiply(mask_csr)
                    # Média sobre as posições imputadas
                    conf_media = float(prob_imp.data.sum() / max(1, total_sentinelas))
                else:
                    conf_media = (
                        float(prob_mat.data.mean()) if prob_mat.nnz > 0 else 0.0
                    )
            else:
                prob_arr = _para_array_denso(prob_raw)
                conf_media = float(np.mean(prob_arr))

        # Esparsidade global da matriz final
        if is_sparse:
            nnz_final = int(_para_csr_matrix(X_mat).nnz)
        else:
            nnz_final = int(np.sum(_para_array_denso(X_mat) > 0.0))
        densidade_final = (
            (nnz_final / total_coordenadas * 100.0) if total_coordenadas > 0 else 0.0
        )

        return {
            "n_celulas": n_celulas,
            "n_genes": n_genes,
            "total_coordenadas": total_coordenadas,
            "total_sentinelas_resolvidos": total_sentinelas,
            "percentual_imputado_global": round(pct_global_imputado, 4),
            "posicoes_ativadas": total_ativados,
            "percentual_ativadas": round(pct_ativados, 2),
            "posicoes_inativadas": total_inativados,
            "percentual_inativadas": round(pct_inativados, 2),
            "confianca_media_imputacao": round(conf_media, 4)
            if conf_media is not None
            else None,
            "densidade_final_matriz_pct": round(densidade_final, 2),
        }

    def auditar_marcadores_biologicos(
        self,
        adata: ad.AnnData,
        classes_reais: Sequence[int | str] | NDArray[Any],
        map_features: Mapping[str, str] | None = None,
        marcadores_custom: Mapping[int, list[str]] | None = None,
    ) -> pd.DataFrame:
        """Audita se genes marcadores canônicos ausentes foram imputados coerentemente por tipo celular.

        Parameters
        ----------
        adata : ad.AnnData
            Objeto AnnData imputado.
        classes_reais : Sequence[int | str] | NDArray
            Vetor com os rótulos verdadeiros de classe de cada célula (ex: clo_alvo).
        map_features : Mapping[str, str] | None, optional
            Mapeamento {Ensembl ID: Gene Symbol}.
        marcadores_custom : Mapping[int, list[str]] | None, optional
            Dicionário customizado {classe: [marcadores]}. Se None, usa marcadores do cérebro.

        Returns
        -------
        pd.DataFrame
            Tabela com a auditoria de cada marcador (presença, taxa de ativação na classe alvo vs outras).
        """
        dict_marcadores = marcadores_custom or MARCADORES_CANONICOS_CEREBRO
        labels = np.asarray(classes_reais)

        # Constrói mapeamento reverso: Gene Symbol -> Coluna Index
        symbol_to_col: dict[str, int] = {}
        for col_idx, eid in enumerate(adata.var_names):
            sym = map_features.get(eid, eid) if map_features else eid
            symbol_to_col[sym.upper()] = col_idx
            symbol_to_col[eid.upper()] = col_idx

        # Identifica quais genes são considerados imputados em var
        genes_imputados_flags: NDArray[np.bool_]
        if "gene_imputado" in adata.var:
            genes_imputados_flags = adata.var["gene_imputado"].to_numpy().astype(bool)
        elif "mascara_imputada" in adata.layers:
            m_csr = _para_csr_matrix(adata.layers["mascara_imputada"])
            genes_imputados_flags = np.asarray(m_csr.sum(axis=0)).ravel() > 0
        else:
            genes_imputados_flags = np.zeros(adata.n_vars, dtype=bool)

        prob_mat = adata.layers.get("probabilidade_imputada")

        linhas: list[dict[str, Any]] = []

        for classe_id, marcadores in dict_marcadores.items():
            nome_classe = NOMES_CLASSES_CEREBRO.get(classe_id, f"Classe {classe_id}")
            mask_classe = labels == classe_id
            mask_outras = ~mask_classe
            n_cel_alvo = int(np.sum(mask_classe))
            n_cel_outras = int(np.sum(mask_outras))

            for gene in marcadores:
                col = symbol_to_col.get(gene.upper())
                if col is None:
                    linhas.append(
                        {
                            "classe_id": classe_id,
                            "tipo_celular": nome_classe,
                            "gene_marcador": gene,
                            "status_no_dataset": "Não Encontrado",
                            "foi_imputado": False,
                            "n_celulas_tipo": n_cel_alvo,
                            "ativacao_no_tipo_pct": 0.0,
                            "ativacao_outros_pct": 0.0,
                            "razao_especificidade": 0.0,
                            "confianca_media": None,
                        }
                    )
                    continue

                foi_imp = bool(genes_imputados_flags[col])
                col_data = _extrair_coluna_1d(adata.X, col)

                val_alvo = col_data[mask_classe] if n_cel_alvo > 0 else np.array([])
                val_outras = col_data[mask_outras] if n_cel_outras > 0 else np.array([])

                pct_alvo = (
                    float(np.mean(val_alvo > 0.0) * 100.0) if len(val_alvo) > 0 else 0.0
                )
                pct_outras = (
                    float(np.mean(val_outras > 0.0) * 100.0)
                    if len(val_outras) > 0
                    else 0.0
                )

                razao = (
                    (pct_alvo / max(0.01, pct_outras))
                    if pct_outras > 0
                    else (pct_alvo / 0.01)
                )

                conf_gene: float | None = None
                if prob_mat is not None:
                    col_prob_all = _extrair_coluna_1d(prob_mat, col)
                    col_prob = (
                        col_prob_all[mask_classe]
                        if len(col_prob_all) > 0
                        else np.array([])
                    )
                    conf_gene = (
                        round(float(np.mean(col_prob)), 4)
                        if len(col_prob) > 0
                        else None
                    )

                status = "Imputado pela Rede" if foi_imp else "Original Observado"

                linhas.append(
                    {
                        "classe_id": classe_id,
                        "tipo_celular": nome_classe,
                        "gene_marcador": gene,
                        "status_no_dataset": status,
                        "foi_imputado": foi_imp,
                        "n_celulas_tipo": n_cel_alvo,
                        "ativacao_no_tipo_pct": round(pct_alvo, 2),
                        "ativacao_outros_pct": round(pct_outras, 2),
                        "razao_especificidade": round(razao, 2),
                        "confianca_media": conf_gene,
                    }
                )

        return pd.DataFrame(linhas)

    def imprimir_relatorio(
        self,
        metricas_globais: dict[str, Any],
        df_marcadores: pd.DataFrame | None = None,
    ) -> None:
        """Renderiza o relatório de auditoria e validação no terminal e output de células.

        Parameters
        ----------
        metricas_globais : dict[str, Any]
            Dicionário retornado por `auditar_imputacao_global()`.
        df_marcadores : pd.DataFrame | None, optional
            Tabela retornada por `auditar_marcadores_biologicos()`.
        """
        print("\n" + "=" * 70)
        print("  RELATÓRIO DE AUDITORIA E VALIDAÇÃO DA IMPUTAÇÃO CROSS-DATASET")
        print("=" * 70)

        # 1. Métricas Globais Quantitativas
        tot_sent = metricas_globais["total_sentinelas_resolvidos"]
        tot_coord = metricas_globais["total_coordenadas"]
        ativ = metricas_globais["posicoes_ativadas"]
        inativ = metricas_globais["posicoes_inativadas"]
        pct_at = metricas_globais["percentual_ativadas"]
        pct_in = metricas_globais["percentual_inativadas"]
        pct_imp = metricas_globais["percentual_imputado_global"]

        print("\n[1. Métricas Quantitativas Globais]")
        print(
            f"  • Dimensões da matriz      : {metricas_globais['n_celulas']:,} células × {metricas_globais['n_genes']:,} genes ({tot_coord:,} coordenadas)"
        )
        print(
            f"  • Sentinelas resolvidos    : {tot_sent:,} ({pct_imp:.2f}% do espaço total)"
        )
        print(f"  • Posições ativadas (1.0)  : {ativ:,} ({pct_at:.2f}%)")
        print(f"  • Posições inativadas (0.0): {inativ:,} ({pct_in:.2f}%)")
        print(
            f"  • Densidade pós-imputação  : {metricas_globais['densidade_final_matriz_pct']:.2f}% de 1s na matriz inteira"
        )

        if metricas_globais.get("confianca_media_imputacao") is not None:
            print(
                f"  • Confiança média da rede  : {metricas_globais['confianca_media_imputacao']:.4f}"
            )

        # 2. Auditoria Biológica de Marcadores
        if df_marcadores is not None and not df_marcadores.empty:
            print("\n[2. Auditoria Biológica por Linhagem Celular]")
            imputados_df = df_marcadores[df_marcadores["foi_imputado"]]
            originais_df = df_marcadores[~df_marcadores["foi_imputado"]]

            print(f"  • Marcadores canônicos avaliados : {len(df_marcadores)}")
            print(f"  • Marcadores que foram imputados : {len(imputados_df)}")
            print(f"  • Marcadores originais medidos   : {len(originais_df)}")

            print("\n  Amostra de Marcadores Imputados pela Hopfield:")
            print(
                f"  {'Tipo Celular':<20} | {'Gene':<8} | {'Ativação Alvo':<14} | {'Outros':<8} | {'Especificidade':<14}"
            )
            print("  " + "-" * 72)
            alvo_imputados = (
                imputados_df if not imputados_df.empty else df_marcadores.head(10)
            )
            for _, r in alvo_imputados.head(12).iterrows():
                print(
                    f"  {r['tipo_celular'][:19]:<20} | "
                    f"{r['gene_marcador']:<8} | "
                    f"{r['ativacao_no_tipo_pct']:>6.1f}%        | "
                    f"{r['ativacao_outros_pct']:>5.1f}%  | "
                    f"{r['razao_especificidade']:>6.2f}x"
                )

        print("\n" + "=" * 70 + "\n")

    def exportar_relatorio(
        self,
        path_relatorio_json: PathType,
        metricas_globais: dict[str, Any],
        df_marcadores: pd.DataFrame | None = None,
    ) -> None:
        """Salva as métricas de validação integradas ao arquivo JSON de auditoria.

        Parameters
        ----------
        path_relatorio_json : PathType
            Caminho do arquivo JSON a ser atualizado ou criado.
        metricas_globais : dict[str, Any]
            Métricas calculadas por `auditar_imputacao_global()`.
        df_marcadores : pd.DataFrame | None, optional
            Tabela de marcadores calculada por `auditar_marcadores_biologicos()`.
        """
        dados_existentes: dict[str, Any] = {}
        if os.path.exists(path_relatorio_json):
            try:
                with open(path_relatorio_json, encoding="utf-8") as f:
                    dados_existentes = json.load(f)
            except Exception:
                dados_existentes = {}

        dados_existentes["auditoria_validacao"] = {
            "metricas_globais": metricas_globais,
            "marcadores_auditados": (
                df_marcadores.to_dict(orient="records")
                if df_marcadores is not None
                else []
            ),
        }

        os.makedirs(
            os.path.dirname(os.path.abspath(path_relatorio_json)), exist_ok=True
        )
        with open(path_relatorio_json, "w", encoding="utf-8") as f_out:
            json.dump(dados_existentes, f_out, indent=2, ensure_ascii=False)
