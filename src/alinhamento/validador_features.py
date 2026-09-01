"""Módulo de validação estrita e detecção de anomalias em arquivos de features e anotações genômicas."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from re import Pattern

import anndata as ad
import polars as pl

PathType = str | os.PathLike[str]


class ValidadorFeatures:
    """Valida integridade, ordenação de colunas e compatibilidade de identificadores de features pré-alinhamento.

    Adota política estrita (Fail-Fast) para detectar precocemente:
    1. Inversão de colunas em arquivos de features (ex: Gene Symbol na col 0 e Ensembl ID na col 1).
    2. Incompatibilidade entre os identificadores do AnnData (var_names) e o mapa de features.
    3. Mismatch severo de sobreposição de genes entre os datasets de referência e alvo.

    Parameters
    ----------
    min_match_pct : float, default=50.0
        Percentual mínimo de compatibilidade esperado entre var_names e features.
    min_genes_comuns : int, default=1000
        Número mínimo aceitável de genes compartilhados entre conjuntos de dados.

    Attributes
    ----------
    min_match_pct : float
        Limiar percentual de correspondência.
    min_genes_comuns : int
        Limiar mínimo de genes em comum.
    """

    ENSEMBL_PATTERN: Pattern[str] = re.compile(r"^ENS[A-Z]*G\d{11}$", re.IGNORECASE)
    ENSEMBL_COM_VERSAO_PATTERN: Pattern[str] = re.compile(
        r"^ENS[A-Z]*G\d{11}(\.\d+)?$", re.IGNORECASE
    )

    def __init__(
        self, min_match_pct: float = 50.0, min_genes_comuns: int = 1000
    ) -> None:
        self.min_match_pct: float = float(min_match_pct)
        self.min_genes_comuns: int = int(min_genes_comuns)

    def validar_arquivo_features(
        self,
        path_features: PathType,
        dataset_name: str = "Dataset",
        sample_size: int = 1000,
    ) -> bool:
        """Valida se o arquivo de features segue o formato padrão 10x [Ensembl ID, Gene Symbol].

        Parameters
        ----------
        path_features : str | os.PathLike[str]
            Caminho para o arquivo TSV/CSV de features.
        dataset_name : str, default="Dataset"
            Nome descritivo do dataset para emissão de logs.
        sample_size : int, default=1000
            Número de linhas para amostragem inicial e inferência.

        Returns
        -------
        bool
            True se a estrutura estiver íntegra e aprovada.

        Raises
        ------
        FileNotFoundError
            Se o arquivo de features não existir.
        ValueError
            Se as colunas estiverem invertidas ou formato inconsistente.
        """
        path: str = str(path_features)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[{dataset_name}] Arquivo de features não encontrado: {path}"
            )

        sep: str = "\t" if (path.endswith(".tsv") or path.endswith(".tsv.gz")) else ","
        df: pl.DataFrame
        try:
            df = pl.read_csv(
                path,
                separator=sep,
                has_header=False,
                n_rows=sample_size,
            )
        except Exception:
            alt_sep: str = "," if sep == "\t" else "\t"
            df = pl.read_csv(
                path,
                separator=alt_sep,
                has_header=False,
                n_rows=sample_size,
            )

        if df.width < 2:
            raise ValueError(
                f"[FALHA NA VALIDAÇÃO] O arquivo de features de {dataset_name} possui apenas {df.width} coluna(s).\n"
                f"Caminho: {path}\n"
                f"Esperado: Pelo menos 2 colunas [Coluna 0: Ensembl ID, Coluna 1: Gene Symbol]."
            )

        col0_vals: list[str] = [
            str(x).strip()
            for x in df.get_column(df.columns[0]).to_list()
            if x is not None
        ]
        col1_vals: list[str] = [
            str(x).strip()
            for x in df.get_column(df.columns[1]).to_list()
            if x is not None
        ]

        col0_ensembl_count: int = sum(
            1 for x in col0_vals if self.ENSEMBL_COM_VERSAO_PATTERN.match(x)
        )
        col1_ensembl_count: int = sum(
            1 for x in col1_vals if self.ENSEMBL_COM_VERSAO_PATTERN.match(x)
        )

        col0_pct: float = (
            (col0_ensembl_count / len(col0_vals) * 100) if col0_vals else 0.0
        )
        col1_pct: float = (
            (col1_ensembl_count / len(col1_vals) * 100) if col1_vals else 0.0
        )

        # Detecção de Colunas Invertidas: Coluna 1 parece Ensembl e Coluna 0 não
        if col1_pct > 50.0 and col0_pct < 20.0:
            amostra_col0: list[str] = col0_vals[:5]
            amostra_col1: list[str] = col1_vals[:5]
            raise ValueError(
                f"\n{'=' * 70}\n"
                f"🚨 [ERRO CRÍTICO: COLUNAS INVERTIDAS EM {dataset_name.upper()}]\n"
                f"{'=' * 70}\n"
                f"O arquivo de features parece estar com as colunas invertidas!\n"
                f"Caminho: {path}\n\n"
                f"• Coluna 0 ({col0_pct:.1f}% Ensembl IDs): {amostra_col0}  <-- Parece conter 'Gene Symbols'\n"
                f"• Coluna 1 ({col1_pct:.1f}% Ensembl IDs): {amostra_col1}  <-- Parece conter 'Ensembl IDs'\n\n"
                f"DIAGNÓSTICO:\n"
                f"O pipeline espera [Coluna 0 = Ensembl ID, Coluna 1 = Gene Symbol].\n"
                f"Se as colunas estiverem trocadas, o mapeamento gerará 0 matches com a referência canônica!\n\n"
                f"COMO CORRIGIR:\n"
                f"1. Inverta as colunas no arquivo TSV/CSV ou utilize colunas corretas ao instanciar LeitorFeatures.\n"
                f"{'=' * 70}\n"
            )

        print(
            f"[ValidadorFeatures] Formato do arquivo ({dataset_name}): OK ({col0_pct:.1f}% Ensembl na Col 0)"
        )
        return True

    def validar_compatibilidade_anndata(
        self,
        path_h5ad_or_adata: PathType | ad.AnnData,
        map_features: Mapping[str, str],
        dataset_name: str = "Dataset",
    ) -> bool:
        """Valida se os var_names da matriz AnnData coincidem com as chaves ou valores do mapa de features.

        Parameters
        ----------
        path_h5ad_or_adata : str | os.PathLike[str] | ad.AnnData
            Objeto AnnData ou caminho para o arquivo .h5ad.
        map_features : Mapping[str, str]
            Mapeamento de features carregado.
        dataset_name : str, default="Dataset"
            Nome descritivo para relatórios.

        Returns
        -------
        bool
            True se a compatibilidade atender aos critérios.
        """
        if not map_features:
            raise ValueError(
                f"[{dataset_name}] O mapa de features fornecido está vazio."
            )

        var_names: list[str] | None = None
        if isinstance(path_h5ad_or_adata, (str, os.PathLike)):
            path_str: str = str(path_h5ad_or_adata)
            if not os.path.exists(path_str):
                raise FileNotFoundError(
                    f"[{dataset_name}] Matriz AnnData não encontrada: {path_str}"
                )
            if path_str.endswith(".h5ad"):
                try:
                    import h5py

                    with h5py.File(path_str, "r") as f:
                        if "var" in f:
                            var_g = f["var"]
                            index_key = var_g.attrs.get("_index", "_index")
                            if index_key in var_g:
                                raw = var_g[index_key][:]
                                var_names = [
                                    x.decode("utf-8")
                                    if isinstance(x, bytes)
                                    else str(x)
                                    for x in raw
                                ]
                            elif "_index" in var_g:
                                raw = var_g["_index"][:]
                                var_names = [
                                    x.decode("utf-8")
                                    if isinstance(x, bytes)
                                    else str(x)
                                    for x in raw
                                ]
                            elif "gene_name" in var_g:
                                raw = var_g["gene_name"][:]
                                var_names = [
                                    x.decode("utf-8")
                                    if isinstance(x, bytes)
                                    else str(x)
                                    for x in raw
                                ]
                except Exception:
                    var_names = None
            if var_names is None:
                _adata = ad.read_h5ad(path_str, backed="r")
                var_names = list(_adata.var_names)
                if hasattr(_adata, "file") and _adata.file is not None:
                    _adata.file.close()
                del _adata
        elif isinstance(path_h5ad_or_adata, ad.AnnData):
            var_names = list(path_h5ad_or_adata.var_names)
        else:
            raise TypeError(
                f"Tipo inválido para path_h5ad_or_adata: {type(path_h5ad_or_adata)}"
            )

        assert var_names is not None
        n_vars: int = len(var_names)
        if n_vars == 0:
            raise ValueError(
                f"[{dataset_name}] Matriz AnnData não contém variáveis (var_names está vazio)."
            )

        # Teste de correspondência com chaves (gene_name) ou valores (ensembl_id)
        vals_set: set[str] = set(map_features.values())
        match_chaves: int = sum(1 for g in var_names if g in map_features)
        match_valores: int = sum(1 for g in var_names if g in vals_set)

        # Consideramos o melhor tipo de match
        melhor_match: int = max(match_chaves, match_valores)
        match_pct: float = (melhor_match / n_vars) * 100.0

        if match_pct < self.min_match_pct:
            amostra_varnames: list[str] = var_names[:5]
            amostra_map_keys: list[str] = list(map_features.keys())[:5]
            amostra_map_vals: list[str] = list(map_features.values())[:5]
            raise ValueError(
                f"\n{'=' * 70}\n"
                f"🚨 [ERRO CRÍTICO: INCOMPATIBILIDADE DE IDENTIFICADORES EM {dataset_name.upper()}]\n"
                f"{'=' * 70}\n"
                f"Os var_names da matriz AnnData NÃO coincidem com o mapa de features!\n"
                f"• Total de variáveis na matriz AnnData: {n_vars:,}\n"
                f"• Total de genes mapeados encontrados: {melhor_match:,} ({match_pct:.2f}% de compatibilidade)\n"
                f"• Limiar mínimo exigido: {self.min_match_pct:.1f}%\n\n"
                f"AMOSTRAS PARA DIAGNÓSTICO:\n"
                f"• var_names da Matriz: {amostra_varnames}\n"
                f"• Chaves do Mapa (Gene Symbol): {amostra_map_keys}\n"
                f"• Valores do Mapa (Ensembl ID): {amostra_map_vals}\n\n"
                f"DIAGNÓSTICO:\n"
                f"A matriz AnnData está usando uma convenção de nomes de genes incompatível com a tabela de features.\n"
                f"Se o alinhamento continuar, quase todas as colunas serão descartadas ou preenchidas com 0.5.\n"
                f"{'=' * 70}\n"
            )

        print(
            f"[ValidadorFeatures] Compatibilidade AnnData x Features ({dataset_name}): OK ({match_pct:.1f}% de correspondência)"
        )
        return True

    def validar_sobreposicao_inter_dataset(
        self,
        map_f: Mapping[str, str],
        map_m: Mapping[str, str],
    ) -> bool:
        """Valida se existe sobreposição biológica plausível entre os dois conjuntos de features.

        Parameters
        ----------
        map_f : Mapping[str, str]
            Mapeamento de features do conjunto referência.
        map_m : Mapping[str, str]
            Mapeamento de features do conjunto alvo.

        Returns
        -------
        bool
            True se a sobreposição for suficiente.
        """
        ids_f: set[str] = set(map_f.values())
        ids_m: set[str] = set(map_m.values())
        comuns: set[str] = ids_f & ids_m

        if len(comuns) < self.min_genes_comuns:
            raise ValueError(
                f"\n{'=' * 70}\n"
                f"🚨 [ERRO CRÍTICO: SOBREPOSIÇÃO GENÔMICA ANORMALMENTE BAIXA]\n"
                f"{'=' * 70}\n"
                f"Os datasets compartilham apenas {len(comuns):,} genes em comum (esperado >= {self.min_genes_comuns:,}).\n"
                f"• Genes Referência (Fujita): {len(ids_f):,}\n"
                f"• Genes Alvo (Mathys)      : {len(ids_m):,}\n"
                f"• Genes em Comum           : {len(comuns):,}\n\n"
                f"DIAGNÓSTICO:\n"
                f"Verifique se ambos os arquivos de features utilizam a mesma versão do genoma de referência.\n"
                f"{'=' * 70}\n"
            )

        print(
            f"[ValidadorFeatures] Sobreposição Inter-Datasets: OK ({len(comuns):,} genes em comum)"
        )
        return True

    def validar_tudo(
        self,
        path_features_ref: PathType,
        path_features_alvo: PathType,
        path_h5ad_ref: PathType | None = None,
        path_h5ad_alvo: PathType | None = None,
        map_f: Mapping[str, str] | None = None,
        map_m: Mapping[str, str] | None = None,
    ) -> bool:
        """Executa a verificação completa de saúde de todos os artefatos de entrada pré-alinhamento.

        Parameters
        ----------
        path_features_ref : str | os.PathLike[str]
            Caminho do TSV de features da referência.
        path_features_alvo : str | os.PathLike[str]
            Caminho do TSV de features do alvo.
        path_h5ad_ref : str | os.PathLike[str] | None, optional
            Caminho do .h5ad da referência.
        path_h5ad_alvo : str | os.PathLike[str] | None, optional
            Caminho do .h5ad do alvo.
        map_f : Mapping[str, str] | None, optional
            Mapeamento lido de features da referência.
        map_m : Mapping[str, str] | None, optional
            Mapeamento lido de features do alvo.

        Returns
        -------
        bool
            True se todas as validações forem concluídas com sucesso.
        """
        print("\n" + "=" * 60)
        print("🔍 [ValidadorFeatures] Iniciando Auditoria de Features e AnnData")
        print("=" * 60)

        # 1. Validação de formato e ordem de colunas
        self.validar_arquivo_features(
            path_features_ref, dataset_name="Referência (Fujita)"
        )
        self.validar_arquivo_features(path_features_alvo, dataset_name="Alvo (Mathys)")

        # 2. Validação AnnData x Features (se os caminhos e mapas forem fornecidos)
        if path_h5ad_ref and map_f:
            self.validar_compatibilidade_anndata(
                path_h5ad_ref, map_f, dataset_name="Referência (Fujita)"
            )
        if path_h5ad_alvo and map_m:
            self.validar_compatibilidade_anndata(
                path_h5ad_alvo, map_m, dataset_name="Alvo (Mathys)"
            )

        # 3. Validação de sobreposição entre datasets
        if map_f and map_m:
            self.validar_sobreposicao_inter_dataset(map_f, map_m)

        print("=" * 60)
        print(
            "✅ [ValidadorFeatures] Todos os arquivos de entrada são compatíveis e válidos!"
        )
        print("=" * 60 + "\n")
        return True

    def __repr__(self) -> str:
        """Representação textual do validador de features."""
        return (
            f"ValidadorFeatures(\n"
            f"  min_match_pct    = {self.min_match_pct}%\n"
            f"  min_genes_comuns = {self.min_genes_comuns}\n"
            f")"
        )
