"""Gerador de Synthetic Ground Truth para Auditoria de scRNA-Seq e Redes Hopfield.

Gera micro-datasets humano-verificáveis (padrão 12 células × 8 genes com 3 tipos celulares)
e matrizes de alta escala (até 36.591 genes) com injeção controlada de dropouts estocásticos,
genes ausentes e ruído sequencial, viabilizando provas reais sem suposições ad-hoc.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray

SaidaMatriz = NDArray[np.float32] | pd.DataFrame | ad.AnnData | sp.csr_matrix


class GeradorGroundTruthSintetico:
    """Gerador parametrizável de matrizes de controle para benchmarking de bioinformática e IA.

    Parameters
    ----------
    n_celulas : int, default=12
        Número total de células na matriz gerada.
    n_genes : int, default=8
        Número total de genes (dimensão de features).
    n_classes : int, default=3
        Número de tipos celulares distintos simulados.
    seed : int, default=42
        Semente para o gerador de números pseudoaleatórios.

    Attributes
    ----------
    n_celulas : int
        Total de células geradas.
    n_genes : int
        Total de genes na assinatura.
    n_classes : int
        Total de classes biológicas ativas.
    seed : int
        Semente de reprodutibilidade.
    rng : np.random.Generator
        Gerador de números aleatórios do NumPy.
    gene_names : list[str]
        Lista com os identificadores canônicos de cada gene.
    cell_names : list[str]
        Lista com os identificadores e anotações de cada célula.
    labels : NDArray[np.int_]
        Array com o tipo celular numérico correspondente a cada célula.
    matriz_pura : NDArray[np.float32]
        Matriz (n_celulas × n_genes) contendo o Ground Truth exato sem ruído.
    """

    def __init__(
        self,
        n_celulas: int = 12,
        n_genes: int = 8,
        n_classes: int = 3,
        seed: int = 42,
    ) -> None:
        self.n_celulas: int = n_celulas
        self.n_genes: int = n_genes
        self.n_classes: int = min(n_classes, n_celulas)
        self.seed: int = seed

        self.rng: np.random.Generator = np.random.default_rng(seed)
        self.gene_names: list[str] = [f"G{j}" for j in range(n_genes)]
        self.cell_names: list[str] = []
        self.labels: NDArray[np.int_] = np.empty(0, dtype=int)
        self.matriz_pura: NDArray[np.float32] = np.zeros(
            (n_celulas, n_genes), dtype=np.float32
        )

        self._construir_ground_truth()

    def _construir_ground_truth(self) -> None:
        """Constrói internamente a matriz pura em blocos bem definidos por classe biológica."""
        matriz: NDArray[np.float32] = np.zeros(
            (self.n_celulas, self.n_genes), dtype=np.float32
        )

        # Alocação balanceada de células para as classes
        celulas_por_classe: int = int(np.ceil(self.n_celulas / self.n_classes))
        genes_por_classe: int = int(np.ceil(self.n_genes / self.n_classes))

        nomes_classes: list[str] = [
            "TipoA",
            "TipoB",
            "TipoC",
            "TipoD",
            "TipoE",
            "TipoF",
            "TipoG",
        ]
        labels_list: list[int] = []
        self.cell_names = []

        for idx in range(self.n_celulas):
            cls_idx: int = min(idx // celulas_por_classe, self.n_classes - 1)
            nome_cls: str = (
                nomes_classes[cls_idx]
                if cls_idx < len(nomes_classes)
                else f"Tipo{cls_idx}"
            )

            self.cell_names.append(f"C{idx}_{nome_cls}")
            labels_list.append(
                cls_idx + 1
            )  # Classes 1, 2, 3... (compatível com clo do pipeline)

            # Ativa um bloco específico de genes para cada classe celular (Assinatura Transcricional)
            g_start: int = cls_idx * genes_por_classe
            g_end: int = min(g_start + genes_por_classe, self.n_genes)
            if g_start < self.n_genes:
                matriz[idx, g_start:g_end] = 1.0
            else:
                # Se houver mais classes que blocos de genes, ativa genes de forma cíclica
                matriz[idx, cls_idx % self.n_genes] = 1.0

        self.labels = np.array(labels_list, dtype=int)
        self.matriz_pura = matriz

    def gerar_matriz_pura(
        self,
        formato: Literal["numpy", "dataframe", "anndata", "esparso"] = "numpy",
        contagem_continua: bool = False,
    ) -> SaidaMatriz:
        """Retorna o Ground Truth real em estado perfeito sem dropouts.

        Parameters
        ----------
        formato : {"numpy", "dataframe", "anndata", "esparso"}, default="numpy"
            Formato estrutural do retorno.
        contagem_continua : bool, default=False
            Se True, simula contagens contínuas ao invés de {0, 1}.

        Returns
        -------
        SaidaMatriz
            Matriz de controle sem perturbações no formato especificado.
        """
        data: NDArray[np.float32] = self.matriz_pura.copy()
        if contagem_continua:
            mask: NDArray[np.bool_] = data > 0
            contagens: NDArray[np.float32] = self.rng.uniform(
                5.0, 25.0, size=data.shape
            ).astype(np.float32)
            data[mask] = contagens[mask]

        return self._formatar_saida(data, formato)

    def gerar_matriz_perturbada(
        self,
        taxa_dropout: float = 0.15,
        genes_remover: Sequence[str] | None = None,
        dropouts_deterministicos: Sequence[tuple[int, int]] | None = None,
        formato: Literal["numpy", "dataframe", "anndata", "esparso"] = "numpy",
    ) -> SaidaMatriz:
        """Gera uma matriz com perturbações bio-realistas (dropouts e genes ausentes).

        Parameters
        ----------
        taxa_dropout : float, default=0.15
            Probabilidade estocástica de dropout em genes ativos.
        genes_remover : Sequence[str] | None, optional
            Lista de nomes de genes para suprimir integralmente.
        dropouts_deterministicos : Sequence[tuple[int, int]] | None, optional
            Coordenadas explícitas (idx_celula, idx_gene) para anular.
        formato : {"numpy", "dataframe", "anndata", "esparso"}, default="numpy"
            Formato do retorno.

        Returns
        -------
        SaidaMatriz
            Matriz perturbada com dropouts ou genes ausentes.
        """
        data: NDArray[np.float32] = self.matriz_pura.copy()
        genes_atuais: list[str] = list(self.gene_names)

        if dropouts_deterministicos is not None:
            for c_idx, g_idx in dropouts_deterministicos:
                if 0 <= c_idx < self.n_celulas and 0 <= g_idx < len(genes_atuais):
                    data[c_idx, g_idx] = 0.0
        elif taxa_dropout > 0:
            mask_ativos: NDArray[np.bool_] = data > 0
            sorteio: NDArray[np.float64] = self.rng.random(size=data.shape)
            drop_mask: NDArray[np.bool_] = mask_ativos & (sorteio < taxa_dropout)
            data[drop_mask] = 0.0

        if genes_remover is not None:
            colunas_manter_idx: list[int] = [
                j for j, g in enumerate(genes_atuais) if g not in genes_remover
            ]
            data = data[:, colunas_manter_idx]
            genes_atuais = [genes_atuais[j] for j in colunas_manter_idx]

        return self._formatar_saida(data, formato, genes_customizados=genes_atuais)

    def _formatar_saida(
        self,
        data: NDArray[np.float32],
        formato: str,
        genes_customizados: Sequence[str] | None = None,
    ) -> SaidaMatriz:
        """Formata o array numérico para a estrutura de dados solicitada."""
        cols: Sequence[str] = (
            genes_customizados if genes_customizados is not None else self.gene_names
        )
        if formato == "numpy":
            return data
        elif formato == "dataframe":
            return pd.DataFrame(data, index=self.cell_names, columns=list(cols))
        elif formato == "anndata":
            obs_df = pd.DataFrame(
                {"clo": self.labels, "cell_name": self.cell_names},
                index=self.cell_names,
            )
            var_df = pd.DataFrame({"gene_name": list(cols)}, index=list(cols))
            adata = ad.AnnData(X=data.copy(), obs=obs_df, var=var_df)
            return adata
        elif formato == "esparso":
            return sp.csr_matrix(data)
        else:
            raise ValueError(f"[GeradorGroundTruth] Formato desconhecido: {formato}")

    def gerar_tabela_markdown(
        self,
        matriz: Any,
        genes_customizados: Sequence[str] | None = None,
        titulo: str | None = None,
    ) -> str:
        """Retorna uma string em formato Markdown com a tabela legível para inspeção humana.

        Parameters
        ----------
        matriz : Any
            Matriz ou objeto AnnData/DataFrame a ser exibido.
        genes_customizados : Sequence[str] | None, optional
            Nomes das colunas caso customizados.
        titulo : str | None, optional
            Título opcional da seção markdown.

        Returns
        -------
        str
            Tabela formatada em Markdown.
        """
        cols: list[str] = (
            list(genes_customizados)
            if genes_customizados is not None
            else list(self.gene_names)
        )
        linhas_nomes: list[str]

        if isinstance(matriz, ad.AnnData):
            data_arr = (
                matriz.X.toarray() if sp.issparse(matriz.X) else np.asarray(matriz.X)
            )
            cols = matriz.var_names.tolist()
            linhas_nomes = matriz.obs_names.tolist()
        elif isinstance(matriz, pd.DataFrame):
            data_arr = matriz.values
            cols = matriz.columns.tolist()
            linhas_nomes = matriz.index.tolist()
        else:
            data_arr = np.asarray(matriz)
            linhas_nomes = self.cell_names[: data_arr.shape[0]]

        lines: list[str] = []
        if titulo:
            lines.append(f"### {titulo}\n")

        header: str = "| Célula / Rótulo | " + " | ".join(cols) + " |"
        divisor: str = "| :--- | " + " | ".join([":---:" for _ in cols]) + " |"
        lines.append(header)
        lines.append(divisor)

        for i, row in enumerate(data_arr):
            nome_linha: str = linhas_nomes[i] if i < len(linhas_nomes) else f"C{i}"
            vals_str: list[str] = []
            for val in row:
                if float(val) == int(val):
                    vals_str.append(f"{int(val)}")
                else:
                    vals_str.append(f"{float(val):.1f}")
            line: str = f"| **{nome_linha}** | " + " | ".join(vals_str) + " |"
            lines.append(line)

        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:
        """Representação textual do gerador de ground truth sintético."""
        return (
            f"GeradorGroundTruthSintetico(celulas={self.n_celulas}, "
            f"genes={self.n_genes}, classes={self.n_classes}, seed={self.seed})"
        )
