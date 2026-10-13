"""Módulo de comparação topológica dimensional cross-dataset (Capítulo 15).

Calcula e plota reduções de dimensionalidade (PCA, UMAP e t-SNE) para avaliar
o alinhamento e separação biológica das linhagens celulares entre a matriz de referência,
a matriz alvo original (com sentinela) e a matriz alvo pós-imputação.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap


@dataclass
class ResultadoComparacaoEmbeddings:
    """Estrutura consolidada contendo as coordenadas 2D e métricas calculadas.

    Attributes
    ----------
    coord_pca_ref : NDArray[np.float32]
        Coordenadas PC1 e PC2 da referência Pan.
    coord_pca_orig : NDArray[np.float32]
        Coordenadas PC1 e PC2 do alvo Mathys original (pré-imputação).
    coord_pca_pos : NDArray[np.float32]
        Coordenadas PC1 e PC2 do alvo Mathys pós-imputação Hopfield.
    coord_umap_ref : NDArray[np.float32]
        Coordenadas UMAP1 e UMAP2 da referência Pan.
    coord_umap_orig : NDArray[np.float32]
        Coordenadas UMAP1 e UMAP2 do alvo Mathys original.
    coord_umap_pos : NDArray[np.float32]
        Coordenadas UMAP1 e UMAP2 do alvo Mathys pós-imputação.
    coord_tsne_ref : NDArray[np.float32]
        Coordenadas t-SNE1 e t-SNE2 da referência Pan.
    coord_tsne_orig : NDArray[np.float32]
        Coordenadas t-SNE1 e t-SNE2 do alvo Mathys original.
    coord_tsne_pos : NDArray[np.float32]
        Coordenadas t-SNE1 e t-SNE2 do alvo Mathys pós-imputação.
    var_exp_pca : tuple[float, float]
        Taxas de variância explicada por PC1 e PC2 na referência.
    y_ref : NDArray[np.int64]
        Rótulos de classe para cada célula da referência.
    y_alvo : NDArray[np.int64]
        Rótulos de classe para cada célula do alvo.
    nomes_classes : list[str]
        Lista ordenada com os nomes canônicos das classes biológicas.
    df_metricas : pd.DataFrame
        Tabela com as métricas quantitativas de Silhouette Score.
    caminho_png : str | None
        Caminho para o painel gráfico PNG gerado em disco.
    caminho_coordenadas : str | None
        Caminho para a tabela de coordenadas 2D salva em disco.
    caminho_metricas : str | None
        Caminho para o arquivo CSV de métricas salvo em disco.
    do_cache : bool
        Indica se os resultados foram recuperados de cache prévio.
    """

    coord_pca_ref: NDArray[np.float32]
    coord_pca_orig: NDArray[np.float32]
    coord_pca_pos: NDArray[np.float32]
    coord_umap_ref: NDArray[np.float32]
    coord_umap_orig: NDArray[np.float32]
    coord_umap_pos: NDArray[np.float32]
    coord_tsne_ref: NDArray[np.float32]
    coord_tsne_orig: NDArray[np.float32]
    coord_tsne_pos: NDArray[np.float32]
    var_exp_pca: tuple[float, float]
    y_ref: NDArray[np.int64]
    y_alvo: NDArray[np.int64]
    nomes_classes: list[str] = field(default_factory=list)
    df_metricas: pd.DataFrame = field(default_factory=pd.DataFrame)
    caminho_png: str | None = None
    caminho_coordenadas: str | None = None
    caminho_metricas: str | None = None
    do_cache: bool = False


class ComparadorEmbeddings:
    """Orquestrador OOM-Safe de comparações dimensionais (PCA, UMAP e t-SNE).

    Ajusta modelos de redução de dimensionalidade sobre a matriz de referência
    (espaço rSWeeP 600D) e projeta as matrizes alvo antes e depois da imputação,
    garantindo eixos uniformes e análise de coerência biológica.

    Parameters
    ----------
    seed : int, default=42
        Semente pseudoaleatória para reprodutibilidade estrita.
    n_neighbors : int, default=30
        Número de vizinhos locais para o algoritmo UMAP.
    min_dist : float, default=0.3
        Distância mínima entre pontos no espaço embutido do UMAP.
    perplexity : float, default=30.0
        Perplexidade estatística do t-SNE.
    tsne_iter : int, default=1000
        Número de iterações do otimizador de gradiente do t-SNE.
    pca_pre_tsne : int, default=50
        Número de componentes PCA lineares intermediárias antes do t-SNE.
    """

    def __init__(
        self,
        seed: int = 42,
        n_neighbors: int = 30,
        min_dist: float = 0.3,
        perplexity: float = 30.0,
        tsne_iter: int = 1000,
        pca_pre_tsne: int = 50,
    ) -> None:
        self.seed: int = int(seed)
        self.n_neighbors: int = int(n_neighbors)
        self.min_dist: float = float(min_dist)
        self.perplexity: float = float(perplexity)
        self.tsne_iter: int = int(tsne_iter)
        self.pca_pre_tsne: int = int(pca_pre_tsne)

    def calcular_pca(
        self,
        x_ref: NDArray[np.float32] | np.ndarray,
        x_orig: NDArray[np.float32] | np.ndarray,
        x_pos: NDArray[np.float32] | np.ndarray,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], tuple[float, float]]:
        """Ajusta o PCA na referência e transforma os alvos pré e pós-imputação.

        Parameters
        ----------
        x_ref : ndarray de shape (N_ref, D)
            Matriz vetorial da referência.
        x_orig : ndarray de shape (N_alvo, D)
            Matriz vetorial do alvo original.
        x_pos : ndarray de shape (N_alvo, D)
            Matriz vetorial do alvo pós-imputação.

        Returns
        -------
        coord_ref : ndarray de shape (N_ref, 2)
            Coordenadas PC1 e PC2 da referência.
        coord_orig : ndarray de shape (N_alvo, 2)
            Coordenadas PC1 e PC2 do alvo original.
        coord_pos : ndarray de shape (N_alvo, 2)
            Coordenadas PC1 e PC2 do alvo pós-imputação.
        var_exp : tuple[float, float]
            Frações da variância explicada por PC1 e PC2.
        """
        pca = PCA(n_components=2, random_state=self.seed)
        coord_ref = pca.fit_transform(x_ref).astype(np.float32)
        coord_orig = pca.transform(x_orig).astype(np.float32)
        coord_pos = pca.transform(x_pos).astype(np.float32)

        var_exp = (
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        )
        return coord_ref, coord_orig, coord_pos, var_exp

    def calcular_umap(
        self,
        x_ref: NDArray[np.float32] | np.ndarray,
        x_orig: NDArray[np.float32] | np.ndarray,
        x_pos: NDArray[np.float32] | np.ndarray,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Ajusta o UMAP na referência e transforma os alvos pré e pós-imputação.

        Parameters
        ----------
        x_ref : ndarray de shape (N_ref, D)
            Matriz vetorial da referência.
        x_orig : ndarray de shape (N_alvo, D)
            Matriz vetorial do alvo original.
        x_pos : ndarray de shape (N_alvo, D)
            Matriz vetorial do alvo pós-imputação.

        Returns
        -------
        coord_ref : ndarray de shape (N_ref, 2)
            Coordenadas UMAP1 e UMAP2 da referência.
        coord_orig : ndarray de shape (N_alvo, 2)
            Coordenadas UMAP1 e UMAP2 do alvo original.
        coord_pos : ndarray de shape (N_alvo, 2)
            Coordenadas UMAP1 e UMAP2 do alvo pós-imputação.
        """
        redutor_umap = umap.UMAP(
            n_components=2,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric="cosine",
            random_state=self.seed,
            transform_seed=self.seed,
        )
        coord_ref = redutor_umap.fit_transform(x_ref).astype(np.float32)
        coord_orig = redutor_umap.transform(x_orig).astype(np.float32)
        coord_pos = redutor_umap.transform(x_pos).astype(np.float32)

        return coord_ref, coord_orig, coord_pos

    def calcular_tsne(
        self,
        x_ref: NDArray[np.float32] | np.ndarray,
        x_orig: NDArray[np.float32] | np.ndarray,
        x_pos: NDArray[np.float32] | np.ndarray,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        """Executa co-fit de t-SNE sobre todas as células com pré-PCA e desmembra coordenadas.

        Parameters
        ----------
        x_ref : ndarray de shape (N_ref, D)
            Matriz vetorial da referência.
        x_orig : ndarray de shape (N_alvo, D)
            Matriz vetorial do alvo original.
        x_pos : ndarray de shape (N_alvo, D)
            Matriz vetorial do alvo pós-imputação.

        Returns
        -------
        coord_ref : ndarray de shape (N_ref, 2)
            Coordenadas t-SNE1 e t-SNE2 da referência.
        coord_orig : ndarray de shape (N_alvo, 2)
            Coordenadas t-SNE1 e t-SNE2 do alvo original.
        coord_pos : ndarray de shape (N_alvo, 2)
            Coordenadas t-SNE1 e t-SNE2 do alvo pós-imputação.
        """
        n_ref = len(x_ref)
        n_orig = len(x_orig)

        # 1. Empilhamento vertical de 100% das células
        x_total = np.vstack([x_ref, x_orig, x_pos])

        # 2. Pré-redução linear PCA para acelerar o cálculo das vizinhanças Barnes-Hut
        n_comp_pca = min(self.pca_pre_tsne, x_total.shape[1], x_total.shape[0] - 1)
        if n_comp_pca > 2 and x_total.shape[1] > n_comp_pca:
            pca_pre = PCA(n_components=n_comp_pca, random_state=self.seed)
            x_total_prep = pca_pre.fit_transform(x_total)
        else:
            x_total_prep = x_total

        # 3. Execução integral do t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=self.perplexity,
            max_iter=self.tsne_iter,
            random_state=self.seed,
            n_jobs=-1,
            init="pca",
        )
        coords_total = tsne.fit_transform(x_total_prep).astype(np.float32)

        # 4. Desmembramento em fatias exatas
        coord_ref = coords_total[:n_ref]
        coord_orig = coords_total[n_ref : n_ref + n_orig]
        coord_pos = coords_total[n_ref + n_orig :]

        return coord_ref, coord_orig, coord_pos

    def calcular_metricas_silhouette(
        self,
        metodo: str,
        coord_ref: NDArray[np.float32],
        coord_orig: NDArray[np.float32],
        coord_pos: NDArray[np.float32],
        y_ref: NDArray[np.int64] | Sequence[int],
        y_alvo: NDArray[np.int64] | Sequence[int],
        sample_size: int = 5000,
    ) -> pd.DataFrame:
        """Calcula o Silhouette Score de forma OOM-Safe com subamostragem determinística.

        Parameters
        ----------
        metodo : str
            Nome do método ('PCA', 'UMAP', 't-SNE').
        coord_ref : ndarray
            Coordenadas 2D da referência.
        coord_orig : ndarray
            Coordenadas 2D do alvo original.
        coord_pos : ndarray
            Coordenadas 2D do alvo pós-imputação.
        y_ref : sequence
            Rótulos de classe da referência.
        y_alvo : sequence
            Rótulos de classe do alvo.
        sample_size : int, default=5000
            Tamanho da subamostra determinística para cálculo rápido e sem OOM.

        Returns
        -------
        pd.DataFrame
            Tabela contendo método, condição e Silhouette Score associado.
        """
        y_ref_arr = np.asarray(y_ref)
        y_alvo_arr = np.asarray(y_alvo)

        # Limita sample_size ao tamanho real da matriz se for menor
        ss_ref = min(sample_size, len(coord_ref)) if sample_size > 0 else None
        ss_alvo = min(sample_size, len(coord_orig)) if sample_size > 0 else None

        score_ref = float(
            silhouette_score(
                coord_ref, y_ref_arr, sample_size=ss_ref, random_state=self.seed
            )
        )
        score_orig = float(
            silhouette_score(
                coord_orig, y_alvo_arr, sample_size=ss_alvo, random_state=self.seed
            )
        )
        score_pos = float(
            silhouette_score(
                coord_pos, y_alvo_arr, sample_size=ss_alvo, random_state=self.seed
            )
        )

        linhas = [
            {"metodo": metodo, "condicao": "Referência (Pan)", "silhouette_score": score_ref},
            {"metodo": metodo, "condicao": "Alvo Original (Mathys)", "silhouette_score": score_orig},
            {"metodo": metodo, "condicao": "Alvo Pós-Imputação (Hopfield)", "silhouette_score": score_pos},
        ]
        return pd.DataFrame(linhas)

    def executar_comparacao(
        self,
        x_ref: NDArray[np.float32] | np.ndarray,
        x_orig: NDArray[np.float32] | np.ndarray,
        x_pos: NDArray[np.float32] | np.ndarray,
        y_ref: NDArray[np.int64] | Sequence[int],
        y_alvo: NDArray[np.int64] | Sequence[int],
        nomes_classes: Sequence[str] | None = None,
        sample_size_silhouette: int = 5000,
    ) -> ResultadoComparacaoEmbeddings:
        """Executa os três algoritmos de redução e consolida coordenadas e métricas.

        Parameters
        ----------
        x_ref : ndarray
            Matriz da referência no espaço rSWeeP 600D.
        x_orig : ndarray
            Matriz do alvo original no espaço rSWeeP 600D com sentinela.
        x_pos : ndarray
            Matriz do alvo pós-imputação no espaço rSWeeP 600D.
        y_ref : sequence
            Classes celulares da referência.
        y_alvo : sequence
            Classes celulares do alvo.
        nomes_classes : sequence of str, optional
            Nomes das linhagens biológicas.
        sample_size_silhouette : int, default=5000
            Amostragem OOM-Safe para avaliação das métricas de clusterização.

        Returns
        -------
        ResultadoComparacaoEmbeddings
            Objeto com todos os dados calculados.
        """
        print("[ComparadorEmbeddings] 1/3 Calculando PCA (Fit na Referência Pan)...")
        c_pca_ref, c_pca_orig, c_pca_pos, var_exp = self.calcular_pca(x_ref, x_orig, x_pos)

        print("[ComparadorEmbeddings] 2/3 Calculando UMAP (Fit na Referência Pan)...")
        c_umap_ref, c_umap_orig, c_umap_pos = self.calcular_umap(x_ref, x_orig, x_pos)

        print(
            f"[ComparadorEmbeddings] 3/3 Calculando t-SNE conjunto sobre {len(x_ref) + len(x_orig) + len(x_pos):,} células..."
        )
        c_tsne_ref, c_tsne_orig, c_tsne_pos = self.calcular_tsne(x_ref, x_orig, x_pos)

        print("[ComparadorEmbeddings] Calculando índices Silhouette de separação...")
        df_pca = self.calcular_metricas_silhouette(
            "PCA", c_pca_ref, c_pca_orig, c_pca_pos, y_ref, y_alvo, sample_size=sample_size_silhouette
        )
        df_umap = self.calcular_metricas_silhouette(
            "UMAP", c_umap_ref, c_umap_orig, c_umap_pos, y_ref, y_alvo, sample_size=sample_size_silhouette
        )
        df_tsne = self.calcular_metricas_silhouette(
            "t-SNE", c_tsne_ref, c_tsne_orig, c_tsne_pos, y_ref, y_alvo, sample_size=sample_size_silhouette
        )
        df_metricas = pd.concat([df_pca, df_umap, df_tsne], ignore_index=True)

        y_ref_arr = np.asarray(y_ref, dtype=np.int64)
        y_alvo_arr = np.asarray(y_alvo, dtype=np.int64)

        if nomes_classes is None:
            max_c = max(int(np.max(y_ref_arr)), int(np.max(y_alvo_arr)))
            nomes_classes = [f"Classe_{i}" for i in range(max_c + 1)]

        return ResultadoComparacaoEmbeddings(
            coord_pca_ref=c_pca_ref,
            coord_pca_orig=c_pca_orig,
            coord_pca_pos=c_pca_pos,
            coord_umap_ref=c_umap_ref,
            coord_umap_orig=c_umap_orig,
            coord_umap_pos=c_umap_pos,
            coord_tsne_ref=c_tsne_ref,
            coord_tsne_orig=c_tsne_orig,
            coord_tsne_pos=c_tsne_pos,
            var_exp_pca=var_exp,
            y_ref=y_ref_arr,
            y_alvo=y_alvo_arr,
            nomes_classes=list(nomes_classes),
            df_metricas=df_metricas,
            do_cache=False,
        )

    def plotar_grid_3x3(
        self,
        resultado: ResultadoComparacaoEmbeddings,
        out_png: str | None = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """Renderiza o Painel Sinótico 3 × 3 padronizado e salva em alta resolução.

        Linhas: PCA, UMAP, t-SNE.
        Colunas: Referência, Alvo Original, Alvo Pós-Imputação.

        Parameters
        ----------
        resultado : ResultadoComparacaoEmbeddings
            Estrutura com as coordenadas 2D e metadados.
        out_png : str | None, optional
            Caminho do arquivo PNG de saída.
        dpi : int, default=300
            Resolução gráfica em pontos por polegada.

        Returns
        -------
        plt.Figure
            Objeto da figura matplotlib gerada.
        """
        # Configuração da figura 3x3
        fig, axs = plt.subplots(3, 3, figsize=(18, 16), sharex=False, sharey=False)

        colunas_nomes = [
            f"Referência (Pan)\nN = {len(resultado.y_ref):,}",
            f"Alvo Original (Mathys pré-Hopfield)\nN = {len(resultado.y_alvo):,}",
            f"Alvo Pós-Imputação (Mathys Hopfield)\nN = {len(resultado.y_alvo):,}",
        ]

        # Paleta de cores discreta e consistente para todas as classes
        n_classes = len(resultado.nomes_classes)
        cmap = plt.get_cmap("tab10")
        cores = [cmap(i % 10) for i in range(n_classes)]

        # Matriz de coordenadas por método
        grid_dados: list[tuple[str, str, str, list[NDArray[np.float32]]]] = [
            (
                "PCA",
                f"PC1 ({resultado.var_exp_pca[0] * 100:.1f}%)",
                f"PC2 ({resultado.var_exp_pca[1] * 100:.1f}%)",
                [resultado.coord_pca_ref, resultado.coord_pca_orig, resultado.coord_pca_pos],
            ),
            (
                "UMAP",
                "UMAP 1",
                "UMAP 2",
                [resultado.coord_umap_ref, resultado.coord_umap_orig, resultado.coord_umap_pos],
            ),
            (
                "t-SNE",
                "t-SNE 1",
                "t-SNE 2",
                [resultado.coord_tsne_ref, resultado.coord_tsne_orig, resultado.coord_tsne_pos],
            ),
        ]

        # Iteração sobre as 3 linhas e 3 colunas
        for row_idx, (nome_metodo, xlabel, ylabel, coords_condicoes) in enumerate(grid_dados):
            for col_idx, coords in enumerate(coords_condicoes):
                ax = axs[row_idx, col_idx]
                rotulos = resultado.y_ref if col_idx == 0 else resultado.y_alvo

                # Plot dos pontos de cada classe biológica
                for c_idx, c_nome in enumerate(resultado.nomes_classes):
                    mask = rotulos == c_idx
                    if np.any(mask):
                        ax.scatter(
                            coords[mask, 0],
                            coords[mask, 1],
                            c=[cores[c_idx]],
                            label=c_nome if (row_idx == 0 and col_idx == 0) else None,
                            s=2.5,
                            alpha=0.45,
                            edgecolors="none",
                            rasterized=True,
                        )

                # Títulos de Coluna apenas na primeira linha
                if row_idx == 0:
                    ax.set_title(colunas_nomes[col_idx], fontsize=13, fontweight="bold", pad=12)

                # Títulos de Linha / Métricas no eixo Y
                ax.set_xlabel(xlabel, fontsize=10, labelpad=5)
                if col_idx == 0:
                    ax.set_ylabel(f"{nome_metodo}\n{ylabel}", fontsize=11, fontweight="bold", labelpad=8)
                else:
                    ax.set_ylabel(ylabel, fontsize=10, labelpad=5)

                # Estética limpa
                ax.grid(True, linestyle=":", alpha=0.5)
                ax.tick_params(labelsize=9)

        # Legenda unificada na base da figura
        handles, labels = axs[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(len(labels), 7),
                bbox_to_anchor=(0.5, 0.02),
                fontsize=11,
                title="Linhagens Celulares Canônicas do Cérebro",
                title_fontsize=12,
                frameon=True,
                markerscale=4.0,
            )

        plt.suptitle(
            "Painel Comparativo de Redução de Dimensionalidade (PCA, UMAP e t-SNE)\n"
            "Espaço rSWeeP 600D — Avaliação de Harmonização Topológica Cross-Dataset",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )

        plt.subplots_adjust(bottom=0.08, top=0.92, hspace=0.28, wspace=0.22)

        if out_png:
            os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
            plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
            print(f"[Visualização] Painel Grid 3x3 salvo com sucesso em: {out_png}")

        return fig

    def processar_e_salvar(
        self,
        x_ref: NDArray[np.float32] | np.ndarray,
        x_orig: NDArray[np.float32] | np.ndarray,
        x_pos: NDArray[np.float32] | np.ndarray,
        y_ref: NDArray[np.int64] | Sequence[int],
        y_alvo: NDArray[np.int64] | Sequence[int],
        nomes_classes: Sequence[str] | None = None,
        out_dir: str = "outputs/07_embeddings_comparativos",
        forcar_recalculo: bool = False,
        sample_size_silhouette: int = 5000,
    ) -> ResultadoComparacaoEmbeddings:
        """Executa a comparação completa com suporte a cache idempotente em disco.

        Parameters
        ----------
        x_ref : ndarray
            Matriz de referência no espaço rSWeeP 600D.
        x_orig : ndarray
            Matriz alvo original no espaço rSWeeP 600D.
        x_pos : ndarray
            Matriz alvo pós-imputação no espaço rSWeeP 600D.
        y_ref : sequence
            Rótulos da referência.
        y_alvo : sequence
            Rótulos do alvo.
        nomes_classes : sequence of str, optional
            Nomes das classes.
        out_dir : str, default='outputs/07_embeddings_comparativos'
            Diretório de gravação dos artefatos.
        forcar_recalculo : bool, default=False
            Se True, ignora arquivos existentes e recalcula do início.
        sample_size_silhouette : int, default=5000
            Amostragem para Silhouette Score.

        Returns
        -------
        ResultadoComparacaoEmbeddings
            Objeto consolidado dos resultados.
        """
        os.makedirs(out_dir, exist_ok=True)
        path_png = os.path.join(out_dir, "grid_3x3_pca_umap_tsne_referencia_vs_alvos.png")
        path_coord = os.path.join(out_dir, "coordenadas_grid_pca_umap_tsne.csv")
        path_metricas = os.path.join(out_dir, "metricas_separacao_clusters_cap15.csv")

        # Verificação de Cache
        if (
            os.path.exists(path_png)
            and os.path.exists(path_coord)
            and os.path.exists(path_metricas)
            and not forcar_recalculo
        ):
            print(f"[Cache Capítulo 15] Artefatos já existentes em {out_dir}. Carregando do disco...")
            df_coord = pd.read_csv(path_coord)
            df_metricas = pd.read_csv(path_metricas)

            # Reconstituição rápida das coordenadas a partir do cache
            def _extrair(metodo: str, cond: str) -> NDArray[np.float32]:
                sub = df_coord[(df_coord["metodo"] == metodo) & (df_coord["condicao"] == cond)]
                return sub[["dim1", "dim2"]].to_numpy(dtype=np.float32)

            res = ResultadoComparacaoEmbeddings(
                coord_pca_ref=_extrair("PCA", "Referência (Pan)"),
                coord_pca_orig=_extrair("PCA", "Alvo Original (Mathys)"),
                coord_pca_pos=_extrair("PCA", "Alvo Pós-Imputação (Hopfield)"),
                coord_umap_ref=_extrair("UMAP", "Referência (Pan)"),
                coord_umap_orig=_extrair("UMAP", "Alvo Original (Mathys)"),
                coord_umap_pos=_extrair("UMAP", "Alvo Pós-Imputação (Hopfield)"),
                coord_tsne_ref=_extrair("t-SNE", "Referência (Pan)"),
                coord_tsne_orig=_extrair("t-SNE", "Alvo Original (Mathys)"),
                coord_tsne_pos=_extrair("t-SNE", "Alvo Pós-Imputação (Hopfield)"),
                var_exp_pca=(0.0, 0.0),
                y_ref=np.asarray(y_ref, dtype=np.int64),
                y_alvo=np.asarray(y_alvo, dtype=np.int64),
                nomes_classes=list(nomes_classes) if nomes_classes else [],
                df_metricas=df_metricas,
                caminho_png=path_png,
                caminho_coordenadas=path_coord,
                caminho_metricas=path_metricas,
                do_cache=True,
            )
            return res

        # Execução completa
        res = self.executar_comparacao(
            x_ref=x_ref,
            x_orig=x_orig,
            x_pos=x_pos,
            y_ref=y_ref,
            y_alvo=y_alvo,
            nomes_classes=nomes_classes,
            sample_size_silhouette=sample_size_silhouette,
        )

        # Montagem do DataFrame consolidado de coordenadas
        linhas_coord: list[dict[str, Any]] = []

        def _adicionar_pontos(metodo: str, condicao: str, coords: NDArray[np.float32], rotulos: NDArray[np.int64]) -> None:
            for i in range(len(coords)):
                c_idx = int(rotulos[i])
                c_nome = res.nomes_classes[c_idx] if c_idx < len(res.nomes_classes) else f"Classe_{c_idx}"
                linhas_coord.append(
                    {
                        "metodo": metodo,
                        "condicao": condicao,
                        "ponto_idx": i,
                        "dim1": float(coords[i, 0]),
                        "dim2": float(coords[i, 1]),
                        "classe_idx": c_idx,
                        "classe_nome": c_nome,
                    }
                )

        _adicionar_pontos("PCA", "Referência (Pan)", res.coord_pca_ref, res.y_ref)
        _adicionar_pontos("PCA", "Alvo Original (Mathys)", res.coord_pca_orig, res.y_alvo)
        _adicionar_pontos("PCA", "Alvo Pós-Imputação (Hopfield)", res.coord_pca_pos, res.y_alvo)

        _adicionar_pontos("UMAP", "Referência (Pan)", res.coord_umap_ref, res.y_ref)
        _adicionar_pontos("UMAP", "Alvo Original (Mathys)", res.coord_umap_orig, res.y_alvo)
        _adicionar_pontos("UMAP", "Alvo Pós-Imputação (Hopfield)", res.coord_umap_pos, res.y_alvo)

        _adicionar_pontos("t-SNE", "Referência (Pan)", res.coord_tsne_ref, res.y_ref)
        _adicionar_pontos("t-SNE", "Alvo Original (Mathys)", res.coord_tsne_orig, res.y_alvo)
        _adicionar_pontos("t-SNE", "Alvo Pós-Imputação (Hopfield)", res.coord_tsne_pos, res.y_alvo)

        df_coord_salvar = pd.DataFrame(linhas_coord)
        df_coord_salvar.to_csv(path_coord, index=False)
        print(f"[Persistência] Coordenadas 2D salvas em: {path_coord}")

        res.df_metricas.to_csv(path_metricas, index=False)
        print(f"[Persistência] Tabela de Silhouette Scores salva em: {path_metricas}")

        # Renderização e salvamento da figura
        fig = self.plotar_grid_3x3(resultado=res, out_png=path_png)
        plt.close(fig)

        res.caminho_png = path_png
        res.caminho_coordenadas = path_coord
        res.caminho_metricas = path_metricas
        return res
