"""Testes unitários para o módulo ComparadorEmbeddings (Capítulo 15)."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.pca.comparador_embeddings import (
    ComparadorEmbeddings,
    ResultadoComparacaoEmbeddings,
)


@pytest.fixture
def dados_sinteticos_sweep() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
]:
    """Gera matrizes sintéticas 600D e rótulos de teste rápidos e determinísticos."""
    rng = np.random.default_rng(42)
    n_dim = 600
    n_ref = 60
    n_alvo_orig = 50

    # Três classes canônicas de teste
    nomes_classes = ["Ast", "Ex", "Oli"]

    y_ref = rng.integers(0, 3, size=n_ref)
    y_alvo = rng.integers(0, 3, size=n_alvo_orig)

    # Matrizes com sinal biológico artificial
    centroides = rng.normal(0, 2, size=(3, n_dim))

    x_ref = np.array(
        [centroides[c] + rng.normal(0, 0.5, size=n_dim) for c in y_ref],
        dtype=np.float32,
    )
    x_alvo_orig = np.array(
        [centroides[c] + rng.normal(0, 1.5, size=n_dim) for c in y_alvo],
        dtype=np.float32,
    )
    x_alvo_pos = np.array(
        [centroides[c] + rng.normal(0, 0.6, size=n_dim) for c in y_alvo],
        dtype=np.float32,
    )

    return x_ref, x_alvo_orig, x_alvo_pos, y_ref, y_alvo, nomes_classes


def test_pca_fit_referencia_transform_alvos(
    dados_sinteticos_sweep: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
    ],
) -> None:
    """Testa se o PCA é ajustado na referência e aplicado uniformemente nos alvos."""
    x_ref, x_orig, x_pos, _, _, _ = dados_sinteticos_sweep
    comparador = ComparadorEmbeddings(seed=42)

    c_ref, c_orig, c_pos, var_exp = comparador.calcular_pca(x_ref, x_orig, x_pos)

    assert c_ref.shape == (len(x_ref), 2)
    assert c_orig.shape == (len(x_orig), 2)
    assert c_pos.shape == (len(x_pos), 2)
    assert len(var_exp) == 2
    assert 0.0 <= sum(var_exp) <= 1.0


def test_umap_fit_referencia_transform_alvos(
    dados_sinteticos_sweep: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
    ],
) -> None:
    """Testa se o UMAP é ajustado na referência e aplicado nos alvos."""
    x_ref, x_orig, x_pos, _, _, _ = dados_sinteticos_sweep
    comparador = ComparadorEmbeddings(seed=42, n_neighbors=15, min_dist=0.3)

    c_ref, c_orig, c_pos = comparador.calcular_umap(x_ref, x_orig, x_pos)

    assert c_ref.shape == (len(x_ref), 2)
    assert c_orig.shape == (len(x_orig), 2)
    assert c_pos.shape == (len(x_pos), 2)


def test_tsne_cofit_conjunto_split(
    dados_sinteticos_sweep: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
    ],
) -> None:
    """Testa se o t-SNE executa co-fit sobre todas as células com pré-PCA e split exato."""
    x_ref, x_orig, x_pos, _, _, _ = dados_sinteticos_sweep
    comparador = ComparadorEmbeddings(
        seed=42, tsne_iter=250, perplexity=10.0, pca_pre_tsne=20
    )

    c_ref, c_orig, c_pos = comparador.calcular_tsne(x_ref, x_orig, x_pos)

    assert c_ref.shape == (len(x_ref), 2)
    assert c_orig.shape == (len(x_orig), 2)
    assert c_pos.shape == (len(x_pos), 2)


def test_calcular_metricas_silhouette(
    dados_sinteticos_sweep: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
    ],
) -> None:
    """Testa o cálculo do Silhouette Score com subamostragem determinística."""
    x_ref, x_orig, x_pos, y_ref, y_alvo, _ = dados_sinteticos_sweep
    comparador = ComparadorEmbeddings(seed=42)

    c_ref, c_orig, c_pos, _ = comparador.calcular_pca(x_ref, x_orig, x_pos)
    df_metricas = comparador.calcular_metricas_silhouette(
        metodo="PCA",
        coord_ref=c_ref,
        coord_orig=c_orig,
        coord_pos=c_pos,
        y_ref=y_ref,
        y_alvo=y_alvo,
        sample_size=40,
    )

    assert isinstance(df_metricas, pd.DataFrame)
    assert len(df_metricas) == 3
    assert "condicao" in df_metricas.columns
    assert "silhouette_score" in df_metricas.columns
    for score in df_metricas["silhouette_score"]:
        assert -1.0 <= score <= 1.0


def test_gerar_painel_grid_3x3(
    tmp_path: Path,
    dados_sinteticos_sweep: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
    ],
) -> None:
    """Testa a geração e salvamento do painel 3x3 em disco."""
    x_ref, x_orig, x_pos, y_ref, y_alvo, nomes_classes = dados_sinteticos_sweep
    out_png = str(tmp_path / "teste_grid_3x3.png")

    comparador = ComparadorEmbeddings(
        seed=42, tsne_iter=250, perplexity=10.0, pca_pre_tsne=20
    )
    resultado = comparador.executar_comparacao(
        x_ref=x_ref,
        x_orig=x_orig,
        x_pos=x_pos,
        y_ref=y_ref,
        y_alvo=y_alvo,
        nomes_classes=nomes_classes,
        sample_size_silhouette=40,
    )

    assert isinstance(resultado, ResultadoComparacaoEmbeddings)
    fig = comparador.plotar_grid_3x3(resultado=resultado, out_png=out_png)

    assert os.path.exists(out_png)
    assert os.path.getsize(out_png) > 1000
    plt.close(fig)


def test_orquestracao_completa_e_cache(
    tmp_path: Path,
    dados_sinteticos_sweep: tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]
    ],
) -> None:
    """Testa se a execução completa salva em cache e evita reprocessamento."""
    x_ref, x_orig, x_pos, y_ref, y_alvo, nomes_classes = dados_sinteticos_sweep
    out_dir = tmp_path / "saida_embeddings"

    comparador = ComparadorEmbeddings(
        seed=42, tsne_iter=250, perplexity=10.0, pca_pre_tsne=20
    )
    res1 = comparador.processar_e_salvar(
        x_ref=x_ref,
        x_orig=x_orig,
        x_pos=x_pos,
        y_ref=y_ref,
        y_alvo=y_alvo,
        nomes_classes=nomes_classes,
        out_dir=str(out_dir),
        forcar_recalculo=False,
        sample_size_silhouette=40,
    )

    assert os.path.exists(res1.caminho_png)
    assert os.path.exists(res1.caminho_coordenadas)
    assert os.path.exists(res1.caminho_metricas)

    # Segunda chamada deve reutilizar o cache
    res2 = comparador.processar_e_salvar(
        x_ref=x_ref,
        x_orig=x_orig,
        x_pos=x_pos,
        y_ref=y_ref,
        y_alvo=y_alvo,
        nomes_classes=nomes_classes,
        out_dir=str(out_dir),
        forcar_recalculo=False,
        sample_size_silhouette=40,
    )

    assert res2.do_cache is True
