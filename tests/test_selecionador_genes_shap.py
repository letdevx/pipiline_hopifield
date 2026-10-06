"""Testes automatizados da seleção de features celulares via SHAP na Modern Hopfield Network.

Verifica a diferenciação automática do wrapper PyTorch, a integração com o GradientExplainer,
a agregação de scores por linhagem celular e a exportação OOM-Safe de matrizes filtradas.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from treinamento.hopfield import ModernHopfieldNetwork
from treinamento.selecionador_genes_shap import (
    HopfieldClassifierWrapper,
    SelecionadorGenesSHAPHopfield,
)


@pytest.fixture
def ambiente_sintetico_hopfield() -> tuple[
    ModernHopfieldNetwork,
    np.ndarray,
    np.ndarray,
    list[str],
    list[int],
    list[tuple[int, int]],
    list[str],
]:
    """Fixture que gera um ambiente sintético com 3 classes biológicas e marcadores definidos."""
    rng = np.random.default_rng(42)
    torch.manual_seed(42)

    n_genes: int = 30
    n_amostras_por_classe: int = 15
    classes: list[int] = [0, 1, 2]
    nomes_classes: list[str] = ["Astro", "Neuro", "Micro"]
    nomes_genes: list[str] = [f"GENE_{i:02d}" for i in range(n_genes)]

    # 2 protótipos por classe (total 6 protótipos)
    prototipos: np.ndarray = np.zeros((6, n_genes), dtype=np.float32)
    meta_padroes: list[tuple[int, int]] = []

    # Marcadores fortes por classe
    # Classe 0 (Astro): genes 0 a 4 ativos
    prototipos[0, 0:5] = 1.0
    prototipos[1, 0:5] = 1.0
    meta_padroes.extend([(0, 0), (0, 1)])

    # Classe 1 (Neuro): genes 10 a 14 ativos
    prototipos[2, 10:15] = 1.0
    prototipos[3, 10:15] = 1.0
    meta_padroes.extend([(1, 0), (1, 1)])

    # Classe 2 (Micro): genes 20 a 24 ativos
    prototipos[4, 20:25] = 1.0
    prototipos[5, 20:25] = 1.0
    meta_padroes.extend([(2, 0), (2, 1)])

    hopfield = ModernHopfieldNetwork(beta=8.0, normalize=False, binary=True)
    hopfield.store(prototipos)

    # Gera células sintéticas com ruído
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for c in classes:
        for _ in range(n_amostras_por_classe):
            cel = rng.binomial(1, 0.05, size=n_genes).astype(np.float32)
            if c == 0:
                cel[0:5] = 1.0
            elif c == 1:
                cel[10:15] = 1.0
            elif c == 2:
                cel[20:25] = 1.0
            X_list.append(cel)
            y_list.append(c)

    X: np.ndarray = np.vstack(X_list)
    y: np.ndarray = np.array(y_list, dtype=int)

    return (
        hopfield,
        X,
        y,
        nomes_genes,
        classes,
        meta_padroes,
        nomes_classes,
    )


def test_hopfield_classifier_wrapper_forward_e_gradientes(
    ambiente_sintetico_hopfield: tuple[
        ModernHopfieldNetwork,
        np.ndarray,
        np.ndarray,
        list[str],
        list[int],
        list[tuple[int, int]],
        list[str],
    ],
) -> None:
    """Valida que o wrapper gera probabilidades válidas e gradientes autograd computáveis."""
    (hopfield, _, _, _, classes, meta_padroes, _) = ambiente_sintetico_hopfield

    wrapper = HopfieldClassifierWrapper(
        hopfield=hopfield,
        classes=classes,
        meta_padroes=meta_padroes,
    )

    # Usa query contínua para evitar saturação extrema no softmax
    x_tensor = torch.rand((5, 30), dtype=torch.float32, requires_grad=True)
    probs = wrapper(x_tensor)

    # Verifica formato: (batch_size, n_classes)
    assert probs.shape == (5, 3)

    # Probabilidades devem somar 1.0 por amostra
    soma_probs = probs.sum(dim=-1).detach().numpy()
    np.testing.assert_allclose(soma_probs, np.ones(5), atol=1e-5)

    # Testa se o gradiente flui corretamente pela atenção Softmax
    loss = probs[:, 0].sum()
    loss.backward()

    assert x_tensor.grad is not None
    assert x_tensor.grad.shape == x_tensor.shape
    # O gradiente dos genes deve ser não nulo
    assert bool((x_tensor.grad != 0.0).any())


def test_selecionador_genes_shap_ciclo_completo_rankings(
    ambiente_sintetico_hopfield: tuple[
        ModernHopfieldNetwork,
        np.ndarray,
        np.ndarray,
        list[str],
        list[int],
        list[tuple[int, int]],
        list[str],
    ],
    tmp_path: Path,
) -> None:
    """Testa o fluxo de explicação SHAP, ranking por classe e consolidação global."""
    (
        hopfield,
        X,
        y,
        nomes_genes,
        classes,
        meta_padroes,
        nomes_classes,
    ) = ambiente_sintetico_hopfield

    selecionador = SelecionadorGenesSHAPHopfield(
        hopfield_net=hopfield,
        classes=classes,
        meta_padroes=meta_padroes,
        nomes_classes=nomes_classes,
        nomes_genes=nomes_genes,
        batch_size=8,
    )

    # Executa a explicação com amostragem controlada
    selecionador.explicar(
        matriz_expressao=X,
        labels=y,
        n_background=15,
        n_amostras_explicar=24,
        seed=42,
    )

    assert selecionador.valores_shap is not None
    assert len(selecionador.valores_shap) == 3  # 3 classes

    # 1. Ranking por Linhagem Celular
    df_linhagem = selecionador.obter_ranking_por_classe(top_n=5)
    assert isinstance(df_linhagem, pl.DataFrame)
    assert df_linhagem.columns == [
        "gene",
        "classe",
        "nome_classe",
        "shap_medio_positivo",
        "frequencia_expressao",
    ]

    # Verifica se os Top genes de cada classe correspondem aos marcadores sintéticos reais
    top_astro = (
        df_linhagem.filter(pl.col("classe") == 0).select("gene").to_series().to_list()
    )
    # Pelo menos os primeiros genes de Astro devem ser do bloco 0 a 4
    assert any(
        g in ["GENE_00", "GENE_01", "GENE_02", "GENE_03", "GENE_04"]
        for g in top_astro[:3]
    )

    # 2. Ranking Consolidado
    df_consolidado = selecionador.obter_genes_consolidados(top_n_por_classe=3)
    assert isinstance(df_consolidado, pl.DataFrame)
    assert "gene" in df_consolidado.columns
    assert "max_shap" in df_consolidado.columns
    assert len(df_consolidado) > 0

    # 3. Teste de Salvamento de Relatórios
    out_dir = tmp_path / "shap_test_output"
    selecionador.salvar_relatorio(out_dir=out_dir)
    assert os.path.exists(out_dir / "top_marcadores_por_classe.csv")
    assert os.path.exists(out_dir / "genes_consolidados_shap.csv")

    # 4. Teste de Filtragem de Matriz
    caminho_npy_in = tmp_path / "matriz_in.npy"
    caminho_npy_out = tmp_path / "matriz_filtrada.npy"
    np.save(caminho_npy_in, X)

    genes_sel = df_consolidado["gene"].to_list()[:5]
    selecionador.filtrar_matriz(
        in_path=caminho_npy_in,
        out_path=caminho_npy_out,
        genes_selecionados=genes_sel,
    )
    mat_filtrada = np.load(caminho_npy_out)
    assert mat_filtrada.shape == (X.shape[0], len(genes_sel))


def test_selecionador_genes_shap_validacoes_e_erros(
    ambiente_sintetico_hopfield: tuple[
        ModernHopfieldNetwork,
        np.ndarray,
        np.ndarray,
        list[str],
        list[int],
        list[tuple[int, int]],
        list[str],
    ],
) -> None:
    """Verifica defesas contra chamadas fora de ordem e parâmetros inválidos."""
    (hopfield, _, _, nomes_genes, classes, meta_padroes, _) = (
        ambiente_sintetico_hopfield
    )

    selecionador = SelecionadorGenesSHAPHopfield(
        hopfield_net=hopfield,
        classes=classes,
        meta_padroes=meta_padroes,
        nomes_genes=nomes_genes,
    )

    # Chamar obter_ranking antes de explicar deve levantar RuntimeError
    with pytest.raises(RuntimeError, match=r"Execute \.explicar"):
        selecionador.obter_ranking_por_classe()

    with pytest.raises(RuntimeError, match=r"Execute \.explicar"):
        selecionador.obter_genes_consolidados()
