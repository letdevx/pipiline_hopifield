"""Testes unitários para CarregadorDados e CarregadorDadosFujita."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.treinamento.carregador_dados_fujita import CarregadorDadosFujita


def test_carregador_dados_fujita_sweep_txt_tab(tmp_path: Path) -> None:
    """Testa se matriz SWeeP .txt tabulada sem cabeçalho (padrão rSWeeP) é lida corretamente."""
    n_amostras = 5
    n_comp = 10
    rng = np.random.default_rng(42)
    dados_sweep = rng.standard_normal((n_amostras, n_comp)).astype(np.float32)

    # Grava arquivo .txt tabulado sem cabeçalho
    path_sweep_txt = tmp_path / "matriz_reduzida_sweepF.txt"
    np.savetxt(path_sweep_txt, dados_sweep, delimiter="\t", fmt="%.7f")

    # Cria matriz dummy e rótulos
    path_matriz_npy = tmp_path / "matriz_dummy.npy"
    matriz_dummy = np.ones((n_amostras, 20), dtype=np.float32)
    np.save(path_matriz_npy, matriz_dummy)

    path_labels_txt = tmp_path / "labels.txt"
    np.savetxt(path_labels_txt, np.array([1, 1, 2, 2, 3]), fmt="%d")

    carregador = CarregadorDadosFujita(
        path_matriz=path_matriz_npy,
        path_genes=["GENE_" + str(i) for i in range(20)],
        path_labels=path_labels_txt,
        path_sweep=path_sweep_txt,
        n_genes=20,
    )
    carregador.carregar()

    assert carregador.Wswp is not None
    assert carregador.Wswp.shape == (n_amostras, n_comp)
    assert carregador.Wswp.dtype == np.float32
    np.testing.assert_allclose(carregador.Wswp, dados_sweep, atol=1e-5)
