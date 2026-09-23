"""Testes unitários para CarregadorDados e CarregadorDadosFujita."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.treinamento.carregador_dados_fujita import (
    CarregadorDadosFujita,
    remapear_labels_canonicos,
)


def test_remapear_labels_canonicos_anti_vies_classe2() -> None:
    """Testa se classes 1 a 7 são preservadas e rótulos espúrios são isolados como 0."""
    labels_brutos = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 99, -1, 2], dtype=int)
    remapeados, stats = remapear_labels_canonicos(labels_brutos)

    # Classes canônicas ativas (1 a 7) devem ser preservadas
    assert remapeados[0] == 1
    assert remapeados[1] == 2
    assert remapeados[2] == 3
    assert remapeados[3] == 4
    assert remapeados[4] == 5
    assert remapeados[5] == 6
    assert remapeados[6] == 7
    assert remapeados[11] == 2

    # Rótulos espúrios (8, 9, 99, -1) DEVEM ser 0 (desconhecido) e NÃO 2
    assert remapeados[7] == 0
    assert remapeados[8] == 0
    assert remapeados[9] == 0
    assert remapeados[10] == 0

    # Contagem da classe 2 deve ser exatamente 2 (e não inflada por 8, 9, 99, -1)
    assert stats["classe_2"] == 2
    assert stats["desconhecidos"] == 4
    assert stats["total"] == 12


def test_remapear_labels_canonicos_com_map_de_para() -> None:
    """Testa remapeamento com dicionário de-para customizado."""
    labels_brutos = np.array([10, 20, 30, 99], dtype=int)
    de_para = {10: 1, 20: 2, 30: 3}
    remapeados, stats = remapear_labels_canonicos(
        labels_brutos,
        classes_validas=[1, 2, 3],
        map_de_para=de_para,
        label_desconhecido=0,
    )

    assert list(remapeados) == [1, 2, 3, 0]
    assert stats["classe_1"] == 1
    assert stats["classe_2"] == 1
    assert stats["classe_3"] == 1
    assert stats["desconhecidos"] == 1


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
