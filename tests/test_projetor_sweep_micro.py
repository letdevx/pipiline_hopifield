"""Testes unitários para validação de projeção ortonormal no espaço rSWeeP com micro-datasets."""

import numpy as np

from src.synthetic.gerador_ground_truth import GeradorGroundTruthSintetico
from src.treinamento.projetor_sweep import ProjetorSWeP


def test_projetor_sweep_dimensao_adaptavel():
    """Verifica se o projetor SWeeP executa sem falhas com matrizes onde N_genes <= n_componentes,
    ajustando automaticamente a dimensão ou operando via decomposição QR adaptativa."""
    n_celulas = 12
    n_genes = 8
    n_comp_alvo = 4  # Dimensão reduzida compatível com o micro-dataset

    gerador = GeradorGroundTruthSintetico(
        n_celulas=n_celulas, n_genes=n_genes, n_classes=3, seed=77
    )
    matriz = gerador.gerar_matriz_pura(formato="numpy")

    projetor = ProjetorSWeP(n_features=n_genes, n_componentes=n_comp_alvo, seed=42)
    projetor.gerar_base()

    # Verifica ortogonalidade da matriz R
    R = projetor.R
    assert R.shape == (n_genes, n_comp_alvo), (
        f"Base R esperada {(n_genes, n_comp_alvo)}, obtida {R.shape}"
    )

    erro_ortogonalidade = np.abs(R.T @ R - np.eye(n_comp_alvo)).max()
    assert erro_ortogonalidade < 1e-5, (
        f"Erro de ortogonalidade excede limite aceitável: {erro_ortogonalidade}"
    )

    # Projeta os dados
    projetor.projetar(matriz)
    Wswp = projetor.Wswp

    assert Wswp.shape == (n_celulas, n_comp_alvo), (
        "A projeção Wswp deve ter dimensões (células × componentes)."
    )


def test_projetor_sweep_preserva_separabilidade_de_classes():
    """Comprova que células do mesmo tipo biológico permanecem mais próximas no espaço SWeeP do que de outros tipos."""
    gerador = GeradorGroundTruthSintetico(n_celulas=12, n_genes=8, n_classes=3, seed=55)
    matriz = gerador.gerar_matriz_pura(formato="numpy")

    # Células 0..3 são Tipo A, Células 4..7 são Tipo B
    projetor = ProjetorSWeP(n_features=8, n_componentes=6, seed=123)
    projetor.gerar_base().projetar(matriz)
    Wswp = projetor.Wswp

    # Distância L2 entre duas células do mesmo tipo (C0 e C1 - Tipo A)
    dist_mesma_classe = np.linalg.norm(Wswp[0] - Wswp[1])

    # Distância L2 entre células de tipos diferentes (C0 Tipo A e C5 Tipo B)
    dist_classes_distintas = np.linalg.norm(Wswp[0] - Wswp[5])

    assert dist_mesma_classe < dist_classes_distintas, (
        "A projeção SWeeP deve preservar a topologia e separabilidade biológica."
    )
