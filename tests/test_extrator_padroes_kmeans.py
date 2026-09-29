"""Testes unitários para extração de padrões representativos via K-Means sobre Ground Truth sintético."""

import numpy as np

from src.synthetic.gerador_ground_truth import GeradorGroundTruthSintetico
from src.treinamento.extrator_padroes import ExtratorPadroesSubcluster
from src.treinamento.projetor_sweep import ProjetorSWeP


def test_extracao_padroes_kmeans_recupera_prototipos():
    """Comprova que o agrupamento K-Means no espaço SWeeP seleciona os vetores binários
    exatos que representam as assinaturas biológicas originais de cada tipo celular."""
    gerador = GeradorGroundTruthSintetico(n_celulas=12, n_genes=8, n_classes=3, seed=42)
    W0 = gerador.gerar_matriz_pura(formato="numpy")
    labels = gerador.labels

    # Projeta para o espaço SWeeP de 6 componentes
    projetor = ProjetorSWeP(n_features=8, n_componentes=6, seed=42)
    projetor.gerar_base().projetar(W0)
    Wswp = projetor.Wswp

    # Para 3 classes (Tipo A, B e C), solicitamos 1 subcluster (nc=1) por classe e amostragem de vizinho mais próximo (k=1)
    extrator = ExtratorPadroesSubcluster(
        W0=W0, labels=labels, classes=[1, 2, 3], nc=1, k=1, seed=42
    )
    extrator.extrair(Wswp)

    padroes = extrator.padroes

    assert padroes.shape == (3, 8), (
        f"Esperado 3 protótipos de 8 genes, obtido {padroes.shape}"
    )

    # Assinaturas esperadas teóricas do Ground Truth:
    # Tipo A (Classe 1): genes G0, G1, G2 ativos
    assinatura_a = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    # Tipo B (Classe 2): genes G3, G4, G5 ativos
    assinatura_b = np.array([0, 0, 0, 1, 1, 1, 0, 0], dtype=np.float32)
    # Tipo C (Classe 3): genes G6, G7 ativos
    assinatura_c = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32)

    np.testing.assert_array_equal(
        padroes[0],
        assinatura_a,
        err_msg="O protótipo da Classe 1 (Tipo A) foi distorcido.",
    )
    np.testing.assert_array_equal(
        padroes[1],
        assinatura_b,
        err_msg="O protótipo da Classe 2 (Tipo B) foi distorcido.",
    )
    np.testing.assert_array_equal(
        padroes[2],
        assinatura_c,
        err_msg="O protótipo da Classe 3 (Tipo C) foi distorcido.",
    )
