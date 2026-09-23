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


def test_calcular_nc_estratificado():
    """Valida o cálculo de alocação de subclusters estratificado por raiz quadrada."""
    from src.treinamento.extrator_padroes import calcular_nc_estratificado

    # Simula distribuição fortemente desbalanceada: Classe 1 abundante (9000), Classe 2 rara (400)
    labels = np.array([1] * 9000 + [2] * 400 + [3] * 2000)
    classes = [1, 2, 3]

    nc_map = calcular_nc_estratificado(
        labels=labels,
        classes=classes,
        total_padroes=90,
        min_nc=10,
        max_nc=45,
    )

    assert nc_map[1] > nc_map[2], (
        "Classe abundante deve ter mais subclusters que classe rara"
    )
    assert nc_map[2] >= 10, "Classe rara deve respeitar cota mínima"
    assert nc_map[1] <= 45, "Classe abundante deve respeitar cota máxima"
    assert sum(nc_map.values()) > 0


def test_extracao_padroes_nc_estratificado_e_consenso_k5():
    """Valida extração com dicionário nc estratificado e consenso k=5 eliminando dropouts."""
    # Gera 2 classes com 10 células cada
    gerador = GeradorGroundTruthSintetico(n_celulas=20, n_genes=8, n_classes=2, seed=88)
    W0 = gerador.gerar_matriz_pura(formato="numpy")
    labels = gerador.labels

    # Insere dropout isolado em uma única célula da Classe 1 no gene G0 (deve ser 1)
    W0_corrompido = W0.copy()
    W0_corrompido[0, 0] = 0.0  # dropout acidental

    projetor = ProjetorSWeP(n_features=8, n_componentes=6, seed=88)
    projetor.gerar_base().projetar(W0_corrompido)
    Wswp = projetor.Wswp

    # Classe 1 recebe 2 clusters, Classe 2 recebe 1 cluster (total = 3 padrões)
    nc_estratificado = {1: 2, 2: 1}
    extrator = ExtratorPadroesSubcluster(
        W0=W0_corrompido,
        labels=labels,
        classes=[1, 2],
        nc=nc_estratificado,
        k=5,
        seed=88,
    )
    extrator.extrair(Wswp)

    padroes = extrator.padroes
    assert padroes is not None
    assert padroes.shape == (3, 8), f"Esperado (3, 8), obtido {padroes.shape}"

    # Graças ao consenso majoritário k=5, o gene G0 na Classe 1 deve ser restaurado como 1.0
    assert padroes[0, 0] == 1.0, (
        "Consenso k=5 deve eliminar o dropout isolado da célula 0"
    )
