"""Testes da camada de memória associativa e reconstrução por Redes Hopfield Modernas."""

import time

import numpy as np

from src.synthetic.gerador_ground_truth import GeradorGroundTruthSintetico
from src.treinamento.hopfield import ModernHopfieldNetwork


def test_reconstrucao_hopfield_elimina_dropouts():
    """Valida se a Rede Hopfield recupera 100% das assinaturas biológicas originais (Ground Truth)
    a partir de uma matriz severamente corrompida por dropouts e genes ausentes substituídos por sentinela 0.5."""
    gerador = GeradorGroundTruthSintetico(n_celulas=12, n_genes=8, n_classes=3, seed=88)
    matriz_pura = gerador.gerar_matriz_pura(formato="numpy")

    # Armazena os 3 protótipos ideais correspondentes às 3 classes biológicas
    prototipos = np.vstack([matriz_pura[0], matriz_pura[4], matriz_pura[8]])

    rede = ModernHopfieldNetwork(beta=15.0, n_iters=1, binary=True)
    rede.store(prototipos)

    # Corrompe intencionalmente as células (dropouts explícitos para teste determinístico)
    dropouts = [
        (1, 0),  # Célula 1 (Tipo A): perde o gene G0
        (2, 1),  # Célula 2 (Tipo A): perde o gene G1
        (5, 4),  # Célula 5 (Tipo B): perde o gene G4
        (10, 6),  # Célula 10 (Tipo C): perde o gene G6
    ]
    matriz_perturbada = gerador.gerar_matriz_perturbada(
        dropouts_deterministicos=dropouts, formato="numpy"
    )

    # Simula também que o gene G2 estava ausente em toda a consulta e recebeu a sentinela neutra 0.5
    matriz_perturbada[:, 2] = 0.5

    # Executa a recuperação associativa via atenção Softmax
    matriz_reconstruida = rede.retrieve(
        matriz_perturbada, batch_size=6, normalize=False
    )

    # Verifica erro de reconstrução residual
    erro_residual = np.abs(matriz_reconstruida - matriz_pura).sum()
    taxa_acertos = (matriz_reconstruida == matriz_pura).mean() * 100.0

    assert erro_residual == 0, (
        f"Erro residual de {erro_residual}. A rede não reconstruiu perfeitamente o Ground Truth."
    )
    assert taxa_acertos == 100.0, (
        "Taxa de acertos deve atingir 100% no teste controlado."
    )


def test_escalabilidade_e_complexidade_tempo_memoria():
    """Teste empírico de escalabilidade computacional O(f(n)) e memória O(g(n)).

    Demonstra a execução estável da Rede Hopfield com lote (batch_size) para prevenir
    estouro de RAM/VRAM ao escalar o número de genes e células."""
    n_celulas_teste = 500
    n_genes_teste = 3000

    gerador = GeradorGroundTruthSintetico(
        n_celulas=n_celulas_teste, n_genes=n_genes_teste, n_classes=5, seed=123
    )
    queries = gerador.gerar_matriz_perturbada(taxa_dropout=0.10, formato="numpy")

    # Extrai 10 protótipos simulados
    prototipos = queries[:10]

    rede = ModernHopfieldNetwork(beta=20.0, n_iters=1, binary=True)
    rede.store(prototipos)

    inicio_tempo = time.time()
    # Executa com batch_size = 128 para testar o particionamento de memória
    resultado = rede.retrieve(queries, batch_size=128)
    tempo_gasto = time.time() - inicio_tempo

    assert resultado.shape == (n_celulas_teste, n_genes_teste), (
        "A saída deve preservar as dimensões do dataset."
    )
    assert tempo_gasto < 10.0, (
        f"O tempo de execução excedeu 10s ({tempo_gasto:.2f}s), indicando gargalo $O(f(n))$."
    )


def test_hopfield_retrieve_com_probabilidades():
    """Valida a extração de probabilidades contínuas [0, 1] com return_probabilities=True."""
    prototipos = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    queries = np.array(
        [
            [1.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 1.0],
        ],
        dtype=np.float32,
    )

    rede = ModernHopfieldNetwork(beta=8.0, n_iters=1, binary=True)
    rede.store(prototipos)

    res_bin, res_prob = rede.retrieve(queries, batch_size=2, return_probabilities=True)

    assert res_bin.shape == (2, 4)
    assert res_prob.shape == (2, 4)
    assert np.all((res_bin == 0.0) | (res_bin == 1.0))
    assert np.all((res_prob >= 0.0) & (res_prob <= 1.0))

    # Primeira query deve ter alta ativação nas colunas 0 e 1 e baixa nas colunas 2 e 3
    assert res_prob[0, 0] > 0.8
    assert res_prob[0, 1] > 0.8
    assert res_prob[0, 2] < 0.2
    assert res_prob[0, 3] < 0.2
