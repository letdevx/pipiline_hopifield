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


def test_hopfield_retrieve_normalizacao_esferica():
    """Valida que normalize=True garante invariância à escala de transcritos (library size)."""
    prototipos = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    # Mesma direção biológica, mas com magnitudes distintas (ex.: 1x vs 20x cópias de mRNA)
    query_pequena = np.array([[1.0, 2.0, 0.0, 0.0]], dtype=np.float32)
    query_grande = np.array([[20.0, 40.0, 0.0, 0.0]], dtype=np.float32)

    rede = ModernHopfieldNetwork(beta=25.0, n_iters=1, binary=False, normalize=True)
    rede.store(prototipos)

    res_peq, prob_peq = rede.retrieve(query_pequena, return_probabilities=True)
    res_grd, prob_grd = rede.retrieve(query_grande, return_probabilities=True)

    # Com normalização esférica ativada, os resultados devem ser idênticos
    np.testing.assert_allclose(prob_peq, prob_grd, atol=1e-5)
    np.testing.assert_allclose(res_peq, res_grd, atol=1e-5)


def test_hopfield_softmax_class_pooling():
    """Valida o mecanismo de Softmax Class Pooling (Atenção Agregada da Hopfield)."""
    from src.treinamento.avaliador_hopfield import AvaliadorHopfield

    # 4 protótipos: 2 da Classe 1 e 2 da Classe 2
    prototipos = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],  # Classe 1 - Subcluster A
            [1.0, 0.9, 0.1, 0.0],  # Classe 1 - Subcluster B
            [0.0, 0.0, 1.0, 1.0],  # Classe 2 - Subcluster A
            [0.0, 0.1, 0.9, 1.0],  # Classe 2 - Subcluster B
        ],
        dtype=np.float32,
    )
    meta = [(1, 0), (1, 1), (2, 0), (2, 1)]

    # Query que compartilha atenção entre os dois subclusters da Classe 1
    query = np.array([[1.0, 0.95, 0.05, 0.0]], dtype=np.float32)

    rede = ModernHopfieldNetwork(beta=5.0, n_iters=1, binary=False)
    rede.store(prototipos)

    # 1. Valida retrieve com return_attention_weights=True
    _res_rec, att = rede.retrieve(query, return_attention_weights=True)
    assert att.shape == (1, 4)
    assert np.isclose(att.sum(), 1.0, atol=1e-4)

    # 2. Valida compute_attention_weights direto
    att_direct = rede.compute_attention_weights(query)
    np.testing.assert_allclose(att, att_direct, atol=1e-5)

    # 3. Valida avaliar_por_atencao
    avaliador = AvaliadorHopfield(
        padroes=prototipos,
        classes=[1, 2],
        nc=2,
        meta=meta,
    )
    avaliador.avaliar_por_atencao(att, labels=[1])

    assert avaliador.acuracia == 1.0
    assert avaliador.f1_macro == 1.0
    assert avaliador.prob_classes is not None
    assert avaliador.prob_classes.shape == (1, 2)
    # A probabilidade somada da Classe 1 deve ser superior à da Classe 2
    assert avaliador.prob_classes[0, 0] > avaliador.prob_classes[0, 1]
    assert avaliador.prob_classes[0, 0] > 0.8


def test_avaliador_hopfield_metricas_por_classe_e_relatorio_completo():
    """Valida o cálculo de precision, recall, f1-score, support e médias globais no AvaliadorHopfield."""
    from src.treinamento.avaliador_hopfield import AvaliadorHopfield

    # 4 protótipos em 2 classes
    prototipos = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 0.9, 1.0],
        ],
        dtype=np.float32,
    )
    meta = [(1, 0), (1, 1), (2, 0), (2, 1)]
    nomes = ["Astrocyte", "Microglia"]

    avaliador = AvaliadorHopfield(
        padroes=prototipos,
        classes=[1, 2],
        nc=2,
        nomes_classes=nomes,
        meta=meta,
    )

    # 4 células de teste: 2 da classe 1 e 2 da classe 2
    # Atenção: células 0 e 1 atendem mais à classe 1; células 2 e 3 atendem mais à classe 2
    # Simulamos 1 erro proposital: célula 3 (verdadeira 2) com atenção maior na classe 1
    att = np.array(
        [
            [0.6, 0.3, 0.05, 0.05],  # Célula 0: pred=1, true=1 (TP classe 1)
            [0.4, 0.5, 0.05, 0.05],  # Célula 1: pred=1, true=1 (TP classe 1)
            [0.05, 0.05, 0.5, 0.4],  # Célula 2: pred=2, true=2 (TP classe 2)
            [
                0.5,
                0.2,
                0.15,
                0.15,
            ],  # Célula 3: pred=1, true=2 (FP classe 1, FN classe 2)
        ],
        dtype=np.float32,
    )
    labels = [1, 1, 2, 2]

    avaliador.avaliar_por_atencao(att, labels=labels)

    # 1. Valida DataFrame por classe
    df_classes = avaliador.metricas_por_classe()
    assert "precision" in df_classes.columns
    assert "recall" in df_classes.columns
    assert "f1_score" in df_classes.columns
    assert "support" in df_classes.columns
    assert "tipo_celular" in df_classes.columns
    assert "classe_id" in df_classes.columns
    # Retrocompatibilidade
    assert "classe" in df_classes.columns
    assert "n_celulas" in df_classes.columns

    assert list(df_classes["tipo_celular"]) == nomes
    assert list(df_classes["support"]) == [2, 2]

    # Classe 1: TP=2, FP=1 -> Precision = 2/3 = 0.6667, Recall = 2/2 = 1.0
    p1 = float(df_classes.loc[df_classes["classe_id"] == 1, "precision"].iloc[0])
    r1 = float(df_classes.loc[df_classes["classe_id"] == 1, "recall"].iloc[0])
    assert np.isclose(p1, 2 / 3, atol=1e-3)
    assert np.isclose(r1, 1.0, atol=1e-3)

    # Classe 2: TP=1, FP=0, FN=1 -> Precision = 1.0, Recall = 1/2 = 0.5
    p2 = float(df_classes.loc[df_classes["classe_id"] == 2, "precision"].iloc[0])
    r2 = float(df_classes.loc[df_classes["classe_id"] == 2, "recall"].iloc[0])
    assert np.isclose(p2, 1.0, atol=1e-3)
    assert np.isclose(r2, 0.5, atol=1e-3)

    # 2. Valida Relatório Completo com Macro e Weighted Avg
    df_completo = avaliador.relatorio_classificacao_completo()
    assert len(df_completo) == 4  # 2 classes + Macro Avg + Weighted Avg
    assert "Macro Avg" in list(df_completo["tipo_celular"])
    assert "Weighted Avg" in list(df_completo["tipo_celular"])

    row_macro = df_completo[df_completo["tipo_celular"] == "Macro Avg"].iloc[0]
    expected_macro_p = (p1 + p2) / 2
    assert np.isclose(float(row_macro["precision"]), expected_macro_p, atol=1e-3)
