"""Testes automatizados do módulo de Algoritmo Genético para Redes Hopfield.

Cobre a geração de indivíduos, operadores genéticos (crossover, mutação),
elitismo, sistema de cache LRU de protótipos e ciclo evolutivo determinístico.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from treinamento.algoritmo_genetico import (
    ConfiguracaoAG,
    IndividuoHopfield,
    OtimizadorGeneticoHopfield,
)


def test_individuo_hopfield_imutabilidade_e_dicionario() -> None:
    """Verifica se o indivíduo é imutável (frozen) e serializável para dicionário."""
    ind = IndividuoHopfield(
        nc=20,
        k_vizinhos=3,
        beta=15.0,
        threshold=0.0,
        n_iters=1,
        normalize=False,
        estrategia="kmeans_fixo",
        fitness=0.82,
        f1_val=0.85,
        gap_generalizacao=0.04,
        f1_ruido=0.78,
        entropia_atencao=0.18,
        quimeras=0,
    )

    assert ind.nc == 20
    assert ind.beta == 15.0
    assert ind.fitness == 0.82

    # Imutabilidade estrita (dataclass frozen)
    with pytest.raises(FrozenInstanceError):
        ind.nc = 30  # type: ignore[misc]

    d = ind.para_dicionario()
    assert isinstance(d, dict)
    assert d["nc"] == 20
    assert d["beta"] == 15.0
    assert d["estrategia"] == "kmeans_fixo"


def test_configuracao_ag_validacoes() -> None:
    """Testa valores padrão e limites da configuração do AG."""
    cfg = ConfiguracaoAG(tam_populacao=10, n_geracoes=3, seed=42)
    assert cfg.tam_populacao == 10
    assert cfg.n_geracoes == 3
    assert cfg.p_crossover == 0.85
    assert cfg.elitismo == 2
    assert cfg.beta_min == 0.1
    assert cfg.beta_max == 5.0
    assert cfg.metrica_f1 == "macro"
    assert cfg.escalar_por_raiz_d is True

    # Erro para população muito pequena
    with pytest.raises(ValueError, match="tam_populacao deve ser pelo menos 4"):
        ConfiguracaoAG(tam_populacao=2)

    # Erro para beta_min <= 0
    with pytest.raises(ValueError, match="beta_min deve ser estritamente positivo"):
        ConfiguracaoAG(beta_min=-0.5)

    # Erro para beta_min > beta_max
    with pytest.raises(ValueError, match="beta_min não pode ser maior que beta_max"):
        ConfiguracaoAG(beta_min=10.0, beta_max=2.0)

    # Erro para metrica_f1 inválida
    with pytest.raises(ValueError, match="metrica_f1 deve ser 'macro' ou 'weighted'"):
        ConfiguracaoAG(metrica_f1="accuracy")  # type: ignore[arg-type]


def test_operadores_crossover_e_mutacao() -> None:
    """Valida que crossover e mutação produzem filhos válidos dentro dos limites."""
    cfg = ConfiguracaoAG(tam_populacao=10, n_geracoes=2, seed=123)
    otimizador = OtimizadorGeneticoHopfield(
        w0=np.zeros((10, 10), dtype=np.float32),
        wswp=np.zeros((10, 5), dtype=np.float32),
        labels=np.ones(10, dtype=int),
        config=cfg,
    )

    pai1 = IndividuoHopfield(
        nc=10,
        k_vizinhos=1,
        beta=1.0,
        threshold=-0.1,
        n_iters=1,
        normalize=False,
        estrategia="kmeans_fixo",
    )
    pai2 = IndividuoHopfield(
        nc=30,
        k_vizinhos=5,
        beta=4.0,
        threshold=0.2,
        n_iters=2,
        normalize=True,
        estrategia="kmeans_dinamico",
    )

    filho = otimizador.cruzar(pai1, pai2)
    assert filho.nc in (10, 30) or (10 <= filho.nc <= 30)
    assert filho.n_iters in (1, 2)
    assert 1.0 <= filho.beta <= 4.0

    mutado = otimizador.mutar(filho)
    # Garante que os limites físicos do genótipo são respeitados pós-mutação
    assert 5 <= mutado.nc <= 45
    assert 1 <= mutado.k_vizinhos <= 10
    assert cfg.beta_min <= mutado.beta <= cfg.beta_max
    assert -0.3 <= mutado.threshold <= 0.5
    assert 1 <= mutado.n_iters <= 3
    assert isinstance(mutado.normalize, bool)
    assert mutado.estrategia in ("kmeans_fixo", "kmeans_dinamico")


def test_cache_prototipos_lru() -> None:
    """Testa se o cache de protótipos evita re-extração desnecessária."""
    cfg = ConfiguracaoAG(tam_populacao=4, n_geracoes=1, seed=42)
    # Matriz sintética com 2 classes
    labels = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    w0 = np.eye(8, dtype=np.float32)
    wswp = np.random.default_rng(42).normal(size=(8, 4)).astype(np.float32)

    otimizador = OtimizadorGeneticoHopfield(
        w0=w0,
        wswp=wswp,
        labels=labels,
        classes=[1, 2],
        config=cfg,
    )

    # Primeira extração: cache miss
    padroes1, meta1 = otimizador.obter_padroes(
        nc=2, k_vizinhos=1, estrategia="kmeans_fixo"
    )
    assert (2, 1, "kmeans_fixo") in otimizador.cache_padroes
    assert otimizador.estatisticas_cache["misses"] == 1
    assert otimizador.estatisticas_cache["hits"] == 0

    # Segunda extração com os mesmos hiperparâmetros: cache hit
    padroes2, meta2 = otimizador.obter_padroes(
        nc=2, k_vizinhos=1, estrategia="kmeans_fixo"
    )
    assert otimizador.estatisticas_cache["hits"] == 1
    np.testing.assert_array_equal(padroes1, padroes2)
    assert meta1 == meta2


def test_otimizador_genetico_ciclo_evolucao_micro() -> None:
    """Valida um ciclo micro-evolutivo determinístico com elitismo."""
    # Gera dados sintéticos separáveis de 2 classes
    rng = np.random.default_rng(42)
    n_celulas = 20
    n_genes = 12

    labels = np.array([1] * 10 + [2] * 10)
    w0 = np.zeros((n_celulas, n_genes), dtype=np.float32)
    w0[:10, :6] = 1.0  # Classe 1 ativa genes 0..5
    w0[10:, 6:] = 1.0  # Classe 2 ativa genes 6..11

    wswp = rng.normal(size=(n_celulas, 6)).astype(np.float32)
    wswp[:10, 0] += 5.0
    wswp[10:, 0] -= 5.0

    # Dados de validação
    x_val = np.copy(w0)
    y_val = np.copy(labels)

    cfg = ConfiguracaoAG(
        tam_populacao=6,
        n_geracoes=2,
        elitismo=2,
        seed=42,
    )

    otimizador = OtimizadorGeneticoHopfield(
        w0=w0,
        wswp=wswp,
        labels=labels,
        classes=[1, 2],
        x_val=x_val,
        y_val=y_val,
        config=cfg,
    )

    resultado = otimizador.evoluir()

    assert resultado["campeao"] is not None
    assert isinstance(resultado["campeao"], IndividuoHopfield)
    assert resultado["campeao"].fitness > 0.0
    assert len(resultado["historico_geracoes"]) == 2

    # Verifica monotonicidade do elitismo (melhor fitness da geração 1 <= geração 2)
    fit_g1 = resultado["historico_geracoes"][0]["melhor_fitness"]
    fit_g2 = resultado["historico_geracoes"][1]["melhor_fitness"]
    assert fit_g2 >= fit_g1 - 1e-6


def test_f1_macro_penaliza_colapso_classe_minoritaria() -> None:
    """Verifica se F1 macro penaliza a perda da classe minoritária mais que F1 ponderado."""
    from sklearn.metrics import f1_score

    # Cenário scRNA-seq desbalanceado: 90 células classe 1, 10 células classe 2
    y_true = np.array([1] * 90 + [2] * 10)

    # Predição com colapso total da classe 2 (todas preditas como 1)
    y_pred_colapso = np.array([1] * 100)

    f1_macro = float(f1_score(y_true, y_pred_colapso, average="macro", zero_division=0))
    f1_weighted = float(
        f1_score(y_true, y_pred_colapso, average="weighted", zero_division=0)
    )

    # F1 ponderado mascara o erro (classe 1 domina, score > 0.85)
    # F1 macro reflete a perda severa da linhagem minoritária (score < 0.50)
    assert f1_weighted > 0.85
    assert f1_macro < 0.50
    assert f1_macro < f1_weighted - 0.35


def test_escalonamento_raiz_d_hopfield() -> None:
    """Valida se scale_by_dim normaliza adequadamente os logits pelo fator sqrt(D)."""
    from treinamento.hopfield import ModernHopfieldNetwork

    d_dim = 10000

    # 1 padrão bipolar
    padrao = np.ones((1, d_dim), dtype=np.float32)
    query = np.ones((1, d_dim), dtype=np.float32)

    rede_pura = ModernHopfieldNetwork(
        beta=1.0, binary=False, normalize=False, scale_by_dim=False
    )
    rede_pura.store(padrao)

    rede_escalonada = ModernHopfieldNetwork(
        beta=1.0, binary=False, normalize=False, scale_by_dim=True
    )
    rede_escalonada.store(padrao)

    # Produto interno puro: 10000
    # Com scale_by_dim: 10000 / sqrt(10000) = 100
    att_pura = rede_pura.compute_attention_weights(query)
    att_esc = rede_escalonada.compute_attention_weights(query)

    assert att_pura.shape == (1, 1)
    assert att_esc.shape == (1, 1)
    np.testing.assert_allclose(att_pura, att_esc, atol=1e-5)
    assert rede_escalonada.scale_by_dim is True


def test_algoritmo_genetico_nc_multiplos_de_cinco() -> None:
    """Verifica se os indivíduos gerados e mutados mantêm nc em múltiplos de 5 para otimização de cache."""
    cfg = ConfiguracaoAG(tam_populacao=12, n_geracoes=2, seed=42)
    otimizador = OtimizadorGeneticoHopfield(
        w0=np.zeros((10, 10), dtype=np.float32),
        wswp=np.zeros((10, 5), dtype=np.float32),
        labels=np.ones(10, dtype=int),
        config=cfg,
    )

    pop = otimizador.gerar_populacao_inicial()
    for ind in pop:
        assert ind.nc % 5 == 0, (
            f"nc={ind.nc} deveria ser múltiplo de 5 para maximizar cache hits"
        )

    # Testa que mutações mantêm múltiplos de 5
    for ind in pop:
        mut = otimizador.mutar(ind)
        assert mut.nc % 5 == 0, f"nc mutado={mut.nc} deveria ser múltiplo de 5"
