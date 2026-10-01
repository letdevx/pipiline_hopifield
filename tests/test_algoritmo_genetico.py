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

    # Erro para população muito pequena
    with pytest.raises(ValueError, match="tam_populacao deve ser pelo menos 4"):
        ConfiguracaoAG(tam_populacao=2)


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
        beta=10.0,
        threshold=-0.1,
        n_iters=1,
        normalize=False,
        estrategia="kmeans_fixo",
    )
    pai2 = IndividuoHopfield(
        nc=30,
        k_vizinhos=5,
        beta=40.0,
        threshold=0.2,
        n_iters=2,
        normalize=True,
        estrategia="kmeans_dinamico",
    )

    filho = otimizador.cruzar(pai1, pai2)
    assert filho.nc in (10, 30) or (10 <= filho.nc <= 30)
    assert filho.n_iters in (1, 2)

    mutado = otimizador.mutar(filho)
    # Garante que os limites físicos do genótipo são respeitados pós-mutação
    assert 5 <= mutado.nc <= 45
    assert 1 <= mutado.k_vizinhos <= 10
    assert 1.0 <= mutado.beta <= 80.0
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
