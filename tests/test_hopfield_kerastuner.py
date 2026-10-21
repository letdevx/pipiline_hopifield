"""Testes unitários para o módulo HopfieldKerasTuner."""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

from pathlib import Path

import numpy as np
import pytest

from src.treinamento.hopfield_kerastuner import HopfieldKerasTuner


@pytest.fixture
def dados_sinteticos_kt():
    """Gera matrizes sintéticas reduzidas para teste rápido do tuner."""
    rng = np.random.default_rng(42)
    n_train = 60
    n_val = 30
    n_genes = 40
    n_dims = 10
    classes = [1, 2, 3]

    # Matriz binária de treino e projeção SWeeP
    W0_train = (rng.random((n_train, n_genes)) > 0.5).astype(np.float32)
    Wswp_train = rng.normal(0.0, 1.0, (n_train, n_dims)).astype(np.float32)
    labels_train = rng.choice(classes, size=n_train).astype(int)

    # Matriz de validação
    W0_val = (rng.random((n_val, n_genes)) > 0.5).astype(np.float32)
    labels_val = rng.choice(classes, size=n_val).astype(int)

    return {
        "w0": W0_train,
        "wswp": Wswp_train,
        "labels": labels_train,
        "x_val": W0_val,
        "y_val": labels_val,
        "classes": classes,
    }


def test_hopfield_kerastuner_busca_e_melhor_modelo(dados_sinteticos_kt, tmp_path: Path):
    """Valida a execução de trials, cache LRU e construção do melhor modelo."""
    dir_kt = tmp_path / "kt_output"

    tuner = HopfieldKerasTuner(
        w0=dados_sinteticos_kt["w0"],
        wswp=dados_sinteticos_kt["wswp"],
        labels=dados_sinteticos_kt["labels"],
        x_val=dados_sinteticos_kt["x_val"],
        y_val=dados_sinteticos_kt["y_val"],
        classes=dados_sinteticos_kt["classes"],
        algorithm="random",
        max_trials=2,
        directory=str(dir_kt),
        project_name="teste_unitario",
        seed=42,
    )

    # Executa a busca
    tuner.search()

    # Checa histórico de trials
    assert len(tuner.historico_trials) == 2
    for t in tuner.historico_trials:
        assert "val_f1_macro" in t["metricas"]
        assert 0.0 <= t["metricas"]["val_f1_macro"] <= 1.0
        assert "nc" in t["hiperparametros"]
        assert "beta" in t["hiperparametros"]

    # Checa melhor configuração
    best_cfg = tuner.obter_melhor_configuracao()
    assert "nc" in best_cfg
    assert "beta" in best_cfg
    assert best_cfg["nc"] >= 10

    # Constrói melhor modelo
    rede, padroes, meta, _cfg = tuner.construir_melhor_modelo()
    assert rede.patterns.shape[0] == padroes.shape[0]
    assert len(meta) == padroes.shape[0]
    assert padroes.shape[1] == dados_sinteticos_kt["w0"].shape[1]


def test_hopfield_kerastuner_exportacao_json_e_painel(
    dados_sinteticos_kt, tmp_path: Path
):
    """Valida a exportação em JSON e a geração do painel de convergência."""
    dir_kt = tmp_path / "kt_output_export"

    tuner = HopfieldKerasTuner(
        w0=dados_sinteticos_kt["w0"],
        wswp=dados_sinteticos_kt["wswp"],
        labels=dados_sinteticos_kt["labels"],
        x_val=dados_sinteticos_kt["x_val"],
        y_val=dados_sinteticos_kt["y_val"],
        classes=dados_sinteticos_kt["classes"],
        algorithm="random",
        max_trials=2,
        directory=str(dir_kt),
        project_name="teste_export",
        seed=42,
    )
    tuner.search()

    path_json = tmp_path / "historico.json"
    tuner.exportar_historico_json(path_json)
    assert path_json.exists()
    assert path_json.stat().st_size > 0

    path_png = tmp_path / "painel_convergencia.png"
    fig = tuner.gerar_painel_convergencia(path_png)
    assert fig is not None
    assert path_png.exists()
    assert path_png.stat().st_size > 0
