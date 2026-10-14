"""Testes unitários para o componente AuditorOverfittingCrossDataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.treinamento.diagnostico_overfitting import (
    AuditorOverfittingCrossDataset,
    ResultadoDiagnosticoCrossDataset,
)


class MockModeloHopfieldPerfeito:
    """Mock da ModernHopfieldNetwork que reproduz perfeitamente o padrão com alta confiança."""

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features

    def retrieve(
        self,
        queries: Any,
        batch_size: int = 1024,
        mask_sentinela_ausentes: Any = None,
        fill_value: float = 0.5,
        return_probabilities: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        queries_arr = queries.toarray() if sp.issparse(queries) else np.asarray(queries)

        rec = (queries_arr > 0.0).astype(np.float32)
        # Probabilidades contínuas correlacionadas com a verdade
        prob = np.where(queries_arr > 0.0, 0.95, 0.05).astype(np.float32)
        return rec, prob


class MockModeloHopfieldAleatorio:
    """Mock da ModernHopfieldNetwork que emite ruído descalibrado (simulando overfitting/colapso)."""

    def __init__(self, n_features: int, seed: int = 42) -> None:
        self.n_features = n_features
        self.rng = np.random.default_rng(seed)

    def retrieve(
        self,
        queries: Any,
        batch_size: int = 1024,
        mask_sentinela_ausentes: Any = None,
        fill_value: float = 0.5,
        return_probabilities: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_q = queries.shape[0]
        # Predição puramente aleatória
        prob = self.rng.uniform(0.0, 1.0, size=(n_q, self.n_features)).astype(
            np.float32
        )
        rec = (prob > 0.5).astype(np.float32)
        return rec, prob


def test_selecionar_genes_holdout_taxa_e_marcadores() -> None:
    """Valida a amostragem determinística com inclusão estrita de marcadores canônicos."""
    n_genes = 100
    # 20 genes ausentes originais
    mask_ausentes = np.zeros(n_genes, dtype=bool)
    mask_ausentes[:20] = True

    # Marcadores das linhagens nos genes presentes (ex.: genes 25 e 30)
    marcadores = {1: [25, 30], 2: [40, 45]}

    auditor = AuditorOverfittingCrossDataset(
        modelo=None,
        seed=42,
    )

    mask_holdout = auditor.selecionar_genes_holdout(
        n_genes=n_genes,
        mask_ausentes=mask_ausentes,
        taxa_mascaramento=0.10,  # 10% dos 80 presentes = 8 genes
        marcadores_indices=marcadores,
    )

    assert mask_holdout.shape == (n_genes,)
    assert mask_holdout.dtype == bool
    # Nenhum gene ausente original pode ser selecionado para holdout
    assert not np.any(mask_holdout[:20])

    # Todos os marcadores presentes devem ter sido incluídos
    assert mask_holdout[25]
    assert mask_holdout[30]
    assert mask_holdout[40]
    assert mask_holdout[45]

    # Total de selecionados deve ser exatamente 8
    assert np.sum(mask_holdout) == 8


def test_executar_auditoria_aprovado_mock(tmp_path: Path) -> None:
    """Valida o diagnóstico completo emitindo status APROVADO em modelo com alta fidelidade."""
    n_celulas = 200
    n_genes = 60
    rng = np.random.default_rng(42)

    # Matriz sintética com esparsidade típica de scRNA-seq (~15% de 1s)
    matriz_densa = (rng.uniform(0, 1, size=(n_celulas, n_genes)) > 0.85).astype(
        np.float32
    )

    # Simula classes 1 e 2
    classes_reais = np.array([1] * 100 + [2] * 100)
    nomes_classes = {1: "Astrocyte", 2: "Microglia"}
    marcadores = {1: [10, 11], 2: [20, 21]}

    # Ativa marcadores nas classes correspondentes
    matriz_densa[:100, 10:12] = 1.0
    matriz_densa[100:, 20:22] = 1.0

    mask_ausentes = np.zeros(n_genes, dtype=bool)
    mask_ausentes[:10] = True  # 10 genes ausentes

    modelo = MockModeloHopfieldPerfeito(n_features=n_genes)
    auditor = AuditorOverfittingCrossDataset(modelo=modelo, seed=42)

    resultado = auditor.executar_auditoria(
        matriz_alvo=matriz_densa,
        mask_ausentes=mask_ausentes,
        classes_reais=classes_reais,
        nomes_classes=nomes_classes,
        marcadores_indices=marcadores,
        taxa_mascaramento=0.10,
        batch_size=50,
    )

    assert isinstance(resultado, ResultadoDiagnosticoCrossDataset)
    assert resultado.status == "APROVADO"
    assert resultado.roc_auc >= 0.70
    assert resultado.razao_pr_auc_basal >= 2.0
    assert resultado.f1_ativo >= 0.45
    assert resultado.n_genes_mascarados > 0
    assert len(resultado.f1_por_tipo_celular) == 2
    assert "Astrocyte" in resultado.f1_por_tipo_celular
    assert "Microglia" in resultado.f1_por_tipo_celular


def test_executar_auditoria_reprovado_aleatorio() -> None:
    """Valida a detecção formal de colapso/overfitting emitindo status REPROVADO em modelo ruidoso."""
    n_celulas = 150
    n_genes = 50
    rng = np.random.default_rng(123)

    matriz = (rng.uniform(0, 1, size=(n_celulas, n_genes)) > 0.85).astype(np.float32)
    mask_ausentes = np.zeros(n_genes, dtype=bool)
    mask_ausentes[:5] = True

    modelo = MockModeloHopfieldAleatorio(n_features=n_genes, seed=999)
    auditor = AuditorOverfittingCrossDataset(
        modelo=modelo,
        limiar_roc_auc_alerta=0.60,
        seed=42,
    )

    resultado = auditor.executar_auditoria(
        matriz_alvo=matriz,
        mask_ausentes=mask_ausentes,
        taxa_mascaramento=0.15,
        batch_size=50,
    )

    assert resultado.status == "REPROVADO"
    assert resultado.roc_auc < 0.60
    assert any("Reprovado:" in a for a in resultado.alertas)
    assert len(resultado.recomendacoes) > 0


def test_executar_auditoria_matriz_esparsa_csr() -> None:
    """Valida a execução OOM-Safe com matriz esparsa CSR do SciPy."""
    n_celulas = 80
    n_genes = 40
    rng = np.random.default_rng(77)

    dados_densos = (rng.uniform(0, 1, size=(n_celulas, n_genes)) > 0.90).astype(
        np.float32
    )
    matriz_csr = sp.csr_matrix(dados_densos)

    mask_ausentes = np.zeros(n_genes, dtype=bool)
    mask_ausentes[:5] = True

    modelo = MockModeloHopfieldPerfeito(n_features=n_genes)
    auditor = AuditorOverfittingCrossDataset(modelo=modelo, seed=42)

    resultado = auditor.executar_auditoria(
        matriz_alvo=matriz_csr,
        mask_ausentes=mask_ausentes,
        taxa_mascaramento=0.10,
        batch_size=20,
    )

    assert resultado.n_coordenadas_auditadas == n_celulas * resultado.n_genes_mascarados
    assert resultado.roc_auc > 0.50


def test_painel_grafico_e_exportacao_relatorio(tmp_path: Path) -> None:
    """Valida a geração de gráficos PNG e exportação em CSV e JSON."""
    n_celulas = 100
    n_genes = 50
    rng = np.random.default_rng(88)

    matriz = (rng.uniform(0, 1, size=(n_celulas, n_genes)) > 0.85).astype(np.float32)
    modelo = MockModeloHopfieldPerfeito(n_features=n_genes)
    auditor = AuditorOverfittingCrossDataset(modelo=modelo, seed=42)

    resultado = auditor.executar_auditoria(
        matriz_alvo=matriz,
        taxa_mascaramento=0.10,
        batch_size=50,
    )

    # 1. Painel Gráfico
    path_png = tmp_path / "painel_teste.png"
    fig = auditor.gerar_painel_grafico(
        resultado=resultado,
        path_saida=path_png,
        titulo="Teste Diagnóstico Overfitting",
    )
    assert fig is not None
    assert path_png.exists()
    assert path_png.stat().st_size > 0

    # 2. Exportação de Relatórios
    path_csv = tmp_path / "metricas_teste.csv"
    path_json = tmp_path / "metricas_teste.json"
    arquivos = auditor.exportar_relatorio(
        resultado=resultado,
        path_csv=path_csv,
        path_json=path_json,
    )

    assert "csv" in arquivos and path_csv.exists()
    assert "json" in arquivos and path_json.exists()

    df = pd.read_csv(path_csv)
    assert "ROC-AUC" in df["metrica"].values
    assert "Status_Final" in df["metrica"].values

    with open(path_json, encoding="utf-8") as f:
        dados_json = json.load(f)
    assert dados_json["status"] == resultado.status
    assert "roc_auc" in dados_json["metricas_globais"]


def test_auditor_cross_varredura_limiar_otimo(tmp_path: Path) -> None:
    """Valida a varredura automática de limiar ótimo e métricas calibradas."""
    n_celulas = 80
    n_genes = 40
    rng = np.random.default_rng(99)

    # Matriz com esparsidade e sinal conhecido
    matriz = (rng.uniform(0, 1, size=(n_celulas, n_genes)) > 0.80).astype(np.float32)
    modelo = MockModeloHopfieldPerfeito(n_features=n_genes)
    auditor = AuditorOverfittingCrossDataset(
        modelo=modelo,
        limiar_roc_auc_aprovado=0.70,
        limiar_razao_pr_auc_aprovado=2.0,
        limiar_f1_ativo_aprovado=0.45,
        seed=42,
    )

    resultado = auditor.executar_auditoria(
        matriz_alvo=matriz,
        taxa_mascaramento=0.15,
        batch_size=40,
    )

    # Verifica novos atributos de calibração
    assert hasattr(resultado, "limiar_otimo")
    assert 0.0 <= resultado.limiar_otimo <= 1.0
    assert hasattr(resultado, "f1_ativo_calibrado")
    assert resultado.f1_ativo_calibrado >= resultado.f1_ativo
    assert hasattr(resultado, "status_calibrado")
    assert resultado.status_calibrado in ["APROVADO", "ALERTA", "REPROVADO"]

    # Exportação deve conter as novas métricas calibradas
    path_csv = tmp_path / "metricas_calibradas.csv"
    path_json = tmp_path / "metricas_calibradas.json"
    auditor.exportar_relatorio(
        resultado=resultado,
        path_csv=path_csv,
        path_json=path_json,
    )

    df = pd.read_csv(path_csv)
    assert "Limiar_Otimo" in df["metrica"].values
    assert "F1_Ativo_Calibrado" in df["metrica"].values
    assert "Status_Calibrado" in df["metrica"].values

    with open(path_json, encoding="utf-8") as f:
        dados_json = json.load(f)
    assert "limiar_otimo" in dados_json["metricas_globais"]
    assert "f1_ativo_calibrado" in dados_json["metricas_globais"]
    assert "status_calibrado" in dados_json
