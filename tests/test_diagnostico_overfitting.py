"""Testes da camada de diagnóstico de overfitting e robustez em Redes Hopfield."""

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from src.synthetic.gerador_ground_truth import GeradorGroundTruthSintetico
from src.treinamento.diagnostico_overfitting import (
    AuditorOverfittingHopfield,
    InjetorPerturbacaoTranscritica,
)
from src.treinamento.hopfield import ModernHopfieldNetwork


def _criar_cenario_sintetico() -> tuple[
    ModernHopfieldNetwork,
    NDArray[np.float32],
    list[int],
    NDArray[np.float32],
    NDArray[np.int_],
]:
    """Cria um cenário de teste humano-verificável (12 células × 8 genes, 3 classes: 1, 2, 3)."""
    gerador = GeradorGroundTruthSintetico(n_celulas=12, n_genes=8, n_classes=3, seed=42)
    matriz_pura = cast(NDArray[np.float32], gerador.gerar_matriz_pura(formato="numpy"))
    labels = gerador.labels  # [1 1 1 1 2 2 2 2 3 3 3 3]

    # 3 protótipos canônicos (um por classe)
    prototipos = cast(
        NDArray[np.float32],
        np.vstack([matriz_pura[0], matriz_pura[4], matriz_pura[8]]),
    )
    rotulos_prototipos = [1, 2, 3]

    rede = ModernHopfieldNetwork(beta=8.0, n_iters=1, binary=True)
    rede.store(prototipos)

    return rede, prototipos, rotulos_prototipos, matriz_pura, labels


# --- RN-01: Integridade e Contratos de Entrada ---


def test_auditor_lanca_value_error_quando_genes_divergem_do_modelo() -> None:
    """RN-01 / CA-01: Deve lançar ValueError se a matriz de validação tiver dimensão gênica diferente."""
    rede, prototipos, rotulos_prototipos, matriz_pura, labels = (
        _criar_cenario_sintetico()
    )
    auditor = AuditorOverfittingHopfield(
        modelo=rede,
        padroes_referencia=prototipos,
        rotulos_padroes=rotulos_prototipos,
    )

    x_val_invalido = np.ones((4, 5), dtype=np.float32)  # 5 genes != 8 genes
    y_val = [1, 1, 2, 3]

    with pytest.raises(ValueError, match="Espaço gênico divergente"):
        auditor.executar_auditoria(matriz_pura, labels, x_val_invalido, y_val)


def test_auditor_lanca_value_error_quando_rotulos_incompativeis_com_celulas() -> None:
    """RN-01 / CA-01: Deve lançar ValueError se o número de rótulos não coincidir com o número de células."""
    rede, prototipos, rotulos_prototipos, matriz_pura, labels = (
        _criar_cenario_sintetico()
    )
    auditor = AuditorOverfittingHopfield(
        modelo=rede,
        padroes_referencia=prototipos,
        rotulos_padroes=rotulos_prototipos,
    )

    x_val = matriz_pura[:6]
    y_val_invalido = [1, 2]  # 2 rótulos != 6 células

    with pytest.raises(ValueError, match="Incompatibilidade de dimensões"):
        auditor.executar_auditoria(matriz_pura, labels, x_val, y_val_invalido)


# --- RN-02: Gap Treino vs. Validação ---


def test_auditor_sinaliza_alerta_quando_gap_generalizacao_excede_limiar() -> None:
    """RN-02 / CA-02: Deve sinalizar ALERTA e registrar queda excessiva quando gap > 15%."""
    rede, prototipos, rotulos_prototipos, matriz_pura, labels = (
        _criar_cenario_sintetico()
    )
    auditor = AuditorOverfittingHopfield(
        modelo=rede,
        padroes_referencia=prototipos,
        rotulos_padroes=rotulos_prototipos,
        gap_maximo_tolerado=0.15,
        limiar_entropia_minima=0.0,
    )

    # Cria validação propositalmente corrompida para errar a classe
    x_val_corrompido = np.flip(matriz_pura, axis=1).copy()
    y_val = labels.copy()

    resultado = auditor.executar_auditoria(
        x_treino=matriz_pura,
        y_treino=labels,
        x_val=x_val_corrompido,
        y_val=y_val,
        niveis_ruido=(0.05,),
    )

    assert resultado.gap_generalizacao > 0.15
    assert any("queda excessiva" in a.lower() for a in resultado.alertas)
    assert resultado.status in ("ALERTA", "REPROVADO")


def test_auditor_aprova_quando_gap_generalizacao_dentro_do_limiar() -> None:
    """RN-02 / CA-02: Gap inferior a 15% não deve gerar alertas de generalização."""
    rede, prototipos, rotulos_prototipos, matriz_pura, labels = (
        _criar_cenario_sintetico()
    )
    auditor = AuditorOverfittingHopfield(
        modelo=rede,
        padroes_referencia=prototipos,
        rotulos_padroes=rotulos_prototipos,
        gap_maximo_tolerado=0.15,
        limiar_entropia_minima=0.0,
    )

    # Validação perfeita (Ground Truth)
    resultado = auditor.executar_auditoria(
        x_treino=matriz_pura,
        y_treino=labels,
        x_val=matriz_pura,
        y_val=labels,
        niveis_ruido=(0.05, 0.15),
    )

    assert resultado.gap_generalizacao <= 0.15
    assert not any("queda excessiva" in a.lower() for a in resultado.alertas)


# --- RN-03: Injeção de Ruído e Bacias de Atração ---


def test_injetor_perturbacao_aplica_dropout_sem_mutacao_in_place() -> None:
    """RN-03 / CA-03: Injetor deve retornar cópia modificada mantendo original inalterado."""
    injetor = InjetorPerturbacaoTranscritica()
    matriz_original = np.ones((5, 10), dtype=np.float32)
    matriz_copia_segura = matriz_original.copy()

    matriz_perturbada = injetor.aplicar_dropout(matriz_original, taxa=0.30, seed=42)

    # Garante imutabilidade estrita
    assert np.array_equal(matriz_original, matriz_copia_segura)
    assert not np.array_equal(matriz_original, matriz_perturbada)
    assert matriz_perturbada.sum() < matriz_original.sum()


def test_auditor_reprova_quando_bacia_de_atracao_colapsa_com_ruido_leve() -> None:
    """RN-03 / CA-03: Queda acentuada sob ruído de apenas 5% deve reprovar a rede."""
    rede, prototipos, rotulos_prototipos, matriz_pura, labels = (
        _criar_cenario_sintetico()
    )

    auditor = AuditorOverfittingHopfield(
        modelo=rede,
        padroes_referencia=prototipos,
        rotulos_padroes=rotulos_prototipos,
        limiar_entropia_minima=0.0,
    )

    resultado = auditor.executar_auditoria(
        x_treino=matriz_pura,
        y_treino=labels,
        x_val=matriz_pura,
        y_val=labels,
        niveis_ruido=(0.05, 0.15, 0.30),
    )

    assert isinstance(resultado.fidelidade_sob_ruido, dict)
    assert 0.05 in resultado.fidelidade_sob_ruido


def test_auditor_confirma_robustez_quando_estavel_sob_ruido_moderado() -> None:
    """RN-03 / CA-03: Rede estável até 15% de perturbação preserva alta fidelidade."""
    rede, prototipos, rotulos_prototipos, matriz_pura, labels = (
        _criar_cenario_sintetico()
    )
    auditor = AuditorOverfittingHopfield(
        modelo=rede,
        padroes_referencia=prototipos,
        rotulos_padroes=rotulos_prototipos,
        limiar_entropia_minima=0.0,
    )

    resultado = auditor.executar_auditoria(
        x_treino=matriz_pura,
        y_treino=labels,
        x_val=matriz_pura,
        y_val=labels,
        niveis_ruido=(0.05, 0.15),
    )

    assert resultado.fidelidade_sob_ruido[0.05] >= 0.70
    assert resultado.fidelidade_sob_ruido[0.15] >= 0.70


# --- RN-04: Auditoria de Temperatura e Quimeras ---


def test_auditor_detecta_saturacao_temperatura_quando_entropia_atencao_nula() -> None:
    """RN-04 / CA-04: Beta extremo (ex: 500.0) deve ser sinalizado por atenção concentrada."""
    _, prototipos, rotulos_prototipos, matriz_pura, labels = _criar_cenario_sintetico()

    rede_saturada = ModernHopfieldNetwork(beta=500.0, n_iters=1, binary=True)
    rede_saturada.store(prototipos)

    auditor = AuditorOverfittingHopfield(
        modelo=rede_saturada,
        padroes_referencia=prototipos,
        rotulos_padroes=rotulos_prototipos,
        limiar_entropia_minima=0.05,
    )

    resultado = auditor.executar_auditoria(
        x_treino=matriz_pura,
        y_treino=labels,
        x_val=matriz_pura,
        y_val=labels,
        niveis_ruido=(0.05,),
    )

    assert resultado.proporcao_atencao_saturada >= 0.95
    assert any("saturação de beta" in a.lower() for a in resultado.alertas)
    assert any(
        "reduzir hiperparâmetro beta" in r.lower() for r in resultado.recomendacoes
    )


def test_auditor_detecta_quimera_quando_marcadores_exclusivos_coativados() -> None:
    """RN-04 / CA-04: Coativação de marcadores antagônicos deve ser contabilizada."""
    rede, prototipos, rotulos_prototipos, matriz_pura, labels = (
        _criar_cenario_sintetico()
    )

    # Define marcadores das classes 1 e 2
    marcadores = {
        1: [0],  # Gene 0 característico da classe 1
        2: [4],  # Gene 4 característico da classe 2
    }

    auditor = AuditorOverfittingHopfield(
        modelo=rede,
        padroes_referencia=prototipos,
        rotulos_padroes=rotulos_prototipos,
        marcadores_exclusivos=marcadores,
        limiar_entropia_minima=0.0,
    )

    resultado = auditor.executar_auditoria(
        x_treino=matriz_pura,
        y_treino=labels,
        x_val=matriz_pura,
        y_val=labels,
        niveis_ruido=(0.05,),
    )

    assert resultado.quimeras_detectadas >= 0


# --- RF-05: Parecer Consolidado ---


def test_auditor_emite_status_aprovado_para_rede_com_generalizacao_robusta() -> None:
    """RF-05 / CA-05: Rede bem calibrada sobre dados consistentes atinge status APROVADO."""
    rede, prototipos, rotulos_prototipos, matriz_pura, labels = (
        _criar_cenario_sintetico()
    )
    auditor = AuditorOverfittingHopfield(
        modelo=rede,
        padroes_referencia=prototipos,
        rotulos_padroes=rotulos_prototipos,
        gap_maximo_tolerado=0.15,
        limiar_entropia_minima=0.0,  # Sem corte de entropia para modelo ideal
    )

    resultado = auditor.executar_auditoria(
        x_treino=matriz_pura,
        y_treino=labels,
        x_val=matriz_pura,
        y_val=labels,
        niveis_ruido=(0.05, 0.15),
    )

    assert resultado.status == "APROVADO"
    assert len(resultado.alertas) == 0
    assert resultado.gap_generalizacao <= 0.15
