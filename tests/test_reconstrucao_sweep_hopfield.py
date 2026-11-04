"""Testes unitários para reconstrução contínua de vetores SWeeP 600D via Modern Hopfield Network."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import polars as pl
import pytest
import scipy.sparse as sp

from src.treinamento.hopfield import ModernHopfieldNetwork
from src.treinamento.reconstrucao_sweep_hopfield import (
    avaliar_no_classificador_pan,
    calcular_metricas_geometricas_600d,
    calibrar_hopfield_sweep_completo,
    carregar_matriz_sweep_600d,
    substituir_colunas_faltantes_esparso,
)


def test_substituir_colunas_faltantes_esparso_preserva_formato_csr() -> None:
    """Valida que a substituição de colunas faltantes por 0.5 opera em formato esparso CSR."""
    rng = np.random.default_rng(42)
    dados_binarios = (rng.random((20, 100)) > 0.8).astype(np.float32)
    matriz_csr = sp.csr_matrix(dados_binarios)
    colunas_faltantes = [3, 17, 42, 88]

    modificada = substituir_colunas_faltantes_esparso(
        matriz=matriz_csr,
        indices_colunas=colunas_faltantes,
        fill_value=0.5,
    )

    assert sp.isspmatrix_csr(modificada)
    assert modificada.shape == (20, 100)

    densa_mod = modificada.toarray()
    for col in colunas_faltantes:
        np.testing.assert_allclose(densa_mod[:, col], 0.5, atol=1e-6)

    colunas_preservadas = [c for c in range(100) if c not in set(colunas_faltantes)]
    np.testing.assert_allclose(
        densa_mod[:, colunas_preservadas],
        dados_binarios[:, colunas_preservadas],
        atol=1e-6,
    )


def test_hopfield_continua_mask_self_leave_one_out_em_lotes() -> None:
    """Valida que mask_self=True zera o peso diagonal (i != j) mesmo quando processado em lotes."""
    # 4 células na memória (2 na direção [1, 0, 0, 0] com escalas/ruídos e 2 na direção [0, 0, 1, 0])
    memoria = np.array(
        [
            [10.0, 0.0, 0.0, 0.0],  # Célula 0 (classe 1)
            [2.0, 0.0, 0.0, 0.0],  # Célula 1 (classe 1, vizinha da 0)
            [0.0, 0.0, 8.0, 0.0],  # Célula 2 (classe 2)
            [0.0, 0.0, 4.0, 0.0],  # Célula 3 (classe 2, vizinha da 2)
        ],
        dtype=np.float32,
    )

    rede = ModernHopfieldNetwork(beta=50.0, n_iters=1, binary=False, normalize=True)
    rede.store(memoria)

    # Sem máscara diagonal: a Célula 0 tem cosseno 1.0 consigo mesma e com a Célula 1 -> média = [6, 0, 0, 0]
    rec_sem_mascara, att_sem_mascara = rede.retrieve(
        memoria,
        batch_size=2,
        mask_self=False,
        return_attention_weights=True,
    )
    assert np.isclose(att_sem_mascara[0, 0], 0.5, atol=1e-3)
    assert np.isclose(rec_sem_mascara[0, 0], 6.0, atol=1e-2)

    # Com máscara diagonal (mask_self=True): a diagonal deve ser 0.0 e a Célula 0 deve recuperar a Célula 1 ([2, 0, 0, 0])
    rec_com_mascara, att_com_mascara = rede.retrieve(
        memoria,
        batch_size=2,
        mask_self=True,
        return_attention_weights=True,
    )
    np.testing.assert_allclose(np.diag(att_com_mascara), 0.0, atol=1e-6)
    assert np.isclose(att_com_mascara[0, 1], 1.0, atol=1e-4)
    assert np.isclose(rec_com_mascara[0, 0], 2.0, atol=1e-3)
    assert np.isclose(rec_com_mascara[2, 2], 4.0, atol=1e-3)
    assert "max_weight_mean" in rede.ultimo_diagnostico_atencao
    assert rede.ultimo_diagnostico_atencao["max_weight_mean"] > 0.99


def test_calcular_metricas_geometricas_600d_e_por_classe() -> None:
    """Valida o cálculo de MSE, MAE, Similaridade de Cosseno e Pearson em vetores contínuos."""
    rng = np.random.default_rng(7)
    ref = rng.normal(0.0, 1.0, size=(30, 16)).astype(np.float32)
    ruido = rng.normal(0.0, 0.1, size=(30, 16)).astype(np.float32)
    Perturbado = (ref + ruido).astype(np.float32)
    rotulos = np.array([1] * 15 + [2] * 15, dtype=np.int32)

    resumo_id, df_classes_id = calcular_metricas_geometricas_600d(
        sweep_referencia=ref,
        sweep_avaliado=ref,
        rotulos=rotulos,
        classes_validas=(1, 2),
    )
    assert np.isclose(resumo_id["mse"], 0.0, atol=1e-7)
    assert np.isclose(resumo_id["mae"], 0.0, atol=1e-7)
    assert np.isclose(resumo_id["cosseno_medio"], 1.0, atol=1e-6)
    assert np.isclose(resumo_id["pearson_medio"], 1.0, atol=1e-6)
    assert len(df_classes_id) == 2

    resumo_pert, _ = calcular_metricas_geometricas_600d(
        sweep_referencia=ref,
        sweep_avaliado=Perturbado,
        rotulos=rotulos,
        classes_validas=(1, 2),
    )
    assert resumo_pert["mse"] > 0.0
    assert 0.90 < resumo_pert["cosseno_medio"] < 1.0


def test_calibrar_hopfield_sweep_completo_reduz_erro_de_sentinela() -> None:
    """Valida que a calibração Grid Search encontra hiperparâmetros que reduzem o MSE do vetor corrompido."""
    rng = np.random.default_rng(123)
    prototipo_c1 = rng.normal(1.0, 0.5, size=(1, 32)).astype(np.float32)
    prototipo_c2 = rng.normal(-1.0, 0.5, size=(1, 32)).astype(np.float32)

    # 20 células da classe 1 e 20 células da classe 2 com micro-variação biológica
    celulas_c1 = prototipo_c1 + rng.normal(0.0, 0.02, size=(20, 32)).astype(np.float32)
    celulas_c2 = prototipo_c2 + rng.normal(0.0, 0.02, size=(20, 32)).astype(np.float32)
    sweep_ref = np.vstack([celulas_c1, celulas_c2]).astype(np.float32)
    rotulos = np.array([1] * 20 + [2] * 20, dtype=np.int32)

    # Simula o viés aditivo de colunas faltantes preenchidas com 0.5
    vies_sentinela = rng.normal(0.0, 0.35, size=(40, 32)).astype(np.float32)
    sweep_faltante = (sweep_ref + vies_sentinela).astype(np.float32)

    df_grid, melhor_cfg = calibrar_hopfield_sweep_completo(
        sweep_referencia=sweep_ref,
        sweep_faltante=sweep_faltante,
        rotulos_referencia=rotulos,
        betas=(10.0, 50.0),
        n_iters_list=(1, 2),
        batch_size=16,
        mask_self=True,
    )

    assert len(df_grid) == 4
    assert melhor_cfg["cosseno_medio"] > melhor_cfg["cosseno_antes"]
    assert melhor_cfg["mse"] < melhor_cfg["mse_antes"]


def test_carregar_matriz_sweep_600d_formatos_txt_csv_npy(tmp_path: Path) -> None:
    """Valida o carregamento de matrizes SWeeP em .txt (TSV sem cabeçalho), .csv (com cabeçalho) e .npy."""
    dados = np.arange(24, dtype=np.float32).reshape(4, 6)

    # 1. NPY
    path_npy = tmp_path / "sweep.npy"
    np.save(path_npy, dados)
    lido_npy = carregar_matriz_sweep_600d(path_npy)
    np.testing.assert_allclose(lido_npy, dados)

    # 2. TXT sem cabeçalho separado por tabulação
    path_txt = tmp_path / "sweep.txt"
    np.savetxt(path_txt, dados, delimiter="\t", fmt="%.5f")
    lido_txt = carregar_matriz_sweep_600d(path_txt)
    np.testing.assert_allclose(lido_txt, dados, atol=1e-4)

    # 3. CSV com cabeçalho V1..V6
    path_csv = tmp_path / "sweep.csv"
    df_csv = pl.DataFrame(dados, schema=[f"V{i}" for i in range(1, 7)])
    df_csv.write_csv(path_csv)
    lido_csv = carregar_matriz_sweep_600d(path_csv, tem_cabecalho=True, separador=",")
    np.testing.assert_allclose(lido_csv, dados, atol=1e-4)

    # 4. Arquivo inexistente deve levantar FileNotFoundError
    with pytest.raises(FileNotFoundError):
        carregar_matriz_sweep_600d(tmp_path / "nao_existe.txt")


def test_avaliar_no_classificador_pan_metricas_e_latente() -> None:
    """Valida filtragem das classes 1..7, cômputo de métricas e extração de latente 32D."""
    # 5 amostras com 600 dimensões, incluindo rótulos válidos (1, 2, 3) e inválidos (0, 9)
    x_sweep = np.ones((5, 600), dtype=np.float32)
    rotulos = np.array([1, 2, 0, 3, 9], dtype=np.int32)
    classes_validas = (1, 2, 3)

    mock_scaler = MagicMock()
    mock_scaler.transform.side_effect = lambda x: x

    mock_modelo = MagicMock()
    probs = np.array(
        [
            [0.85, 0.10, 0.05],  # Predição 1 (argmax 0 + 1 -> 1)
            [0.05, 0.85, 0.10],  # Predição 2 (argmax 1 + 1 -> 2)
            [0.10, 0.10, 0.80],  # Predição 3 (argmax 2 + 1 -> 3)
        ],
        dtype=np.float32,
    )
    mock_modelo.predict.return_value = probs

    camada_penultima = MagicMock()
    camada_penultima.output = "output_penultima_camada"
    camada_saida = MagicMock()
    mock_modelo.layers = [camada_penultima, camada_saida]
    mock_modelo.inputs = ["input_layer"]

    mock_tf = MagicMock()
    mock_extrator = MagicMock()
    latente_simulado = np.zeros((3, 32), dtype=np.float32)
    mock_extrator.predict.return_value = latente_simulado
    mock_tf.keras.Model.return_value = mock_extrator

    with patch.dict("sys.modules", {"tensorflow": mock_tf}):
        resultado = avaliar_no_classificador_pan(
            sweep_600d=x_sweep,
            rotulos=rotulos,
            modelo_keras=mock_modelo,
            scaler=mock_scaler,
            classes_validas=classes_validas,
            extrair_latente_32d=True,
        )

    # Filtragem das classes válidas (restando exatamente 3 de 5)
    assert len(resultado["y_real"]) == 3
    np.testing.assert_array_equal(resultado["y_real"], np.array([1, 2, 3]))
    np.testing.assert_array_equal(resultado["y_pred"], np.array([1, 2, 3]))

    # Métricas
    assert np.isclose(resultado["acuracia"], 1.0)
    assert np.isclose(resultado["f1_macro"], 1.0)
    assert np.isclose(resultado["f1_weighted"], 1.0)
    assert resultado["matriz_confusao"].shape == (3, 3)
    np.testing.assert_array_equal(resultado["matriz_confusao"], np.eye(3, dtype=int))

    # Relatório e latente
    assert len(resultado["df_relatorio_classes"]) == 3
    assert resultado["latente_32d"] is not None
    assert resultado["latente_32d"].shape == (3, 32)
    np.testing.assert_allclose(resultado["latente_32d"], latente_simulado)


def test_avaliar_no_classificador_pan_falha_latente_emite_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Valida observabilidade obrigatória com log warning quando a extração latente falha."""
    x_sweep = np.ones((2, 600), dtype=np.float32)
    rotulos = np.array([1, 2], dtype=np.int32)
    classes_validas = (1, 2)

    mock_scaler = MagicMock()
    mock_scaler.transform.side_effect = lambda x: x

    mock_modelo = MagicMock()
    mock_modelo.predict.return_value = np.array(
        [[0.9, 0.1], [0.1, 0.9]], dtype=np.float32
    )
    camada_penultima = MagicMock()
    camada_penultima.output = "output_penultima_camada"
    mock_modelo.layers = [camada_penultima, MagicMock()]
    mock_modelo.inputs = ["input_layer"]

    mock_tf = MagicMock()
    mock_tf.keras.Model.side_effect = RuntimeError("Falha simulada na camada Keras")

    with patch.dict("sys.modules", {"tensorflow": mock_tf}), caplog.at_level("WARNING"):
        resultado = avaliar_no_classificador_pan(
            sweep_600d=x_sweep,
            rotulos=rotulos,
            modelo_keras=mock_modelo,
            scaler=mock_scaler,
            classes_validas=classes_validas,
            extrair_latente_32d=True,
        )

    # Execução não quebra e retorna latente_32d como None
    assert resultado["latente_32d"] is None
    assert np.isclose(resultado["acuracia"], 1.0)
    # Log de advertência foi emitido
    assert any(
        "Falha ao extrair espaço latente 32D" in record.message
        for record in caplog.records
    )


def test_hopfield_mask_self_lote_parcial_e_indices_fora_do_intervalo() -> None:
    """Valida que mask_self=True opera seguramente em lotes parciais e com índices fora do intervalo 0..N-1."""
    # Memória com 3 padrões
    memoria = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rede = ModernHopfieldNetwork(beta=10.0, n_iters=1, binary=False, normalize=True)
    rede.store(memoria)

    # 5 queries, batch_size=2 -> lotes de tamanhos 2, 2 e 1 (último lote parcial)
    queries = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Inclui índices negativos (-1) e índices maiores ou iguais a n_patterns (99)
    query_indices_com_invalidos = np.array([0, -1, 99, 1, 2], dtype=np.int64)

    # Não deve disparar IndexError
    rec = rede.retrieve(
        queries,
        batch_size=2,
        mask_self=True,
        query_indices=query_indices_com_invalidos,
    )
    assert rec.shape == (5, 3)
    assert not np.isnan(rec).any()

    # Valida compute_attention_weights com mask_self
    att = rede.compute_attention_weights(
        queries,
        batch_size=2,
        mask_self=True,
        query_indices=query_indices_com_invalidos,
    )
    assert att.shape == (5, 3)
    # Amostra 0 teve query_index 0 -> peso na coluna 0 deve ser zero
    assert np.isclose(att[0, 0], 0.0, atol=1e-6)

    # Valida que tamanho incompatível de query_indices dispara ValueError
    with pytest.raises(
        ValueError, match="query_indices deve ter comprimento igual a n_queries"
    ):
        rede.retrieve(queries, mask_self=True, query_indices=[0, 1])

    with pytest.raises(
        ValueError, match="query_indices deve ter comprimento igual a n_queries"
    ):
        rede.compute_attention_weights(queries, mask_self=True, query_indices=[0, 1])
