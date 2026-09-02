"""Testes unitários para a classe ValidadorImputacao."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.treinamento.validador_imputacao import ValidadorImputacao


def test_validador_imputacao_global_e_marcadores(tmp_path: Path) -> None:
    """Valida o cálculo de métricas globais e a auditoria de marcadores biológicos."""
    n_celulas = 20
    n_genes = 6
    # 2 classes: 1 (Astrocytes - células 0..9) e 3 (Excitatory Neurons - células 10..19)
    classes_reais = np.array([1] * 10 + [3] * 10)

    # Matriz X: binária {0, 1}
    X = np.zeros((n_celulas, n_genes), dtype=np.float32)
    # Gene 0: GFAP (marcador de Astrocyte)
    # Gene 1: SLC17A7 (marcador de Excitatory)
    # Gene 2: ausente e imputado ativo em astrócitos
    # Gene 3: ausente e inativado
    # Gene 4: outro gene
    # Gene 5: outro gene

    # GFAP ativo em 90% dos astrócitos (9/10) e 10% dos neurônios (1/10)
    X[:9, 0] = 1.0
    X[10, 0] = 1.0

    # SLC17A7 ativo em 100% dos neurônios e 0% dos astrócitos
    X[10:, 1] = 1.0

    # Gene 2 (imputado): ativo em 8 dos 10 astrócitos
    X[:8, 2] = 1.0

    # Máscara de imputação: colunas 2 e 3 foram inteiramente imputadas
    mascara_imp = np.zeros((n_celulas, n_genes), dtype=np.float32)
    mascara_imp[:, 2] = 1.0
    mascara_imp[:, 3] = 1.0

    # Probabilidades
    prob_imp = np.zeros((n_celulas, n_genes), dtype=np.float32)
    prob_imp[:8, 2] = 0.85
    prob_imp[8:, 2] = 0.15
    prob_imp[:, 3] = 0.05

    var_df = pd.DataFrame(
        {
            "gene_symbol": ["GFAP", "SLC17A7", "AQP4", "GAD1", "GENE_4", "GENE_5"],
            "gene_imputado": [False, False, True, True, False, False],
        },
        index=["ENSG0", "ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG5"],
    )

    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame({"tipo_celular_real": classes_reais}),
        var=var_df,
    )
    adata.layers["mascara_imputada"] = sp.csr_matrix(mascara_imp)
    adata.layers["probabilidade_imputada"] = sp.csr_matrix(prob_imp)

    validador = ValidadorImputacao()

    # 1. Auditoria global
    metricas = validador.auditar_imputacao_global(adata)
    assert metricas["n_celulas"] == 20
    assert metricas["n_genes"] == 6
    # 20 células * 2 colunas imputadas = 40 coordenadas
    assert metricas["total_sentinelas_resolvidos"] == 40
    # Ativados nas posições imputadas: 8 posições no gene 2
    assert metricas["posicoes_ativadas"] == 8
    assert metricas["posicoes_inativadas"] == 32
    assert metricas["percentual_ativadas"] == 20.0
    assert metricas["percentual_inativadas"] == 80.0
    assert metricas["confianca_media_imputacao"] is not None

    # 2. Auditoria biológica de marcadores
    map_features = {
        "ENSG0": "GFAP",
        "ENSG1": "SLC17A7",
        "ENSG2": "AQP4",
        "ENSG3": "GAD1",
    }
    df_marcadores = validador.auditar_marcadores_biologicos(
        adata=adata,
        classes_reais=classes_reais,
        map_features=map_features,
    )
    assert not df_marcadores.empty
    gfap_row = df_marcadores[df_marcadores["gene_marcador"] == "GFAP"].iloc[0]
    assert gfap_row["ativacao_no_tipo_pct"] == 90.0
    assert gfap_row["ativacao_outros_pct"] == 10.0
    assert gfap_row["razao_especificidade"] == 9.0
    assert not gfap_row["foi_imputado"]

    aqp4_row = df_marcadores[df_marcadores["gene_marcador"] == "AQP4"].iloc[0]
    assert aqp4_row["foi_imputado"]
    assert aqp4_row["ativacao_no_tipo_pct"] == 80.0
    assert aqp4_row["ativacao_outros_pct"] == 0.0

    # 3. Impressão de relatório (não deve levantar exceção)
    validador.imprimir_relatorio(metricas, df_marcadores)

    # 4. Exportação para JSON
    path_json = tmp_path / "relatorio_teste.json"
    validador.exportar_relatorio(path_json, metricas, df_marcadores)
    assert path_json.exists()

    with open(path_json, encoding="utf-8") as f:
        dados_json = json.load(f)
    assert "auditoria_validacao" in dados_json
    assert (
        dados_json["auditoria_validacao"]["metricas_globais"][
            "total_sentinelas_resolvidos"
        ]
        == 40
    )
