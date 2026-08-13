"""Testes unitários para o módulo Binarizador utilizando Synthetic Ground Truth."""

import os
import shutil
import tempfile
import pytest
import numpy as np
import anndata as ad

from src.synthetic.gerador_ground_truth import GeradorGroundTruthSintetico
from src.preprocessing.binarizador import Binarizador


@pytest.fixture
def temp_dir():
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


def test_binarizacao_preserva_assinaturas(temp_dir):
    """Verifica se contagens contínuas (ex: 15.4, 8.2) são convertidas em 1 e zeros em 0 sem alterar o formato."""
    gerador = GeradorGroundTruthSintetico(n_celulas=12, n_genes=8, n_classes=3, seed=101)
    
    # Gera matriz com contagens de expressão simuladas (5.0 a 25.0 nos ativos)
    adata_continuo = gerador.gerar_matriz_pura(formato="anndata", contagem_continua=True)
    
    # Salva o arquivo h5ad temporário
    path_input = os.path.join(temp_dir, "dataset_continuo.h5ad")
    adata_continuo.write_h5ad(path_input)
    
    # Executa a binarização
    binarizador = Binarizador(path_input, out_dir=temp_dir, out_dir_binarizada=temp_dir)
    binarizador.binarizar(nome_arquivo="binarizado_resultado.h5ad")
    
    adata_resultado = binarizador.carregar_binarizada()
    
    # Matriz pura binária de referência (Ground Truth)
    matriz_referencia = gerador.gerar_matriz_pura(formato="numpy", contagem_continua=False)
    
    # Asserções de conformidade
    assert adata_resultado.shape == (12, 8), "As dimensões da matriz sofreram mutação indevida."
    np.testing.assert_array_equal(
        adata_resultado.X, 
        matriz_referencia.astype(np.int8), 
        err_msg="A binarização desviou da matriz de Ground Truth teórica."
    )
    assert adata_resultado.X.dtype == np.int8, "O tipo de dado gerado deve ser int8 para economia de memória."
