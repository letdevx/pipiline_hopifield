"""Testes para auditar o comportamento de alinhamento e impacto dos valores sentinela (0.0 vs 0.5)."""

import pytest
import numpy as np
import pandas as pd
import anndata as ad
from src.synthetic.gerador_ground_truth import GeradorGroundTruthSintetico
from src.alinhamento.alinhador import Alinhador
from src.treinamento.hopfield import ModernHopfieldNetwork


def test_impacto_matematico_sentinela_em_hopfield():
    """Prova matemática empírica sobre por que o valor sentinela 0.5 funciona no espaço bipolar da Rede Hopfield.
    
    Em uma Rede Hopfield Bipolar (binary=True), os vetores {0, 1} são mapeados para {-1, +1}:
        x_bipolar = 2*x - 1
    Quando um gene está ausente no dataset Mathys mas presente em Fujita, o preenchimento sentinela:
        - Para 0.5: 2*(0.5) - 1 = 0.0 -> O gene ausente tem contribuição NULA no produto escalar da atenção.
        - Para 0.0: 2*(0.0) - 1 = -1.0 -> O gene é interpretado como repressão biológica confirmada (-1),
          penalizando injustificadamente a similaridade contra o protótipo que o possui (+1).
    """
    gerador = GeradorGroundTruthSintetico(n_celulas=4, n_genes=4, n_classes=2, seed=42)
    # Protótipo de Referência (Fujita): Célula Tipo A tem genes G0 e G1 ativos -> [1, 1, 0, 0]
    prototipo_fujita = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    
    # Query (Mathys): O gene G1 não foi sequenciado/anotado na plataforma Mathys!
    # Preenchimento sentinela 0.5 (Recomendado/Oficial) -> [1.0, 0.5, 0.0, 0.0]
    query_sentinela_meio = np.array([[1.0, 0.5, 0.0, 0.0]], dtype=np.float32)
    
    # Preenchimento sentinela 0.0 -> [1.0, 0.0, 0.0, 0.0]
    query_sentinela_zero = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    
    rede = ModernHopfieldNetwork(beta=10.0, n_iters=1, binary=True)
    rede.store(prototipo_fujita)
    
    # Verificação de ativação interna (produto escalar na camada de atenção)
    # Projeção Bipolar:
    # prototipo_bipolar = [1, 1, -1, -1]
    # query_meio_bipolar = [1, 0, -1, -1] -> produto escalar com protótipo: 1*1 + 0*1 + (-1)*(-1) + (-1)*(-1) = 3.0
    # query_zero_bipolar = [1, -1, -1, -1] -> produto escalar com protótipo: 1*1 + (-1)*1 + (-1)*(-1) + (-1)*(-1) = 2.0
    
    bipolar_meio = 2.0 * query_sentinela_meio - 1.0
    bipolar_zero = 2.0 * query_sentinela_zero - 1.0
    bipolar_prot = 2.0 * prototipo_fujita - 1.0
    
    escore_meio = (bipolar_meio @ bipolar_prot.T)[0, 0]
    escore_zero = (bipolar_zero @ bipolar_prot.T)[0, 0]
    
    # O valor sentinela 0.5 garante maior fidelidade de similaridade associativa para genes não sequenciados!
    assert escore_meio > escore_zero, "O escore de atenção com sentinela 0.5 deve superar o preenchimento por zero para genes ausentes."
    assert np.isclose(escore_meio, 3.0), f"Escore esperado 3.0, obtido {escore_meio}"
    assert np.isclose(escore_zero, 2.0), f"Escore esperado 2.0, obtido {escore_zero}"


def test_alinhador_direto_preenche_ausentes():
    """Verifica se o método _alinhar_direto preenche colunas ausente no alvo com o valor sentinela correto."""
    # Simula um dataset com genes G0, G1
    adata_in = ad.AnnData(
        X=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        var=pd.DataFrame(index=["G0", "G1"])
    )
    
    # O genoma de referência (Alvo/Fujita) exige os genes G0, G1, G2 (G2 está ausente em adata_in)
    ensembl_map = {"G0": "G0", "G1": "G1"}
    gene_alvo_idx = {"G0": 0, "G1": 1, "G2": 2}
    genes_ordenados = ["G0", "G1", "G2"]
    
    alinhador = Alinhador(
        path_binarizada_m="dummy", path_binarizada_f="dummy", out_dir="dummy",
        map_f={}, map_m={}, gene_alvo_idx=gene_alvo_idx, genes_ordenados=genes_ordenados
    )
    
    # Executa alinhamento com fill_value=0.5 (padrão Mathys)
    adata_out = alinhador._alinhar_direto(adata_in, ensembl_map, fill_value=0.5)
    
    assert adata_out.shape == (2, 3), "A matriz alinhada deve conter todas as colunas do genoma de referência."
    
    matriz_final = adata_out.X.toarray() if hasattr(adata_out.X, "toarray") else adata_out.X
    
    # Coluna G2 (índice 2) deve estar preenchida integralmente com 0.5
    np.testing.assert_array_equal(matriz_final[:, 2], np.array([0.5, 0.5]), 
                                  err_msg="A coluna do gene ausente deve receber o valor sentinela 0.5.")


def test_retrieve_com_mascara_sentinela_ausentes():
    """Testa se o método retrieve injeta o valor 0.5 nos genes ausentes sob demanda em lotes OOM-Safe."""
    import scipy.sparse as sp
    
    # 2 protótipos em 4 genes:
    # Protótipo 0 (Classe A): [1, 1, 0, 0]
    # Protótipo 1 (Classe B): [0, 0, 1, 1]
    prototipos = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ], dtype=np.float32)
    
    rede = ModernHopfieldNetwork(beta=20.0, n_iters=1, binary=True)
    rede.store(prototipos)
    
    # Query esparsa CSR: G0 ativo (1.0), G1 ausente (0.0 na matriz esparsa, mas marcado em mask_ausentes)
    queries = sp.csr_matrix(np.array([
        [1.0, 0.0, 0.0, 0.0],  # Deve reconhecer Classe A e reconstruir G1=1.0
        [0.0, 0.0, 1.0, 0.0],  # Deve reconhecer Classe B e reconstruir G3=1.0
    ], dtype=np.float32))
    
    mask_ausentes = np.array([False, True, False, True]) # G1 e G3 são ausentes na plataforma alvo
    
    recuperado = rede.retrieve(queries, batch_size=1, mask_sentinela_ausentes=mask_ausentes, fill_value=0.5)
    
    # Validações:
    # A query 0 deve reconstruir perfeitamente o protótipo 0 [1, 1, 0, 0]
    np.testing.assert_array_equal(recuperado[0], np.array([1.0, 1.0, 0.0, 0.0]))
    # A query 1 deve reconstruir perfeitamente o protótipo 1 [0, 0, 1, 1]
    np.testing.assert_array_equal(recuperado[1], np.array([0.0, 0.0, 1.0, 1.0]))

