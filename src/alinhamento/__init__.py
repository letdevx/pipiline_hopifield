"""Módulo de Alinhamento Genômico e Mapeamento de Features scRNA-Seq.

Oferece utilitários para leitura de arquivos de features TSV/CSV, validação
de schemas Ensembl ID/Gene Symbol, seleção de genes de alta frequência/diferenciais,
análise de sobreposição e alinhamento dimensional de matrizes (esparso e denso).
"""

from .leitor_features import LeitorFeatures
from .analisador_sobreposicao import AnalisadorSobreposicao
from .alinhador import Alinhador
from .alinhador_esparso import AlinhadorEsparso
from .validador_alinhamento import ValidadorAlinhamento
from .validador_features import ValidadorFeatures
from .selecionador_genes_frequentes import SelecionadorGenesFrequentes
from .selecionador_genes_diferenciais import SelecionadorGenesDiferenciais
from .analisador_cobertura import AnalisadorCobertura

__all__ = [
    "LeitorFeatures",
    "AnalisadorSobreposicao",
    "Alinhador",
    "AlinhadorEsparso",
    "ValidadorAlinhamento",
    "ValidadorFeatures",
    "SelecionadorGenesFrequentes",
    "SelecionadorGenesDiferenciais",
    "AnalisadorCobertura",
]


