"""Módulo de Alinhamento Genômico e Mapeamento de Features scRNA-Seq.

Oferece utilitários para leitura de arquivos de features TSV/CSV, validação
de schemas Ensembl ID/Gene Symbol, seleção de genes de alta frequência/diferenciais,
análise de sobreposição e alinhamento dimensional de matrizes (esparso e denso).
"""

from .alinhador import Alinhador
from .alinhador_esparso import AlinhadorEsparso
from .analisador_cobertura import AnalisadorCobertura
from .analisador_sobreposicao import AnalisadorSobreposicao
from .exportador_mtx import ExportadorMTX
from .injetor_mtx_sentinela import InjetorMTXSentinela
from .leitor_features import LeitorFeatures
from .marcador_sentinela import MarcadorSentinela
from .selecionador_genes_diferenciais import SelecionadorGenesDiferenciais
from .selecionador_genes_frequentes import SelecionadorGenesFrequentes
from .validador_alinhamento import ValidadorAlinhamento
from .validador_features import ValidadorFeatures
from .validador_ordem_genes import ValidadorOrdemGenes

__all__ = [
    "Alinhador",
    "AlinhadorEsparso",
    "AnalisadorCobertura",
    "AnalisadorSobreposicao",
    "ExportadorMTX",
    "InjetorMTXSentinela",
    "LeitorFeatures",
    "MarcadorSentinela",
    "SelecionadorGenesDiferenciais",
    "SelecionadorGenesFrequentes",
    "ValidadorAlinhamento",
    "ValidadorFeatures",
    "ValidadorOrdemGenes",
]
