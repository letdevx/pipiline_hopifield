"""Pacote principal do Pipeline Hopfield Expandido para scRNA-Seq.

Exporta as classes centrais de pré-processamento, alinhamento genômico,
projeção dimensional e redes associativas Modern Hopfield.
"""

from .preprocessing import Binarizador
from .alinhamento import (
    LeitorFeatures,
    AnalisadorSobreposicao,
    Alinhador,
    AlinhadorEsparso,
    ValidadorAlinhamento,
    ValidadorFeatures,
    SelecionadorGenesFrequentes,
    AnalisadorCobertura,
)
from .treinamento import GeradorConjuntoTreinamento

__all__ = [
    "Binarizador",
    "LeitorFeatures",
    "AnalisadorSobreposicao",
    "Alinhador",
    "AlinhadorEsparso",
    "ValidadorAlinhamento",
    "ValidadorFeatures",
    "SelecionadorGenesFrequentes",
    "AnalisadorCobertura",
    "GeradorConjuntoTreinamento",
]


