"""Pacote principal do Pipeline Hopfield Expandido para scRNA-Seq.

Exporta as classes centrais de pré-processamento, alinhamento genômico,
projeção dimensional e redes associativas Modern Hopfield.
"""

from .alinhamento import (
    Alinhador,
    AlinhadorEsparso,
    AnalisadorCobertura,
    AnalisadorSobreposicao,
    LeitorFeatures,
    SelecionadorGenesFrequentes,
    ValidadorAlinhamento,
    ValidadorFeatures,
)
from .preprocessing import Binarizador
from .treinamento import GeradorConjuntoTreinamento

__all__ = [
    "Alinhador",
    "AlinhadorEsparso",
    "AnalisadorCobertura",
    "AnalisadorSobreposicao",
    "Binarizador",
    "GeradorConjuntoTreinamento",
    "LeitorFeatures",
    "SelecionadorGenesFrequentes",
    "ValidadorAlinhamento",
    "ValidadorFeatures",
]
