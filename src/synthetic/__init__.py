"""Módulo de geração de dados sintéticos de controle (Synthetic Ground Truth).

Fornece matrizes de expressão gênica sintética controlada (micro-datasets humano-verificáveis
e matrizes parametrizáveis em alta escala) para auditoria teórica e técnica do pipeline.
"""

from .gerador_ground_truth import GeradorGroundTruthSintetico

__all__ = ["GeradorGroundTruthSintetico"]
