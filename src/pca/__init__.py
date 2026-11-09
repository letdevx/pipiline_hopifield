"""Módulo de redução de dimensionalidade por PCA e comparação de embeddings."""

from .comparador_embeddings import (
    ComparadorEmbeddings,
    ResultadoComparacaoEmbeddings,
)

__all__ = ["ComparadorEmbeddings", "ResultadoComparacaoEmbeddings"]
