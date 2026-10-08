"""Módulo de utilitários gerais do pipeline.

Contém ferramentas de suporte, diagnóstico e validação de ambiente.
"""

from .validador_git_colab import (
    atualizar_repositorio_colab,
    detectar_ambiente_colab,
    obter_info_commit,
    validar_commit_head,
)

__all__ = [
    "atualizar_repositorio_colab",
    "detectar_ambiente_colab",
    "obter_info_commit",
    "validar_commit_head",
]
