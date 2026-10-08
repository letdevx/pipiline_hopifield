"""Instalador do hook Git post-commit para automação Colab.

Configura o hook .git/hooks/post-commit para atualizar automaticamente
o EXPECTED_COMMIT no notebook e executar git push origin a cada commit.

Uso:
    uv run python scripts/instalar_hook.py
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOKS_DIR = REPO_ROOT / ".git" / "hooks"
POST_COMMIT_HOOK = HOOKS_DIR / "post-commit"

HOOK_CONTENT = """#!/bin/sh
# Hook automático pós-commit: atualiza EXPECTED_COMMIT no notebook e executa push
echo "[Git Hook post-commit] Atualizando EXPECTED_COMMIT no notebook..."
uv run python scripts/sincronizar_colab.py --push
"""


def instalar_hook() -> None:
    """Instala ou atualiza o hook post-commit no repositório."""
    if not HOOKS_DIR.exists():
        raise RuntimeError(
            f"Diretório de hooks '{HOOKS_DIR}' não encontrado. "
            "Certifique-se de que o repositório Git foi inicializado."
        )

    POST_COMMIT_HOOK.write_text(HOOK_CONTENT, encoding="utf-8")

    with contextlib.suppress(Exception):
        os.chmod(POST_COMMIT_HOOK, 0o755)

    print("=" * 70)
    print("  HOOK GIT POST-COMMIT INSTALADO COM SUCESSO!")
    print("=" * 70)
    print(f"Arquivo do hook : {POST_COMMIT_HOOK}")
    print("\nFluxo Automatizado Ativo:")
    print("1. Toda vez que você fizer 'git commit' (no terminal ou pela IDE):")
    print("   • O hash do commit é capturado;")
    print("   • O EXPECTED_COMMIT em pipeline_generico_Pan_Mathys.py é atualizado;")
    print("   • O notebook .ipynb é sincronizado via Jupytext;")
    print("   • Um 'git push origin <branch>' é disparado automaticamente.")
    print("2. Ao clicar em 'Run All' na IDE conectada ao Colab:")
    print("   • O Colab puxa o novo commit e valida a integridade com sucesso.")
    print("=" * 70)


if __name__ == "__main__":
    instalar_hook()
