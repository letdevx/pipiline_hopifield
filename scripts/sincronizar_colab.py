"""Script CLI para sincronização de commit e push pré-Google Colab.

Permite inspecionar o HEAD local, atualizar automaticamente a variável
EXPECTED_COMMIT no script do notebook emparelhado com o Jupytext,
sincronizar o .ipynb e opcionalmente realizar o git push.

Uso:
    uv run python scripts/sincronizar_colab.py [--push]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PY = REPO_ROOT / "pipeline_generico_Pan_Mathys.py"


def _exec_git(args: list[str]) -> str:
    """Executa comando git na raiz do repositório."""
    res = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Erro ao executar git {' '.join(args)}:\n{res.stderr.strip()}"
        )
    return res.stdout.strip()


def obter_status_git() -> dict[str, str]:
    """Coleta dados do repositório local."""
    hash_curto = _exec_git(["rev-parse", "--short", "HEAD"])
    hash_completo = _exec_git(["rev-parse", "HEAD"])
    branch = _exec_git(["branch", "--show-current"])
    mensagem = _exec_git(["log", "-1", "--format=%s"])
    status = _exec_git(["status", "--short"])
    return {
        "hash_curto": hash_curto,
        "hash_completo": hash_completo,
        "branch": branch,
        "mensagem": mensagem,
        "dirty": "sim" if status else "não",
    }


def atualizar_expected_commit_no_notebook(novo_hash: str) -> bool:
    """Atualiza a constante EXPECTED_COMMIT em pipeline_generico_Pan_Mathys.py."""
    if not NOTEBOOK_PY.exists():
        print(f"[Aviso] Arquivo '{NOTEBOOK_PY.name}' não encontrado.")
        return False

    conteudo = NOTEBOOK_PY.read_text(encoding="utf-8")
    padrao = r'EXPECTED_COMMIT\s*=\s*"[^"]*"'
    novo_conteudo = re.sub(padrao, f'EXPECTED_COMMIT = "{novo_hash}"', conteudo)

    if conteudo == novo_conteudo:
        print(f"[Notebook] EXPECTED_COMMIT já está atualizado com '{novo_hash}'.")
        return False

    NOTEBOOK_PY.write_text(novo_conteudo, encoding="utf-8")
    print(f"[Notebook] EXPECTED_COMMIT atualizado para '{novo_hash}'.")

    # Sincroniza o .ipynb via Jupytext
    print("[Jupytext] Sincronizando com o notebook .ipynb...")
    res_jupy = subprocess.run(
        ["uv", "run", "jupytext", "--sync", str(NOTEBOOK_PY)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if res_jupy.returncode != 0:
        print(f"[Jupytext Erro] Falha na sincronização:\n{res_jupy.stderr}")
        return False

    print("[Jupytext] Sincronização concluída com sucesso.")
    return True


def main() -> None:
    """Ponto de entrada do utilitário."""
    parser = argparse.ArgumentParser(
        description="Utilitário de preparação e sincronização de commit para Google Colab."
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Executa automaticamente 'git push origin <branch>' após atualizar o notebook.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  SINCRONIZADOR PRÉ-GOOGLE COLAB — PIPELINE HOPFIELD")
    print("=" * 70)

    try:
        info = obter_status_git()
    except Exception as e:
        print(f"Erro ao consultar git: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"• Branch ativa   : {info['branch']}")
    print(f"• HEAD atual     : {info['hash_curto']} ({info['hash_completo']})")
    print(f"• Mensagem       : {info['mensagem']}")
    print(
        f"• Modificações   : {'Pendentes' if info['dirty'] == 'sim' else 'Nenhuma (Working Tree Clean)'}"
    )
    print("-" * 70)

    atualizado = atualizar_expected_commit_no_notebook(info["hash_curto"])

    if args.push:
        print(f"\n[Git Push] Enviando alterações para 'origin/{info['branch']}'...")
        res_push = subprocess.run(
            ["git", "push", "origin", info["branch"]],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if res_push.returncode != 0:
            print(f"[Erro no Push]:\n{res_push.stderr}", file=sys.stderr)
            sys.exit(res_push.returncode)
        print("[Git Push] Concluído com sucesso!")
    else:
        print("\n[Lembrete]")
        if atualizado:
            print(f"O arquivo '{NOTEBOOK_PY.name}' foi alterado. Faça commit e push:")
            print(f"    git add {NOTEBOOK_PY.name} pipeline_generico_Pan_Mathys.ipynb")
            print('    git commit -m "chore: atualiza hash esperado para Colab"')
            print(f"    git push origin {info['branch']}")
        else:
            print(
                f"Certifique-se de que a branch 'origin/{info['branch']}' está atualizada:"
            )
            print(f"    git push origin {info['branch']}")

    print("=" * 70)


if __name__ == "__main__":
    main()
