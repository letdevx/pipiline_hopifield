"""Validador de integridade e versão do repositório Git para Google Colab.

Implementa salvaguardas baseadas no princípio Fail Fast para assegurar
que o ambiente de nuvem execute exatamente o commit hash esperado,
prevenindo execuções com código defasado por esquecimento de 'git push'.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def detectar_ambiente_colab() -> bool:
    """Verifica se a execução atual está ocorrendo no Google Colab.

    Returns
    -------
    bool
        True se estiver no ambiente Google Colab, False caso contrário.
    """
    if "google.colab" in sys.modules:
        return True
    if os.path.exists("/content"):
        return True
    if os.environ.get("COLAB_GPU") is not None:
        return True
    return os.environ.get("COLAB_RELEASE_TAG") is not None


def obter_info_commit(repo_path: str | Path = ".") -> dict[str, str]:
    """Obtém metadados do commit HEAD no repositório especificado.

    Parameters
    ----------
    repo_path : str | Path, default="."
        Caminho local para o repositório Git.

    Returns
    -------
    dict[str, str]
        Dicionário contendo os seguintes campos:
        - 'hash_completo': SHA-1 integral de 40 caracteres.
        - 'hash_curto': SHA-1 abreviado (7 caracteres).
        - 'autor': Nome do autor do commit.
        - 'data': Data e hora do commit (formato ISO).
        - 'mensagem': Título da mensagem do commit.
        - 'branch': Nome da branch atualmente ativa.

    Raises
    ------
    RuntimeError
        Se o comando git falhar ou o caminho não for um repositório git válido.
    """
    path_str = str(repo_path)
    if not os.path.isdir(path_str):
        raise RuntimeError(f"Diretório do repositório não encontrado: '{path_str}'.")

    def _exec_git(args: list[str]) -> str:
        res = subprocess.run(
            ["git", "-C", path_str, *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"Erro ao executar git {' '.join(args)} em '{path_str}':\n"
                f"{res.stderr.strip()}"
            )
        return res.stdout.strip()

    hash_completo = _exec_git(["rev-parse", "HEAD"])
    hash_curto = _exec_git(["rev-parse", "--short", "HEAD"])
    autor = _exec_git(["log", "-1", "--format=%an"])
    data = _exec_git(["log", "-1", "--format=%ad", "--date=iso"])
    mensagem = _exec_git(["log", "-1", "--format=%s"])

    branch = ""
    try:
        branch = _exec_git(["branch", "--show-current"])
    except RuntimeError:
        branch = "HEAD (desanexado)"

    return {
        "hash_completo": hash_completo,
        "hash_curto": hash_curto,
        "autor": autor,
        "data": data,
        "mensagem": mensagem,
        "branch": branch,
    }


def atualizar_repositorio_colab(
    repo_url: str,
    dest_path: str | Path,
    branch: str = "reconstrução_Pan_Mathys",
) -> None:
    """Clona ou atualiza o repositório no caminho de destino da VM.

    Executa git clone caso a pasta não exista, seguido de checkout
    da branch alvo e fetch/pull remoto para garantir sincronização.

    Parameters
    ----------
    repo_url : str
        URL do repositório remoto Git.
    dest_path : str | Path
        Caminho de destino do clone (ex.: '/content/pipiline_hopifield').
    branch : str, default="reconstrução_Pan_Mathys"
        Nome da branch Git a ser sincronizada.

    Raises
    ------
    RuntimeError
        Se qualquer operação de clone, checkout ou pull falhar.
    """
    dest_str = str(dest_path)

    if not os.path.exists(dest_str):
        print(f"[Colab Git] Clonando '{repo_url}' em '{dest_str}'...")
        res_clone = subprocess.run(
            ["git", "clone", repo_url, dest_str],
            capture_output=True,
            text=True,
            check=False,
        )
        if res_clone.returncode != 0:
            raise RuntimeError(
                f"Falha ao clonar repositório:\n{res_clone.stderr.strip()}"
            )

    print(f"[Colab Git] Sincronizando branch '{branch}' em '{dest_str}'...")
    res_checkout = subprocess.run(
        ["git", "-C", dest_str, "checkout", branch],
        capture_output=True,
        text=True,
        check=False,
    )
    if res_checkout.returncode != 0:
        raise RuntimeError(
            f"Falha ao mudar para a branch '{branch}':\n{res_checkout.stderr.strip()}"
        )

    res_fetch = subprocess.run(
        ["git", "-C", dest_str, "fetch", "origin", branch],
        capture_output=True,
        text=True,
        check=False,
    )
    if res_fetch.returncode != 0:
        raise RuntimeError(
            f"Falha ao executar 'git fetch origin {branch}':\n"
            f"{res_fetch.stderr.strip()}"
        )

    res_pull = subprocess.run(
        ["git", "-C", dest_str, "pull", "origin", branch],
        capture_output=True,
        text=True,
        check=False,
    )
    if res_pull.returncode != 0:
        raise RuntimeError(
            f"Falha ao executar 'git pull origin {branch}':\n{res_pull.stderr.strip()}"
        )


def validar_commit_head(
    repo_path: str | Path,
    expected_commit: str | None,
    forcar_no_local: bool = False,
) -> dict[str, str]:
    """Valida se o HEAD do repositório coincide com o commit hash esperado.

    Seguindo o princípio Fail Fast, interrompe imediatamente com
    RuntimeError caso haja divergência ou se o commit esperado não
    estiver preenchido quando executando no Google Colab.

    Parameters
    ----------
    repo_path : str | Path
        Caminho para o repositório Git.
    expected_commit : str | None
        Hash esperado (prefixo de 7+ caracteres ou SHA-1 completo de 40).
    forcar_no_local : bool, default=False
        Se True, aplica a validação estrita mesmo fora do Google Colab.

    Returns
    -------
    dict[str, str]
        Metadados do commit HEAD validado.

    Raises
    ------
    RuntimeError
        Se o commit esperado for nulo/vazio no Colab ou se o commit atual
        divergir do esperado.
    """
    em_colab = detectar_ambiente_colab()

    if not em_colab and not forcar_no_local:
        print(
            "[Git Validador] Execução em ambiente local detectada. "
            "Validação estrita de commit do Colab ignorada."
        )
        try:
            return obter_info_commit(repo_path)
        except Exception:
            return {
                "hash_completo": "local_dev",
                "hash_curto": "local",
                "autor": "local_user",
                "data": "n/a",
                "mensagem": "Execução local sem rastreamento estrito",
                "branch": "local",
            }

    if expected_commit is None or not expected_commit.strip():
        raise RuntimeError(
            "\n"
            + "=" * 80
            + "\n[FALHA DE INTEGRIDADE - FAIL FAST] COMMIT ESPERADO NÃO INFORMADO\n"
            + "=" * 80
            + "\nNo ambiente do Google Colab, a variável EXPECTED_COMMIT é obrigatória!\n"
            + "Defina o hash do commit desejado antes de executar o notebook para evitar\n"
            + "execuções não reproduzíveis ou versões obsoletas do código.\n"
            + "\nExemplo:\n"
            + "    EXPECTED_COMMIT = '058e839'  # Hash obtido via 'git rev-parse --short HEAD'\n"
            + "=" * 80
        )

    exp_clean = expected_commit.strip().lower()
    if len(exp_clean) < 7:
        raise RuntimeError(
            f"O hash informado '{expected_commit}' possui menos de 7 caracteres. "
            "Forneça pelo menos 7 caracteres para identificação inequívoca do commit."
        )

    info = obter_info_commit(repo_path)
    head_completo = info["hash_completo"].lower()
    head_curto = info["hash_curto"].lower()

    # Compara por prefixo ou igualdade exata
    coincide = head_completo.startswith(exp_clean) or (exp_clean == head_curto)

    if not coincide:
        # Se o hash esperado for ancestral do HEAD, o repositório já contém o commit esperado
        res_ancestor = subprocess.run(
            [
                "git",
                "-C",
                str(repo_path),
                "merge-base",
                "--is-ancestor",
                exp_clean,
                "HEAD",
            ],
            capture_output=True,
            check=False,
        )
        if res_ancestor.returncode == 0:
            coincide = True

    if not coincide:
        msg_erro = (
            "\n"
            + "=" * 80
            + "\n[FALHA DE INTEGRIDADE - FAIL FAST] DIVERGÊNCIA DE COMMIT NO GOOGLE COLAB\n"
            + "=" * 80
            + f"\n• Commit Esperado : {expected_commit}"
            + f"\n• Commit no Colab : {info['hash_curto']} ({info['hash_completo']})"
            + f"\n• Branch atual    : {info['branch']}"
            + f"\n• Autor do Commit : {info['autor']}"
            + f"\n• Data do Commit  : {info['data']}"
            + f"\n• Mensagem        : {info['mensagem']}"
            + "\n"
            + "-" * 80
            + "\nMOTIVO DA INTERRUPÇÃO:"
            + "\nO repositório clonado na VM do Google Colab não está no commit esperado."
            + "\nVocê provavelmente esqueceu de fazer push dos commits locais do seu laptop!"
            + "\n"
            + "\nCOMO CORRIGIR:"
            + "\n1. No terminal do seu laptop, envie os commits mais recentes:"
            + "\n       git status"
            + f"\n       git push origin {info['branch']}"
            + "\n2. No Google Colab, re-execute esta célula para atualizar a VM e prosseguir."
            + "\n"
            + "=" * 80
        )
        raise RuntimeError(msg_erro)

    print(
        f"[Git Validador] Integridade confirmada! HEAD no commit esperado: "
        f"{info['hash_curto']} - '{info['mensagem']}' (Branch: {info['branch']})"
    )
    return info
