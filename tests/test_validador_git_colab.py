"""Testes unitários para o validador de integridade Git no Google Colab."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.utils.validador_git_colab import (
    atualizar_repositorio_colab,
    detectar_ambiente_colab,
    obter_info_commit,
    validar_commit_head,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_detectar_ambiente_colab_local() -> None:
    """Verifica detecção quando fora do Colab."""
    with (
        patch.dict(sys.modules),
        patch.dict("os.environ", {}, clear=True),
        patch("os.path.exists", return_value=False),
    ):
        if "google.colab" in sys.modules:
            del sys.modules["google.colab"]
        assert detectar_ambiente_colab() is False


def test_detectar_ambiente_colab_positivo() -> None:
    """Verifica detecção positiva por sys.modules ou variável de ambiente."""
    with patch.dict(sys.modules, {"google.colab": MagicMock()}):
        assert detectar_ambiente_colab() is True

    with patch.dict("os.environ", {"COLAB_GPU": "1"}):
        assert detectar_ambiente_colab() is True

    with patch.dict("os.environ", {"COLAB_RELEASE_TAG": "colab-2026"}):
        assert detectar_ambiente_colab() is True


def test_obter_info_commit_repositorio_real() -> None:
    """Verifica obtenção de metadados em repositório Git local ativo."""
    info = obter_info_commit(REPO_ROOT)
    assert len(info["hash_completo"]) == 40
    assert len(info["hash_curto"]) >= 7
    assert len(info["autor"]) > 0
    assert len(info["mensagem"]) > 0


def test_obter_info_commit_diretorio_invalido(tmp_path: Path) -> None:
    """Verifica erro ao apontar para diretório inexistente ou sem git."""
    inexistente = tmp_path / "nao_existe"
    with pytest.raises(RuntimeError, match="Diretório do repositório não encontrado"):
        obter_info_commit(inexistente)

    sem_git = tmp_path / "vazio"
    sem_git.mkdir()
    with pytest.raises(RuntimeError, match="Erro ao executar git"):
        obter_info_commit(sem_git)


def test_validar_commit_head_ambiente_local() -> None:
    """Verifica que no ambiente local (sem forçar) a validação não bloqueia."""
    with patch(
        "src.utils.validador_git_colab.detectar_ambiente_colab", return_value=False
    ):
        # Mesmo com None, deve passar localmente
        info = validar_commit_head(
            REPO_ROOT, expected_commit=None, forcar_no_local=False
        )
        assert "hash_curto" in info


def test_validar_commit_head_falha_sem_commit() -> None:
    """Verifica disparo de RuntimeError quando commit esperado é None ou vazio no Colab."""
    with patch(
        "src.utils.validador_git_colab.detectar_ambiente_colab", return_value=True
    ):
        with pytest.raises(RuntimeError, match="COMMIT ESPERADO NÃO INFORMADO"):
            validar_commit_head(REPO_ROOT, expected_commit=None)

        with pytest.raises(RuntimeError, match="COMMIT ESPERADO NÃO INFORMADO"):
            validar_commit_head(REPO_ROOT, expected_commit="   ")


def test_validar_commit_head_hash_muito_curto() -> None:
    """Verifica recusa de hashes com menos de 7 caracteres."""
    with pytest.raises(RuntimeError, match="menos de 7 caracteres"):
        validar_commit_head(REPO_ROOT, expected_commit="abc12", forcar_no_local=True)


def test_validar_commit_head_sucesso() -> None:
    """Verifica validação bem-sucedida para hash curto e completo."""
    info_real = obter_info_commit(REPO_ROOT)
    hash_curto = info_real["hash_curto"]
    hash_completo = info_real["hash_completo"]

    # Teste com hash curto
    res_curto = validar_commit_head(
        REPO_ROOT, expected_commit=hash_curto, forcar_no_local=True
    )
    assert res_curto["hash_curto"] == hash_curto

    # Teste com hash completo
    res_comp = validar_commit_head(
        REPO_ROOT, expected_commit=hash_completo, forcar_no_local=True
    )
    assert res_comp["hash_completo"] == hash_completo


def test_validar_commit_head_divergencia_fail_fast() -> None:
    """Verifica disparo e conteúdo detalhado do erro Fail Fast ao divergir."""
    commit_ficticio = "0000000deadbeef"
    with pytest.raises(RuntimeError) as exc_info:
        validar_commit_head(
            REPO_ROOT, expected_commit=commit_ficticio, forcar_no_local=True
        )

    msg = str(exc_info.value)
    assert "DIVERGÊNCIA DE COMMIT NO GOOGLE COLAB" in msg
    assert commit_ficticio in msg
    assert "git push origin" in msg
    assert "Você provavelmente esqueceu de fazer push" in msg


def test_atualizar_repositorio_colab_sucesso(tmp_path: Path) -> None:
    """Verifica fluxo de clone e pull com mock de subprocess."""
    caminho_dest = tmp_path / "repo_teste"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        atualizar_repositorio_colab(
            repo_url="https://github.com/exemplo/repo.git",
            dest_path=caminho_dest,
            branch="main",
        )
        assert mock_run.call_count == 4  # clone, checkout, fetch, pull


def test_atualizar_repositorio_colab_erro_clone(tmp_path: Path) -> None:
    """Verifica tratamento de erro quando git clone falha."""
    caminho_dest = tmp_path / "repo_teste"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Repository not found"
        )
        with pytest.raises(RuntimeError, match="Falha ao clonar repositório"):
            atualizar_repositorio_colab(
                repo_url="https://github.com/exemplo/repo.git",
                dest_path=caminho_dest,
                branch="main",
            )
