"""Testes unitários para o módulo global src.config."""

import os
from collections.abc import Generator
from pathlib import Path

import pytest

import src.config as cfg
from src.config import (
    _extrair_nome_dataset,
    configurar_diretorios,
)


@pytest.fixture(autouse=True)
def resetar_config() -> Generator[None]:
    """Restaura as configurações originais do módulo config após cada teste."""
    orig_ref = cfg.PATH_REFERENCIA
    orig_alvo = cfg.PATH_ALVO
    orig_nome_ref = cfg.NOME_REF
    orig_nome_alvo = cfg.NOME_ALVO
    orig_base = cfg.PATH_BASE
    yield
    configurar_diretorios(
        path_referencia=orig_ref,
        path_alvo=orig_alvo,
        nome_ref=orig_nome_ref,
        nome_alvo=orig_nome_alvo,
        path_base=orig_base,
        criar_diretorios=False,
    )


def test_extrair_nome_dataset() -> None:
    """Verifica a extração e higienização do nome a partir do caminho."""
    assert _extrair_nome_dataset("dados/pan_anotado.h5ad") == "pan_anotado"
    assert (
        _extrair_nome_dataset(r"C:\workspace\Mathys 2019-filtrado.h5ad")
        == "Mathys_2019-filtrado"
    )
    assert _extrair_nome_dataset("dataset.teste.v2.csv") == "dataset_teste_v2"
    assert _extrair_nome_dataset("") == "dados"


def test_valores_padrao_config() -> None:
    """Verifica se os valores padrão do config refletem o formato dinâmico output_<ref>_<alvo>."""
    assert cfg.OUTPUTS.endswith(f"output_{cfg.NOME_REF}_{cfg.NOME_ALVO}")
    assert os.path.join(cfg.OUTPUTS, "binarizacao") == cfg.OUT_BINARIZACAO
    assert os.path.join(cfg.OUTPUTS, "alinhamento") == cfg.OUT_ALINHAMENTO
    assert cfg.OUT_SWEEP_POS_IMPUTACAO == cfg.OUT_SWEEP_ALVO_POS_IMPUTACAO
    assert (
        os.path.join(cfg.OUT_TREINAMENTO, "matriz_reduzida_sweepREF.txt")
        == cfg.PATH_SWEEP_REFERENCIA
    )
    assert (
        os.path.join(cfg.OUT_TREINAMENTO, "matriz_reduzida_sweepALVO.txt")
        == cfg.PATH_SWEEP_ALVO
    )


def test_configurar_diretorios_com_novos_caminhos() -> None:
    """Verifica se a troca dos caminhos de entrada altera automaticamente o nome de output."""
    novo_outputs = configurar_diretorios(
        path_referencia="/data/fujita_control.h5ad",
        path_alvo="/data/mathys_ad.h5ad",
    )

    assert cfg.NOME_REF == "fujita_control"
    assert cfg.NOME_ALVO == "mathys_ad"
    assert novo_outputs.endswith("output_fujita_control_mathys_ad")
    assert novo_outputs == cfg.OUTPUTS
    assert os.path.join(novo_outputs, "binarizacao") == cfg.OUT_BINARIZACAO
    assert (
        os.path.join(novo_outputs, "treinamento", "matriz_reduzida_sweepREF.txt")
        == cfg.PATH_SWEEP_REFERENCIA
    )


def test_configurar_diretorios_com_nomes_customizados() -> None:
    """Verifica se nomes amigáveis explícitos têm precedência sobre a extração automática."""
    novo_outputs = configurar_diretorios(
        path_referencia="/data/arquivo_com_nome_muito_longo_v1_final.h5ad",
        path_alvo="/data/alvo_longo_complicado.h5ad",
        nome_ref="pan",
        nome_alvo="mathys",
    )

    assert cfg.NOME_REF == "pan"
    assert cfg.NOME_ALVO == "mathys"
    assert novo_outputs.endswith("output_pan_mathys")
    assert novo_outputs == cfg.OUTPUTS


def test_configurar_diretorios_criar_diretorios(tmp_path: Path) -> None:
    """Verifica se criar_diretorios=True gera as pastas fisicamente no disco."""
    base_temp = str(tmp_path)
    novo_outputs = configurar_diretorios(
        nome_ref="teste_ref",
        nome_alvo="teste_alvo",
        path_base=base_temp,
        criar_diretorios=True,
    )

    assert os.path.isdir(novo_outputs)
    assert os.path.isdir(cfg.OUT_BINARIZACAO)
    assert os.path.isdir(cfg.OUT_ALINHAMENTO)
    assert os.path.isdir(cfg.OUT_TOP_GENES)
    assert os.path.isdir(cfg.OUT_TREINAMENTO)
    assert os.path.isdir(cfg.OUT_HOPFIELD)
    assert os.path.isdir(cfg.OUT_IMPUTACAO)
    assert os.path.isdir(cfg.OUT_RELATORIO)
    assert os.path.isdir(cfg.OUT_SWEEP_ALVO_POS_IMPUTACAO)
    assert os.path.isdir(cfg.OUT_MTX_REFERENCIA)
    assert os.path.isdir(cfg.OUT_MTX_ALVO_SENTINELA)
    assert os.path.isdir(cfg.OUT_MTX_ALVO_IMPUTADO)
