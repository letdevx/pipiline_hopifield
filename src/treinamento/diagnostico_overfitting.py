"""Módulo de Diagnóstico de Overfitting e Robustez em Redes Hopfield.

Fornece componentes para auditoria de generalização fora da amostra, teste de
estresse por injeção de ruído sintético, auditoria de temperatura/atenção Softmax
e detecção de quimeras transcricionais espúrias em dados de scRNA-seq.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .avaliador_hopfield import AvaliadorHopfield

PathType = str | os.PathLike[str]


@dataclass(frozen=True)
class ResultadoDiagnosticoOverfitting:
    """Contrato imutável de resultado da auditoria de overfitting.

    Parameters
    ----------
    status : {"APROVADO", "ALERTA", "REPROVADO"}
        Parecer consolidado final da auditoria.
    gap_generalizacao : float
        Diferença de F1-Score biológico entre treino e validação.
    f1_treino : float
        F1-Score ponderado obtido na recuperação do conjunto de treino.
    f1_validacao : float
        F1-Score ponderado obtido na recuperação do conjunto de validação.
    fidelidade_sob_ruido : dict[float, float]
        Mapeamento nível de perturbação -> fidelidade transcricional média.
    ponto_ruptura_ruido : float or None
        Nível de ruído em que ocorreu colapso de atratores, se houver.
    entropia_media_atencao : float
        Entropia de Shannon média da distribuição de atenção Softmax.
    proporcao_atencao_saturada : float
        Fração de células cuja atenção concentrou peso ~1.0 em um único protótipo.
    quimeras_detectadas : int
        Número de células recuperadas com coativação de marcadores antagônicos.
    alertas : list[str], optional
        Lista de avisos e inconformidades detectadas durante o diagnóstico.
    recomendacoes : list[str], optional
        Lista de ações recomendadas de calibração paramétrica.
    """

    status: Literal["APROVADO", "ALERTA", "REPROVADO"]
    gap_generalizacao: float
    f1_treino: float
    f1_validacao: float
    fidelidade_sob_ruido: dict[float, float]
    ponto_ruptura_ruido: float | None
    entropia_media_atencao: float
    proporcao_atencao_saturada: float
    quimeras_detectadas: int
    alertas: list[str] = field(default_factory=list)
    recomendacoes: list[str] = field(default_factory=list)


class InjetorPerturbacaoTranscritica:
    """Injetor funcional de ruído estocástico e dropout artificial.

    Gera cópias imutáveis das matrizes transcricionais aplicando perturbações
    controladas sem alteração in-place dos dados originais.
    Suporta matrizes densas NumPy e esparsas SciPy (csr_matrix) de modo OOM-safe.
    """

    def aplicar_dropout(
        self,
        matriz: NDArray[Any] | sp.spmatrix,
        taxa: float,
        seed: int | None = None,
    ) -> NDArray[np.float32] | sp.spmatrix:
        """Zera estocasticamente uma fração dos genes ativos sem mutação in-place.

        Suporta matrizes densas NumPy e esparsas SciPy (csr_matrix), preservando a
        esparsidade e operando de modo OOM-safe sobre os dados não nulos.

        Parameters
        ----------
        matriz : NDArray[Any] | sp.spmatrix
            Matriz de expressão binária de dimensões (n_celulas × n_genes).
        taxa : float
            Fração de genes ativos a serem desativados (0.0 a 1.0).
        seed : int or None, optional
            Semente para reproducibilidade estocástica.

        Returns
        -------
        NDArray[np.float32] | sp.spmatrix
            Cópia da matriz com o dropout artificial aplicado.

        Raises
        ------
        ValueError
            Se a taxa não estiver no intervalo [0.0, 1.0].
        """
        if not (0.0 <= taxa <= 1.0):
            raise ValueError(
                f"Taxa de dropout deve estar entre 0.0 e 1.0, recebido {taxa}"
            )

        if sp.issparse(matriz):
            copia_csr = sp.csr_matrix(matriz, copy=True)
            if taxa == 0.0:
                return copia_csr

            rng = np.random.default_rng(seed)
            # Em matrizes binárias esparsas scRNA-seq, o vetor data contém os genes ativos (>0)
            n_ativos = len(copia_csr.data)
            if n_ativos > 0:
                n_remover = int(np.round(n_ativos * taxa))
                indices_sorteados = rng.choice(n_ativos, size=n_remover, replace=False)
                copia_csr.data[indices_sorteados] = 0.0
                copia_csr.eliminate_zeros()

            return copia_csr

        copia = np.array(matriz, dtype=np.float32, copy=True)
        if taxa == 0.0:
            return copia

        rng = np.random.default_rng(seed)
        linhas_ativas, cols_ativas = np.where(copia > 0.0)
        n_ativos = len(linhas_ativas)

        if n_ativos > 0:
            n_remover = int(np.round(n_ativos * taxa))
            indices_sorteados = rng.choice(n_ativos, size=n_remover, replace=False)
            copia[linhas_ativas[indices_sorteados], cols_ativas[indices_sorteados]] = (
                0.0
            )

        return copia

    def aplicar_bit_flip(
        self,
        matriz: NDArray[Any] | sp.spmatrix,
        taxa: float,
        seed: int | None = None,
    ) -> NDArray[np.float32] | sp.spmatrix:
        """Inverte aleatoriamente uma fração dos valores da matriz binária.

        Parameters
        ----------
        matriz : NDArray[Any] | sp.spmatrix
            Matriz de expressão binária de dimensões (n_celulas × n_genes).
        taxa : float
            Fração total de posições a terem seus bits invertidos (0.0 a 1.0).
        seed : int or None, optional
            Semente para reproducibilidade estocástica.

        Returns
        -------
        NDArray[np.float32] | sp.spmatrix
            Cópia da matriz com inversões estocásticas aplicadas.

        Raises
        ------
        ValueError
            Se a taxa não estiver no intervalo [0.0, 1.0].
        """
        if not (0.0 <= taxa <= 1.0):
            raise ValueError(
                f"Taxa de bit flip deve estar entre 0.0 e 1.0, recebido {taxa}"
            )

        if sp.issparse(matriz):
            copia_csr = sp.csr_matrix(matriz, copy=True)
            if taxa == 0.0:
                return copia_csr

            rng = np.random.default_rng(seed)
            n_rows, n_cols = copia_csr.shape
            n_total = n_rows * n_cols
            n_inverter = int(np.round(n_total * taxa))

            if n_inverter > 0:
                copia_lil = copia_csr.tolil()
                indices_lineares = rng.choice(n_total, size=n_inverter, replace=False)
                idx_linhas, idx_cols = np.unravel_index(
                    indices_lineares, (n_rows, n_cols)
                )
                for r, c in zip(idx_linhas, idx_cols, strict=True):
                    copia_lil[r, c] = 0.0 if copia_lil[r, c] > 0.5 else 1.0
                return copia_lil.tocsr()

            return copia_csr

        copia = np.array(matriz, dtype=np.float32, copy=True)
        if taxa == 0.0:
            return copia

        rng = np.random.default_rng(seed)
        n_total = copia.size
        n_inverter = int(np.round(n_total * taxa))

        if n_inverter > 0:
            indices_lineares = rng.choice(n_total, size=n_inverter, replace=False)
            idx_linhas, idx_cols = np.unravel_index(indices_lineares, copia.shape)
            copia[idx_linhas, idx_cols] = np.where(
                copia[idx_linhas, idx_cols] > 0.5, 0.0, 1.0
            )

        return copia


class AuditorOverfittingHopfield:
    """Orquestrador de auditoria diagnóstica de generalização e estabilidade.

    Conduz testes de validação fora da amostra (holdout), teste de estresse de
    bacias de atração via perturbações sintéticas e análise de saturação de
    temperatura e quimeras biológicas.

    Parameters
    ----------
    modelo : Any
        Instância de rede de memória associativa (ex.: ModernHopfieldNetwork).
    padroes_referencia : NDArray[Any]
        Matriz de protótipos armazenados de dimensões (n_prototipos × n_genes).
    rotulos_padroes : Sequence[int] or NDArray[Any]
        Rótulo de linhagem celular associado a cada protótipo.
    gap_maximo_tolerado : float, default=0.15
        Diferença máxima aceitável de F1-Score entre treino e validação.
    limiar_entropia_minima : float, default=0.05
        Limiar mínimo de entropia Softmax abaixo do qual a atenção é considerada saturada.
    marcadores_exclusivos : Mapping[int, Sequence[int]] or None, optional
        Mapeamento classe -> índices de genes marcadores antagônicos.
    seed : int, default=42
        Semente para inicializações pseudoaleatórias do auditor.
    """

    def __init__(
        self,
        modelo: Any,
        padroes_referencia: NDArray[Any],
        rotulos_padroes: Sequence[int] | NDArray[Any],
        gap_maximo_tolerado: float = 0.15,
        limiar_entropia_minima: float = 0.05,
        marcadores_exclusivos: Mapping[int, Sequence[int]] | None = None,
        seed: int = 42,
    ) -> None:
        self.modelo: Any = modelo
        self.padroes: NDArray[np.float32] = np.asarray(
            padroes_referencia, dtype=np.float32
        )
        self.rotulos_padroes: list[int] = [int(r) for r in rotulos_padroes]
        self.gap_maximo: float = float(gap_maximo_tolerado)
        self.limiar_entropia: float = float(limiar_entropia_minima)
        self.marcadores_exclusivos: dict[int, list[int]] = (
            {int(k): [int(g) for g in v] for k, v in marcadores_exclusivos.items()}
            if marcadores_exclusivos is not None
            else {}
        )
        self.seed: int = int(seed)
        self.injetor: InjetorPerturbacaoTranscritica = InjetorPerturbacaoTranscritica()
        self.classes: list[int] = sorted(list(set(self.rotulos_padroes)))

    def _obter_f1(
        self,
        recuperados: NDArray[np.float32],
        rotulos_esperados: list[int],
    ) -> float:
        """Calcula o F1-Score ponderado através do AvaliadorHopfield."""
        meta = [(rot, i) for i, rot in enumerate(self.rotulos_padroes)]
        avaliador = AvaliadorHopfield(
            padroes=self.padroes,
            classes=self.classes,
            meta=meta,
        )
        avaliador.avaliar(recuperados, rotulos_esperados)
        return float(
            avaliador.f1_weighted if avaliador.f1_weighted is not None else 0.0
        )

    def executar_auditoria(
        self,
        x_treino: NDArray[Any] | sp.spmatrix,
        y_treino: Sequence[int] | NDArray[Any],
        x_val: NDArray[Any] | sp.spmatrix,
        y_val: Sequence[int] | NDArray[Any],
        niveis_ruido: Sequence[float] = (0.05, 0.15, 0.30),
    ) -> ResultadoDiagnosticoOverfitting:
        """Executa a bateria completa de testes diagnósticos de overfitting.

        Parameters
        ----------
        x_treino : NDArray[Any] | sp.spmatrix
            Matriz transcricional de treino (n_celulas_treino × n_genes).
        y_treino : Sequence[int] or NDArray[Any]
            Rótulos celulares verdadeiros das células de treino.
        x_val : NDArray[Any] | sp.spmatrix
            Matriz transcricional de validação holdout (n_celulas_val × n_genes).
        y_val : Sequence[int] or NDArray[Any]
            Rótulos celulares verdadeiros das células de validação.
        niveis_ruido : Sequence[float], default=(0.05, 0.15, 0.30)
            Taxas de perturbação a serem injetadas para teste de bacias de atração.

        Returns
        -------
        ResultadoDiagnosticoOverfitting
            Resultado estruturado com métricas, alertas e parecer consolidado.

        Raises
        ------
        ValueError
            Se as dimensões gênicas de x_val divergirem do modelo ou se houver
            incompatibilidade entre células e rótulos.
        """
        x_val_arr: NDArray[np.float32] | sp.csr_matrix = (
            sp.csr_matrix(x_val)
            if sp.issparse(x_val)
            else np.asarray(x_val, dtype=np.float32)
        )
        x_treino_arr: NDArray[np.float32] | sp.csr_matrix = (
            sp.csr_matrix(x_treino)
            if sp.issparse(x_treino)
            else np.asarray(x_treino, dtype=np.float32)
        )
        y_val_list = [int(r) for r in y_val]
        y_treino_list = [int(r) for r in y_treino]

        # RN-01: Verificação contratual de formato e anotações
        n_genes_esperados = int(self.padroes.shape[1])
        if x_val_arr.shape[1] != n_genes_esperados:
            raise ValueError(
                f"Espaço gênico divergente: modelo espera {n_genes_esperados} genes, "
                f"recebido {x_val_arr.shape[1]}"
            )
        if x_val_arr.shape[0] != len(y_val_list):
            raise ValueError(
                f"Incompatibilidade de dimensões: x_val possui {x_val_arr.shape[0]} células, "
                f"mas y_val possui {len(y_val_list)} rótulos"
            )
        if x_val_arr.shape[0] == 0:
            raise ValueError("Conjunto de validação x_val não pode ser vazio")

        alertas: list[str] = []
        recomendacoes: list[str] = []

        # RN-02: Recuperação de treino e validação
        recup_treino = self.modelo.retrieve(x_treino_arr)
        recup_val, atencao_val = self.modelo.retrieve(
            x_val_arr,
            return_attention_weights=True,
        )

        f1_treino = self._obter_f1(recup_treino, y_treino_list)
        f1_val = self._obter_f1(recup_val, y_val_list)

        gap_generalizacao = float(max(0.0, f1_treino - f1_val))
        if gap_generalizacao > self.gap_maximo:
            alertas.append(
                f"Alerta: Queda excessiva de generalização (Gap de {gap_generalizacao * 100:.1f}%)"
            )
            recomendacoes.append(
                "Reduzir número de subclusters por classe ou calibrar representação dos protótipos."
            )

        # RN-03: Teste de Estresse por Perturbação de Ruído
        fidelidade_sob_ruido: dict[float, float] = {}
        ponto_ruptura_ruido: float | None = None

        for taxa in niveis_ruido:
            x_perturbado = self.injetor.aplicar_dropout(
                x_val_arr, taxa=taxa, seed=self.seed
            )
            recup_ruido = self.modelo.retrieve(x_perturbado)
            fid = self._obter_f1(recup_ruido, y_val_list)
            fidelidade_sob_ruido[float(taxa)] = fid

            # Se com ruído leve (≤ 5%) há queda acentuada (> 25% em relação ao f1_val)
            if taxa <= 0.05 and (f1_val - fid) > 0.25:
                ponto_ruptura_ruido = float(taxa)
                alertas.append(
                    "Alerta: Atratores frágeis / Bacias de atração colapsadas com ruído leve (≤ 5%)"
                )
                recomendacoes.append(
                    "Aumentar o valor de beta ou refinar o centróide dos protótipos."
                )

        # RN-04: Auditoria de Atenção Softmax e Quimeras Transcricionais
        atencao_arr = np.asarray(atencao_val, dtype=np.float32)
        eps = 1e-12
        entropia_por_celula = -np.sum(atencao_arr * np.log(atencao_arr + eps), axis=1)
        n_prototipos = int(atencao_arr.shape[1])

        if n_prototipos > 1:
            entropia_norm = entropia_por_celula / np.log(n_prototipos)
        else:
            entropia_norm = entropia_por_celula

        entropia_media_atencao = float(np.mean(entropia_norm))
        proporcao_atencao_saturada = float(
            np.mean(entropia_norm <= self.limiar_entropia)
        )

        if (
            proporcao_atencao_saturada >= 0.95
            and n_prototipos > 1
            and self.limiar_entropia > 0.0
        ):
            alertas.append(
                "Alerta: Saturação de Beta / Memorização Rígida de Ruído (efeito 1-NN puro)"
            )
            recomendacoes.append(
                "Reduzir hiperparâmetro beta para suavizar as bacias de atração da atenção Softmax."
            )

        # Detecção de Quimeras Biológicas
        quimeras_detectadas = 0
        if self.marcadores_exclusivos and len(self.marcadores_exclusivos) >= 2:
            recup_val_arr = np.asarray(recup_val, dtype=np.float32)
            for celula in recup_val_arr:
                classes_ativas = [
                    classe_id
                    for classe_id, genes in self.marcadores_exclusivos.items()
                    if all(celula[g] > 0.5 for g in genes)
                ]
                if len(classes_ativas) > 1:
                    quimeras_detectadas += 1

            if quimeras_detectadas > 0:
                alertas.append(
                    f"Alerta: Estado Espúrio / Quimera Transcricional detectada em "
                    f"{quimeras_detectadas} células"
                )
                recomendacoes.append(
                    "Limpar assinaturas sobrepostas nos protótipos de treino e revisar marcadores canônicos."
                )

        # RF-05: Parecer Consolidado
        if ponto_ruptura_ruido is not None or quimeras_detectadas > 0:
            status = "REPROVADO"
        elif len(alertas) > 0:
            status = "ALERTA"
        else:
            status = "APROVADO"

        return ResultadoDiagnosticoOverfitting(
            status=status,
            gap_generalizacao=gap_generalizacao,
            f1_treino=f1_treino,
            f1_validacao=f1_val,
            fidelidade_sob_ruido=fidelidade_sob_ruido,
            ponto_ruptura_ruido=ponto_ruptura_ruido,
            entropia_media_atencao=entropia_media_atencao,
            proporcao_atencao_saturada=proporcao_atencao_saturada,
            quimeras_detectadas=quimeras_detectadas,
            alertas=alertas,
            recomendacoes=recomendacoes,
        )


@dataclass(frozen=True)
class ResultadoDiagnosticoCrossDataset:
    """Contrato imutável de resultado da auditoria de overfitting na imputação cross-dataset.

    Parameters
    ----------
    status : {"APROVADO", "ALERTA", "REPROVADO"}
        Parecer consolidado final da auditoria.
    roc_auc : float
        Área sob a curva ROC calculada sobre as probabilidades contínuas nos genes mascarados.
    pr_auc : float
        Área sob a curva Precision-Recall (Average Precision) nos genes mascarados.
    prevalencia_basal : float
        Fração real de expressão ativa (1s) nas coordenadas mascaradas.
    razao_pr_auc_basal : float
        Razão entre PR-AUC e a prevalência basal (pr_auc / max(1e-6, prevalencia_basal)).
    f1_ativo : float
        F1-Score da classe positiva/ativa (1.0) sobre os valores recuperados binarizados.
    precisao_ativo : float
        Precisão da classe ativa (1.0) sobre os valores recuperados binarizados.
    recall_ativo : float
        Sensibilidade / Recall da classe ativa (1.0).
    acuracia_binaria : float
        Acurácia global de predição nas coordenadas mascaradas.
    n_genes_mascarados : int
        Total de genes artificialmente ocultados com sentinela 0.5.
    n_coordenadas_auditadas : int
        Total de coordenadas avaliadas (n_celulas × n_genes_mascarados).
    especificidade_marcadores : dict[str, float]
        Métrica de especificidade por linhagem celular nos marcadores canônicos mascarados.
    f1_por_tipo_celular : dict[str, float]
        F1-Score da classe ativa estratificado por tipo celular nos genes mascarados.
    alertas : list[str], optional
        Lista de avisos e inconformidades detectadas durante o diagnóstico.
    recomendacoes : list[str], optional
        Lista de ações recomendadas de calibração paramétrica.
    curva_roc : dict[str, list[float]] or None, optional
        Pontos decimados da curva ROC (fpr, tpr).
    curva_pr : dict[str, list[float]] or None, optional
        Pontos decimados da curva Precision-Recall (recall, precision).
    limiar_otimo : float, default=0.50
        Limiar de corte contínuo que maximiza a média harmônica F1 no holdout.
    f1_ativo_calibrado : float, default=0.0
        F1-Score obtido sob o limiar ótimo calibrado.
    precisao_calibrada : float, default=0.0
        Precisão obtida sob o limiar ótimo calibrado.
    recall_calibrado : float, default=0.0
        Recall obtido sob o limiar ótimo calibrado.
    status_calibrado : {"APROVADO", "ALERTA", "REPROVADO"}, default="ALERTA"
        Parecer consolidado sob o ponto de operação do limiar ótimo.
    """

    status: Literal["APROVADO", "ALERTA", "REPROVADO"]
    roc_auc: float
    pr_auc: float
    prevalencia_basal: float
    razao_pr_auc_basal: float
    f1_ativo: float
    precisao_ativo: float
    recall_ativo: float
    acuracia_binaria: float
    n_genes_mascarados: int
    n_coordenadas_auditadas: int
    especificidade_marcadores: dict[str, float]
    f1_por_tipo_celular: dict[str, float]
    alertas: list[str] = field(default_factory=list)
    recomendacoes: list[str] = field(default_factory=list)
    curva_roc: dict[str, list[float]] | None = None
    curva_pr: dict[str, list[float]] | None = None
    limiar_otimo: float = 0.50
    f1_ativo_calibrado: float = 0.0
    precisao_calibrada: float = 0.0
    recall_calibrado: float = 0.0
    status_calibrado: Literal["APROVADO", "ALERTA", "REPROVADO"] = "ALERTA"


def _extrair_bloco_matriz_generica(
    mat: Any, inicio: int, tamanho: int
) -> sp.csr_matrix | NDArray[Any]:
    """Extrai um lote de células de forma transparente a partir de AnnData, CSR ou NumPy.

    Parameters
    ----------
    mat : Any
        Matriz de entrada (AnnData, CSR/CSC SciPy ou NDArray NumPy).
    inicio : int
        Índice da linha inicial.
    tamanho : int
        Número de linhas a extrair.

    Returns
    -------
    sp.csr_matrix or NDArray[Any]
        Fatia correspondente em formato esparso CSR ou array denso NumPy.
    """
    fim = inicio + tamanho
    alvo = mat.X if hasattr(mat, "X") else mat

    chunk = alvo[inicio:fim]
    if hasattr(chunk, "to_memory"):
        return sp.csr_matrix(chunk.to_memory())
    if sp.issparse(chunk):
        return sp.csr_matrix(chunk)
    return chunk


class AuditorOverfittingCrossDataset:
    """Auditor de overfitting e generalização biológica na imputação cross-dataset via Masked Holdout.

    Executa o protocolo experimental padrão-ouro em bioinformática de scRNA-seq:
    1. Oculta artificialmente uma fração fixa (ex.: 10%) de genes medidos no alvo com a sentinela 0.5.
    2. Realiza a inferência em lotes seguros contra estouro de memória (OOM-Safe).
    3. Avalia o poder discriminativo contínuo via ROC-AUC e PR-AUC em relação à prevalência basal.
    4. Audita o F1-Score da classe ativa e a especificidade por tipo celular contra o Ground Truth.
    5. Emite veredito consolidado padronizado (APROVADO, ALERTA ou REPROVADO).

    Parameters
    ----------
    modelo : Any
        Instância treinada de ModernHopfieldNetwork com método .retrieve().
    limiar_roc_auc_aprovado : float, default=0.70
        Limiar mínimo de ROC-AUC para status APROVADO.
    limiar_roc_auc_alerta : float, default=0.60
        Limiar mínimo de ROC-AUC abaixo do qual o modelo é REPROVADO.
    limiar_razao_pr_auc_aprovado : float, default=2.0
        Razão mínima de ganho do PR-AUC em relação à prevalência basal (P0) para APROVADO.
    limiar_razao_pr_auc_alerta : float, default=1.5
        Razão mínima de ganho do PR-AUC abaixo da qual o modelo é REPROVADO.
    limiar_f1_ativo_aprovado : float, default=0.45
        Limiar de F1-Score na classe ativa para consideração de generalização aceitável.
    seed : int, default=42
        Semente pseudoaleatória para reprodutibilidade estocástica do mascaramento.
    """

    def __init__(
        self,
        modelo: Any,
        limiar_roc_auc_aprovado: float = 0.70,
        limiar_roc_auc_alerta: float = 0.60,
        limiar_razao_pr_auc_aprovado: float = 2.0,
        limiar_razao_pr_auc_alerta: float = 1.5,
        limiar_f1_ativo_aprovado: float = 0.45,
        seed: int = 42,
    ) -> None:
        self.modelo: Any = modelo
        self.limiar_roc_auc_aprovado: float = float(limiar_roc_auc_aprovado)
        self.limiar_roc_auc_alerta: float = float(limiar_roc_auc_alerta)
        self.limiar_razao_pr_auc_aprovado: float = float(limiar_razao_pr_auc_aprovado)
        self.limiar_razao_pr_auc_alerta: float = float(limiar_razao_pr_auc_alerta)
        self.limiar_f1_ativo_aprovado: float = float(limiar_f1_ativo_aprovado)
        self.seed: int = int(seed)

    def selecionar_genes_holdout(
        self,
        n_genes: int,
        mask_ausentes: NDArray[Any] | Sequence[int] | None,
        taxa_mascaramento: float = 0.10,
        marcadores_indices: Mapping[int | str, Sequence[int]] | None = None,
        seed: int | None = None,
    ) -> NDArray[np.bool_]:
        """Seleciona deterministicamente os genes presentes para mascaramento artificial.

        Parameters
        ----------
        n_genes : int
            Total de genes no espaço canônico alinhado.
        mask_ausentes : NDArray[Any] or Sequence[int] or None
            Máscara dos genes ausentes originais no dataset alvo.
        taxa_mascaramento : float, default=0.10
            Fração dos genes presentes a serem mascarados (0.0 a 1.0).
        marcadores_indices : Mapping[int | str, Sequence[int]] or None, optional
            Mapeamento de tipos celulares para índices de marcadores canônicos.
        seed : int or None, optional
            Semente para amostragem pseudoaleatória.

        Returns
        -------
        NDArray[np.bool_]
            Vetor booleano de dimensão (n_genes,) com True nas posições de holdout.
        """
        if not (0.0 < taxa_mascaramento < 1.0):
            raise ValueError(
                f"taxa_mascaramento deve estar entre 0.0 e 1.0, recebido {taxa_mascaramento}"
            )

        rng = np.random.default_rng(self.seed if seed is None else int(seed))

        if mask_ausentes is None:
            mask_aus_arr = np.zeros(n_genes, dtype=bool)
        else:
            mask_aus_arr = np.asarray(mask_ausentes, dtype=bool)
            if len(mask_aus_arr) != n_genes:
                # Se for lista de índices
                temp_mask = np.zeros(n_genes, dtype=bool)
                temp_mask[np.asarray(mask_ausentes, dtype=int)] = True
                mask_aus_arr = temp_mask

        indices_presentes = np.where(~mask_aus_arr)[0]
        n_presentes = len(indices_presentes)
        if n_presentes == 0:
            raise ValueError("Não há genes presentes disponíveis para mascaramento.")

        n_alvo_holdout = max(1, round(n_presentes * taxa_mascaramento))

        selecionados_set: set[int] = set()

        # 1. Prioriza a inclusão de marcadores canônicos presentes para teste biológico
        if marcadores_indices is not None:
            for _, genes in marcadores_indices.items():
                for g in genes:
                    g_idx = int(g)
                    if 0 <= g_idx < n_genes and not mask_aus_arr[g_idx]:
                        selecionados_set.add(g_idx)

        # 2. Amostra o restante dos genes presentes para atingir a taxa configurada
        candidatos_restantes = [
            int(idx) for idx in indices_presentes if idx not in selecionados_set
        ]

        faltantes = n_alvo_holdout - len(selecionados_set)
        if faltantes > 0 and len(candidatos_restantes) > 0:
            qtd_amostrar = min(faltantes, len(candidatos_restantes))
            amostrados = rng.choice(
                candidatos_restantes, size=qtd_amostrar, replace=False
            )
            for g in amostrados:
                selecionados_set.add(int(g))

        mask_holdout = np.zeros(n_genes, dtype=bool)
        for idx in selecionados_set:
            mask_holdout[idx] = True

        return mask_holdout

    def executar_auditoria(
        self,
        matriz_alvo: Any,
        mask_ausentes: NDArray[Any] | Sequence[int] | None = None,
        mask_genes_holdout: NDArray[Any] | None = None,
        classes_reais: Sequence[int | str] | NDArray[Any] | None = None,
        nomes_classes: Mapping[int | str, str] | None = None,
        taxa_mascaramento: float = 0.10,
        marcadores_indices: Mapping[int | str, Sequence[int]] | None = None,
        batch_size: int = 5000,
        seed: int | None = None,
    ) -> ResultadoDiagnosticoCrossDataset:
        """Executa a auditoria completa de overfitting cross-dataset em lotes OOM-safe.

        Parameters
        ----------
        matriz_alvo : Any
            Matriz de expressão do conjunto alvo (NDArray, sp.csr_matrix ou ad.AnnData).
        mask_ausentes : NDArray[Any] or Sequence[int] or None, optional
            Máscara de genes ausentes originais do dataset alvo.
        mask_genes_holdout : NDArray[Any] or None, optional
            Máscara pré-definida de genes a serem mascarados. Se None, é gerada automaticamente.
        classes_reais : Sequence[int | str] or NDArray[Any] or None, optional
            Rótulos verdadeiros de tipo celular das células do alvo.
        nomes_classes : Mapping[int | str, str] or None, optional
            Dicionário mapeando IDs de classe para nomes biológicos legíveis.
        taxa_mascaramento : float, default=0.10
            Fração dos genes presentes a serem mascarados (se mask_genes_holdout for None).
        marcadores_indices : Mapping[int | str, Sequence[int]] or None, optional
            Mapeamento de classes para índices de marcadores canônicos conhecidos.
        batch_size : int, default=5000
            Tamanho de lote para processamento em streaming com memória constante.
        seed : int or None, optional
            Semente para reprodutibilidade.

        Returns
        -------
        ResultadoDiagnosticoCrossDataset
            Contrato estruturado contendo métricas, alertas e veredito formal.
        """
        # Dimensões da matriz alvo
        if hasattr(matriz_alvo, "shape"):
            n_celulas = int(matriz_alvo.shape[0])
            n_genes = int(matriz_alvo.shape[1])
        elif hasattr(matriz_alvo, "n_obs"):
            n_celulas = int(matriz_alvo.n_obs)
            n_genes = int(matriz_alvo.n_vars)
        else:
            raise TypeError("matriz_alvo não possui formato com atributos de dimensão.")

        if n_celulas == 0 or n_genes == 0:
            raise ValueError("matriz_alvo vazia.")

        # Máscara de ausentes original
        if mask_ausentes is None:
            mask_aus_arr = np.zeros(n_genes, dtype=bool)
        else:
            mask_aus_arr = np.asarray(mask_ausentes, dtype=bool)
            if len(mask_aus_arr) != n_genes:
                temp_mask = np.zeros(n_genes, dtype=bool)
                temp_mask[np.asarray(mask_ausentes, dtype=int)] = True
                mask_aus_arr = temp_mask

        # Máscara de holdout experimental
        if mask_genes_holdout is None:
            mask_holdout_arr = self.selecionar_genes_holdout(
                n_genes=n_genes,
                mask_ausentes=mask_aus_arr,
                taxa_mascaramento=taxa_mascaramento,
                marcadores_indices=marcadores_indices,
                seed=seed,
            )
        else:
            mask_holdout_arr = np.asarray(mask_genes_holdout, dtype=bool)
            if len(mask_holdout_arr) != n_genes:
                raise ValueError(
                    f"Dimensão de mask_genes_holdout ({len(mask_holdout_arr)}) diverge de n_genes ({n_genes})"
                )

        indices_holdout = np.where(mask_holdout_arr)[0]
        n_holdout = len(indices_holdout)
        if n_holdout == 0:
            raise ValueError("Nenhum gene selecionado para holdout experimental.")

        # Máscara combinada para injeção de 0.5 durante a inferência Hopfield
        mask_combinada = mask_aus_arr | mask_holdout_arr

        # Pré-aloca matrizes de acumulação OOM-Safe apenas para as colunas mascaradas
        y_true_acumulado = np.empty((n_celulas, n_holdout), dtype=np.uint8)
        y_scores_acumulado = np.empty((n_celulas, n_holdout), dtype=np.float32)
        y_pred_acumulado = np.empty((n_celulas, n_holdout), dtype=np.uint8)

        print(
            f"[AuditorOverfittingCrossDataset] Executando auditoria em {n_celulas:,} células "
            f"com {n_holdout:,} genes mascarados (lotes de {batch_size:,})..."
        )

        for s in range(0, n_celulas, batch_size):
            chunk = _extrair_bloco_matriz_generica(matriz_alvo, s, batch_size)
            fim = min(s + batch_size, n_celulas)
            tam_chunk = fim - s

            # 1. Extração do Ground Truth nos genes mascarados
            if sp.issparse(chunk):
                chunk_csr = sp.csr_matrix(chunk)
                y_true_sub = chunk_csr[:, mask_holdout_arr].toarray()
            else:
                chunk_np = np.asarray(chunk)
                y_true_sub = chunk_np[:, mask_holdout_arr]

            y_true_acumulado[s:fim] = (y_true_sub > 0.0).astype(np.uint8)

            # 2. Inferência Hopfield com injeção transitória da sentinela 0.5
            res_inf = self.modelo.retrieve(
                queries=chunk,
                batch_size=tam_chunk,
                mask_sentinela_ausentes=mask_combinada,
                fill_value=0.5,
                return_probabilities=True,
            )

            if isinstance(res_inf, tuple):
                recup_chunk = res_inf[0]
                prob_chunk = res_inf[1]
            else:
                recup_chunk = res_inf
                prob_chunk = res_inf

            # 3. Fatiamento restrito aos genes mascarados
            prob_sub = prob_chunk[:, mask_holdout_arr]
            recup_sub = recup_chunk[:, mask_holdout_arr]

            y_scores_acumulado[s:fim] = np.asarray(prob_sub, dtype=np.float32)
            y_pred_acumulado[s:fim] = (recup_sub > 0.5).astype(np.uint8)

        # 4. Cálculo Estatístico Global
        total_coordenadas = n_celulas * n_holdout
        prevalencia_basal = float(np.mean(y_true_acumulado == 1))

        # Amostragem determinística para curvas ROC/PR se volume for gigantesco
        LIMITE_PONTOS_CURVA = 5_000_000
        rng_eval = np.random.default_rng(self.seed)

        if total_coordenadas > LIMITE_PONTOS_CURVA:
            idx_amostra = rng_eval.choice(
                total_coordenadas, size=LIMITE_PONTOS_CURVA, replace=False
            )
            y_true_eval = y_true_acumulado.ravel()[idx_amostra]
            y_scores_eval = y_scores_acumulado.ravel()[idx_amostra]
            y_pred_eval = y_pred_acumulado.ravel()[idx_amostra]
        else:
            y_true_eval = y_true_acumulado.ravel()
            y_scores_eval = y_scores_acumulado.ravel()
            y_pred_eval = y_pred_acumulado.ravel()

        # Validação contra classes monocromáticas
        if len(np.unique(y_true_eval)) < 2:
            roc_auc = 0.5
            pr_auc = prevalencia_basal
            fpr = np.array([0.0, 1.0])
            tpr = np.array([0.0, 1.0])
            precision_pts = np.array([prevalencia_basal, prevalencia_basal])
            recall_pts = np.array([0.0, 1.0])
            thresholds_pr = np.array([0.5])
        else:
            roc_auc = float(roc_auc_score(y_true_eval, y_scores_eval))
            pr_auc = float(average_precision_score(y_true_eval, y_scores_eval))
            fpr, tpr, _ = roc_curve(y_true_eval, y_scores_eval)
            precision_pts, recall_pts, thresholds_pr = precision_recall_curve(
                y_true_eval, y_scores_eval
            )

        f1_ativo = float(
            f1_score(y_true_eval, y_pred_eval, pos_label=1, zero_division=cast(Any, 0))
        )
        precisao_ativo = float(
            precision_score(
                y_true_eval, y_pred_eval, pos_label=1, zero_division=cast(Any, 0)
            )
        )
        recall_ativo = float(
            recall_score(
                y_true_eval, y_pred_eval, pos_label=1, zero_division=cast(Any, 0)
            )
        )
        acuracia_binaria = float(accuracy_score(y_true_eval, y_pred_eval))
        razao_pr_auc_basal = float(pr_auc / max(1e-6, prevalencia_basal))

        # Varredura vetorial instantânea de limiar ótimo de corte na curva PR
        if len(precision_pts) > 1 and len(thresholds_pr) > 0:
            f1_curva = (2.0 * precision_pts[:-1] * recall_pts[:-1]) / np.maximum(
                precision_pts[:-1] + recall_pts[:-1], 1e-9
            )
            idx_otimo = int(np.argmax(f1_curva))
            limiar_otimo = float(thresholds_pr[idx_otimo])
            f1_ativo_calibrado = float(f1_curva[idx_otimo])
            precisao_calibrada = float(precision_pts[idx_otimo])
            recall_calibrado = float(recall_pts[idx_otimo])
        else:
            limiar_otimo = 0.50
            f1_ativo_calibrado = f1_ativo
            precisao_calibrada = precisao_ativo
            recall_calibrado = recall_ativo

        # Se o corte padrão 0.50 já for superior por arredondamento
        if f1_ativo > f1_ativo_calibrado:
            f1_ativo_calibrado = f1_ativo
            precisao_calibrada = precisao_ativo
            recall_calibrado = recall_ativo
            limiar_otimo = 0.50

        # Decimação de pontos para visualização e persistência compacta
        MAX_PTS = 200
        if len(fpr) > MAX_PTS:
            passo_roc = max(1, len(fpr) // MAX_PTS)
            fpr = fpr[::passo_roc]
            tpr = tpr[::passo_roc]
        if len(precision_pts) > MAX_PTS:
            passo_pr = max(1, len(precision_pts) // MAX_PTS)
            precision_pts = precision_pts[::passo_pr]
            recall_pts = recall_pts[::passo_pr]

        curva_roc = {
            "fpr": [float(v) for v in fpr],
            "tpr": [float(v) for v in tpr],
        }
        curva_pr = {
            "recall": [float(v) for v in recall_pts],
            "precision": [float(v) for v in precision_pts],
        }

        # 5. Avaliação Estratificada por Linhagem Celular
        f1_por_tipo: dict[str, float] = {}
        if classes_reais is not None and len(classes_reais) == n_celulas:
            classes_arr = np.asarray(classes_reais)
            for c in sorted(list(set(classes_arr))):
                idx_c = np.where(classes_arr == c)[0]
                if len(idx_c) == 0:
                    continue
                y_t_c = y_true_acumulado[idx_c].ravel()
                y_p_c = y_pred_acumulado[idx_c].ravel()
                nome_c = str(nomes_classes.get(c, c)) if nomes_classes else str(c)
                f1_c = float(
                    f1_score(y_t_c, y_p_c, pos_label=1, zero_division=cast(Any, 0))
                )
                f1_por_tipo[nome_c] = round(f1_c, 4)

        # 6. Auditoria de Especificidade de Marcadores Canônicos
        especificidade_marcadores: dict[str, float] = {}
        if (
            marcadores_indices is not None
            and classes_reais is not None
            and len(classes_reais) == n_celulas
        ):
            classes_arr = np.asarray(classes_reais)
            col_map_holdout: dict[int, int] = {
                int(orig_idx): pos for pos, orig_idx in enumerate(indices_holdout)
            }

            for classe_id, genes in marcadores_indices.items():
                colunas_marc = [
                    col_map_holdout[g] for g in genes if g in col_map_holdout
                ]
                if not colunas_marc:
                    continue

                idx_classe = np.where(classes_arr == classe_id)[0]
                idx_outras = np.where(classes_arr != classe_id)[0]

                if len(idx_classe) > 0 and len(idx_outras) > 0:
                    taxa_classe = float(
                        np.mean(y_pred_acumulado[idx_classe][:, colunas_marc] == 1)
                    )
                    taxa_outras = float(
                        np.mean(y_pred_acumulado[idx_outras][:, colunas_marc] == 1)
                    )
                    razao_esp = float(taxa_classe / max(1e-6, taxa_outras))
                    nome_classe_str = (
                        str(nomes_classes.get(classe_id, classe_id))
                        if nomes_classes
                        else str(classe_id)
                    )
                    especificidade_marcadores[nome_classe_str] = round(razao_esp, 2)

        # 7. Regras Formais de Decisão e Alertas
        alertas: list[str] = []
        recomendacoes: list[str] = []

        if roc_auc < self.limiar_roc_auc_alerta:
            alertas.append(
                f"Reprovado: ROC-AUC nos genes mascarados ({roc_auc:.4f} < {self.limiar_roc_auc_alerta:.2f}) "
                "indica falha severa de discriminação (discriminação próxima ao acaso)."
            )
            recomendacoes.append(
                "Verificar se a temperatura Beta da rede Hopfield está alta demais ou se houve colapso de atratores."
            )
        elif roc_auc < self.limiar_roc_auc_aprovado:
            alertas.append(
                f"Alerta: ROC-AUC moderado ({roc_auc:.4f} < {self.limiar_roc_auc_aprovado:.2f})."
            )

        if razao_pr_auc_basal < self.limiar_razao_pr_auc_alerta:
            alertas.append(
                f"Reprovado: PR-AUC ({pr_auc:.4f}) próximo ao limiar de prevalência basal ({prevalencia_basal:.4f}), "
                f"com ganho de {razao_pr_auc_basal:.2f}x < {self.limiar_razao_pr_auc_alerta:.1f}x."
            )
            recomendacoes.append(
                "O modelo está memorizando a proporção geral de zeros sem recuperar a estrutura esparsa dos genes ativos."
            )
        elif razao_pr_auc_basal < self.limiar_razao_pr_auc_aprovado:
            alertas.append(
                f"Alerta: Ganho de PR-AUC ({razao_pr_auc_basal:.2f}x) abaixo do limiar ótimo de 2.0x prevalência basal."
            )

        if f1_ativo < self.limiar_f1_ativo_aprovado:
            alertas.append(
                f"Alerta: F1-Score da classe ativa ({f1_ativo:.4f} < {self.limiar_f1_ativo_aprovado:.2f}) "
                "aponta recuperação conservadora das coordenadas ativas."
            )

        if any(a.startswith("Reprovado:") for a in alertas):
            status: Literal["APROVADO", "ALERTA", "REPROVADO"] = "REPROVADO"
        elif len(alertas) > 0:
            status = "ALERTA"
        else:
            status = "APROVADO"

        # 8. Parecer sob o limiar ótimo calibrado
        alertas_calibrados: list[str] = []
        if roc_auc < self.limiar_roc_auc_alerta:
            alertas_calibrados.append("Reprovado: ROC-AUC")
        elif roc_auc < self.limiar_roc_auc_aprovado:
            alertas_calibrados.append("Alerta: ROC-AUC moderado")

        if razao_pr_auc_basal < self.limiar_razao_pr_auc_alerta:
            alertas_calibrados.append("Reprovado: PR-AUC")
        elif razao_pr_auc_basal < self.limiar_razao_pr_auc_aprovado:
            alertas_calibrados.append("Alerta: PR-AUC moderado")

        if f1_ativo_calibrado < self.limiar_f1_ativo_aprovado:
            alertas_calibrados.append("Alerta: F1 ativo calibrado")

        if any(a.startswith("Reprovado:") for a in alertas_calibrados):
            status_calibrado: Literal["APROVADO", "ALERTA", "REPROVADO"] = "REPROVADO"
        elif len(alertas_calibrados) > 0:
            status_calibrado = "ALERTA"
        else:
            status_calibrado = "APROVADO"

        return ResultadoDiagnosticoCrossDataset(
            status=status,
            roc_auc=round(roc_auc, 4),
            pr_auc=round(pr_auc, 4),
            prevalencia_basal=round(prevalencia_basal, 4),
            razao_pr_auc_basal=round(razao_pr_auc_basal, 2),
            f1_ativo=round(f1_ativo, 4),
            precisao_ativo=round(precisao_ativo, 4),
            recall_ativo=round(recall_ativo, 4),
            acuracia_binaria=round(acuracia_binaria, 4),
            n_genes_mascarados=n_holdout,
            n_coordenadas_auditadas=total_coordenadas,
            especificidade_marcadores=especificidade_marcadores,
            f1_por_tipo_celular=f1_por_tipo,
            alertas=alertas,
            recomendacoes=recomendacoes,
            curva_roc=curva_roc,
            curva_pr=curva_pr,
            limiar_otimo=round(limiar_otimo, 4),
            f1_ativo_calibrado=round(f1_ativo_calibrado, 4),
            precisao_calibrada=round(precisao_calibrada, 4),
            recall_calibrado=round(recall_calibrado, 4),
            status_calibrado=status_calibrado,
        )

    def gerar_painel_grafico(
        self,
        resultado: ResultadoDiagnosticoCrossDataset,
        path_saida: PathType | None = None,
        titulo: str = "Diagnóstico de Overfitting — Imputação Cross-Dataset (Mathys)",
    ) -> Any:
        """Gera o painel gráfico com Curva ROC, Curva Precision-Recall e F1 por Linhagem.

        Parameters
        ----------
        resultado : ResultadoDiagnosticoCrossDataset
            Objeto com os resultados estatísticos da auditoria.
        path_saida : PathType or None, optional
            Caminho de saída para salvar a imagem PNG (resolução 300 DPI).
        titulo : str, default="Diagnóstico de Overfitting — Imputação Cross-Dataset (Mathys)"
            Título superior principal da figura.

        Returns
        -------
        matplotlib.figure.Figure
            Instância da figura Matplotlib criada.
        """
        fig, (ax_roc, ax_pr, ax_f1) = plt.subplots(1, 3, figsize=(19, 5.5))

        # 1. Curva ROC
        if resultado.curva_roc is not None:
            fpr = resultado.curva_roc["fpr"]
            tpr = resultado.curva_roc["tpr"]
            ax_roc.plot(
                fpr,
                tpr,
                color="#2b5c8f",
                lw=2.5,
                label=f"Hopfield (AUC = {resultado.roc_auc:.4f})",
            )
        ax_roc.plot(
            [0, 1],
            [0, 1],
            color="#888888",
            linestyle="--",
            lw=1.5,
            label="Aleatório (AUC = 0.50)",
        )
        ax_roc.set_xlim([-0.02, 1.02])
        ax_roc.set_ylim([-0.02, 1.05])
        ax_roc.set_xlabel("FPR (1 - Especificidade)", fontsize=10)
        ax_roc.set_ylabel("TPR (Sensibilidade)", fontsize=10)
        ax_roc.set_title("Curva ROC (Masked Holdout)", fontsize=11, fontweight="bold")
        ax_roc.legend(loc="lower right", frameon=True)
        ax_roc.grid(True, linestyle=":", alpha=0.6)

        # 2. Curva Precision-Recall
        if resultado.curva_pr is not None:
            rec = resultado.curva_pr["recall"]
            prec = resultado.curva_pr["precision"]
            ax_pr.plot(
                rec,
                prec,
                color="#2a9d8f",
                lw=2.5,
                label=f"Hopfield (PR-AUC = {resultado.pr_auc:.4f})",
            )
        ax_pr.axhline(
            resultado.prevalencia_basal,
            color="#e76f51",
            linestyle="--",
            lw=1.8,
            label=f"Prevalência P0 ({resultado.prevalencia_basal:.4f})",
        )
        if resultado.f1_ativo_calibrado > 0:
            ax_pr.plot(
                resultado.recall_calibrado,
                resultado.precisao_calibrada,
                marker="o",
                markersize=8,
                color="#e76f51",
                label=f"Limiar Ótimo ({resultado.limiar_otimo:.2f} | F1 = {resultado.f1_ativo_calibrado:.3f})",
            )
        ax_pr.set_xlim([-0.02, 1.02])
        ax_pr.set_ylim([-0.02, 1.05])
        ax_pr.set_xlabel("Recall (Sensibilidade)", fontsize=10)
        ax_pr.set_ylabel("Precisão", fontsize=10)
        ax_pr.set_title(
            f"Curva Precision-Recall (Ganho = {resultado.razao_pr_auc_basal:.2f}× P0)",
            fontsize=11,
            fontweight="bold",
        )
        ax_pr.legend(loc="upper right", frameon=True, fontsize=8)
        ax_pr.grid(True, linestyle=":", alpha=0.6)

        # 3. F1-Score por Linhagem Celular
        if resultado.f1_por_tipo_celular:
            nomes = list(resultado.f1_por_tipo_celular.keys())
            f1s = list(resultado.f1_por_tipo_celular.values())
            x_pos = np.arange(len(nomes))

            ax_f1.bar(
                x_pos,
                f1s,
                color="#457b9d",
                edgecolor="#1d3557",
                width=0.55,
                label="F1 Linhagem",
            )
            ax_f1.axhline(
                resultado.f1_ativo,
                color="#e63946",
                linestyle="--",
                lw=1.8,
                label=f"F1 Global Padrão ({resultado.f1_ativo:.4f})",
            )
            if resultado.f1_ativo_calibrado > resultado.f1_ativo:
                ax_f1.axhline(
                    resultado.f1_ativo_calibrado,
                    color="#2a9d8f",
                    linestyle=":",
                    lw=1.8,
                    label=f"F1 Calibrado ({resultado.f1_ativo_calibrado:.4f})",
                )
            ax_f1.set_ylim([0, 1.15])
            ax_f1.set_ylabel("F1-Score da Classe Ativa", fontsize=10)
            ax_f1.set_title(
                "F1-Score por Linhagem (Genes Mascarados)",
                fontsize=11,
                fontweight="bold",
            )
            ax_f1.set_xticks(x_pos)
            ax_f1.set_xticklabels(nomes, rotation=35, ha="right", fontsize=9)

            for i, v in enumerate(f1s):
                ax_f1.annotate(
                    f"{v:.3f}",
                    xy=(i, v + 0.02),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
            ax_f1.legend(loc="upper right", frameon=True, fontsize=8)
            ax_f1.grid(axis="y", linestyle=":", alpha=0.6)
        else:
            ax_f1.text(
                0.5,
                0.5,
                "Estratificação celular não informada",
                ha="center",
                va="center",
                fontsize=11,
            )

        cor_status = {
            "APROVADO": "#2a9d8f",
            "ALERTA": "#e76f51",
            "REPROVADO": "#d62828",
        }.get(resultado.status, "#333333")

        titulo_completo = f"{titulo} — Parecer: [{resultado.status}]"
        if resultado.status_calibrado != resultado.status:
            titulo_completo += f" | Calibrado: [{resultado.status_calibrado}]"

        fig.suptitle(
            titulo_completo,
            fontsize=13,
            fontweight="bold",
            color=cor_status,
            y=1.02,
        )
        plt.tight_layout()

        if path_saida is not None:
            os.makedirs(os.path.dirname(os.path.abspath(path_saida)), exist_ok=True)
            plt.savefig(path_saida, dpi=300, bbox_inches="tight")

        return fig

    def exportar_relatorio(
        self,
        resultado: ResultadoDiagnosticoCrossDataset,
        path_csv: PathType | None = None,
        path_json: PathType | None = None,
    ) -> dict[str, str]:
        """Exporta os resultados e métricas consolidadas em formatos CSV e JSON.

        Parameters
        ----------
        resultado : ResultadoDiagnosticoCrossDataset
            Objeto com os resultados da auditoria.
        path_csv : PathType or None, optional
            Caminho do arquivo CSV de saída.
        path_json : PathType or None, optional
            Caminho do arquivo JSON de saída.

        Returns
        -------
        dict[str, str]
            Caminhos dos arquivos exportados com sucesso.
        """
        arquivos_gerados: dict[str, str] = {}

        dados_tabela = [
            {
                "metrica": "ROC-AUC",
                "valor": resultado.roc_auc,
                "referencia_padrao": "≥ 0.70 (Aprovado) / ≥ 0.60 (Alerta)",
            },
            {
                "metrica": "PR-AUC",
                "valor": resultado.pr_auc,
                "referencia_padrao": "≥ 2.0x Prevalência Basal P0",
            },
            {
                "metrica": "Prevalencia_Basal_P0",
                "valor": resultado.prevalencia_basal,
                "referencia_padrao": "Taxa natural de 1s nos dados",
            },
            {
                "metrica": "Razao_PRAUC_Prevalencia",
                "valor": resultado.razao_pr_auc_basal,
                "referencia_padrao": "≥ 2.0x (Aprovado) / ≥ 1.5x (Alerta)",
            },
            {
                "metrica": "F1_Classe_Ativa",
                "valor": resultado.f1_ativo,
                "referencia_padrao": "≥ 0.45",
            },
            {
                "metrica": "Precisao_Classe_Ativa",
                "valor": resultado.precisao_ativo,
                "referencia_padrao": "Alta",
            },
            {
                "metrica": "Recall_Classe_Ativa",
                "valor": resultado.recall_ativo,
                "referencia_padrao": "Alta",
            },
            {
                "metrica": "Limiar_Otimo",
                "valor": resultado.limiar_otimo,
                "referencia_padrao": "Argmax F1 no holdout",
            },
            {
                "metrica": "F1_Ativo_Calibrado",
                "valor": resultado.f1_ativo_calibrado,
                "referencia_padrao": "≥ 0.45 (Aprovado)",
            },
            {
                "metrica": "Precisao_Calibrada",
                "valor": resultado.precisao_calibrada,
                "referencia_padrao": "No limiar ótimo",
            },
            {
                "metrica": "Recall_Calibrado",
                "valor": resultado.recall_calibrado,
                "referencia_padrao": "No limiar ótimo",
            },
            {
                "metrica": "Acuracia_Binaria",
                "valor": resultado.acuracia_binaria,
                "referencia_padrao": "≥ 0.85",
            },
            {
                "metrica": "Genes_Mascarados",
                "valor": resultado.n_genes_mascarados,
                "referencia_padrao": "10% dos genes presentes",
            },
            {
                "metrica": "Coordenadas_Auditadas",
                "valor": resultado.n_coordenadas_auditadas,
                "referencia_padrao": "N_celulas × N_genes_holdout",
            },
            {
                "metrica": "Status_Final",
                "valor": resultado.status,
                "referencia_padrao": "APROVADO",
            },
            {
                "metrica": "Status_Calibrado",
                "valor": resultado.status_calibrado,
                "referencia_padrao": "APROVADO sob calibração",
            },
        ]
        df_metricas = pd.DataFrame(dados_tabela)

        if path_csv is not None:
            os.makedirs(os.path.dirname(os.path.abspath(path_csv)), exist_ok=True)
            df_metricas.to_csv(path_csv, index=False)
            arquivos_gerados["csv"] = str(path_csv)

        if path_json is not None:
            os.makedirs(os.path.dirname(os.path.abspath(path_json)), exist_ok=True)
            dados_json = {
                "status": resultado.status,
                "status_calibrado": resultado.status_calibrado,
                "metricas_globais": {
                    "roc_auc": resultado.roc_auc,
                    "pr_auc": resultado.pr_auc,
                    "prevalencia_basal": resultado.prevalencia_basal,
                    "razao_pr_auc_basal": resultado.razao_pr_auc_basal,
                    "f1_ativo": resultado.f1_ativo,
                    "precisao_ativo": resultado.precisao_ativo,
                    "recall_ativo": resultado.recall_ativo,
                    "limiar_otimo": resultado.limiar_otimo,
                    "f1_ativo_calibrado": resultado.f1_ativo_calibrado,
                    "precisao_calibrada": resultado.precisao_calibrada,
                    "recall_calibrado": resultado.recall_calibrado,
                    "acuracia_binaria": resultado.acuracia_binaria,
                    "n_genes_mascarados": resultado.n_genes_mascarados,
                    "n_coordenadas_auditadas": resultado.n_coordenadas_auditadas,
                },
                "especificidade_marcadores": resultado.especificidade_marcadores,
                "f1_por_tipo_celular": resultado.f1_por_tipo_celular,
                "alertas": resultado.alertas,
                "recomendacoes": resultado.recomendacoes,
            }
            with open(path_json, "w", encoding="utf-8") as f:
                json.dump(dados_json, f, indent=2, ensure_ascii=False)
            arquivos_gerados["json"] = str(path_json)

        return arquivos_gerados
