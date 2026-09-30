"""Módulo de Diagnóstico de Overfitting e Robustez em Redes Hopfield.

Fornece componentes para auditoria de generalização fora da amostra, teste de
estresse por injeção de ruído sintético, auditoria de temperatura/atenção Softmax
e detecção de quimeras transcricionais espúrias em dados de scRNA-seq.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .avaliador_hopfield import AvaliadorHopfield


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
    """

    def aplicar_dropout(
        self,
        matriz: NDArray[Any],
        taxa: float,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Zera estocasticamente uma fração dos genes ativos sem mutação in-place.

        Parameters
        ----------
        matriz : NDArray[Any]
            Matriz de expressão binária de dimensões (n_celulas × n_genes).
        taxa : float
            Fração de genes ativos a serem desativados (0.0 a 1.0).
        seed : int or None, optional
            Semente para reproducibilidade estocástica.

        Returns
        -------
        NDArray[np.float32]
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
        matriz: NDArray[Any],
        taxa: float,
        seed: int | None = None,
    ) -> NDArray[np.float32]:
        """Inverte aleatoriamente uma fração dos valores da matriz binária.

        Parameters
        ----------
        matriz : NDArray[Any]
            Matriz de expressão binária de dimensões (n_celulas × n_genes).
        taxa : float
            Fração total de posições a terem seus bits invertidos (0.0 a 1.0).
        seed : int or None, optional
            Semente para reproducibilidade estocástica.

        Returns
        -------
        NDArray[np.float32]
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
        x_treino: NDArray[Any],
        y_treino: Sequence[int] | NDArray[Any],
        x_val: NDArray[Any],
        y_val: Sequence[int] | NDArray[Any],
        niveis_ruido: Sequence[float] = (0.05, 0.15, 0.30),
    ) -> ResultadoDiagnosticoOverfitting:
        """Executa a bateria completa de testes diagnósticos de overfitting.

        Parameters
        ----------
        x_treino : NDArray[Any]
            Matriz transcricional de treino (n_celulas_treino × n_genes).
        y_treino : Sequence[int] or NDArray[Any]
            Rótulos celulares verdadeiros das células de treino.
        x_val : NDArray[Any]
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
        x_val_arr = np.asarray(x_val, dtype=np.float32)
        x_treino_arr = np.asarray(x_treino, dtype=np.float32)
        y_val_list = [int(r) for r in y_val]
        y_treino_list = [int(r) for r in y_treino]

        # RN-01: Verificação contratual de formato e anotações
        n_genes_esperados = int(self.padroes.shape[1])
        if x_val_arr.shape[1] != n_genes_esperados:
            raise ValueError(
                f"Espaço gênico divergente: modelo espera {n_genes_esperados} genes, "
                f"recebido {x_val_arr.shape[1]}"
            )
        if len(x_val_arr) != len(y_val_list):
            raise ValueError(
                f"Incompatibilidade de dimensões: x_val possui {len(x_val_arr)} células, "
                f"mas y_val possui {len(y_val_list)} rótulos"
            )
        if len(x_val_arr) == 0:
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
