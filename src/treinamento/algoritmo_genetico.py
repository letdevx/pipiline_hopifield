"""Módulo de Otimização de Hiperparâmetros via Algoritmo Genético.

Explora o espaço combinatório da Modern Hopfield Network e do Extrator de
Padrões SWeeP para descobrir configurações de máxima generalização,
resistência a ruído e equilíbrio termodinâmico de atenção.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import scipy.sparse as sp
import torch
from numpy.typing import NDArray
from sklearn.metrics import f1_score

from .diagnostico_overfitting import InjetorPerturbacaoTranscritica
from .estrategias_clusterizacao import EstrategiaKMeansDinamico, EstrategiaKMeansFixo
from .extrator_padroes import ExtratorPadroesSubcluster
from .hopfield import ModernHopfieldNetwork


@dataclass(frozen=True)
class IndividuoHopfield:
    """Representação imutável de um cromossomo de hiperparâmetros e suas métricas.

    Parameters
    ----------
    nc : int
        Número de protótipos/centróides por classe biológica.
    k_vizinhos : int
        Número de vizinhos locais consolidados para seleção do protótipo.
    beta : float
        Temperatura inversa da atenção Softmax Hopfield.
    threshold : float
        Limiar de corte para ativação binarizada pós-recuperação.
    n_iters : int
        Número de iterações síncronas de atualização da rede.
    normalize : bool
        Indica o uso de projeção esférica unitária (similaridade cosseno L2).
    estrategia : {"kmeans_fixo", "kmeans_dinamico"}
        Estratégia de agrupamento no espaço latente SWeeP.
    fitness : float, default=0.0
        Pontuação consolidada de aptidão do indivíduo.
    f1_val : float, default=0.0
        F1-Score (Macro por padrão) obtido na amostra de validação holdout.
    gap_generalizacao : float, default=0.0
        Diferença absoluta de F1 entre treino e validação.
    f1_ruido : float, default=0.0
        F1-Score sob injeção de ruído/dropout sintético de estresse.
    entropia_atencao : float, default=0.0
        Entropia normalizada da distribuição de atenção Softmax.
    quimeras : int, default=0
        Contagem de células espúrias com coativação de marcadores antagônicos.
    """

    nc: int
    k_vizinhos: int
    beta: float
    threshold: float
    n_iters: int
    normalize: bool
    estrategia: Literal["kmeans_fixo", "kmeans_dinamico"]
    fitness: float = 0.0
    f1_val: float = 0.0
    gap_generalizacao: float = 0.0
    f1_ruido: float = 0.0
    entropia_atencao: float = 0.0
    quimeras: int = 0

    def para_dicionario(self) -> dict[str, Any]:
        """Converte os atributos do indivíduo em um dicionário serializável.

        Returns
        -------
        dict[str, Any]
            Dicionário com hiperparâmetros e métricas do indivíduo.
        """
        return {
            "nc": int(self.nc),
            "k_vizinhos": int(self.k_vizinhos),
            "beta": float(self.beta),
            "threshold": float(self.threshold),
            "n_iters": int(self.n_iters),
            "normalize": bool(self.normalize),
            "estrategia": str(self.estrategia),
            "fitness": float(self.fitness),
            "f1_val": float(self.f1_val),
            "gap_generalizacao": float(self.gap_generalizacao),
            "f1_ruido": float(self.f1_ruido),
            "entropia_atencao": float(self.entropia_atencao),
            "quimeras": int(self.quimeras),
        }


@dataclass(frozen=True)
class ConfiguracaoAG:
    """Parâmetros operacionais do Algoritmo Genético e pesos da função de aptidão.

    Parameters
    ----------
    tam_populacao : int, default=24
        Quantidade de indivíduos por geração (mínimo 4).
    n_geracoes : int, default=15
        Número total de ciclos evolutivos.
    p_crossover : float, default=0.85
        Probabilidade de cruzamento entre pares selecionados.
    p_mutacao : float, default=0.20
        Probabilidade de mutação por gene individual.
    elitismo : int, default=2
        Quantidade de melhores indivíduos preservados integralmente para a próxima geração.
    torneio_k : int, default=3
        Tamanho do torneio na seleção de pais.
    seed : int, default=42
        Semente pseudoaleatória para reprodutibilidade.
    max_cache_padroes : int, default=25
        Capacidade máxima do cache LRU para matrizes de protótipos SWeeP.
    beta_min : float, default=0.1
        Limite inferior do espaço de busca da temperatura inversa beta.
    beta_max : float, default=5.0
        Limite superior do espaço de busca da temperatura inversa beta.
    metrica_f1 : {"macro", "weighted"}, default="macro"
        Estratégia de cálculo do F1-score (macro pondera uniformemente todas as classes).
    escalar_por_raiz_d : bool, default=True
        Se True, normaliza o produto interno por sqrt(D) em ModernHopfieldNetwork quando normalize=False.
    w_f1 : float, default=0.50
        Peso do F1 de validação holdout.
    w_ruido : float, default=0.25
        Peso do F1 sob degradação por ruído.
    w_gap : float, default=0.15
        Peso da penalidade por excesso de gap de generalização (> 0.15).
    w_sat : float, default=0.05
        Peso da penalidade por saturação de atenção (entropia < 0.05).
    w_qui : float, default=0.05
        Peso da penalidade por detecção de quimeras celulares.
    w_parc : float, default=0.02
        Peso da penalidade suave de complexidade (parcimônia de memória por nc).
    taxa_ruido_estresse : float, default=0.15
        Fração de dropout sintético injetado no teste de estresse de bacias.
    """

    tam_populacao: int = 24
    n_geracoes: int = 15
    p_crossover: float = 0.85
    p_mutacao: float = 0.20
    elitismo: int = 2
    torneio_k: int = 3
    seed: int = 42
    max_cache_padroes: int = 25
    beta_min: float = 0.1
    beta_max: float = 5.0
    metrica_f1: Literal["macro", "weighted"] = "macro"
    escalar_por_raiz_d: bool = True
    w_f1: float = 0.50
    w_ruido: float = 0.25
    w_gap: float = 0.15
    w_sat: float = 0.05
    w_qui: float = 0.05
    w_parc: float = 0.02
    taxa_ruido_estresse: float = 0.15

    def __post_init__(self) -> None:
        """Valida os limites dos hiperparâmetros de configuração."""
        if self.tam_populacao < 4:
            raise ValueError(
                f"tam_populacao deve ser pelo menos 4, recebido {self.tam_populacao}"
            )
        if not (0.0 <= self.p_crossover <= 1.0):
            raise ValueError("p_crossover deve estar no intervalo [0.0, 1.0]")
        if not (0.0 <= self.p_mutacao <= 1.0):
            raise ValueError("p_mutacao deve estar no intervalo [0.0, 1.0]")
        if self.elitismo >= self.tam_populacao:
            raise ValueError("elitismo deve ser menor que o tamanho da população")
        if self.beta_min <= 0.0:
            raise ValueError("beta_min deve ser estritamente positivo (> 0.0)")
        if self.beta_min > self.beta_max:
            raise ValueError("beta_min não pode ser maior que beta_max")
        if self.metrica_f1 not in ("macro", "weighted"):
            raise ValueError(
                f"metrica_f1 deve ser 'macro' ou 'weighted', recebido '{self.metrica_f1}'"
            )


class OtimizadorGeneticoHopfield:
    """Motor de otimização evolutiva de hiperparâmetros para Redes Hopfield Modernas.

    Parameters
    ----------
    w0 : NDArray | sp.spmatrix
        Matriz de expressão binarizada original (n_celulas × n_genes).
    wswp : NDArray
        Matriz de projeção dimensional latente SWeeP (n_celulas × n_componentes).
    labels : NDArray | Sequence[int]
        Vetor com os rótulos biológicos canônicos das células.
    classes : Sequence[int] | None, optional
        Lista de identificadores das classes canônicas a processar.
    x_val : NDArray | None, optional
        Matriz de expressão do conjunto holdout não-visto para validação.
    y_val : NDArray | None, optional
        Rótulos correspondentes ao conjunto de validação.
    marcadores_exclusivos : dict[int, list[int]] | None, optional
        Mapeamento classe -> índices de genes marcadores canônicos para detecção de quimeras.
    config : ConfiguracaoAG | None, optional
        Instância imutável com parâmetros do algoritmo genético.
    """

    def __init__(
        self,
        w0: NDArray[Any] | sp.spmatrix,
        wswp: NDArray[Any],
        labels: NDArray[Any] | Sequence[int],
        classes: Sequence[int] | None = None,
        x_val: NDArray[Any] | None = None,
        y_val: NDArray[Any] | None = None,
        marcadores_exclusivos: dict[int, list[int]] | None = None,
        config: ConfiguracaoAG | None = None,
    ) -> None:
        if sp.issparse(w0):
            self.w0: NDArray[np.float32] = (
                sp.csr_matrix(w0).toarray().astype(np.float32)
            )
        else:
            self.w0 = np.asarray(w0, dtype=np.float32)
        self.wswp: NDArray[np.float32] = np.asarray(wswp, dtype=np.float32)
        self.labels: NDArray[np.int_] = np.asarray(labels, dtype=int)
        self.classes: list[int] = (
            list(classes)
            if classes is not None
            else sorted(np.unique(self.labels).tolist())
        )
        self.marcadores_exclusivos: dict[int, list[int]] = (
            marcadores_exclusivos if marcadores_exclusivos is not None else {}
        )
        self.config: ConfiguracaoAG = config if config is not None else ConfiguracaoAG()

        # Configura conjunto de validação
        if x_val is not None and y_val is not None:
            self.x_val: NDArray[np.float32] = np.asarray(x_val, dtype=np.float32)
            self.y_val: NDArray[np.int_] = np.asarray(y_val, dtype=int)
        else:
            self.x_val = self.w0
            self.y_val = self.labels

        # Injetor de ruído estocástico pré-computado para estabilidade
        injetor = InjetorPerturbacaoTranscritica()
        self.x_val_ruido: NDArray[np.float32] = injetor.aplicar_dropout(
            matriz=self.x_val,
            taxa=self.config.taxa_ruido_estresse,
            seed=self.config.seed,
        )

        # Gerador pseudoaleatório determinístico
        self.rng: np.random.Generator = np.random.default_rng(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Cache LRU de padrões extraídos
        self.cache_padroes: OrderedDict[
            tuple[int, int, str], tuple[NDArray[np.float32], list[tuple[int, int]]]
        ] = OrderedDict()
        self.estatisticas_cache: dict[str, int] = {"hits": 0, "misses": 0}
        self.historico_geracoes: list[dict[str, Any]] = []

    def obter_padroes(
        self,
        nc: int,
        k_vizinhos: int,
        estrategia: Literal["kmeans_fixo", "kmeans_dinamico"],
    ) -> tuple[NDArray[np.float32], list[tuple[int, int]]]:
        """Recupera protótipos de memória do cache LRU ou executa o extrator.

        Parameters
        ----------
        nc : int
            Número de subclusters por classe.
        k_vizinhos : int
            Contagem de vizinhos locais.
        estrategia : {"kmeans_fixo", "kmeans_dinamico"}
            Estratégia de particionamento.

        Returns
        -------
        tuple[NDArray[np.float32], list[tuple[int, int]]]
            Matriz de protótipos (n_padroes × n_genes) e lista de tuplas (classe, idx).
        """
        chave = (int(nc), int(k_vizinhos), str(estrategia))
        if chave in self.cache_padroes:
            self.estatisticas_cache["hits"] += 1
            self.cache_padroes.move_to_end(chave)
            return self.cache_padroes[chave]

        self.estatisticas_cache["misses"] += 1
        estrat_inst = (
            EstrategiaKMeansFixo(n_clusters=nc, seed=self.config.seed)
            if estrategia == "kmeans_fixo"
            else EstrategiaKMeansDinamico(k_range=[nc], seed=self.config.seed)
        )

        extrator = ExtratorPadroesSubcluster(
            W0=self.w0,
            labels=self.labels,
            classes=self.classes,
            estrategia=estrat_inst,
            seed=self.config.seed,
            k=k_vizinhos,
            nc=nc,
        )
        extrator.extrair(self.wswp)

        assert extrator.padroes is not None and extrator.meta is not None
        padroes = extrator.padroes.astype(np.float32)
        meta = list(extrator.meta)

        # Evicção LRU se exceder o teto
        if len(self.cache_padroes) >= self.config.max_cache_padroes:
            self.cache_padroes.popitem(last=False)

        self.cache_padroes[chave] = (padroes, meta)
        return padroes, meta

    def gerar_populacao_inicial(self) -> list[IndividuoHopfield]:
        """Gera a população inicial combinando sementes consagradas e amostragem variada.

        Returns
        -------
        list[IndividuoHopfield]
            Lista inicial de indivíduos não-avaliados.
        """
        populacao: list[IndividuoHopfield] = []

        # 1. Sementes canônicas calibradas na escala estável de beta
        sementes = [
            IndividuoHopfield(
                nc=30,
                k_vizinhos=1,
                beta=1.0,
                threshold=0.0,
                n_iters=1,
                normalize=False,
                estrategia="kmeans_fixo",
            ),
            IndividuoHopfield(
                nc=30,
                k_vizinhos=5,
                beta=0.5,
                threshold=0.0,
                n_iters=1,
                normalize=False,
                estrategia="kmeans_fixo",
            ),
            IndividuoHopfield(
                nc=20,
                k_vizinhos=3,
                beta=2.5,
                threshold=0.0,
                n_iters=1,
                normalize=True,
                estrategia="kmeans_dinamico",
            ),
            IndividuoHopfield(
                nc=15,
                k_vizinhos=1,
                beta=4.0,
                threshold=-0.1,
                n_iters=1,
                normalize=False,
                estrategia="kmeans_fixo",
            ),
        ]
        sementes_ajustadas = [
            IndividuoHopfield(
                nc=s.nc,
                k_vizinhos=s.k_vizinhos,
                beta=float(np.clip(s.beta, self.config.beta_min, self.config.beta_max)),
                threshold=s.threshold,
                n_iters=s.n_iters,
                normalize=s.normalize,
                estrategia=s.estrategia,
            )
            for s in sementes
        ]
        populacao.extend(sementes_ajustadas[: self.config.tam_populacao])

        # 2. Amostragem complementar dentro dos limites do espaço de busca
        while len(populacao) < self.config.tam_populacao:
            nc = int(self.rng.integers(5, 46))
            k_viz = int(self.rng.integers(1, 11))
            beta = float(self.rng.uniform(self.config.beta_min, self.config.beta_max))
            threshold = float(self.rng.uniform(-0.2, 0.4))
            n_iters = int(self.rng.integers(1, 4))
            normalize = bool(self.rng.choice([True, False]))
            estrategia: Literal["kmeans_fixo", "kmeans_dinamico"] = (
                "kmeans_fixo" if self.rng.random() < 0.5 else "kmeans_dinamico"
            )

            populacao.append(
                IndividuoHopfield(
                    nc=nc,
                    k_vizinhos=k_viz,
                    beta=beta,
                    threshold=threshold,
                    n_iters=n_iters,
                    normalize=normalize,
                    estrategia=estrategia,
                )
            )

        return populacao

    def avaliar_individuo(self, ind: IndividuoHopfield) -> IndividuoHopfield:
        """Calcula o fitness multiobjetivo e as métricas de auditoria para um indivíduo.

        Parameters
        ----------
        ind : IndividuoHopfield
            Indivíduo com hiperparâmetros a avaliar.

        Returns
        -------
        IndividuoHopfield
            Novo indivíduo preenchido com as métricas de fitness computadas.
        """
        padroes, meta = self.obter_padroes(
            nc=ind.nc, k_vizinhos=ind.k_vizinhos, estrategia=ind.estrategia
        )

        # Instanciação da Modern Hopfield Network
        rede = ModernHopfieldNetwork(
            beta=ind.beta,
            n_iters=ind.n_iters,
            binary=True,
            threshold=ind.threshold,
            normalize=ind.normalize,
            scale_by_dim=self.config.escalar_por_raiz_d,
        )
        rede.store(padroes)

        # 1. Avaliação no conjunto de validação holdout
        batch_size = 4096
        w_rec_val, att_val = rede.retrieve(
            self.x_val, batch_size=batch_size, return_attention_weights=True
        )

        # Softmax Class Pooling
        classes_padroes = np.array([m[0] for m in meta])
        y_pred_val = self._predizer_por_pooling(att_val, classes_padroes)
        f1_val = float(
            f1_score(
                self.y_val,
                y_pred_val,
                average=self.config.metrica_f1,
                zero_division=cast(Any, 0),
            )
        )

        # 2. Avaliação de treino aproximada para detecção de gap
        # Avalia amostra rápida do treino para cálculo de generalização
        n_amostra_treino = min(1000, len(self.w0))
        _, att_tr = rede.retrieve(
            self.w0[:n_amostra_treino],
            batch_size=batch_size,
            return_attention_weights=True,
        )
        y_pred_tr = self._predizer_por_pooling(att_tr, classes_padroes)
        f1_treino = float(
            f1_score(
                self.labels[:n_amostra_treino],
                y_pred_tr,
                average=self.config.metrica_f1,
                zero_division=cast(Any, 0),
            )
        )
        gap = float(abs(f1_treino - f1_val))

        # 3. Teste de estresse sob ruído/dropout sintético
        _, att_ruido = rede.retrieve(
            self.x_val_ruido, batch_size=batch_size, return_attention_weights=True
        )
        y_pred_ruido = self._predizer_por_pooling(att_ruido, classes_padroes)
        f1_ruido = float(
            f1_score(
                self.y_val,
                y_pred_ruido,
                average=self.config.metrica_f1,
                zero_division=cast(Any, 0),
            )
        )

        # 4. Entropia Normalizada da Atenção Softmax
        n_padroes = len(padroes)
        entropia_norm = self._calcular_entropia_atencao(att_val, n_padroes)

        # 5. Detecção de Quimeras Transcricionais
        quimeras = self._contar_quimeras(w_rec_val)

        # 6. Cálculo da Função de Aptidão Ponderada com Barreira
        penalidade_gap = max(0.0, gap - 0.15)
        penalidade_sat = max(0.0, 0.05 - entropia_norm)
        penalidade_qui = float(quimeras) / max(1.0, float(len(self.x_val)))
        penalidade_parc = float(ind.nc) / 45.0

        fitness = float(
            (self.config.w_f1 * f1_val)
            + (self.config.w_ruido * f1_ruido)
            - (self.config.w_gap * penalidade_gap)
            - (self.config.w_sat * penalidade_sat)
            - (self.config.w_qui * penalidade_qui)
            - (self.config.w_parc * penalidade_parc)
        )

        return IndividuoHopfield(
            nc=ind.nc,
            k_vizinhos=ind.k_vizinhos,
            beta=ind.beta,
            threshold=ind.threshold,
            n_iters=ind.n_iters,
            normalize=ind.normalize,
            estrategia=ind.estrategia,
            fitness=fitness,
            f1_val=f1_val,
            gap_generalizacao=gap,
            f1_ruido=f1_ruido,
            entropia_atencao=entropia_norm,
            quimeras=quimeras,
        )

    def _predizer_por_pooling(
        self, att_weights: NDArray[np.float32], classes_padroes: NDArray[np.int_]
    ) -> NDArray[np.int_]:
        """Agrega os pesos de atenção por classe celular e extrai a predição majoritária."""
        n_amostras = att_weights.shape[0]
        classes_unicas = self.classes
        somas = np.zeros((n_amostras, len(classes_unicas)), dtype=np.float32)

        for i, cl in enumerate(classes_unicas):
            mascara_cl = classes_padroes == cl
            if np.any(mascara_cl):
                somas[:, i] = np.sum(att_weights[:, mascara_cl], axis=1)

        idx_max = np.argmax(somas, axis=1)
        return np.array([classes_unicas[i] for i in idx_max], dtype=int)

    def _calcular_entropia_atencao(
        self, att_weights: NDArray[np.float32], n_padroes: int
    ) -> float:
        """Calcula a entropia de Shannon normalizada média dos pesos de atenção."""
        eps = 1e-12
        p = np.clip(att_weights, eps, 1.0)
        entropia_por_celula = -np.sum(p * np.log(p), axis=1)
        entropia_max = np.log(max(2, n_padroes))
        return float(np.mean(entropia_por_celula) / entropia_max)

    def _contar_quimeras(self, w_recuperado: NDArray[np.float32]) -> int:
        """Identifica células recuperadas que coativaram marcadores mutuamente exclusivos."""
        if len(self.marcadores_exclusivos) < 2:
            return 0

        # Verifica binarização das células recuperadas
        w_bin = (w_recuperado > 0.0).astype(int)
        quimeras = 0

        # Identifica ativação por par de classes antagônicas
        classes_marcadas = list(self.marcadores_exclusivos.keys())
        for idx_c1 in range(len(classes_marcadas)):
            for idx_c2 in range(idx_c1 + 1, len(classes_marcadas)):
                c1, c2 = classes_marcadas[idx_c1], classes_marcadas[idx_c2]
                genes_c1 = self.marcadores_exclusivos[c1]
                genes_c2 = self.marcadores_exclusivos[c2]

                ativo_c1 = np.all(w_bin[:, genes_c1] == 1, axis=1)
                ativo_c2 = np.all(w_bin[:, genes_c2] == 1, axis=1)
                quimeras += int(np.sum(ativo_c1 & ativo_c2))

        return quimeras

    def selecionar_torneio(
        self, populacao: list[IndividuoHopfield]
    ) -> IndividuoHopfield:
        """Seleciona o indivíduo vencedor de um subconjunto de tamanho torneio_k."""
        k = min(self.config.torneio_k, len(populacao))
        indices = self.rng.choice(len(populacao), size=k, replace=False)
        candidatos = [populacao[i] for i in indices]
        return max(candidatos, key=lambda ind: ind.fitness)

    def cruzar(
        self, pai1: IndividuoHopfield, pai2: IndividuoHopfield
    ) -> IndividuoHopfield:
        """Executa crossover uniforme entre dois pais com probabilidade p_crossover."""
        if self.rng.random() > self.config.p_crossover:
            return pai1

        nc = pai1.nc if self.rng.random() < 0.5 else pai2.nc
        k_viz = pai1.k_vizinhos if self.rng.random() < 0.5 else pai2.k_vizinhos
        # Recombinação linear para parâmetros contínuos
        alfa = float(self.rng.uniform(0.0, 1.0))
        beta = float(alfa * pai1.beta + (1.0 - alfa) * pai2.beta)
        threshold = float(alfa * pai1.threshold + (1.0 - alfa) * pai2.threshold)
        n_iters = pai1.n_iters if self.rng.random() < 0.5 else pai2.n_iters
        normalize = pai1.normalize if self.rng.random() < 0.5 else pai2.normalize
        estrategia = pai1.estrategia if self.rng.random() < 0.5 else pai2.estrategia

        return IndividuoHopfield(
            nc=int(nc),
            k_vizinhos=int(k_viz),
            beta=beta,
            threshold=threshold,
            n_iters=int(n_iters),
            normalize=normalize,
            estrategia=estrategia,
        )

    def mutar(self, ind: IndividuoHopfield) -> IndividuoHopfield:
        """Aplica mutação Gaussiana e discreta nos genes respeitando os limites físicos."""
        nc = ind.nc
        k_viz = ind.k_vizinhos
        beta = ind.beta
        threshold = ind.threshold
        n_iters = ind.n_iters
        normalize = ind.normalize
        estrategia = ind.estrategia

        p_mut = self.config.p_mutacao

        if self.rng.random() < p_mut:
            nc = int(np.clip(nc + self.rng.choice([-5, -2, 2, 5]), 5, 45))

        if self.rng.random() < p_mut:
            k_viz = int(np.clip(k_viz + self.rng.choice([-1, 1]), 1, 10))

        if self.rng.random() < p_mut:
            beta = float(
                np.clip(
                    beta + self.rng.normal(0.0, 0.3),
                    self.config.beta_min,
                    self.config.beta_max,
                )
            )

        if self.rng.random() < p_mut:
            threshold = float(
                np.clip(threshold + self.rng.normal(0.0, 0.05), -0.3, 0.5)
            )

        if self.rng.random() < p_mut:
            n_iters = int(np.clip(n_iters + self.rng.choice([-1, 1]), 1, 3))

        if self.rng.random() < p_mut:
            normalize = not normalize

        if self.rng.random() < p_mut:
            estrategia = (
                "kmeans_dinamico" if estrategia == "kmeans_fixo" else "kmeans_fixo"
            )

        return IndividuoHopfield(
            nc=nc,
            k_vizinhos=k_viz,
            beta=beta,
            threshold=threshold,
            n_iters=n_iters,
            normalize=normalize,
            estrategia=estrategia,
        )

    def evoluir(
        self,
        callback_geracao: Callable[[int, IndividuoHopfield, float], None] | None = None,
    ) -> dict[str, Any]:
        """Executa o ciclo evolutivo completo do Algoritmo Genético.

        Parameters
        ----------
        callback_geracao : Callable[[int, IndividuoHopfield, float], None] | None, optional
            Função opcional invocada ao fim de cada geração (geracao, campeao_atual, media_fit).

        Returns
        -------
        dict[str, Any]
            Dicionário com o indivíduo campeão, histórico de evolução e telemetria de cache.
        """
        populacao = self.gerar_populacao_inicial()
        # Avalia geração inicial
        populacao = [self.avaliar_individuo(ind) for ind in populacao]
        populacao.sort(key=lambda ind: ind.fitness, reverse=True)

        self.historico_geracoes = []

        for gen in range(self.config.n_geracoes):
            campeao_geracao = populacao[0]
            media_fitness = float(np.mean([ind.fitness for ind in populacao]))

            registro_gen = {
                "geracao": gen + 1,
                "melhor_fitness": float(campeao_geracao.fitness),
                "media_fitness": media_fitness,
                "melhor_f1_val": float(campeao_geracao.f1_val),
                "melhor_f1_ruido": float(campeao_geracao.f1_ruido),
                "melhor_gap": float(campeao_geracao.gap_generalizacao),
                "melhores_parametros": campeao_geracao.para_dicionario(),
            }
            self.historico_geracoes.append(registro_gen)

            if callback_geracao is not None:
                callback_geracao(gen + 1, campeao_geracao, media_fitness)

            # Nova geração com elitismo
            nova_populacao: list[IndividuoHopfield] = list(
                populacao[: self.config.elitismo]
            )

            while len(nova_populacao) < self.config.tam_populacao:
                pai1 = self.selecionar_torneio(populacao)
                pai2 = self.selecionar_torneio(populacao)
                filho = self.cruzar(pai1, pai2)
                filho_mutado = self.mutar(filho)
                filho_avaliado = self.avaliar_individuo(filho_mutado)
                nova_populacao.append(filho_avaliado)

            nova_populacao.sort(key=lambda ind: ind.fitness, reverse=True)
            populacao = nova_populacao

        campeao_final = populacao[0]
        return {
            "campeao": campeao_final,
            "historico_geracoes": self.historico_geracoes,
            "estatisticas_cache": dict(self.estatisticas_cache),
        }

    def exportar_historico_json(self, path_json: str | os.PathLike[str]) -> None:
        """Persiste o relatório estruturado da evolução em formato JSON.

        Parameters
        ----------
        path_json : str | PathLike
            Caminho de saída para gravação do arquivo JSON.
        """
        dados = {
            "historico_geracoes": self.historico_geracoes,
            "estatisticas_cache": self.estatisticas_cache,
            "configuracao": {
                "tam_populacao": self.config.tam_populacao,
                "n_geracoes": self.config.n_geracoes,
                "p_crossover": self.config.p_crossover,
                "p_mutacao": self.config.p_mutacao,
                "elitismo": self.config.elitismo,
                "seed": self.config.seed,
                "beta_min": self.config.beta_min,
                "beta_max": self.config.beta_max,
                "metrica_f1": self.config.metrica_f1,
                "escalar_por_raiz_d": self.config.escalar_por_raiz_d,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(path_json)), exist_ok=True)
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(dados, f, indent=2, ensure_ascii=False)
