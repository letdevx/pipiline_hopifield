import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, f1_score, classification_report,
    precision_recall_fscore_support,
)

from .hopfield_utils import closervects


class AvaliadorHopfield:
    """Avalia os resultados da rede Hopfield com métricas de classificação.

    Compara os padrões recuperados pela rede com os rótulos verdadeiros,
    mapeando cada padrão recuperado para a classe mais próxima em perf35.

    Atributos
    ---------
    padroes       : padrões de protótipos usados na rede (perf35)
    classes       : lista ordenada de classes (mesma ordem usada na extração)
    nc            : número de subclusters por classe
    nomes_classes : nomes legíveis das classes (opcional)
    acuracia      : acurácia calculada após .avaliar()
    f1_macro      : F1 macro após .avaliar()
    f1_weighted   : F1 ponderado após .avaliar()
    matriz_conf   : matriz de confusão após .avaliar()
    y_true        : rótulos verdadeiros usados na avaliação
    y_pred        : rótulos preditos pela rede
    """

    def __init__(self, padroes, classes, nc=10, nomes_classes=None, meta=None):
        self.padroes = np.asarray(padroes, dtype=np.float64)
        self.classes = list(classes)
        self.nc = nc
        self.nomes_classes = nomes_classes
        # meta: lista de (classe, idx_célula) gerada pelo ExtratorPadroesSubcluster
        # quando fornecida, substitui o mapeamento por idx_proto // nc
        self._pattern_classes = np.array([m[0] for m in meta]) if meta is not None else None
        self.acuracia          = None
        self.f1_macro          = None
        self.f1_weighted       = None
        self.taxa_reconstrucao = None
        self.semelhanca_media  = None
        self.matriz_conf       = None
        self.y_true            = None
        self.y_pred            = None

    def avaliar(self, Wrecuperado, labels):
        """Avalia a recuperação comparando com os rótulos verdadeiros.

        Para cada padrão recuperado, encontra o protótipo mais próximo em
        perf35 e mapeia ao índice de classe correspondente.

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        classes_arr = np.array(self.classes)
        labels = np.asarray(labels, dtype=int)

        print("[AvaliadorHopfield] Mapeando padrões recuperados para classes...")
        perf_f = self.padroes.astype(np.float64)
        W_f = np.asarray(Wrecuperado, dtype=np.float64)

        a2 = (W_f ** 2).sum(axis=1, keepdims=True)
        b2 = (perf_f ** 2).sum(axis=1, keepdims=True).T
        idx_proto = (a2 + b2 - 2 * (W_f @ perf_f.T)).argmin(axis=1)
        if self._pattern_classes is not None:
            pred = self._pattern_classes[idx_proto]
        else:
            pred = classes_arr[idx_proto // self.nc]

        # Taxa de reconstrução: fração com distância de Hamming = 0 ao protótipo mais próximo
        prototipos = perf_f[idx_proto]
        hamming    = (W_f != prototipos).mean(axis=1)

        mask = np.isin(labels, self.classes)
        self.y_true = labels[mask]
        self.y_pred = pred[mask]

        self.acuracia          = (self.y_true == self.y_pred).mean()
        self.f1_macro          = f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        self.f1_weighted       = f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
        self.taxa_reconstrucao = (hamming[mask] == 0).mean()
        self.semelhanca_media  = (1 - hamming[mask]).mean()
        self.matriz_conf       = confusion_matrix(self.y_true, self.y_pred, labels=self.classes)

        print(f"[AvaliadorHopfield] Acurácia: {self.acuracia * 100:.2f}% "
              f"(n={mask.sum():,})")
        print(f"[AvaliadorHopfield] F1 macro={self.f1_macro:.4f}, "
              f"F1 ponderado={self.f1_weighted:.4f}")
        print(f"[AvaliadorHopfield] Taxa de reconstrução exata : {self.taxa_reconstrucao * 100:.2f}%")
        print(f"[AvaliadorHopfield] Semelhança média ao protótipo: {self.semelhanca_media:.4f}")
        print(classification_report(self.y_true, self.y_pred,
                                    labels=self.classes,
                                    target_names=[str(c) for c in self.classes],
                                    zero_division=0))
        return self

    def plotar(self, titulo="Matriz de Confusão — rede35", normalizado=False, ax=None):
        """Plota a matriz de confusão como heatmap.

        Parâmetros
        ----------
        titulo      : título do gráfico
        normalizado : se True, normaliza por linha (taxa por classe real)
        ax          : eixo matplotlib externo; se None, cria figura própria

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        if self.matriz_conf is None:
            raise RuntimeError("[AvaliadorHopfield] Execute .avaliar() antes de .plotar().")

        rotulos = self.nomes_classes if self.nomes_classes else [str(c) for c in self.classes]

        mat = self.matriz_conf.astype(float)
        if normalizado:
            totais = mat.sum(axis=1, keepdims=True)
            totais[totais == 0] = 1
            mat = mat / totais
            fmt = ".1%"
        else:
            fmt = "d"
            mat = mat.astype(int)

        criar_figura = ax is None
        if criar_figura:
            fig, ax = plt.subplots(figsize=(max(6, len(self.classes)), max(5, len(self.classes))))

        sns.heatmap(mat, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=rotulos, yticklabels=rotulos, ax=ax)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(titulo)

        if criar_figura:
            plt.tight_layout()
            plt.show()
        return self

    def metricas_resumo(self, nome=""):
        """Retorna dict com métricas globais da avaliação.

        Útil para comparar múltiplos datasets em uma tabela consolidada.
        """
        if self.acuracia is None:
            raise RuntimeError("[AvaliadorHopfield] Execute .avaliar() antes de .metricas_resumo().")
        return {
            "dataset":           nome,
            "n_celulas":         int(len(self.y_true)),
            "acuracia":          round(float(self.acuracia), 4),
            "f1_macro":          round(float(self.f1_macro), 4),
            "f1_weighted":       round(float(self.f1_weighted), 4),
            "taxa_reconstrucao": round(float(self.taxa_reconstrucao), 4),
            "semelhanca_media":  round(float(self.semelhanca_media), 4),
        }

    def metricas_por_classe(self):
        """Retorna DataFrame com precision, recall e F1 por classe.

        Retorna
        -------
        pd.DataFrame com colunas: classe, n_celulas, precisao, recall, f1
        """
        if self.y_true is None:
            raise RuntimeError("[AvaliadorHopfield] Execute .avaliar() antes de .metricas_por_classe().")
        rotulos = self.nomes_classes if self.nomes_classes else [str(c) for c in self.classes]
        p, r, f, s = precision_recall_fscore_support(
            self.y_true, self.y_pred,
            labels=self.classes,
            zero_division=0,
        )
        return pd.DataFrame({
            "classe":    rotulos,
            "n_celulas": s.astype(int),
            "precisao":  p.round(4),
            "recall":    r.round(4),
            "f1":        f.round(4),
        })

    def __repr__(self):
        acc = f"{self.acuracia * 100:.2f}%"          if self.acuracia          is not None else "não avaliado"
        f1m = f"{self.f1_macro:.4f}"                  if self.f1_macro          is not None else "—"
        f1w = f"{self.f1_weighted:.4f}"               if self.f1_weighted       is not None else "—"
        rec = f"{self.taxa_reconstrucao * 100:.2f}%"  if self.taxa_reconstrucao is not None else "—"
        sim = f"{self.semelhanca_media:.4f}"           if self.semelhanca_media  is not None else "—"
        return (
            f"AvaliadorHopfield(\n"
            f"  padroes            = {self.padroes.shape}\n"
            f"  classes            = {self.classes}\n"
            f"  nc                 = {self.nc}\n"
            f"  acuracia           = {acc}\n"
            f"  f1_macro           = {f1m}\n"
            f"  f1_weighted        = {f1w}\n"
            f"  taxa_reconstrucao  = {rec}\n"
            f"  semelhanca_media   = {sim}\n"
            f")"
        )
