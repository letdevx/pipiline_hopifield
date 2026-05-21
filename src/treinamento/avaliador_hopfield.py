import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score, classification_report

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

    def __init__(self, padroes, classes, nc=10, nomes_classes=None):
        self.padroes = np.asarray(padroes, dtype=np.float64)
        self.classes = list(classes)
        self.nc = nc
        self.nomes_classes = nomes_classes
        self.acuracia = None
        self.f1_macro = None
        self.f1_weighted = None
        self.matriz_conf = None
        self.y_true = None
        self.y_pred = None

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
        pred = classes_arr[idx_proto // self.nc]

        mask = labels != 0
        self.y_true = labels[mask]
        self.y_pred = pred[mask]

        self.acuracia = (self.y_true == self.y_pred).mean()
        self.f1_macro = f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        self.f1_weighted = f1_score(self.y_true, self.y_pred, average="weighted", zero_division=0)
        self.matriz_conf = confusion_matrix(self.y_true, self.y_pred, labels=self.classes)

        print(f"[AvaliadorHopfield] Acurácia: {self.acuracia * 100:.2f}% "
              f"(n={mask.sum():,})")
        print(f"[AvaliadorHopfield] F1 macro={self.f1_macro:.4f}, "
              f"F1 ponderado={self.f1_weighted:.4f}")
        print(classification_report(self.y_true, self.y_pred,
                                    labels=self.classes,
                                    target_names=[str(c) for c in self.classes],
                                    zero_division=0))
        return self

    def plotar(self, titulo="Matriz de Confusão — rede35"):
        """Plota a matriz de confusão como heatmap.

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        if self.matriz_conf is None:
            raise RuntimeError("[AvaliadorHopfield] Execute .avaliar() antes de .plotar().")

        rotulos = self.nomes_classes if self.nomes_classes else [str(c) for c in self.classes]
        fig, ax = plt.subplots(figsize=(max(6, len(self.classes)), max(5, len(self.classes))))
        sns.heatmap(self.matriz_conf, annot=True, fmt="d", cmap="Blues",
                    xticklabels=rotulos, yticklabels=rotulos, ax=ax)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_title(titulo)
        plt.tight_layout()
        plt.show()
        return self

    def __repr__(self):
        acc = f"{self.acuracia * 100:.2f}%" if self.acuracia is not None else "não avaliado"
        f1m = f"{self.f1_macro:.4f}" if self.f1_macro is not None else "—"
        f1w = f"{self.f1_weighted:.4f}" if self.f1_weighted is not None else "—"
        return (
            f"AvaliadorHopfield(\n"
            f"  padroes     = {self.padroes.shape}\n"
            f"  classes     = {self.classes}\n"
            f"  nc          = {self.nc}\n"
            f"  acuracia    = {acc}\n"
            f"  f1_macro    = {f1m}\n"
            f"  f1_weighted = {f1w}\n"
            f")"
        )
