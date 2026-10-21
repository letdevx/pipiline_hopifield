"""Módulo de Treinamento, Redes Hopfield Modernas e Avaliação de Imputação."""

from .algoritmo_genetico import (
    ConfiguracaoAG,
    IndividuoHopfield,
    OtimizadorGeneticoHopfield,
)
from .avaliador_hopfield import AvaliadorHopfield
from .carregador_dados_fujita import (
    CarregadorDados,
    CarregadorDadosFujita,
    carregar_labels,
    remapear_labels_canonicos,
)
from .diagnostico_overfitting import (
    AuditorOverfittingCrossDataset,
    AuditorOverfittingHopfield,
    InjetorPerturbacaoTranscritica,
    ResultadoDiagnosticoCrossDataset,
    ResultadoDiagnosticoOverfitting,
)
from .estrategias_clusterizacao import (
    EstrategiaHDBSCAN,
    EstrategiaKMeansDinamico,
    EstrategiaKMeansFixo,
)
from .exportador_imputacao import ExportadorImputacao
from .extrator_padroes import ExtratorPadroesSubcluster
from .gerador_conjunto_treinamento import GeradorConjuntoTreinamento
from .gerador_relatorio import GeradorRelatorio
from .hopfield import ModernHopfieldNetwork
from .projetor_sweep import ProjetorSWeePR, ProjetorSWeP

try:
    from .hopfield_kerastuner import HopfieldKerasTuner
except ImportError:
    HopfieldKerasTuner = None  # type: ignore[assignment,misc]
from .selecionador_genes_shap import (
    HopfieldClassifierWrapper,
    SelecionadorGenesSHAPHopfield,
)
from .validador_imputacao import ValidadorImputacao

__all__ = [
    "AuditorOverfittingCrossDataset",
    "AuditorOverfittingHopfield",
    "AvaliadorHopfield",
    "CarregadorDados",
    "CarregadorDadosFujita",
    "ConfiguracaoAG",
    "EstrategiaHDBSCAN",
    "EstrategiaKMeansDinamico",
    "EstrategiaKMeansFixo",
    "ExportadorImputacao",
    "ExtratorPadroesSubcluster",
    "GeradorConjuntoTreinamento",
    "GeradorRelatorio",
    "HopfieldClassifierWrapper",
    "HopfieldKerasTuner",
    "IndividuoHopfield",
    "InjetorPerturbacaoTranscritica",
    "ModernHopfieldNetwork",
    "OtimizadorGeneticoHopfield",
    "ProjetorSWeP",
    "ProjetorSWeePR",
    "ResultadoDiagnosticoCrossDataset",
    "ResultadoDiagnosticoOverfitting",
    "SelecionadorGenesSHAPHopfield",
    "ValidadorImputacao",
    "carregar_labels",
    "remapear_labels_canonicos",
]
