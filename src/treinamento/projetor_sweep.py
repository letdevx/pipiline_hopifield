"""Módulo de Projeção Espectral Dimensional SWeeP e rSWeeP.

Projeta vetores de expressão genômica em subespaços ortogonais compactos
utilizando decomposição QR ou pontes de integração externa via Rscript.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any, Sequence, Union

import anndata as ad
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import scipy.sparse as sp

from .hopfield_utils import princomp_

PathType = Union[str, os.PathLike[str]]


class ProjetorSWeP:
    """Projeta dados binários no espaço SWeeP usando base ortogonal rSWeeP.

    Pode gerar uma base sintética via decomposição QR (quando a base R5k pré-determinada
    não está disponível) ou carregar a base a partir de arquivo. Aplica opcionalmente PCA sem
    centralização sobre as projeções geradas.

    Parameters
    ----------
    n_features : int
        Número de genes de entrada (linhas da base R).
    n_componentes : int, default=600
        Dimensão do espaço latente SWeeP (colunas da base R).
    seed : int, default=42
        Semente pseudoaleatória para geração da base ortonormal.

    Attributes
    ----------
    n_features : int
        Dimensão dos genes.
    n_componentes : int
        Dimensão da projeção.
    seed : int
        Semente aleatória.
    R : NDArray[np.float32] | None
        Matriz de base ortonormal (n_features × n_componentes).
    Wswp : NDArray[np.float32] | None
        Projeções SWeeP das células (células × n_componentes).
    componentes : NDArray[np.float32] | None
        Loadings calculados via PCA (n_componentes × n_componentes).
    Wpc : NDArray[np.float32] | None
        Scores das componentes principais (células × n_componentes).
    """

    def __init__(self, n_features: int, n_componentes: int = 600, seed: int = 42) -> None:
        self.n_features: int = int(n_features)
        self.n_componentes: int = int(n_componentes)
        self.seed: int = int(seed)
        self.R: NDArray[np.float32] | None = None
        self.Wswp: NDArray[np.float32] | None = None
        self.componentes: NDArray[np.float32] | None = None
        self.Wpc: NDArray[np.float32] | None = None

    def gerar_base(self) -> ProjetorSWeP:
        """Gera base ortonormal sintética via decomposição QR.

        Returns
        -------
        ProjetorSWeP
            A própria instância com o atributo R preenchido.
        """
        print(f"[ProjetorSWeP] Gerando base sintética QR "
              f"({self.n_features} × {self.n_componentes}, seed={self.seed})...")
        rng = np.random.default_rng(self.seed)
        Q, _ = np.linalg.qr(rng.standard_normal((self.n_features, self.n_componentes)))
        self.R = Q.astype(np.float32)
        erro: float = float(np.abs(self.R.T @ self.R - np.eye(self.n_componentes)).max())
        print(f"[ProjetorSWeP] Base gerada. Erro de ortogonalidade: {erro:.2e}")
        return self

    def carregar_base(self, path: PathType) -> ProjetorSWeP:
        """Carrega base R pré-existente de arquivo .txt ou .npy.

        Parameters
        ----------
        path : str | os.PathLike[str]
            Caminho do arquivo contendo a matriz de projeção.

        Returns
        -------
        ProjetorSWeP
            A própria instância com a base carregada.
        """
        path_str: str = str(path)
        print(f"[ProjetorSWeP] Carregando base: {path_str}")
        if path_str.endswith(".npy"):
            self.R = np.load(path_str).astype(np.float32)
        else:
            self.R = np.loadtxt(path_str, dtype=np.float32)
        print(f"[ProjetorSWeP] Base carregada: {self.R.shape}")
        return self

    def projetar(self, W: Union[NDArray[Any], sp.spmatrix]) -> ProjetorSWeP:
        """Projeta matriz W no espaço SWeeP: Wswp = W @ R.

        Parameters
        ----------
        W : NDArray | sp.spmatrix
            Matriz de expressão binarizada (n_células × n_features).

        Returns
        -------
        ProjetorSWeP
            A própria instância com `Wswp` preenchido.
        """
        if self.R is None:
            raise RuntimeError("[ProjetorSWeP] Execute .gerar_base() ou .carregar_base() primeiro.")
        print(f"[ProjetorSWeP] Projetando W {W.shape} → SWeeP...")
        if sp.issparse(W):
            self.Wswp = sp.csr_matrix(W).dot(self.R).astype(np.float32, copy=False)
        else:
            self.Wswp = (np.asarray(W, dtype=np.float32) @ self.R).astype(np.float32)
        print(f"[ProjetorSWeP] Wswp shape: {self.Wswp.shape}")
        return self

    def usar_sweep_precomputado(self, Wswp: Union[NDArray[Any], Sequence[Sequence[float]]]) -> ProjetorSWeP:
        """Configura projeções SWeeP já calculadas externamente.

        Parameters
        ----------
        Wswp : NDArray | Sequence
            Matriz de coordenadas latentes.

        Returns
        -------
        ProjetorSWeP
            A própria instância.
        """
        self.Wswp = np.asarray(Wswp, dtype=np.float32)
        print(f"[ProjetorSWeP] SWeeP pré-computado definido: {self.Wswp.shape}")
        return self

    def aplicar_pca(self) -> ProjetorSWeP:
        """Aplica PCA sem centralização sobre Wswp (equivalente a princomp_ do MATLAB).

        Returns
        -------
        ProjetorSWeP
            A própria instância com `componentes` e `Wpc` calculados.
        """
        if self.Wswp is None:
            raise RuntimeError("[ProjetorSWeP] Execute .projetar() ou .usar_sweep_precomputado() primeiro.")
        print("[ProjetorSWeP] Aplicando PCA sem centralização...")
        self.componentes = princomp_(self.Wswp)
        self.Wpc = self.Wswp @ self.componentes
        print(f"[ProjetorSWeP] Wpc shape: {self.Wpc.shape}")
        return self

    def __repr__(self) -> str:
        """Representação textual do projetor SWeeP."""
        r_shape = self.R.shape if self.R is not None else "não gerada"
        wswp = self.Wswp.shape if self.Wswp is not None else "não gerada"
        wpc = self.Wpc.shape if self.Wpc is not None else "não gerada"
        return (
            f"ProjetorSWeP(\n"
            f"  n_features    = {self.n_features}\n"
            f"  n_componentes = {self.n_componentes}\n"
            f"  seed          = {self.seed}\n"
            f"  R             = {r_shape}\n"
            f"  Wswp          = {wswp}\n"
            f"  Wpc           = {wpc}\n"
            f")"
        )


class ProjetorSWeePR:
    """Executa a projeção rSWeeP via script R externo (subprocess) ou fallback NumPy.

    Lê a matriz binária de entrada, executa o script R `projetar_sweep.R`
    e salva o resultado em formato CSV.

    Parameters
    ----------
    path_matriz : str | os.PathLike[str]
        Caminho do arquivo de expressão de entrada.
    path_saida : str | os.PathLike[str]
        Caminho do CSV de destino.
    n_componentes : int, default=600
        Dimensão do espaço latente.
    seed : int, default=42
        Semente pseudoaleatória.

    Attributes
    ----------
    path_matriz : str
        Caminho de entrada.
    path_saida : str
        Caminho de saída.
    n_componentes : int
        Dimensão.
    seed : int
        Semente.
    Wswp : NDArray[np.float32] | None
        Matriz de projeção latente calculada.
    """

    _R_SCRIPT: str = os.path.join(os.path.dirname(__file__), "projetar_sweep.R")

    def __init__(
        self,
        path_matriz: PathType,
        path_saida: PathType,
        n_componentes: int = 600,
        seed: int = 42,
    ) -> None:
        self.path_matriz: str = str(path_matriz)
        self.path_saida: str = str(path_saida)
        self.n_componentes: int = int(n_componentes)
        self.seed: int = int(seed)
        self.Wswp: NDArray[np.float32] | None = None

    def projetar(self) -> ProjetorSWeePR:
        """Executa a projeção rSWeeP via Rscript e carrega o resultado.

        Pula a execução se o arquivo de saída já existir.

        Returns
        -------
        ProjetorSWeePR
            A própria instância com os dados projetados.
        """
        if os.path.exists(self.path_saida):
            print(f"[ProjetorSWeePR] Arquivo já existe, carregando: {self.path_saida}")
            self._carregar()
            return self

        if self.path_matriz.endswith(".npy") or self.path_matriz.endswith(".h5ad"):
            print(f"[ProjetorSWeePR] Matriz binária (.npy/.h5ad) detectada: {self.path_matriz}")
            print("[ProjetorSWeePR] Executando rSWeeP via motor nativo Python/NumPy QR (otimização OOM sem parser de texto)...")
            self._fallback_python()
            return self

        print("[ProjetorSWeePR] Executando rSWeeP via R...")
        print(f"  entrada : {self.path_matriz}")
        print(f"  saída   : {self.path_saida}")
        print(f"  dim_proj: {self.n_componentes}, seed: {self.seed}")

        cmd: list[str] = [
            "Rscript",
            self._R_SCRIPT,
            self.path_matriz,
            self.path_saida,
            str(self.n_componentes),
            str(self.seed),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("[ProjetorSWeePR] R falhou — usando fallback Python (QR).")
            print(f"[ProjetorSWeePR] Detalhe R:\n{result.stderr}")
            self._fallback_python()
            return self

        print(result.stdout)
        self._carregar()
        return self

    def _fallback_python(self) -> None:
        """Fallback: projeção via base ortogonal QR gerada em Python (100% OOM-Safe)."""
        print(f"[ProjetorSWeePR] Carregando matriz no motor Python/NumPy: {self.path_matriz}")

        W: Union[NDArray[Any], sp.spmatrix]
        n_features: int
        if self.path_matriz.endswith(".npy"):
            W = np.load(self.path_matriz, mmap_mode="r")
            n_features = int(W.shape[1])
        elif self.path_matriz.endswith(".h5ad"):
            adata_tmp: ad.AnnData = ad.read_h5ad(self.path_matriz)
            W = adata_tmp.X  # Mantém formato esparso CSR se disponível
            n_features = int(adata_tmp.n_vars)
            del adata_tmp
        else:
            W = pd.read_csv(self.path_matriz, index_col=0).to_numpy(dtype=np.float32)
            n_features = int(W.shape[1])

        print(f"[ProjetorSWeePR] Matriz {W.shape} → projetando para {self.n_componentes} componentes (QR)...")
        rng = np.random.default_rng(self.seed)
        Q, _ = np.linalg.qr(rng.standard_normal((n_features, self.n_componentes)))
        R: NDArray[np.float32] = Q.astype(np.float32)

        proj: NDArray[np.float32]
        if sp.issparse(W):
            proj = sp.csr_matrix(W).dot(R).astype(np.float32, copy=False)
        else:
            proj = (np.asarray(W, dtype=np.float32) @ R).astype(np.float32, copy=False)

        os.makedirs(os.path.dirname(os.path.abspath(self.path_saida)), exist_ok=True)
        pd.DataFrame(proj).to_csv(self.path_saida, index=False)
        print(f"[ProjetorSWeePR] Projeção SWeeP salva com sucesso em: {self.path_saida}")
        self._carregar()

    def _carregar(self) -> None:
        self.Wswp = pd.read_csv(self.path_saida).to_numpy(dtype=np.float32)
        assert self.Wswp is not None
        print(f"[ProjetorSWeePR] Wswp carregado: {self.Wswp.shape}")

    def __repr__(self) -> str:
        """Representação textual do projetor SWeeP em R."""
        wswp = self.Wswp.shape if self.Wswp is not None else "não gerada"
        return (
            f"ProjetorSWeePR(\n"
            f"  path_matriz   = {self.path_matriz}\n"
            f"  path_saida    = {self.path_saida}\n"
            f"  n_componentes = {self.n_componentes}\n"
            f"  seed          = {self.seed}\n"
            f"  Wswp          = {wswp}\n"
            f")"
        )


# Alias para tolerância a grafias com 1 ou 2 'e's
ProjetorSWePR = ProjetorSWeePR

