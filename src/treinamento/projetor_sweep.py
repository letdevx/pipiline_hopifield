"""Módulo de Projeção Espectral Dimensional SWeeP e rSWeeP.

Projeta vetores de expressão genômica em subespaços ortogonais compactos
utilizando estritamente a biblioteca oficial rSWeeP da UFPR via Rscript.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Sequence
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
import scipy.io as sio
import scipy.sparse as sp
from numpy.typing import NDArray

try:
    from src.config import PATH_ORTHBASE_RDS  # pyright: ignore[reportMissingImports]
except ImportError:
    try:
        from config import PATH_ORTHBASE_RDS  # type: ignore[import-not-found]
    except ImportError:
        from ..config import PATH_ORTHBASE_RDS  # type: ignore[import-not-found]

from .hopfield_utils import princomp_

PathType = str | os.PathLike[str]


class ProjetorSWeP:
    """Consome representações no espaço SWeeP e gerencia base ortogonal rSWeeP.

    Permite carregar bases pré-existentes, definir projeções pré-computadas
    e calcular PCA sem centralização sobre as coordenadas projetadas.

    Parameters
    ----------
    n_features : int
        Número de genes de entrada (linhas da base R).
    n_componentes : int, default=600
        Dimensão do espaço latente SWeeP (colunas da base R).
    seed : int, default=42
        Semente pseudoaleatória para geração da base.

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

    def __init__(
        self, n_features: int, n_componentes: int = 600, seed: int = 42
    ) -> None:
        self.n_features: int = int(n_features)
        self.n_componentes: int = int(n_componentes)
        self.seed: int = int(seed)
        self.R: NDArray[np.float32] | None = None
        self.Wswp: NDArray[np.float32] | None = None
        self.componentes: NDArray[np.float32] | None = None
        self.Wpc: NDArray[np.float32] | None = None

    def gerar_base(self) -> ProjetorSWeP:
        """Gera base ortonormal sintética em memória estritamente para testes de unidade locais.

        Aviso: Para treinamento e pipelines científicos de produção, o uso do ProjetorSWeePR
        com o pacote oficial rSWeeP é obrigatório por diretriz mandatória de projeto.

        Returns
        -------
        ProjetorSWeP
            A própria instância com o atributo R preenchido.
        """
        print(
            f"[ProjetorSWeP] Base em memória para testes "
            f"({self.n_features} × {self.n_componentes}, seed={self.seed})..."
        )
        rng = np.random.default_rng(self.seed)
        Q, _ = np.linalg.qr(rng.standard_normal((self.n_features, self.n_componentes)))
        self.R = Q.astype(np.float32)
        erro: float = float(
            np.abs(self.R.T @ self.R - np.eye(self.n_componentes)).max()
        )
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
        assert self.R is not None
        print(f"[ProjetorSWeP] Base carregada: {self.R.shape}")
        return self

    def projetar(self, W: NDArray[Any] | sp.spmatrix) -> ProjetorSWeP:
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
            raise RuntimeError(
                "[ProjetorSWeP] Execute .gerar_base() ou .carregar_base() primeiro."
            )
        print(f"[ProjetorSWeP] Projetando W {W.shape} → SWeeP...")
        if sp.issparse(W):
            self.Wswp = np.asarray(sp.csr_matrix(W).dot(self.R), dtype=np.float32)
        else:
            self.Wswp = np.asarray(W, dtype=np.float32) @ self.R
        assert self.Wswp is not None
        print(f"[ProjetorSWeP] Wswp shape: {self.Wswp.shape}")
        return self

    def usar_sweep_precomputado(
        self, Wswp: NDArray[Any] | Sequence[Sequence[float]]
    ) -> ProjetorSWeP:
        """Configura projeções SWeeP já calculadas externamente via rSWeeP em R.

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
            raise RuntimeError(
                "[ProjetorSWeP] Execute .projetar() ou .usar_sweep_precomputado() primeiro."
            )
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
    """Executa a projeção rSWeeP exclusivamente via script R externo (subprocess).

    Garante a conformidade estrita com a regra mandatória da pesquisa, executando
    o algoritmo canônico orthBase() + SWeeP() da biblioteca R oficial rSWeeP (AIBIALab/UFPR).
    Todos os fallbacks sintéticos foram sumariamente eliminados.

    Parameters
    ----------
    path_matriz : str | os.PathLike[str]
        Caminho do arquivo de expressão de entrada (.mtx, .h5ad ou .npy).
    path_saida : str | os.PathLike[str]
        Caminho do arquivo de texto tabulado (.txt) de destino.
    n_componentes : int, default=600
        Dimensão do espaço latente.
    seed : int, default=42
        Semente pseudoaleatória.
    path_orthbase : str | os.PathLike[str] | None, default=None
        Caminho do arquivo .rds para congelamento e reutilização da base ortonormal.
        Se None, assume automaticamente a base canônica configurada em `src.config.PATH_ORTHBASE_RDS`.
    forcar_recriacao : bool, default=False
        Se True, ignora a base existente em disco e força a geração e sobrescrita
        de uma nova base ortonormal canônica via `orthBase()`.

    Attributes
    ----------
    path_matriz : str
        Caminho de entrada.
    path_saida : str
        Caminho de saída (.txt tabulado).
    n_componentes : int
        Dimensão latente.
    seed : int
        Semente aleatória.
    path_orthbase : str
        Caminho da base congelada em RDS.
    forcar_recriacao : bool
        Indica se a base será regenerada e sobrescrita.
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
        path_orthbase: PathType | None = None,
        forcar_recriacao: bool = False,
    ) -> None:
        self.path_matriz: str = str(path_matriz)
        self.path_saida: str = str(path_saida)
        self.n_componentes: int = int(n_componentes)
        self.seed: int = int(seed)
        self.path_orthbase: str = (
            str(path_orthbase) if path_orthbase is not None else str(PATH_ORTHBASE_RDS)
        )
        self.forcar_recriacao: bool = bool(forcar_recriacao)
        self.Wswp: NDArray[np.float32] | None = None

    @staticmethod
    def verificar_e_instalar_dependencias_r() -> None:
        """Verifica e instala o pacote oficial rSWeeP (Bioconductor) e Matrix no ambiente R.

        Configurado com flags não-interativas (ask=FALSE, update=FALSE) para compatibilidade
        estrita com Google Colab e ambientes em nuvem.
        """
        import shutil

        if shutil.which("Rscript") is None:
            raise RuntimeError(
                "[ProjetorSWeePR] Rscript não foi encontrado no PATH do sistema. "
                "Certifique-se de que o R está instalado e disponível."
            )

        r_code = """
options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager")
}
if (!requireNamespace("rSWeeP", quietly = TRUE)) {
    message("[ProjetorSWeePR] Instalando rSWeeP via BiocManager (não-interativo)...")
    BiocManager::install("rSWeeP", update = FALSE, ask = FALSE)
}
if (!requireNamespace("Matrix", quietly = TRUE)) {
    install.packages("Matrix")
}
library(rSWeeP)
cat("[ProjetorSWeePR] Ambiente R pronto. rSWeeP versao:", as.character(packageVersion("rSWeeP")), "\\n")
"""
        print("[ProjetorSWeePR] Verificando dependências R no ambiente...")
        res = subprocess.run(["Rscript", "-e", r_code], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"[ProjetorSWeePR] Falha ao instalar rSWeeP no ambiente R:\n{res.stderr}"
            )
        print(res.stdout)

    def projetar(self) -> ProjetorSWeePR:
        """Executa a projeção rSWeeP via Rscript e carrega o resultado.

        Garante a conversão para Matrix Market (.mtx) se a entrada for h5ad/npy,
        executa o script R oficial sem fallbacks e dispara RuntimeError se houver falha.

        Returns
        -------
        ProjetorSWeePR
            A própria instância com os dados projetados carregados em Wswp.

        Raises
        ------
        RuntimeError
            Se o ambiente R, o script ou a projeção falharem.
        """
        if os.path.exists(self.path_saida) and not self.forcar_recriacao:
            print(
                f"[ProjetorSWeePR] Arquivo de projeção já existente, carregando: {self.path_saida}"
            )
            self._carregar()
            return self

        # 1. Preparação da Matriz de Entrada no formato Matrix Market (.mtx) OOM-Safe
        path_mtx: str

        if os.path.isdir(self.path_matriz):
            candidato_mtx = os.path.join(self.path_matriz, "matrix.mtx")
            candidato_gz = os.path.join(self.path_matriz, "matrix.mtx.gz")
            if os.path.exists(candidato_mtx):
                path_mtx = candidato_mtx
            elif os.path.exists(candidato_gz):
                path_mtx = candidato_gz
            else:
                raise FileNotFoundError(
                    f"[ProjetorSWeePR] Nenhum arquivo matrix.mtx ou matrix.mtx.gz "
                    f"encontrado no diretório: {self.path_matriz}"
                )
        elif self.path_matriz.endswith(".mtx") or self.path_matriz.endswith(".mtx.gz"):
            path_mtx = self.path_matriz
        elif self.path_matriz.endswith(".h5ad"):
            path_mtx = self.path_matriz.rsplit(".h5ad", 1)[0] + ".mtx"
            if not os.path.exists(path_mtx):
                print(
                    f"[ProjetorSWeePR] Exportando camada esparsa do h5ad para Matrix Market: {path_mtx}..."
                )
                adata_tmp: ad.AnnData = ad.read_h5ad(self.path_matriz, backed="r")
                x_raw: Any = adata_tmp.X
                x_mat: Any = x_raw.to_memory() if hasattr(x_raw, "to_memory") else x_raw

                if not sp.issparse(x_mat):
                    x_mat = sp.csr_matrix(x_mat)
                os.makedirs(os.path.dirname(os.path.abspath(path_mtx)), exist_ok=True)
                sio.mmwrite(path_mtx, x_mat)
                adata_tmp.file.close()
                del adata_tmp, x_raw, x_mat
            else:
                print(
                    f"[ProjetorSWeePR] Arquivo .mtx correspondente já disponível: {path_mtx}"
                )
        elif self.path_matriz.endswith(".npy"):
            path_mtx = self.path_matriz.rsplit(".npy", 1)[0] + ".mtx"
            if not os.path.exists(path_mtx):
                print(
                    f"[ProjetorSWeePR] Convertendo array NumPy para Matrix Market esparso: {path_mtx}..."
                )
                arr = np.load(self.path_matriz, mmap_mode="r")
                sp_arr = sp.csr_matrix(arr)
                os.makedirs(os.path.dirname(os.path.abspath(path_mtx)), exist_ok=True)
                sio.mmwrite(path_mtx, sp_arr)
                del arr, sp_arr
            else:
                print(
                    f"[ProjetorSWeePR] Arquivo .mtx correspondente já disponível: {path_mtx}"
                )
        else:
            path_mtx = self.path_matriz

        # 2. Configuração e Disparo do Subprocesso R Canônico com Auditoria
        print(
            "[ProjetorSWeePR] ========================================================="
        )
        print("[ProjetorSWeePR] [AUDITORIA] Executando projeção oficial rSWeeP em R...")
        print(f"  script R         : {self._R_SCRIPT}")
        print(f"  entrada          : {path_mtx}")
        print(f"  saída            : {self.path_saida}")
        print(f"  dim_proj         : {self.n_componentes}, seed: {self.seed}")
        print(f"  base RDS padrão  : {self.path_orthbase}")
        print(f"  forçar recriação : {self.forcar_recriacao}")
        print(
            "[ProjetorSWeePR] ========================================================="
        )

        cmd: list[str] = [
            "Rscript",
            self._R_SCRIPT,
            path_mtx,
            self.path_saida,
            str(self.n_componentes),
            str(self.seed),
            self.path_orthbase,
            str(self.forcar_recriacao).upper(),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            msg_erro: str = (
                f"[ProjetorSWeePR] FALHA CRÍTICA no subprocesso R (código {result.returncode}).\n"
                f"Por diretriz mandatória do projeto (ADR 019), nenhum fallback sintético é tolerado.\n"
                f"--- ERRO DO R (stderr) ---\n{result.stderr}\n"
                f"--- SAÍDA DO R (stdout) ---\n{result.stdout}"
            )
            if "rSWeeP" in result.stderr and (
                "não encontrado" in result.stderr or "not found" in result.stderr
            ):
                msg_erro += (
                    "\n\n[DICA DE RESOLUÇÃO NO COLAB/LINUX]\n"
                    "O pacote rSWeeP não está instalado no seu ambiente R.\n"
                    "Para instalá-lo com segurança e sem travamentos, execute no notebook:\n"
                    "  from treinamento.projetor_sweep import ProjetorSWeePR\n"
                    "  ProjetorSWeePR.verificar_e_instalar_dependencias_r()\n"
                    "Ou em uma célula bash:\n"
                    '  !Rscript -e \'options(repos = c(CRAN = "https://cloud.r-project.org")); if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager"); BiocManager::install("rSWeeP", update = FALSE, ask = FALSE)\'\n'
                )
            raise RuntimeError(msg_erro)

        print(result.stdout)

        # 3. Carregamento do Resultado Gerado em R
        self._carregar()
        return self

    def _carregar(self) -> None:
        """Carrega a matriz projetada gravada pelo script R em formato tabulado."""
        if not os.path.exists(self.path_saida):
            raise FileNotFoundError(
                f"[ProjetorSWeePR] Arquivo de saída não encontrado: {self.path_saida}"
            )

        # Tenta ler com Polars primeiro por velocidade; se falhar, usa Pandas
        try:
            df = pl.read_csv(self.path_saida, separator="\t", has_header=False)
            self.Wswp = df.to_numpy().astype(np.float32)
        except Exception:
            self.Wswp = pd.read_csv(self.path_saida, sep=r"\s+", header=None).to_numpy(
                dtype=np.float32
            )

        assert self.Wswp is not None
        print(
            f"[ProjetorSWeePR] Wswp carregado com sucesso absoluto: {self.Wswp.shape}"
        )

    def __repr__(self) -> str:
        """Representação textual do projetor SWeeP em R."""
        wswp = self.Wswp.shape if self.Wswp is not None else "não gerada"
        return (
            f"ProjetorSWeePR(\n"
            f"  path_matriz      = {self.path_matriz}\n"
            f"  path_saida       = {self.path_saida}\n"
            f"  n_componentes    = {self.n_componentes}\n"
            f"  seed             = {self.seed}\n"
            f"  base_rds         = {self.path_orthbase}\n"
            f"  forcar_recriacao = {self.forcar_recriacao}\n"
            f"  Wswp             = {wswp}\n"
            f")"
        )


# Alias para tolerância a grafias com 1 ou 2 'e's
ProjetorSWePR = ProjetorSWeePR
