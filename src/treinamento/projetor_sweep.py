import os
import subprocess
import numpy as np
import pandas as pd

from .hopfield_utils import princomp_


class ProjetorSWeP:
    """Projeta dados binários no espaço SWeeP usando base ortogonal rSWeeP.

    Pode gerar uma base sintética via QR (quando R5k real não está disponível)
    ou carregar a base a partir de arquivo. Aplica opcionalmente PCA sem
    centralização sobre as projeções geradas.

    Atributos
    ---------
    n_features    : número de genes de entrada (linhas da base R)
    n_componentes : dimensão do espaço SWeeP (colunas da base R)
    seed          : semente para geração sintética da base
    R             : base ortonormal (n_features × n_componentes)
    Wswp          : projeções SWeeP (células × n_componentes)
    componentes   : loadings da PCA (n_componentes × n_componentes)
    Wpc           : scores da PCA (células × n_componentes)
    """

    def __init__(self, n_features, n_componentes=600, seed=42):
        self.n_features = n_features
        self.n_componentes = n_componentes
        self.seed = seed
        self.R = None
        self.Wswp = None
        self.componentes = None
        self.Wpc = None

    def gerar_base(self):
        """Gera base ortonormal sintética via decomposição QR.

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        print(f"[ProjetorSWeP] Gerando base sintética QR "
              f"({self.n_features} × {self.n_componentes}, seed={self.seed})...")
        rng = np.random.default_rng(self.seed)
        Q, _ = np.linalg.qr(rng.standard_normal((self.n_features, self.n_componentes)))
        self.R = Q.astype(np.float32)
        erro = np.abs(self.R.T @ self.R - np.eye(self.n_componentes)).max()
        print(f"[ProjetorSWeP] Base gerada. Erro de ortogonalidade: {erro:.2e}")
        return self

    def carregar_base(self, path):
        """Carrega base R5k de arquivo .txt ou .npy.

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        print(f"[ProjetorSWeP] Carregando base: {path}")
        if path.endswith(".npy"):
            self.R = np.load(path).astype(np.float32)
        else:
            self.R = np.loadtxt(path, dtype=np.float32)
        print(f"[ProjetorSWeP] Base carregada: {self.R.shape}")
        return self

    def projetar(self, W):
        """Projeta W no espaço SWeeP: Wswp = W @ R.

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        if self.R is None:
            raise RuntimeError("[ProjetorSWeP] Execute .gerar_base() ou .carregar_base() primeiro.")
        print(f"[ProjetorSWeP] Projetando W {W.shape} → SWeeP...")
        self.Wswp = np.asarray(W, dtype=np.float32) @ self.R
        print(f"[ProjetorSWeP] Wswp shape: {self.Wswp.shape}")
        return self

    def usar_sweep_precomputado(self, Wswp):
        """Usa projeções SWeeP já calculadas externamente.

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        self.Wswp = np.asarray(Wswp, dtype=np.float32)
        print(f"[ProjetorSWeP] SWeeP pré-computado definido: {self.Wswp.shape}")
        return self

    def aplicar_pca(self):
        """Aplica PCA sem centralização sobre Wswp (equivalente a princomp_ do MATLAB).

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        if self.Wswp is None:
            raise RuntimeError("[ProjetorSWeP] Execute .projetar() ou .usar_sweep_precomputado() primeiro.")
        print("[ProjetorSWeP] Aplicando PCA sem centralização...")
        self.componentes = princomp_(self.Wswp)
        self.Wpc = self.Wswp @ self.componentes
        print(f"[ProjetorSWeP] Wpc shape: {self.Wpc.shape}")
        return self

    def __repr__(self):
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
    """Executa a projeção rSWeeP via script R externo (subprocess).

    Lê a matriz binária de entrada, chama o script R projetar_sweep.R
    e salva o resultado como CSV. Implementa skip-if-exists.

    Atributos
    ---------
    path_matriz   : caminho para o TXT de entrada (células × genes, sem cabeçalho)
    path_saida    : caminho do CSV de saída (células × n_componentes)
    n_componentes : dimensão da projeção SWeeP (padrão: 600)
    seed          : semente aleatória para reprodutibilidade
    Wswp          : matriz de projeções carregada após .projetar()
    """

    _R_SCRIPT = os.path.join(os.path.dirname(__file__), "projetar_sweep.R")

    def __init__(self, path_matriz, path_saida, n_componentes=600, seed=42):
        self.path_matriz = path_matriz
        self.path_saida = path_saida
        self.n_componentes = n_componentes
        self.seed = seed
        self.Wswp = None

    def projetar(self):
        """Executa a projeção rSWeeP via Rscript e carrega o resultado.

        Pula a execução se o arquivo de saída já existir.
        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        if os.path.exists(self.path_saida):
            print(f"[ProjetorSWeePR] Arquivo já existe, carregando: {self.path_saida}")
            self._carregar()
            return self

        print(f"[ProjetorSWeePR] Executando rSWeeP via R...")
        print(f"  entrada : {self.path_matriz}")
        print(f"  saída   : {self.path_saida}")
        print(f"  dim_proj: {self.n_componentes}, seed: {self.seed}")

        cmd = [
            "Rscript", self._R_SCRIPT,
            self.path_matriz,
            self.path_saida,
            str(self.n_componentes),
            str(self.seed),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"[ProjetorSWeePR] Erro no script R:\n{result.stderr}"
            )

        print(result.stdout)
        self._carregar()
        return self

    def _carregar(self):
        self.Wswp = pd.read_csv(self.path_saida).to_numpy(dtype=np.float32)
        print(f"[ProjetorSWeePR] Wswp carregado: {self.Wswp.shape}")

    def __repr__(self):
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
