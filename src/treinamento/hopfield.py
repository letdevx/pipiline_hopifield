import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernHopfieldNetwork(nn.Module):
    """Rede de Hopfield Moderna (Ramsauer et al., 2020).

    Substitui hopf_tr (treino) e hopf_ts (teste) do script MATLAB original.
    Capacidade de armazenamento exponencial em vez de linear; recuperação
    equivalente a um passo de attention: softmax(β·Ξ·ξ)·Ξᵀ.

    Atributos
    ---------
    beta       : temperatura inversa do softmax (maior → winner-takes-all)
    n_iters    : número de iterações da regra de atualização
    binary     : se True, aplica limiar {0,1} na saída
    threshold  : limiar para binarizar saída quando binary=True
    patterns   : tensor com os padrões armazenados
    """

    def __init__(self, beta=8.0, n_iters=1, binary=True, threshold=0.0, normalize=False):
        super().__init__()
        self.beta = beta
        self.n_iters = n_iters
        self.binary = binary
        self.threshold = threshold
        self.normalize = normalize
        self.register_buffer("patterns", torch.empty(0))

    def store(self, patterns):
        """Armazena os padrões na rede.

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        K = torch.as_tensor(np.asarray(patterns), dtype=torch.float32)

        if self.binary:
            K = 2.0 * K - 1.0

        self.patterns = K.to(torch.device("cpu"))
        print(f"[ModernHopfieldNetwork] {self.patterns.shape[0]} padrões armazenados "
              f"({self.patterns.shape[1]} genes, device=cpu)")
        return self

    @torch.no_grad()
    def retrieve(self, queries, batch_size=1024, normalize=None, subspace_mask=None,
                 mask_sentinela_ausentes=None, fill_value=0.5, out_buffer=None):
        """Recupera o padrão mais próximo para cada query.
        
        Parâmetros
        ----------
        queries                 : matriz de entrada (esparsa CSR ou densa numpy)
        batch_size              : tamanho do lote de processamento OOM-Safe
        normalize               : se True, aplica normalização L2 nos vetores
        subspace_mask           : máscara de colunas para cálculo da atenção
        mask_sentinela_ausentes : máscara booleana ou lista de índices de genes ausentes.
                                  Se fornecida, atribui fill_value (0.5) a essas colunas em cada lote.
        fill_value              : valor atribuído aos genes ausentes (padrão 0.5 -> 0.0 no espaço bipolar)
        out_buffer              : buffer pré-alocado opcional para escrita direta
        """
        if self.patterns.numel() == 0:
            raise RuntimeError("[ModernHopfieldNetwork] Execute .store() antes de .retrieve().")

        if normalize is None:
            normalize = getattr(self, "normalize", False)

        import scipy.sparse as sp

        Xi = self.patterns.to(dtype=torch.float32, device='cpu')
        if sp.issparse(queries):
            n_queries, n_features = queries.shape
            is_sparse = True
            queries_np = None
        else:
            queries_np = np.asarray(queries)
            n_queries, n_features = queries_np.shape
            is_sparse = False

        if subspace_mask is not None:
            if isinstance(subspace_mask, np.ndarray):
                subspace_mask = torch.from_numpy(subspace_mask).to(device='cpu')
            elif isinstance(subspace_mask, torch.Tensor):
                subspace_mask = subspace_mask.to(device='cpu')

        if out_buffer is None:
            target_dtype = np.float32 if is_sparse else (queries_np.dtype if queries_np.dtype in (np.float16, np.float32) else np.float32)
            out_buffer = np.empty((n_queries, n_features), dtype=target_dtype)

        for s in range(0, n_queries, batch_size):
            if is_sparse:
                chunk_np = queries[s:s + batch_size].toarray().astype(np.float32)
            else:
                chunk_np = queries_np[s:s + batch_size].astype(np.float32, copy=True if mask_sentinela_ausentes is not None else False)

            if mask_sentinela_ausentes is not None:
                chunk_np[:, mask_sentinela_ausentes] = fill_value

            x = torch.from_numpy(chunk_np).to(device='cpu')

            if self.binary:
                x = 2.0 * x - 1.0

            for _ in range(self.n_iters):
                if subspace_mask is not None:
                    x_att = x[:, subspace_mask]
                    Xi_att = Xi[:, subspace_mask]
                else:
                    x_att = x
                    Xi_att = Xi

                if normalize:
                    x_norm = F.normalize(x_att, p=2, dim=-1, eps=1e-8)
                    Xi_norm = F.normalize(Xi_att, p=2, dim=-1, eps=1e-8)
                    scores = self.beta * (x_norm @ Xi_norm.T)
                else:
                    scores = self.beta * (x_att @ Xi_att.T)
                weights = torch.softmax(scores, dim=-1)
                x = weights @ Xi

            if self.binary:
                x = (x > self.threshold).float()

            res_np = x.numpy()
            if out_buffer.dtype != res_np.dtype:
                res_np = res_np.astype(out_buffer.dtype, copy=False)
            out_buffer[s:s + batch_size] = res_np

        print(f"[ModernHopfieldNetwork] Recuperação concluída: {out_buffer.shape} (dtype={out_buffer.dtype})")
        return out_buffer

    def salvar(self, path):
        if self.patterns.numel() == 0:
            raise RuntimeError("[ModernHopfieldNetwork] Execute .store() antes de salvar.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "beta":      self.beta,
            "n_iters":   self.n_iters,
            "binary":    self.binary,
            "threshold": self.threshold,
            "normalize": getattr(self, "normalize", False),
            "patterns":  self.patterns.cpu()
        }, path)
        print(f"[ModernHopfieldNetwork] Rede salva em: {path} "
              f"({self.patterns.shape[0]} padrões)")
        return self

    @classmethod
    def carregar(cls, path):
        data = torch.load(path, map_location="cpu")
        rede = cls(
            beta      = data["beta"],
            n_iters   = data["n_iters"],
            binary    = data["binary"],
            threshold = data["threshold"],
            normalize = data.get("normalize", False)
        )
        rede.patterns = data["patterns"]
        print(f"[ModernHopfieldNetwork] Rede carregada de: {path} "
              f"({rede.patterns.shape[0]} padrões)")
        return rede

    def salvar_com_metadados(self, path_pt, path_meta, meta, classes=None, nc=None):
        """Salva a rede (.pt) e o arquivo JSON de metadados (.json)."""
        import json
        self.salvar(path_pt)

        meta_serializavel = [list(item) if isinstance(item, (tuple, list, np.ndarray)) else item for item in meta]
        n_patterns = self.patterns.shape[0] if self.patterns.numel() else 0
        n_genes = self.patterns.shape[1] if self.patterns.numel() else 0

        info = {
            "meta": meta_serializavel,
            "classes": classes if classes is not None else [1, 2, 3, 4, 5, 6, 7],
            "nc": nc if nc is not None else (n_patterns // len(classes) if classes else 30),
            "n_patterns": n_patterns,
            "n_genes": n_genes
        }

        os.makedirs(os.path.dirname(os.path.abspath(path_meta)), exist_ok=True)
        with open(path_meta, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

        print(f"[ModernHopfieldNetwork] Metadados salvos em: {path_meta} "
              f"({n_patterns} padrões, {n_genes} genes)")
        return self

    @classmethod
    def carregar_com_metadados(cls, path_pt, path_meta):
        """Carrega a rede (.pt) e o arquivo JSON de metadados (.json)."""
        import json
        rede = cls.carregar(path_pt)

        with open(path_meta, "r", encoding="utf-8") as f:
            meta_json = json.load(f)

        meta_eval = [tuple(x) for x in meta_json["meta"]]

        print(f"[ModernHopfieldNetwork] Metadados carregados de: {path_meta} "
              f"(classes={meta_json.get('classes')}, nc={meta_json.get('nc')}, n_patterns={meta_json.get('n_patterns')})")
        return rede, meta_eval, meta_json


    def hopf_tr(self, patterns):
        """Alias compatível com o script MATLAB original."""
        return self.store(patterns)

    def hopf_ts(self, queries, **kw):
        """Alias compatível com o script MATLAB original."""
        return self.retrieve(queries, **kw)

    forward = retrieve

    def __repr__(self):
        n_pad = self.patterns.shape[0] if self.patterns.numel() else 0
        dim   = self.patterns.shape[1] if self.patterns.numel() else 0
        return (
            f"ModernHopfieldNetwork(\n"
            f"  beta       = {self.beta}\n"
            f"  n_iters    = {self.n_iters}\n"
            f"  binary     = {self.binary}\n"
            f"  threshold  = {self.threshold}\n"
            f"  normalize  = {getattr(self, 'normalize', False)}\n"
            f"  patterns   = {n_pad} × {dim}\n"
            f")"
        )
