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
    binary     : se True, converte {0,1} ↔ {-1,+1} internamente
    threshold  : limiar para binarizar saída quando binary=True
    patterns   : tensor com os padrões armazenados (preenchido em .store())
    """

    def __init__(self, beta=8.0, n_iters=1, binary=True, threshold=0.0):
        super().__init__()
        self.beta = beta
        self.n_iters = n_iters
        self.binary = binary
        self.threshold = threshold
        self.register_buffer("patterns", torch.empty(0))

    def store(self, patterns):
        """Armazena os padrões na rede (equivalente a hopf_tr do MATLAB).

        Retorna o próprio objeto para permitir encadeamento de chamadas.
        """
        P = torch.as_tensor(np.asarray(patterns), dtype=torch.float32)
        if self.binary:
            P = 2.0 * P - 1.0
        device = self.patterns.device if self.patterns.numel() else torch.device("cpu")
        self.patterns = P.to(device)
        print(f"[ModernHopfieldNetwork] {self.patterns.shape[0]} padrões armazenados "
              f"(dim={self.patterns.shape[1]}, device={device})")
        return self

    @torch.no_grad()
    def retrieve(self, queries, batch_size=2048):
        """Recupera o padrão mais próximo para cada query (equivalente a hopf_ts do MATLAB).

        Retorna array numpy com os padrões recuperados.
        """
        if self.patterns.numel() == 0:
            raise RuntimeError("[ModernHopfieldNetwork] Execute .store(patterns) antes de .retrieve().")

        Xi = self.patterns
        Q = torch.as_tensor(np.asarray(queries), dtype=torch.float32)
        if self.binary:
            Q = 2.0 * Q - 1.0
        Q = Q.to(Xi.device)

        out = torch.empty_like(Q)
        for s in range(0, Q.shape[0], batch_size):
            x = Q[s:s + batch_size]
            for _ in range(self.n_iters):
                scores = self.beta * x @ Xi.T
                weights = torch.softmax(scores, dim=-1)
                x = weights @ Xi
            out[s:s + batch_size] = x

        if self.binary:
            out = (out > self.threshold).float()

        print(f"[ModernHopfieldNetwork] Recuperação concluída: {out.shape}")
        return out.cpu().numpy()

    def hopf_tr(self, patterns):
        """Alias compatível com o script MATLAB original."""
        return self.store(patterns)

    def hopf_ts(self, queries, **kw):
        """Alias compatível com o script MATLAB original."""
        return self.retrieve(queries, **kw)

    forward = retrieve

    def __repr__(self):
        n_pad = self.patterns.shape[0] if self.patterns.numel() else 0
        dim = self.patterns.shape[1] if self.patterns.numel() else 0
        return (
            f"ModernHopfieldNetwork(\n"
            f"  beta       = {self.beta}\n"
            f"  n_iters    = {self.n_iters}\n"
            f"  binary     = {self.binary}\n"
            f"  threshold  = {self.threshold}\n"
            f"  padrões    = {n_pad} × {dim}\n"
            f")"
        )
