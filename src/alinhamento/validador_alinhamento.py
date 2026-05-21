import anndata as ad


class ValidadorAlinhamento:
    """Valida que dois AnnData alinhados possuem exatamente a mesma lista de genes na mesma ordem."""

    def __init__(self, path_f_alinhado, path_m_alinhado, genes_ordenados):
        self.path_f_alinhado = path_f_alinhado
        self.path_m_alinhado = path_m_alinhado
        self.genes_ordenados = genes_ordenados

    def validar(self):
        print("[ValidadorAlinhamento] Carregando metadados...")
        _f = ad.read_h5ad(self.path_f_alinhado, backed='r')
        _m = ad.read_h5ad(self.path_m_alinhado, backed='r')
        genes_f = _f.var_names.tolist()
        genes_m = _m.var_names.tolist()
        _f.file.close()
        _m.file.close()
        del _f, _m

        if len(genes_f) != len(self.genes_ordenados):
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Fujita alinhado tem {len(genes_f)} genes, "
                f"esperado {len(self.genes_ordenados)}."
            )
        if len(genes_m) != len(self.genes_ordenados):
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Mathys alinhado tem {len(genes_m)} genes, "
                f"esperado {len(self.genes_ordenados)}."
            )

        divs_f = [(i, self.genes_ordenados[i], genes_f[i])
                  for i in range(len(self.genes_ordenados))
                  if genes_f[i] != self.genes_ordenados[i]]
        if divs_f:
            msg = "\n  ".join(
                f"pos {i}: esperado={e!r} encontrado={e2!r}" for i, e, e2 in divs_f[:5]
            )
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Fujita diverge em {len(divs_f)} posição(ões):\n  {msg}"
            )

        divs_m = [(i, self.genes_ordenados[i], genes_m[i])
                  for i in range(len(self.genes_ordenados))
                  if genes_m[i] != self.genes_ordenados[i]]
        if divs_m:
            msg = "\n  ".join(
                f"pos {i}: esperado={e!r} encontrado={e2!r}" for i, e, e2 in divs_m[:5]
            )
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Mathys diverge em {len(divs_m)} posição(ões):\n  {msg}"
            )

        divs_fm = [(i, genes_f[i], genes_m[i])
                   for i in range(len(genes_f))
                   if genes_f[i] != genes_m[i]]
        if divs_fm:
            msg = "\n  ".join(
                f"pos {i}: F={gf!r} M={gm!r}" for i, gf, gm in divs_fm[:5]
            )
            raise ValueError(
                f"[VALIDAÇÃO FALHOU] Fujita e Mathys divergem em {len(divs_fm)} posição(ões):\n  {msg}"
            )

        print(f"✓ Número de genes idêntico: {len(self.genes_ordenados)}")
        print(f"✓ Fujita alinhado == ordem de referência")
        print(f"✓ Mathys alinhado == ordem de referência")
        print(f"✓ Fujita alinhado == Mathys alinhado")
        print("[ValidadorAlinhamento] Validação concluída com sucesso.")
        return self

    def __repr__(self):
        return (
            f"ValidadorAlinhamento(\n"
            f"  path_f_alinhado  = {self.path_f_alinhado}\n"
            f"  path_m_alinhado  = {self.path_m_alinhado}\n"
            f"  genes_ordenados  = {len(self.genes_ordenados)} genes\n"
            f")"
        )
