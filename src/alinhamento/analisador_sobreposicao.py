class AnalisadorSobreposicao:
    """Calcula sobreposição de espaços gênicos e define a ordem canônica baseada no Fujita."""

    def __init__(self, map_f, map_m, var_names_f_original):
        self.map_f = map_f
        self.map_m = map_m
        self.var_names_f_original = var_names_f_original
        self.ids_comuns = None
        self.ids_so_f = None
        self.ids_so_m = None
        self.genes_ordenados = None
        self.gene_alvo_idx = None

    def analisar(self):
        ids_f = set(self.map_f.values())
        ids_m = set(self.map_m.values())
        self.ids_comuns = ids_f & ids_m
        self.ids_so_f   = ids_f - ids_m
        self.ids_so_m   = ids_m - ids_f

        seen = set()
        self.genes_ordenados = []
        for gene_name in self.var_names_f_original:
            eid = self.map_f.get(gene_name, gene_name)
            if eid not in seen:
                self.genes_ordenados.append(eid)
                seen.add(eid)

        self.gene_alvo_idx = {g: i for i, g in enumerate(self.genes_ordenados)}

        print(f"[AnalisadorSobreposicao] Em comum  : {len(self.ids_comuns)}")
        print(f"[AnalisadorSobreposicao] Só Fujita : {len(self.ids_so_f)}")
        print(f"[AnalisadorSobreposicao] Só Mathys : {len(self.ids_so_m)}  ← serão excluídos")
        print(f"[AnalisadorSobreposicao] Espaço gênico final: {len(self.genes_ordenados)} genes")
        return self

    def __repr__(self):
        def _n(x): return len(x) if x is not None else 'não calculado'
        return (
            f"AnalisadorSobreposicao(\n"
            f"  ids_comuns      = {_n(self.ids_comuns)}\n"
            f"  ids_so_f        = {_n(self.ids_so_f)}\n"
            f"  ids_so_m        = {_n(self.ids_so_m)}\n"
            f"  genes_ordenados = {_n(self.genes_ordenados)}\n"
            f")"
        )
