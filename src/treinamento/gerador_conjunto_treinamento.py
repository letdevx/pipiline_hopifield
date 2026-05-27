import os

import polars as pl


class GeradorConjuntoTreinamento:
    """Filtra arquivos TXT alinhados para manter somente os genes do conjunto de treinamento.

    Usa Polars streaming (sink_csv) para suportar arquivos massivos sem carregá-los na memória.
    Mantém a ordem original das colunas do arquivo alinhado.
    """

    def __init__(self, path_top_genes_csv, out_dir, chunk=3000):
        self.path_top_genes_csv = path_top_genes_csv
        self.out_dir            = out_dir
        self.chunk              = chunk  # mantido por compatibilidade
        self.genes_selecionados = None
        self._carregar_genes()

    def _carregar_genes(self):
        df = pl.read_csv(self.path_top_genes_csv)
        self.genes_selecionados = set(df['gene'].to_list())
        print(f"[GeradorConjuntoTreinamento] {len(self.genes_selecionados)} genes carregados de: {self.path_top_genes_csv}")

    def gerar(self, path_txt):
        with open(path_txt, encoding='utf-8') as f:
            todos_genes = f.readline().strip().split(',')

        genes_filtrados = [g for g in todos_genes if g in self.genes_selecionados]
        n = len(genes_filtrados)
        nome       = os.path.splitext(os.path.basename(path_txt))[0]
        path_saida = os.path.join(self.out_dir, f"{nome}_top{n}.txt")
        os.makedirs(self.out_dir, exist_ok=True)

        if os.path.exists(path_saida):
            print(f"[GeradorConjuntoTreinamento] Arquivo já existe, pulando: {path_saida}")
            self.path_saida = path_saida
            return self

        path_tmp = path_saida + ".tmp"
        if os.path.exists(path_tmp):
            os.remove(path_tmp)

        print(f"\n[GeradorConjuntoTreinamento] Processando: {path_txt}")
        print(f"  Genes encontrados no arquivo: {n} de {len(self.genes_selecionados)} selecionados")
        print(f"  Escrevendo via Polars streaming...")

        (
            pl.scan_csv(path_txt, infer_schema_length=1)
            .select(genes_filtrados)
            .sink_csv(path_tmp)
        )

        os.rename(path_tmp, path_saida)
        print(f"[GeradorConjuntoTreinamento] Salvo: {path_saida}  ({n} genes)")
        self.path_saida = path_saida
        return self

    def __repr__(self):
        n = len(self.genes_selecionados) if self.genes_selecionados else 'não carregado'
        return (
            f"GeradorConjuntoTreinamento(\n"
            f"  path_top_genes_csv = {self.path_top_genes_csv}\n"
            f"  out_dir            = {self.out_dir}\n"
            f"  genes_selecionados = {n}\n"
            f")"
        )
