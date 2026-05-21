import os
import pandas as pd


class GeradorConjuntoTreinamento:
    """Filtra arquivos TXT alinhados para manter somente os genes do conjunto de treinamento.

    Lê o TXT em chunks para suportar arquivos grandes.
    Mantém a ordem original das colunas do arquivo alinhado.
    """

    def __init__(self, path_top_genes_csv, out_dir, chunk=3000):
        self.path_top_genes_csv = path_top_genes_csv
        self.out_dir            = out_dir
        self.chunk              = chunk
        self.genes_selecionados = None
        self._carregar_genes()

    def _carregar_genes(self):
        df = pd.read_csv(self.path_top_genes_csv)
        self.genes_selecionados = set(df['gene'].tolist())
        print(f"[GeradorConjuntoTreinamento] {len(self.genes_selecionados)} genes carregados de: {self.path_top_genes_csv}")

    def gerar(self, path_txt):
        # Lê somente o cabeçalho (primeira linha) para calcular o caminho de saída — operação barata
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

        total = 0
        with open(path_tmp, 'w', buffering=64 * 1024 * 1024) as fout:
            fout.write(','.join(genes_filtrados) + '\n')
            for chunk_df in pd.read_csv(path_txt, chunksize=self.chunk,
                                        header=0, usecols=genes_filtrados):
                chunk_df[genes_filtrados].to_csv(fout, header=False, index=False)
                total += len(chunk_df)
                if total % (self.chunk * 10) == 0:
                    print(f"  {total} células processadas...")

        os.rename(path_tmp, path_saida)
        print(f"[GeradorConjuntoTreinamento] Salvo: {path_saida}  ({total} células × {n} genes)")
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
