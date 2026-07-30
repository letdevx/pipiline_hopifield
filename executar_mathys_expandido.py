import sys
import os
import gc
import numpy as np
import pandas as pd
import polars as pl
import anndata as ad
import scipy.sparse as sp

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    PATH_M, PATH_F, PATH_FEATURES_F, PATH_FEATURES_M,
    OUT_BINARIZACAO, OUT_ALINHAMENTO, OUT_TOP_GENES
)
from preprocessing.binarizador import Binarizador
from alinhamento.leitor_features import LeitorFeatures

print("=== 1. Carregando e Mapeando Features ===")
leitor = LeitorFeatures(PATH_FEATURES_F, PATH_FEATURES_M)
leitor.ler()

path_genes_target = os.path.join(OUT_TOP_GENES, 'genes_expandidos_frequentes.csv')
print(f"Lendo genes alvo a partir de: {path_genes_target}")
df_genes = pd.read_csv(path_genes_target)
genes_ordenados = df_genes['gene'].tolist()

# Precisamos do map_f e map_m do leitor para mapear o nome(original) para a posicao do gene ordenado
gene_alvo_idx = {}
for i, eid in enumerate(genes_ordenados):
    gene_alvo_idx[eid] = i

print(f"Total de genes na lista de intersecção: {len(genes_ordenados)}")

print("\n=== 2. Binarização da Matriz de Entrada (Mathys) ===")
binarizador_m = Binarizador(path_h5ad=PATH_M, out_dir=OUT_BINARIZACAO)
binarizador_m.binarizar()

print("\n=== 3. Alinhamento e Inserção de 0.5 para Falhas Cruzadas ===")
# Em vez de invocar toda a classe Alinhador que requer F e M juntos, vamos instanciar
# apenas para alinhar Mathys. Mas a classe Alinhador processa ambos. Vamos instanciar 
# passando null para F ou sobrescrevendo temporariamente para usar a função direta _alinhar_direto
from alinhamento.alinhador import Alinhador
alinhador = Alinhador(
    path_binarizada_m=binarizador_m.path_binarizada,
    path_binarizada_f="fake",  
    out_dir=OUT_ALINHAMENTO,
    map_f=leitor.map_f,
    map_m=leitor.map_m,
    gene_alvo_idx=gene_alvo_idx,
    genes_ordenados=genes_ordenados
)

print("[Orquestrador] Lendo h5ad binarizado do Mathys original...")
adatam = ad.read_h5ad(binarizador_m.path_binarizada)

print("[Orquestrador] Alinhando matriz aos genes alvo e mapeando missing values -> 0.5")
# fill_value=0.5 -> genes nao presentes no map do Mathys, mas presentes nos genes alvo
adatam_alinhado = alinhador._alinhar_direto(adatam, leitor.map_m, fill_value=0.5)

print("\n=== 4. Verificação Métrica de Dimensionalidade ===")
n_celulas = adatam_alinhado.shape[0]
n_genes = adatam_alinhado.shape[1]
print(f"============================================================")
print(f"✅ VALIDAÇÃO: Matriz Mathys Finalizada com Sucesso!")
print(f"📊 DIMENSIONALIDADE: {n_celulas} células X {n_genes} genes")
print(f"============================================================")

if n_genes != len(genes_ordenados):
    raise ValueError(f"Dimensão de genes incorreta! Esperava {len(genes_ordenados)}, encontrou {n_genes}")

print("\n=== 5. Exportação para *.TXT em lotes (Polars) ===")
out_txt = os.path.join(OUT_ALINHAMENTO, "Mathys_Binarizado_Alinhado_Expandido.txt")
path_tmp = out_txt + ".tmp"

chunk = 5000
total = 0

with open(path_tmp, 'w', buffering=128 * 1024 * 1024) as fout:
    fout.write(','.join(genes_ordenados) + '\n')
    for start in range(0, n_celulas, chunk):
        end = min(start + chunk, n_celulas)
        X = adatam_alinhado.X[start:end]
        if sp.issparse(X):
            X = X.toarray()
        fout.write(pl.from_numpy(np.asfortranarray(X.astype(np.float32))).write_csv(include_header=False))
        total += end - start
        if total % (chunk * 2) == 0:
            print(f"  {total} células processadas e salvas no disco...")

os.rename(path_tmp, out_txt)
print(f"\nFinalizado! Matriz em disco salva com êxito em:\n -> {out_txt}")
