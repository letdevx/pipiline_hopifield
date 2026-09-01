# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import pandas as pd

# %%
pan = pd.read_csv(r"C:\Users\Leticia\Documents\Letworkspace\pipiline_hopifield\imputs\featuresPAN.tsv", sep="\t")
pan.head

# %%
pan = pd.read_csv(r"C:\Users\Leticia\Documents\Letworkspace\pipiline_hopifield\imputs\featuresPAN.tsv", sep="\t")
pan.columns = pan.columns.str.strip()
pan['gene_id'] = [gene.split(".")[0].strip() for gene in pan['gene_id']]
pan.head(5)


# %%
pan = pan[['gene_id','gene_name']]

# %%
pan.to_csv('featuresPANcorrigido.tsv',header=None, sep="\t", index=False)

# %%
pan.head(5)

# %%
M = pd.read_csv(r"C:\Users\Leticia\Documents\Letworkspace\pipiline_hopifield\imputs\featuresM.tsv\features.tsv", sep="\t", header=None)
M[0] = M[0].str.strip()
M.head(5)


# %%


pan.head(5)

# %%
pan_set = set(pan["gene_id"])
m_set = set(M[0])
interseccao = pan_set & m_set
print(len(interseccao))

# %%
math_set = set(M[1])
pan_setGN = set(pan["gene_name"])
inter_pan_math = pan_set & math_set
print(len(inter_pan_math))

# %%

# %%
list(pan_set)[:5]

# %%
fuji = pd.read_csv(r"C:\Users\Leticia\Documents\Letworkspace\pipiline_hopifield\imputs\features.tsv",sep="\t",header=None)
fuji[0] = fuji[0].str.strip().
fuji.head(5)

# %%
fuji_set = set(fuji[0])

# %%

# %%
inter_pan_fuj =pan_set & fuji_set
print(len(inter_pan_fuj))

# %%
diferenca_pan_fuji = pan_set - fuji_set
print(len(diferenca_pan_fuji))

# %%
diferenca_pan_M = pan_set - m_set
print(len(diferenca_pan_M))

# %%
pan_features = pd.DataFrame(pan_set)
pan_features.head(5)


# %%
pan_features.to_csv('featuresPANcorrigido.tsv',header=None, sep="\t", index=False)

# %%
import numpy as np
dados = np.load(r"C:\Users\Leticia\Downloads\X_mathys_IMPUTADO_rede35.npy") 
print(dados)

# %%
from scipy import io, sparse

# Converte o array denso para matriz esparsa CSR
matriz_esparsa = sparse.csr_matrix(dados)

# 3. Salva no formato .mtx (Matrix Market)
io.mmwrite(r"C:\Users\Leticia\Downloads\matriz_esparsa.mtx", matriz_esparsa)

