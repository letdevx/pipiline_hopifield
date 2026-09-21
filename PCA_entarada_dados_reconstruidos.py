# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python (venv)
#     language: python
#     name: meu_venv
# ---

# %% [markdown]
# #### PCA dados de entrada e pos imputação 
#
# A intenção  plotar graficos PCAs para ver como os dados se comportam após a passagem pela reconstução

# %%
import numpy as np 
import sklearn as sk
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns 
import matplotlib.pyplot as plt


# %% [markdown]
# #### Importando dados para projeção

# %%
matriz_pan_antes_imputação = pd.read_csv(r"C:\Users\Leticia\Downloads\matriz_reduzida_sweepREF.txt", sep="\t")
matriz_pan_apos_imputacao = pd.read_csv(r"C:\Users\Leticia\Downloads\sweep_alvo_pos_imputacao.txt", sep="\t")
rotulos_pan = pd.read_csv(r'C:\Users\Leticia\Documents\Letworkspace\Treinamento-MLP-Mathys\Inputs\rotulos\PanNumerico.csv')

# %%

X  = matriz_pan_apos_imputacao### passagem dos dados para  variavel
y = rotulos_pan
 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA (n_components= 2) #### Definição da quntidade de componentes
X_pca = pca.fit_transform(X_scaled)


print("Variabilidade em cada cordenada", pca.explained_variance_ratio_) #verificando variancia nas cordenadas

# 5. Visualizar com Seaborn
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['classe'] = y.values


df_pca['classe'] = df_pca['classe'].astype(str)
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='classe')

plt.title('Projeção PCA PAN Faltantes  (2 Componentes) Após imputação da Rede')
plt.show()


# %%
# 1. Verifica se as classes 2 e 5 realmente existem no df_pca
print("Contagem por classe no df_pca:")
print(df_pca['classe'].value_counts(dropna=False))

# 2. Verifica se há valores nulos (NaN)
print("\nValores nulos na coluna classe:", df_pca['classe'].isna().sum())


# %%

X  = matriz_pan_antes_imputação### passagem dos dados para  variavel
y = rotulos_pan
 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA (n_components= 2) #### Definição da quntidade de componentes
X_pca = pca.fit_transform(X_scaled)


print("Variabilidade em cada cordenada", pca.explained_variance_ratio_) #verificando variancia nas cordenadas

# 5. Visualizar com Seaborn
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['classe'] = y.values


df_pca['classe'] = df_pca['classe'].astype(str)
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='classe')

plt.title('Projeção PCA PAN Faltantes  (2 Componentes) Antes da Imputação')
plt.show()


# %% [raw]
#
