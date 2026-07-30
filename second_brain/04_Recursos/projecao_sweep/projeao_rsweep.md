---
tipo: recurso
tags: [recurso, rsweep, sweep, algoritmo, embeddings]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Nota de recurso sobre a ferramenta e biblioteca da projeção rSWeeP (AIBIALab)."
---

# Recurso: Projeção rSWeeP (Random SWeeP Embedding)

> **Recurso:** Projeção SWeeP  
> **Idioma:** PT-BR  

---

## 1. O Conceito da Projeção rSWeeP

A projeção **rSWeeP** (desenvolvida no laboratório AIBIALab) é uma técnica de redução de dimensionalidade linear baseada na projeção de vetores de alta dimensão sobre uma matriz de base ortonormal aleatória gerada via decomposição QR.

$$W_{\text{swp}} = W_0 \times R_{\text{expandido}}$$

Onde:
- $W_0 \in \mathbb{R}^{C \times G}$: Matriz de expressão de $C$ células por $G \approx 11.000$ genes.
- $R_{\text{expandido}} \in \mathbb{R}^{G \times 600}$: Matriz de projeção com colunas ortonormais.
- $W_{\text{swp}} \in \mathbb{R}^{C \times 600}$: Matriz compacta de embeddings de 600 dimensões.

Veja o **[[03_Conhecimento/projecao_rsweep_600d|Conceito Atômico da Projeção rSWeeP 600D]]** e o **[[04_Recursos/adrs/adr_004_projecao_rsweep_600d_kmeans|ADR 004]]**.

---

## 2. Vantagens Computacionais e Biológicas

1. **Aceleração da Clusterização:** Agrupa 40.000 células em 600 dimensões em frações de segundo via K-Means, evitando o gargalo de rodar em 11.000 dimensões.
2. **Preservação de Distância (Lema de Johnson-Lindenstrauss):** Projeções aleatórias ortonormais preservam as distâncias relativas entre pontos no espaço de alta dimensão com baixíssimo erro.
3. **Independência de Re-Treinamento:** Ao contrário de Autoencoders ou PCA que exigem ajuste de autovetores a cada alteração da base, a matriz rSWeeP pode ser gerada deterministicamente via semente (`seed=42`).
