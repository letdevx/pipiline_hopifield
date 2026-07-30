---
tipo: conceito
tags: [conceito, rsweep, sweep, embeddings, reducao-dimensionalidade]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Conceito atômico da técnica de redução de dimensionalidade rSWeeP via matrizes de projeção ortonormal."
---

# Projeção rSWeeP 600D

## Definição
A **Projeção rSWeeP 600D** (Random SWeeP Embedding) é um algoritmo de redução de dimensionalidade linear que multiplica a matriz de expressão gênica de alta dimensão ($G \approx 11.000$) por uma matriz de base ortonormal aleatória $R \in \mathbb{R}^{G \times 600}$ gerada por decomposição QR.

$$W_{\text{swp}} = W_0 \times R_{\text{expandido}}$$

## Propriedades Claves
1. **Preservação de Distância (Lema de Johnson-Lindenstrauss):** A projeção em base ortonormal preserva com altíssima precisão as distâncias relativas entre pares de células.
2. **Eficiência Computacional:** Acelera algoritmos de agrupamento como o K-Means em mais de 18 vezes comparado à execução no espaço gênico completo.

## Conexões
- Recursos: [[04_Recursos/projecao_sweep/projeao_rsweep|Recurso Projeção rSWeeP]]
- Decisão Formada: [[04_Recursos/adrs/adr_004_projecao_rsweep_600d_kmeans|ADR 004: Projeção rSWeeP 600D]]
- Conceito Relacionado: [[03_Conhecimento/amostragem_prototipos_kmeans|Amostragem de Protótipos via K-Means]]
