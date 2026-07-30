---
tipo: conceito
tags: [conceito, kmeans, prototipos, amostragem, subclusters]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Conceito atômico da amostragem de protótipos de memória usando K-Means em espaço vetorial reduzido."
---

# Amostragem de Protótipos via K-Means

## Definição
A **amostragem de protótipos via K-Means** é a estratégia de particionamento da população celular de cada classe biológica em $nc$ subgrupos (no espaço de baixa dimensão rSWeeP 600D) e seleção da célula real binária mais próxima do centroide no espaço original de alta dimensão.

## Vantagens
1. **Representatividade Intra-Classe:** Permite que a rede Hopfield guarde sub-estados celulares e variações funcionais internas das classes.
2. **Redução de Redundância:** Substitui dezenas de milhares de células por poucas centenas de padrões prototípicos de alta fidelidade ($210$ padrões no total).

## Conexões
- Área: [[02_Areas/modern_hopfield_networks/index|Redes Hopfield Modernas]]
- Conceito Relacionado: [[03_Conhecimento/projecao_rsweep_600d|Projeção rSWeeP 600D]]
- Decisão Formada: [[04_Recursos/adrs/adr_004_projecao_rsweep_600d_kmeans|ADR 004: Projeção rSWeeP 600D]]
