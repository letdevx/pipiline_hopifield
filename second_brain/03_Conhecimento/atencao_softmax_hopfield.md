---
tipo: conceito
tags: [conceito, hopfield, softmax, atencao, redes-neurais]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Conceito atômico do mecanismo de atualização por Atenção Softmax em Redes Hopfield Modernas."
---

# Atenção Softmax Hopfield

## Definição
A **Atenção Softmax Hopfield** é a regra de atualização de estados em Redes Hopfield Modernas com memória contínua (Ramsauer et al., 2020). É formalmente idêntica à camada de atenção autônoma (*Self-Attention*) de modelos Transformer:

$$\xi^{\text{novo}} = \Xi^T \cdot \text{softmax}(\beta \cdot \Xi \cdot \xi)$$

Onde $\xi$ é o vetor de consulta, $\Xi$ é a matriz de memórias armazenadas e $\beta$ é a temperatura inversa.

## Propriedades Claves
1. **Capacidade Exponencial:** Consegue armazenar um número de padrões que cresce exponencialmente com a dimensão do espaço de entrada.
2. **Convergência em 1 Passo:** Para valores elevados de $\beta$ (ex: $\beta = 50.0$), a atualização converge em apenas um passo de computação.

## Conexões
- Área: [[02_Areas/modern_hopfield_networks/index|Redes Hopfield Modernas]]
- Decisão Formada: [[04_Recursos/adrs/adr_005_rede_hopfield_moderna_parametros|ADR 005: Hiperparâmetros Hopfield]]
- Conceito Relacionado: [[03_Conhecimento/amostragem_prototipos_kmeans|Amostragem de Protótipos via K-Means]]
