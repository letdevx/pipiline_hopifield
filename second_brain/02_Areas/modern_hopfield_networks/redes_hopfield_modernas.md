---
tipo: area
tags: [area, hopfield, softmax-attention, memoria-associativa]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Nota técnica sobre a teoria e implementação das Redes Hopfield Modernas (Ramsauer et al., 2020)."
---

# Nota Técnica: Redes Hopfield Modernas (Modern Hopfield Networks)

> **Área:** Redes Hopfield Modernas  
> **Idioma:** PT-BR  

---

## 1. Fundamentos Teóricos (Ramsauer et al., 2020)

As **Redes Hopfield Modernas** (Modern Hopfield Networks) estendem a arquitetura clássica de Hopfield (densa e binária) para estados contínuos com capacidade de armazenamento exponencial. A regra de atualização de estados equivale formalmente a uma camada de **[[03_Conhecimento/atencao_softmax_hopfield|Atenção Softmax]]**:

$$\xi^{\text{novo}} = \Xi^T \cdot \text{softmax}(\beta \cdot \Xi \cdot \xi)$$

Onde:
- $\xi \in \mathbb{R}^N$: Vetor query (célula de entrada com $N \approx 11.000$ genes).
- $\Xi \in \mathbb{R}^{M \times N}$: Matriz de padrões armazenados na memória ($M = 210$ protótipos).
- $\beta$: Parâmetro de temperatura inversa.
- $\text{softmax}(\cdot)$: Operador de normalização Softmax.

---

## 2. Sintonia do Parâmetro de Temperatura $\beta$

- **$\beta \to \infty$ (Winner-Takes-All):** A rede foca exclusivamente no protótipo de memória com maior similaridade (menor distância L2 / maior produto escalar), ignorando ruídos moderados. Ideal para conjuntos de dados de alta dimensionalidade onde a identidade celular é bem definida. Veja **[[04_Recursos/adrs/adr_005_rede_hopfield_moderna_parametros|ADR 005]]**.
- **$\beta$ Moderado ($\beta \sim 5.0 - 10.0$):** A rede realiza uma média ponderada suave entre memórias próximas, útil para interpolação entre estados de diferenciação celular contínua.
- **Configuração Escolhida:** $\beta = 50.0$, garantindo alta seletividade entre os 210 protótipos de subtipos celulares.

---

## 3. Extração de Protótipos via K-Means em Espaço SWeeP

Armazenar todas as 40.000 células brutas como memórias geraria redundância e alto custo de busca. Por isso, aplica-se o algoritmo **[[03_Conhecimento/amostragem_prototipos_kmeans|K-Means]]** no espaço reduzido **[[03_Conhecimento/projecao_rsweep_600d|rSWeeP 600D]]** com $nc=30$ subclusters por classe biológica. O vetor binário real de 11.000 genes mais próximo de cada centroide é selecionado como o protótipo representativo final.
