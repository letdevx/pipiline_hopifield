---
tipo: adr
tags: [adr, hopfield, hiperparametros, beta, subclusters, decisao-arquitetura]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "ADR 005: Decisão dos hiperparâmetros da Modern Hopfield Network (beta=50.0, nc=30, k=1)."
---

# ADR 005: Hiperparâmetros da Modern Hopfield Network ($\beta=50.0$, $nc=30$, $k=1$)

> **Status:** Aceito  
> **Data:** 30/07/2026  
> **Decisores:** Equipe de Bioinformática & Agente AI  

---

## 1. Contexto

A acurácia e a capacidade de generalização da rede Hopfield dependem diretamente da quantidade de padrões armazenados na memória ($\Xi$) e da temperatura inversa $\beta$ que controla a nitidez da regra de atenção Softmax. O modelo anterior com $nc=10$ (70 memórias) sofria de underfitting, alcançando apenas 68,15% de acurácia.

## 2. Decisão

Fixar os seguintes hiperparâmetros otimizados por busca em grade (Grid Search):
- **$nc = 30$ subclusters por classe** (7 classes $\times$ 30 = **210 padrões de memória** em 11k genes).
- **$k = 1$ indivíduo real mais próximo por centroide**.
- **$\beta = 50.0$ (Temperatura Inversa)**: Força um comportamento *winner-takes-all* nítido na recuperação.
- **$\text{iters} = 1$ iteração**: Suficiente para convergência direta em 1 passo de atenção contínua.
- **$\text{threshold} = 0.8$**: Limiar de binarização da resposta contínua recuperada.

Veja o **[[03_Conhecimento/atencao_softmax_hopfield|Conceito Atômico de Atenção Softmax Hopfield]]**.

## 3. Consequências Biológicas

- **Vantagens:** O aumento para $nc=30$ elevou a acurácia de classificação de 68,15% para 72,65% (+4,5 pontos percentuais) e o F1-score de 0,625 para 0,688, permitindo que a rede capture sub-estados transcricionais (ex: sub-camadas de neurônios excitatórios).

## 4. Consequências Técnicas

- **Desempenho:** Armazenar 210 padrões em 11.279 genes ocupa apenas ~9,4 MB de memória, tornando o armazenamento leve enquanto maximiza o raio de atração das memórias.
