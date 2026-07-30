---
tipo: adr
tags: [adr, binarizacao, scrnaseq, decisao-arquitetura]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "ADR 001: Decisão de converter matrizes contínuas scRNA-seq para estado binário (>0 -> 1)."
---

# ADR 001: Binarização de Expressão Gênica Single-Cell ($> 0 \rightarrow 1$)

> **Status:** Aceito  
> **Data:** 30/07/2026  
> **Decisores:** Equipe de Bioinformática & Agente AI  

---

## 1. Contexto

Os dados de sequenciamento de RNA de célula única (scRNA-seq) apresentam altíssima esparsidade (taxas de *dropout* variando entre 70% e 90%), além de variações técnicas de profundidade de sequenciamento entre bibliotecas. Em redes de memória associativa Hopfield, valores contínuos sujeitos a ruído de magnitude de leitura podem desestabilizar as bacias de atração e dificultar a convergência.

## 2. Decisão

Adotou-se a **binarização estrita** da matriz de expressão gênica:
$$x_{i,j} = \begin{cases} 1 & \text{se } \text{expressão}_{i,j} > 0 \\ 0 & \text{se } \text{expressão}_{i,j} \le 0 \end{cases}$$

Veja o **[[03_Conhecimento/binarizacao_expressao_genica|Conceito Atômico de Binarização]]**.

## 3. Consequências Biológicas

- **Vantagens:** Remove ruídos de escala contínua e amplificações estocásticas do sequenciamento. Foca o modelo na presença/ausência de programas transcricionais ON/OFF, que definem a identidade fundamental dos tipos celulares.
- **Limitações:** Descarta a gradação quantitativa de expressão de genes constitutivos com variação de nível.

## 4. Consequências Técnicas

- **Vantagens:** Reduz drasticamente o ruído na computação dos produtos internos do vetor de query $\xi$ com a matriz de memória $\Xi$. Permite representações altamente eficientes na memória.
- **Impacto em Memória/Desempenho:** Permite armazenamento compacto e simplifica o cálculo de distâncias na rede Hopfield.
