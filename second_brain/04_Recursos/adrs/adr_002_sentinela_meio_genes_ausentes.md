---
tipo: adr
tags: [adr, sentinela, alinhamento, mathys, decisao-arquitetura]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "ADR 002: Decisão de usar o valor neutro 0.5 para genes ausentes no alinhamento cross-dataset Mathys -> Fujita."
---

# ADR 002: Preenchimento de Genes Ausentes com Valor Sentinela Neutral 0.5 no Alinhamento Cross-Dataset

> **Status:** Aceito  
> **Data:** 30/07/2026  
> **Decisores:** Equipe de Bioinformática & Agente AI  

---

## 1. Contexto

No alinhamento dos datasets Fujita (referência de treinamento) e Mathys (alvo de imputação), aproximadamente 6.289 genes presentes no genoma do Fujita não estão presentes na anotação ou foram descartados no Mathys. Zerar esses genes antes da recuperação implicaria assumir incorretamente que eles estão biológica e comprovadamente desligados ($0$).

## 2. Decisão

Adotou-se o valor neutro $0.5$ como **sentinela de incerteza biológica** para todos os genes ausentes no Mathys durante a construção da matriz alinhada:
$$W_{\text{mathys}}[i, j] = 0.5 \quad \forall j \in \text{Genes Ausentes}$$

Veja o **[[03_Conhecimento/sentinela_meio_genes_ausentes|Conceito Atômico de Sentinela Neutra]]**.

Durante o passo de binarização da resposta da rede (`threshold=0.8`), os valores sentinelas são binarizados e substituídos pelas memórias reconstruídas da rede Hopfield.

## 3. Consequências Biológicas

- **Vantagens:** Não penaliza genes não sequenciados como sendo inativos ($0$). Permite que a rede Hopfield reconstrua o valor biológico esperado ($0$ ou $1$) com base na combinação dos outros ~5.000 genes ativos presentes na célula query.

## 4. Consequências Técnicas

- **Vantagens:** Ao passar o vetor query pela produto interno $\xi \cdot \Xi^T$, o valor $0.5$ contribui com um peso médio neutro para todas as memórias, permitindo que a decisão de similaridade seja guiada exclusivamente pelos genes de fato observados ($0$ ou $1$).
