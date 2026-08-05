---
tipo: adr
tags: [adr, hopfield, cosseno, normalizacao, kmeans, prototipos-consolidados, rsweep, mathys]
criado: 2026-08-03
atualizado: 2026-08-03
resumo: "Decisão Arquitetural sobre adoção opcional de atenção normalizada por cosseno na Rede Hopfield Moderna, amostragem consolidada de protótipos no rSWeeP (k>1) e harmonização de métricas de avaliação para reconstrução cross-dataset."
---

# ADR 010: Harmonização por Similaridade de Cosseno, Protótipos Consolidados ($k>1$) e Otimização Trans-Dataset

## Status
Aceito (Implementado em 2026-08-03)

## Contexto e Problema
Durante a reconstrução do dataset **Mathys** a partir da memória associativa constituída por protótipos do dataset **Fujita**, foram detectados gargalos de precisão que limitavam o desempenho de generalização e a qualidade da imputação de dados faltantes:

1. **Viés de Esparsidade e Profundidade (Sparsity / Library Size Bias):** No produto escalar não normalizado ($x \cdot \Xi^T$) da [[03_Conhecimento/atencao_softmax_hopfield|Atenção Softmax Hopfield]], memórias com maior número total de genes ativos ($1$s) possuem normas maiores e acabam dominando as probabilidades do Softmax. Como os dois estudos clínicos possuem variações técnicas nas taxas de *dropout*, as consultas do Mathys poderiam ser erroneamente atraídas para células mais ricas de leituras totais ao invés do correto subtipo transcricional.
2. **Ruído de Célula Única no K-Means ($k=1$):** A extração convencional no espaço [[03_Conhecimento/projecao_rsweep_600d|rSWeeP (600D)]] selecionava a única célula binária real do Fujita mais próxima do centroide de cada cluster ($k=1$). Esse vizinho isolado inevitavelmente transportava seus próprios ruídos biológicos e *dropouts* estocásticos individuais para dentro dos pesos permanentes da memória associativa.
3. **Inconsistência de Métricas no Avaliador:** Enquanto a rede toma decisões em alta dimensionalidade (11k a 36k genes) via produto interno com representação de spin $\{-1, +1\}$, o avaliador calculava distâncias Euclidianas quadradas na escala $\{0, 1\}$, sofrendo punições severas pela maldição da dimensionalidade.

## Decisão
A equipe técnica e científica deliberou por introduzir três evoluções arquiteturais nos módulos `src/treinamento/`:

1. **Ativação Explícita de Atenção Normalizada por Cosseno (`normalize=True`):** O método `.retrieve()` da `ModernHopfieldNetwork` agora possui a opção de normalizar os vetores pela norma $L_2$ antes do produto vetorial, invalidando o efeito da quantidade absoluta de genes ligados:
   $$\text{scores} = \beta \cdot \left(\frac{x}{\|x\|_2} \cdot \frac{\Xi^T}{\|\Xi\|_2}\right)$$
2. **Estudo Empírico Comparativo de Protótipos Consolidados ($k \in [1, 3, 5, 10]$):** Em substituição a um valor fixo engessado, permitimos e encorajamos a experimentação com consolidações por voto majoritário dos $k$ vizinhos mais próximos dos centroides SWeeP. A configuração ótima $k^*$ é selecionada empiricamente de acordo com o pico de F1-Score e acurácia.
3. **Harmonização com Avaliação via Cosseno (`metrica="cosseno"`):** O módulo `AvaliadorHopfield` agora suporta classificar as previsões recuperadas com base em distância cosseno em substituição à distância euclidiana, alinhando a avaliação analítica perfeitamente com a mecânica quási-energética da rede neuronal de Hopfield.

```mermaid
flowchart TD
    subgraph SWeeP_Consolidado ["1. Extração no Espaço SWeeP (k > 1)"]
        A["Centroide K-Means 600D"] -->|Busca de Vizinhos| B["k Vizinhos no Espaço Completo de Genes"]
        B -->|Voto Majoritário > 0.5| C["Protótipo Consolidado (Sem Ruído/Dropout)"]
    end

    subgraph Hopfield_Cosseno ["2. Atenção Normalizada (normalize=True)"]
        Q["Query Mathys (Sentinelas 0.5 vira 0.0)"] -->|Normalização L2| Q_NORM["Query Normalizada"]
        C -->|Normalização L2| C_NORM["Memória Normalizada"]
        Q_NORM & C_NORM -->|Produto Escalar| SIM["Similaridade de Cosseno Pura (Sem Viés de Esparsidade)"]
        SIM -->|Ativação Softmax| IMP["Matriz Reconstruida / Imputada"]
    end

    subgraph Avaliacao_Harmonizada ["3. Avaliação Estatística"]
        IMP -->|AvaliadorHopfield (metrica = 'cosseno')| RES["Mapeamento Prototípico Harmonizado"]
    end

    style C fill:#4CAF50,stroke:#388E3C,color:#fff
    style SIM fill:#2196F3,stroke:#1565C0,color:#fff
    style RES fill:#9C27B0,stroke:#6A1B9A,color:#fff
```

## Consequências
* **Positivas:**
  - Robustez comprovada perante artefatos técnicos de sequenciamento diferentes entre laboratórios (Fujita vs. Mathys).
  - Memórias associativas significativamente mais depuradas e biotipicamente puras com $k > 1$.
  - Modularidade preservada através da ativação explícita por parâmetro (`normalize=False` e `metrica="euclidiana"` continuam acessíveis para reprodutibilidade de ensaios anteriores).
* **Negativas:**
  - O cálculo adicional de normas $L_2$ em lotes adiciona milissegundos adicionais de processamento tensor no PyTorch na etapa de inferência `.retrieve()`.

## Conexões e Referências
- Arquitetura Mestre do Sistema: [[01_Projetos/pipeline_hopfield_expandido/arquitetura_do_sistema|Arquitetura do Sistema Expandido]]
- Conceito Atômico da Atenção: [[03_Conhecimento/atencao_softmax_hopfield|Atenção Softmax Hopfield]]
- Redução Vetorial: [[03_Conhecimento/projecao_rsweep_600d|Projeção rSWeeP 600D]]
- Decisão sobre Sentinelas de Inspecção: [[04_Recursos/adrs/adr_002_sentinela_meio_genes_ausentes|ADR 002]]
