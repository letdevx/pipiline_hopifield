---
tipo: adr
tags: [adr, hopfield, consenso, kmeans, sweeep, subclusters, v3]
criado: 2026-08-05
atualizado: 2026-08-05
resumo: "Decisão de avaliar a amostragem por consenso de vizinhos no espaço SWeeP (k=5) com expansão para nc=30 subclusters por classe (210 padrões totais) no pipeline v3."
---

# 🏛️ ADR 012: Experimento com Amostragem por Consenso ($k=5$) e 210 Padrões no Pipeline v3

## 1. Status
**Aceito** (Em validação experimental ativa no notebook [[01_Projetos/pipeline_hopfield_expandido/index|Pipeline Hopfield v3]]).

---

## 2. Contexto
As versões anteriores do pipeline (ex: `pipeline_hopfield_v2`) utilizam uma estratégia de extração de protótipos onde, após a clusterização K-Means no subespaço rSWeeP (600D), seleciona-se apenas o indivíduo binário mais próximo do centroide de cada cluster ($k=1$).

Embora eficaz para uma aproximação inicial de baixo custo computacional ($nc=10$ resultando em 70 padrões totais para 7 classes), a escolha de uma única célula representativa por centroide ($k=1$) expõe o atlas de memória da Rede Hopfield Moderna a **ruídos técnicos individuais de amostragem** e ao fenômeno de ***dropout* transcricional** comum em tecnologias *single-cell RNA sequencing* (scRNA-seq).

Além disso, limitar cada classe celular a apenas 10 protótipos comprime artificialmente a rica diversidade sub-funcional e populacional (ex: gradientes corticais de neurônios excitatórios ou subtipos reativos de microglias), reduzindo o poder discricionário da atenção Softmax durante testes de imputação trans-dataset (Fujita $\rightarrow$ Mathys).

---

## 3. Decisão
Implementamos e configuramos no notebook `pipeline_hopfield_v3` (via edição OOM-Safe com Jupytext) a expansão da capacidade associativa e da robustez de extração:
1. **Amostragem de Consenso de Vizinhança ($k=5$):** Em vez de registrar a expressão bruta de uma única célula, o `ExtratorPadroesSubcluster` seleciona os 5 vizinhos mais próximos a cada centroide K-Means no espaço SWeeP 600D e aplica um **voto de maioria transcricional** ($\text{média} \ge 0.5$).
2. **Alta Resolução de Subclusters ($nc=30$):** Expandimos para 30 subclusters por classe transcricional, resultando no armazenamento de **210 padrões prototípicos de consenso** ($7 \text{ classes} \times 30 \text{ subclusters}$).
3. **Persistência Isolada do Modelo:** Os novos pesos sinápticos consolidados são exportados para o artefato dedicado `rede35_v3_k5_210padroes.pt` (com correspondente arquivo de metadados em JSON), preservando sem colisões o baseline de 70 padrões legados.

```mermaid
graph TD
    subgraph SWeeP ["Espaço SWeeP 600D (Clusterização)"]
        K ["Centroide do Subcluster (nc=1, ..., 30)"]
        V1 ["Vizinho 1 (d=d1)"]
        V2 ["Vizinho 2 (d=d2)"]
        V3 ["Vizinho 3 (d=d3)"]
        V4 ["Vizinho 4 (d=d4)"]
        V5 ["Vizinho 5 (d=d5)"]
        K ---> V1 & V2 & V3 & V4 & V5
    end

    subgraph Consenso ["Espaço Gênico Original (5.000 genes binários)"]
        M ["Média Transcricional dos k=5 Vizinhos"]
        T ["Voto de Maioria: (Média >= 0.5) -> {0, 1}"]
        M --> T
    end

    V1 & V2 & V3 & V4 & V5 ---> M
    T ---> MHN ["Protótipo Lógico Armazenado (Modern Hopfield Network)"]

    style SWeeP fill:#e3f2fd,stroke:#1565c0
    style Consenso fill:#e8f5e9,stroke:#2e7d32
    style MHN fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
```

---

## 4. Consequências Biológicas
* **Mitigação de Dropouts:** Um gene importante expresso de forma intermitente que teria sofrido *dropout* na célula mais próxima ($d_1$) é devidamente reconstruído se os outros vizinhos transcricionalmente próximos o expressarem, gerando um perfil contundente da linhagem.
* **Preservação de Sublinhagens Raras e Complexas:** Com $nc=30$, o modelo preserva subestados celulares heterogêneos de linhagens complexas sem fundi-los de maneira espúria em protótipos excessivamente genéricos.
* **Fidelidade na Imputação:** Durante a recuperação cross-dataset (Mathys com sentinela $0.5$), a matriz de atenção compara a query com memórias filtradas e biologicamente coerentes, eliminando artefatos de ruído técnico pontual.

---

## 5. Consequências Técnicas
* **Consumo de Memória Moderado e Controlado:** Embora a matriz de padrões cresça de $(70, 5000)$ para $(210, 5000)$, o consumo em RAM para operações tensorais com a Rede Hopfield em 5.000 genes continua bem abaixo do limite da máquina (~4 a 8 GB em operações vetoriais completas com `batch_size=4096`).
* **Sincronização Bidirecional (Jupytext):** Todas as edições foram garantidas sem tocar no JSON via `jupytext --set-formats ipynb,py:percent` e `jupytext --sync`, garantindo rastreabilidade no controle de versão Git.
* **Compatibilidade de Metadados:** O arquivo `rede35_v3_k5_210padroes_metadata.json` contém a especificação correta do `nc=30` e mapeia cada um dos 210 padrões à sua classe e índice original no Fujita.

---

## 6. Referências no Grafo do Second Brain
* **Conceito Teórico:** [[03_Conhecimento/atencao_como_memoria_hopfield|Atenção Softmax na Rede Hopfield]]
* **ADR Relacionado:** [[04_Recursos/adrs/adr_010_harmonizacao_cosseno_e_prototipos_consolidados|ADR 010: Protótipos Consolidados ($k>1$)]]
* **Projeto Mestre:** [[01_Projetos/pipeline_hopfield_expandido/index|Arquitetura do Pipeline Hopfield Expandido]]
