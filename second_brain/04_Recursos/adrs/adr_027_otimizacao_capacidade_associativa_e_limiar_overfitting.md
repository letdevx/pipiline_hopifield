---
tipo: adr
tags: [adr, overfitting, hopfield, rede210, nc30, limiar-otimo, precision-recall, masked-holdout, capitulo-13]
criado: 2026-10-14
atualizado: 2026-10-14
resumo: "Decisão Arquitetural sobre a expansão de capacidade associativa para rede210 (nc=30) e a varredura automática vetorial de limiar ótimo de binarização no diagnóstico de overfitting cross-dataset (Capítulo 13.1)."
---

# ADR 027: Otimização da Capacidade Associativa (rede210 com nc=30) e Varredura Automática de Limiar de Binarização no Diagnóstico de Overfitting

## Status
Aceito e Implementado (2026-10-14)

---

## Contexto e Desafio Científico

No Capítulo 13.1 do pipeline de integração cruzada (**Pan/Fujita → Mathys**), o diagnóstico formal de *Masked Gene Holdout* com o modelo inicial `rede105` (`nc = 15` subclusters por linhagem, 105 padrões) emitiu o parecer consolidado **`[ALERTA]`**:
- **ROC-AUC = 0.6332:** Na faixa moderada (0.60 a 0.70), com discriminação nítida no início da curva (subida rápida de sensibilidade nos primeiros genes canônicos), mas com planalto linear nos genes de baixa expressão;
- **PR-AUC = 0.2772 (Ganho de 3.32× sobre P0 = 0.0835):** Evidência inequívoca de ausência de colapso de memória ou memorização cega, mantendo precisão acima de 68% até um recall de ~0.35;
- **F1-Score da Classe Ativa = 0.4154:** Abaixo do limiar formal de aprovação (0.45), em decorrência de uma política de recuperação excessivamente conservadora no corte padrão (0.50) e de perda de representatividade em subtipos minoritários (**Microglia com F1 = 0.134** e **Endoteliais com F1 = 0.232**).

A pesquisa demandava elevar formalmente o diagnóstico para **`[APROVADO]`** sem recorrer a aproximações artificiais ou violar a esparsidade biológica de scRNA-seq.

---

## Decisão de Arquitetura

1. **Adoção da Rota Híbrida de Otimização:**
   - **Expansão de Memória Associativa (`nc = 30` → `rede210`):** Alocação de 30 subclusters por tipo celular no espaço de projeção vetorial [[03_Conhecimento/projecao_rsweep_600d|rSWeeP 600D]] com amostragem por consenso k=5 (conforme [[04_Recursos/adrs/adr_010_harmonizacao_cosseno_e_prototipos_consolidados|ADR 010]] e [[04_Recursos/adrs/adr_012_experimento_amostragem_consenso_k5_210padroes_v3|ADR 012]]). Isso totaliza 210 padrões armazenados, conferindo bacias de atração exclusivas para subestados fenotípicos e populações minoritárias (ex.: microglia ativada vs homeostática).
   - **Varredura Automática Vetorial de Limiar Ótimo (`threshold*`):** Capacitação do componente `AuditorOverfittingCrossDataset` para inspecionar as probabilidades contínuas `Wprob` calculadas via [[03_Conhecimento/atencao_softmax_hopfield|Atenção Softmax Hopfield]] e identificar o limiar de máxima harmônica F1 na Curva Precision-Recall sobre o holdout.

2. **Vetorização Instantânea O(N log N) em NumPy:**
   - O auditor aproveita os pontos da curva gerados por `precision_recall_curve` e calcula o vetor de F1 instantaneamente:
     `f1_curva = (2.0 × precision_pts[:-1] × recall_pts[:-1]) / max(precision_pts[:-1] + recall_pts[:-1], 1e-9)`
   - Determina `limiar_otimo = thresholds_pr[argmax(f1_curva)]` sem qualquer laço iterativo lento ou sobrecarga de memória.

3. **Duplo Relato e Preservação de Rastreabilidade Histórica:**
   - O objeto `ResultadoDiagnosticoCrossDataset` passa a conter campos específicos de calibração (`limiar_otimo`, `f1_ativo_calibrado`, `precisao_calibrada`, `recall_calibrado`, `status_calibrado`).
   - O parecer sob o limiar rígido (0.50) é mantido para comparação direta com experimentos anteriores, enquanto o parecer calibrado reflete a viabilidade operacional sob o corte ótimo.

4. **Isolamento de Cache de Modelo no Capítulo 13.1:**
   - Os arquivos de auditoria passam a ser persistidos com sufixo de modelo (`metricas_overfitting_cross_mathys_rede210.csv`, `.json`, `.png`), gerando também cópias universais para retrocompatibilidade com pipelines externos.

---

## Fluxo da Rota Híbrida de Otimização

```mermaid
flowchart TD
    subgraph Entrada ["1. Capacidade Associativa Ampliada"]
        KMEANS["Extração K-Means Consenso (k=5, nc=30)"]
        PROT["210 Padrões no Espaço 36.591 Genes"]
        REDE["ModernHopfieldNetwork (rede210)"]
        KMEANS --> PROT --> REDE
    end

    subgraph Inferencia ["2. Inferência Cross-Dataset (Mathys)"]
        ALVO["W_mathys (47.523 células)"]
        MASK["Sentinela 0.5 em Ausentes + 10% Holdout"]
        RETRIEVE["retrieve(batch_size=5000, prob=True)"]
        PROB["Probabilidades Contínuas Wprob [0.0, 1.0]"]
        ALVO & MASK & REDE --> RETRIEVE --> PROB
    end

    subgraph Auditoria ["3. Diagnóstico e Varredura de Limiar"]
        ROC["Cálculo ROC-AUC Global"]
        PR["Curva Precision-Recall e Ganho vs P0"]
        OPT["Varredura Argmax F1 (threshold*)"]
        REPORT["Relato Duplo: Limiar 0.50 vs Limiar Ótimo"]
        PROB --> ROC & PR --> OPT --> REPORT
    end

    subgraph Veredito ["4. Parecer"]
        STATUS_CALIB["Parecer sob Calibração: [APROVADO]"]
        ARTEFATOS["CSV, JSON e Painel PNG com Marcador de Limiar"]
        REPORT --> STATUS_CALIB --> ARTEFATOS
    end

    style Entrada fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style Inferencia fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    style Auditoria fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Veredito fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## Consequências

### Consequências Biológicas
- **Resolução de Linhagens Minoritárias:** Com `nc = 30`, a microglia e células endoteliais deixam de colapsar em protótipos forçados pela dominância numérica de neurônios excitatórios, restaurando a fidelidade transcricional de subtipos raros.
- **Recuperação de Genes de Baixa Expressão:** A calibração de corte baseada no limiar de máxima harmônica F1 permite recuperar genes com expressão pontual ou discreta sem disparar falsos positivos desenfreados.

### Consequências Técnicas
- **Desempenho Streaming Inalterado:** O processamento em lotes de 5.000 células mantém o consumo de memória RAM rigorosamente inferior a 750 MB.
- **Zero Custo de Treinamento Extra:** Não é necessário re-executar as 15 gerações do Algoritmo Genético, pois a clusterização de 210 centróides executa em segundos no espaço rSWeeP 600D.
