---
tipo: adr
tags: [adr, alinhamento, features, validacao, fail-fast, scrnaseq, ensembl]
criado: 2026-08-28
atualizado: 2026-08-28
resumo: "Decisão de implementar a classe ValidadorFeatures com validação estrita (Fail-Fast), detecção de colunas invertidas via regex e verificação de compatibilidade AnnData x Features pré-alinhamento."
---

# 🏛️ ADR 013: Validação Estrita de Features e Identificadores Genômicos Pré-Alinhamento

## 1. Status
**Aceito** (Implementado em `src/alinhamento/validador_features.py` e integrado no Capítulo 3 dos pipelines).

---

## 2. Contexto
No processamento de conjuntos de dados independentes de *single-cell RNA sequencing* (scRNA-seq), como **Fujita** e **Mathys**, a correta equivalência de características transcricionais depende do mapeamento entre símbolos gênicos (*Gene Symbols*) e identificadores estáveis do Ensembl (*Ensembl IDs*).

Durante execuções no pipeline genérico, observou-se uma anomalia em que apenas **7 genes** do dataset alvo (Mathys) foram mapeados para o espaço canônico de 61.541 genes. A investigação revelou que:
1. Um arquivo de features com colunas invertidas (ou com convenções de nomes divergentes dos `var_names` da matriz `.h5ad`) passava despercebido pelo leitor e pelo alinhador;
2. Como a matriz esparsa de alinhamento simplesmente preenchia com sentinela `0.5` ou zeros os genes não encontrados, o pipeline continuava a execução sem emitir erros explícitos, resultando em tensores de memória Hopfield degradados e corrupção silenciosa da imputação transcricional.

---

## 3. Decisão
Implementamos a classe [[01_Projetos/pipeline_hopfield_expandido/arquitetura_do_sistema|ValidadorFeatures]] em `src/alinhamento/validador_features.py` adotando uma arquitetura **Fail-Fast** estrita:

1. **Detecção de Schema e Inversão de Colunas (Regex Ensembl):**
   - Inspeciona as primeiras linhas do arquivo de features (`.tsv`, `.tsv.gz`, `.csv`).
   - Avalia a presença do padrão Ensembl (`^ENS[A-Z]*G\d+`). Se a coluna 1 contiver identificadores Ensembl enquanto a coluna 0 contiver símbolos gênicos, o pipeline é interrompido imediatamente com um alerta diagnóstico de colunas invertidas.
2. **Compatibilidade AnnData x Features:**
   - Compara os `var_names` da matriz `.h5ad` binarizada com os identificadores do mapa de features de forma OOM-Safe (`backed='r'`).
   - Exige taxa de compatibilidade mínima configurável (padrão $\ge 50\%$). Caso contrário, levanta `ValueError` detalhado.
3. **Validação de Sobreposição Biológica Inter-Datasets:**
   - Exige que os datasets compartilhem pelo menos um número mínimo de genes homólogos (padrão $\ge 1.000$ genes), evitando o cruzamento acidental de anotações ou espécies distintas.

```mermaid
flowchart TD
    subgraph Entradas ["Arquivos Brutos de Entrada"]
        F1["features_referencia.tsv (.gz)"]
        F2["features_alvo.tsv (.gz)"]
        H1["adata_ref_binarizado.h5ad"]
        H2["adata_alvo_binarizado.h5ad"]
    end

    subgraph Validador ["ValidadorFeatures (Fail-Fast)"]
        V1{"Col 0 = Ensembl ID?<br/>(Regex ^ENSG...)"}
        V2{"var_names coincidem<br/>com mapa (>= 50%)?"}
        V3{"Genes em comum<br/>>= 1.000?"}
    end

    F1 & F2 --> V1
    V1 -- "Invertido / Inválido" --> E1["❌ ValueError + Diagnóstico Estruturado"]
    V1 -- "OK" --> V2
    H1 & H2 --> V2
    V2 -- "Mismatch < 50%" --> E2["❌ ValueError + Dicas de Correção"]
    V2 -- "OK" --> V3
    V3 -- "Sobreposição < 1000" --> E3["❌ ValueError + Alerta de Anotação"]
    V3 -- "Aprovado" --> SUC["✅ AlinhadorEsparso (Projeção Canônica Segura)"]

    style Entradas fill:#f9f9f9,stroke:#333,stroke-width:1px
    style Validador fill:#e1f5fe,stroke:#0288d1,stroke-width:1px
    style SUC fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style E1 fill:#ffebee,stroke:#d32f2f,stroke-width:1px
    style E2 fill:#ffebee,stroke:#d32f2f,stroke-width:1px
    style E3 fill:#ffebee,stroke:#d32f2f,stroke-width:1px
```

---

## 4. Consequências Biológicas e Técnicas

### Positivas
* **Eliminação de Corrupção Silenciosa:** Impede que experimentos de imputação cross-dataset operem sobre espaços vazios preenchidos por sentinela falso-positivo.
* **Diagnóstico Rico:** Em caso de falha, fornece tabelas comparativas no console com as primeiras linhas observadas e instruções explícitas de correção.
* **Reprodutibilidade Científica:** Assegura que todos os pipelines construam seus espaços canônicos estritamente sobre matrizes com correspondência gênica verificada.

### Negativas / Restrições
* Requer que os arquivos de features e matrizes AnnData estejam estritamente padronizados antes do início do alinhamento.
