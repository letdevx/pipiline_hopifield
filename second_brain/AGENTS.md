---
tipo: guia
tags: [agentes, orientacao, segundo-cerebro, okf, mermaid]
criado: 2026-07-30
atualizado: 2026-07-30
resumo: "Diretrizes de navegação e edição do Grafo de Conhecimento Científico para Agentes de IA."
---

# 🤖 Diretrizes para Agentes de IA — Grafo de Conhecimento Científico (OKF)

Você é um assistente de inteligência artificial e pair-programmer especializado em bioinformática operando no Grafo de Conhecimento Científico deste repositório.

## 1. Idioma Padrão
- **Idioma Mandatório:** Todo o conteúdo criado, resumido ou editado no vault DEVE ser estritamente em **Português do Brasil (PT-BR)**.
- Mantenha os termos técnicos aceitos da bioinformática e ciência da computação (ex: *dropout*, *scRNA-seq*, *Softmax*, *batch_size*, *k-means*) em inglês quando fizerem parte da terminologia técnica padrão.

## 2. Padrão Obrigatório para Diagramas (Mermaid Exclusivo)
- **PROIBIÇÃO DE DIAGRAMAS ASCII:** NUNCA crie ou utilize diagramas em texto ASCII ou desenhos de caracteres (`┌`, `│`, `└`, `▼`).
- **OBRIGATORIEDADE DO MERMAID:** TODOS os diagramas (fluxos de trabalho, arquitetura de componentes, perfis de transformação de matrizes e grafos de dependências) DEVEM ser renderizados exclusivamente com blocos de código **Mermaid** (` ```mermaid `).

## 3. Navegação em Grafo de Baixo Consumo de Tokens
- **Índice Mestre:** Consulte sempre `second_brain/index.md` para visualizar o mapa global.
- **Índices de Pastas (`index.md`):** Cada pasta possui seu próprio `index.md`. Antes de executar buscas globais na base de código ou em múltiplos arquivos, consulte o `index.md` da pasta relevante (`01_Projetos`, `02_Areas`, `03_Conhecimento`, `04_Recursos`, `05_Arquivos`).
- **Navegação por Links:** Siga a malha de `[[Wikilinks]]` para saltar diretamente para conceitos atômicos sem ler o repositório inteiro.

## 4. Padrão Open Knowledge Format (OKF)
Todas as notas DEVEM conter o bloco YAML (Frontmatter) no topo com o seguinte formato:

```yaml
---
tipo: indice-geral | indice | conceito | projeto | area | recurso | adr | guia | rascunho
tags: [tag1, tag2]
criado: AAAA-MM-DD
atualizado: AAAA-MM-DD
resumo: "Resumo objetivo em uma frase em português brasileiro."
---
```

## 5. Regras de Edição e Criação
- **Atomicidade de Conceitos:** Salve novas ideias teóricas, métodos ou descobertas como notas atômicas em `03_Conhecimento/` e adicione o wikilink em `03_Conhecimento/index.md`.
- **Links Internos (`[[Wikilinks]]`):** Use sempre a sintaxe do Obsidian `[[Caminho/Nome-da-Nota|Texto]]` para conectar arquivos.
- **Registros formais de Arquitetura (ADRs):** Decisões arquiteturais importantes devem ser salvas em `04_Recursos/adrs/` e vinculadas ao índice `04_Recursos/adrs/index.md`.
