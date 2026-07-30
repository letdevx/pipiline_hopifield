# Agent Rules & Directives — Scientific Second Brain (PARA + OKF + Grafo)

## Jupyter Notebook Editing via Jupytext

To prevent file corruption and formatting errors associated with direct manipulation of `.ipynb` files:

1. **NEVER EDIT `.ipynb` DIRECTLY**: Never edit JSON structure in `.ipynb` files directly (neither manually nor via scripts).
2. **VERIFY & PAIR WITH JUPYTEXT**: Whenever requested to create or edit a Jupyter Notebook (`.ipynb`):
   - Check if a paired `.py` file exists.
   - If not paired, pair it using Jupytext:
     ```bash
     jupytext --set-formats ipynb,py:percent <notebook>.ipynb
     ```
3. **EDIT THE `.py` SCRIPT**: Perform all code edits on the paired `.py` script (percent format).
4. **SYNCHRONIZE NOTEBOOK**: Immediately after modifying the `.py` script, execute synchronization to update the `.ipynb` file:
   ```bash
   jupytext --sync <notebook>.py
   ```

---

## Protocolo de Conhecimento Científico (PARA + OKF + Wikilinks em PT-BR)

Para garantir o acúmulo contínuo de inteligência, reprodutibilidade e contexto arquitetural para pesquisadores humanos e agentes de IA:

1. **CONSULTA PRÉVIA DE ALTA EFICIÊNCIA**: Antes de realizar qualquer pesquisa, refatoração ou planejamento, o agente DEVE consultar o mapa de entrada do Second Brain em `second_brain/index.md` e o guia `second_brain/AGENTS.md` para navegar pelo grafo de conhecimento reutilizando contextos existentes sem sobrecarregar a janela de tokens.
2. **FORMATO OKF & WIKILINKS**: Toda nota criada ou editada DEVE conter o cabeçalho YAML OKF (`tipo`, `tags`, `criado`, `atualizado`, `resumo`) e utilizar a sintaxe de `[[Wikilinks]]` para conectar conceitos atômicos (`second_brain/03_Conhecimento/`), projetos (`second_brain/01_Projetos/`), áreas (`second_brain/02_Areas/`), recursos e ADRs (`second_brain/04_Recursos/adrs/`).
3. **ATOMICIDADE**: Mantenha conceitos teóricos e biológicos isolados como notas atômicas em `second_brain/03_Conhecimento/`. Ao criar um novo conceito, adicione seu link no índice `second_brain/03_Conhecimento/index.md`.
4. **REGISTRO DE DECISÕES DE ARQUITETURA (ADRs)**: Sempre que uma decisão importante de arquitetura for tomada, registre um documento ADR em `second_brain/04_Recursos/adrs/` e vincule-o no índice correspondente.
5. **MANUTENÇÃO DA ARQUITETURA DO SISTEMA**: Mantenha o documento `second_brain/01_Projetos/pipeline_hopfield_expandido/arquitetura_do_sistema.md` sincronizado com os componentes de código em `src/`.
6. **PROIBIÇÃO RIGOROSA DE DIAGRAMAS ASCII**: NUNCA crie diagramas usando apenas texto ASCII ou caixa de caracteres (`┌`, `│`, `└`, `▼`). TODOS os diagramas (arquitetura, fluxo de dados, transformação de matrizes, perfis de memória e hierarquia de componentes) DEVEM ser criados estritamente em **Mermaid** (` ```mermaid `).
7. **IDIOMA MANDATÓRIO (PT-BR)**: Toda a documentação, notas, metadados, comentários em ADRs e arquivos do Second Brain DEVEM ser redigidos estritamente em **Português do Brasil (PT-BR)**.
