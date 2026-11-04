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
8. **PROIBIÇÃO RIGOROSA DE SINTAXE LATEX MATH (`$$...$$` OU `$...$`):** NUNCA utilize notação matemática em LaTeX com delimitadores (`$$ ... $$` ou `$ ... $`) em respostas, documentações ou notas, pois essa sintaxe não renderiza na interface do Google Antigravity IDE. Para descrever formulações matemáticas, operações vetoriais e transformações matriciais, utilize SEMPRE alternativas 100% compatíveis com Markdown padrão:
   - **Código inline ou blocos de código:** ex.: `proj = W × Q (40.913 células × 600 dimensões)`;
   - **Caracteres Unicode legíveis:** símbolos diretos de texto como `×`, `∈`, `≈`, `→`, `·`, subscritos/sobrescritos e dimensões explícitas (ex.: `Wswp = W0 × R_expandido`, matriz de `40.913 × 600`);
   - **Tabelas ou listas estruturadas:** para mapear dimensões, variáveis e operações matriciais;
   - **Diagramas Mermaid:** para ilustrar etapas e transformações dimensionais.
9. **OBRIGATORIEDADE IRREVOGÁVEL DO ALGORITMO rSWeeP (PACOTE R OFICIAL DA UFPR) E PROIBIÇÃO DE FALLBACKS:**
   - Esta pesquisa científica fundamenta-se obrigatoriamente na metodologia SWeeP desenvolvida pelo laboratório AIBIALab/UFPR (De Pierri et al., 2020). É TERMINANTEMENTE PROIBIDO substituir o algoritmo do pacote R `rSWeeP` por qualquer aproximação, projeção aleatória direta, decomposição QR sintética em Python ou qualquer outro método, mesmo diante de lentidão ou limitações de memória. Substituir o SWeeP não é uma opção e nunca deve ser considerado.
   - A projeção DEVE ser executada estritamente através do pacote R `rSWeeP` via script R, aplicando a base gerada por `orthBase()` e projetada por `SWeeP()`, com suporte a congelamento de base em arquivo `.rds` (`path_orthbase`).
   - Caso o ambiente R, dependências ou a execução falhem, o sistema DEVE disparar `RuntimeError` imediatamente com o log do R (`stderr`) e interromper o pipeline. Todo e qualquer fallback ou mecanismo de bypass está expressamente proibido.

---

## Padrões de Ferramental e Ciclo de Qualidade (uv, pytest, ruff, pyrefly)

Para garantir integridade, desempenho, reprodutibilidade e consistência estrita de tipos em toda a base de código:

1. **GERENCIAMENTO EXCLUSIVO VIA `uv`**:
   - NUNCA utilize `pip install` ou manipulação manual de ambientes.
   - Toda execução de scripts, comandos, testes e ferramentas de qualidade DEVE ser realizada prefixada por `uv run` (ex.: `uv run pytest`, `uv run ruff check .`, `uv run pyrefly check`, `uv run python script.py`).
   - Novas dependências primárias devem ser adicionadas via `uv add <pacote>` (ou declaradas no `pyproject.toml`) e sincronizadas com `uv sync` / `uv lock`. Nunca adicione dependências transitivas ao `pyproject.toml`.

2. **LINTING E FORMATAÇÃO COM `ruff`**:
   - Utilize o Ruff como o linter e formatador exclusivo do projeto.
   - O código DEVE estar sempre aderente às regras configuradas em `pyproject.toml` (`[tool.ruff]`).

3. **ANOTAÇÕES E CHECAGEM DE TIPOS COM `pyrefly` E `pyright`**:
   - Todas as funções, classes, métodos e módulos de `src/` DEVEM conter anotações estritas de tipo (`Type Hints`) e docstrings completas (padrão NumPy).
   - Utilize `uv run pyrefly check` e `uv run pyright` para validar ausência de violações de tipo antes de concluir tarefas.

4. **TESTES AUTOMATIZADOS COM `pytest`**:
   - A suíte de testes em `tests/` DEVE manter 100% de taxa de sucesso (`uv run pytest`).
   - Ao adicionar novas funcionalidades ou refatorar componentes existentes em `src/`, crie ou atualize os testes unitários correspondentes.

5. **VERIFICAÇÃO OBRIGATÓRIA PÓS-IMPLEMENTAÇÃO**:
   - Imediatamente após qualquer implementação, alteração de código ou refatoração, o agente DEVE executar o ciclo completo de verificação:
     1. **Testes Unitários:** `uv run pytest`
     2. **Lint e Formatação:** `uv run ruff check .` e `uv run ruff format --check .`
     3. **Tipagem Estática:** `uv run pyrefly check` (e/ou `uv run pyright`)
   - NUNCA finalize uma tarefa deixando testes quebrados ou erros de formatação/tipagem pendentes.

---

## Protocolo de Escalabilidade e Boas Práticas em Bioinformática (RNA-Seq Single-Cell & Hardware Modesto)

Para viabilizar a execução do pipeline científico mesmo em hardware modesto (ex.: laptops locais ou instâncias de nuvem com memória limitada) e prevenir falhas de estouro de memória (*Out Of Memory* — OOM) ou tempos de processamento excessivos:

1. **AVALIAÇÃO ASSINTÓTICA PRÉVIA DE COMPLEXIDADE**:
   - Sempre que o desenvolvedor solicitar a implementação ou alteração de uma funcionalidade, algoritmo ou etapa de processamento, o agente DEVE avaliar previamente a escalabilidade da solução proposta.
   - Verifique a complexidade de tempo `O(...)` e de espaço/memória `O(...)` em relação à escala real dos dados biológicos (ex.: dezenas de milhares de células `N` × dezenas de milhares de genes `M`).

2. **PRESERVAÇÃO ESTRITA DE MATRIZES ESPARSAS E STREAMING**:
   - No domínio de scRNA-seq, matrizes de contagem possuem esparsidade típica superior a 90-95%. É expressamente PROIBIDO materializar matrizes densas gigantes na memória RAM (ex.: chamadas a `.toarray()`, `.todense()` ou `np.array()` sobre a matriz global).
   - Mantenha os dados em formatos esparsos compactos (`scipy.sparse.csr_matrix` ou `csc_matrix`) e utilize processamento em lote (*batching* / *streaming*) ou *backed mode* do AnnData (`h5ad` / `zarr`).

3. **I/O OOM-SAFE E VETO A ARQUIVOS INTERMEDIÁRIOS GIGANTES**:
   - É proibido gerar arquivos intermediários gigantes em disco (ex.: matrizes MTX descompactadas de dezenas de gigabytes) quando operações analíticas equivalentes, projeções diretas em memória ou decomposições algébricas puderem ser empregadas.
   - O pipeline deve ser otimizado para viabilizar execução em hardware modesto sem saturação de disco ou lentidão por escrita e leitura de I/O desnecessárias.

4. **EQUIVALÊNCIA ESTRITA DE COMPORTAMENTO**:
   - Toda proposta de otimização DEVE preservar rigorosamente o comportamento funcional, biológico e numérico esperado pela solução original. Nenhuma aproximação que degrade a exatidão dos resultados deve ser aplicada sem fundamentação matemática comprovada.

5. **DIÁLOGO DIDÁTICO E TOMADA DE DECISÃO INTERATIVA VIA `ask_question`**:
   - Caso o agente identifique que o dev propôs uma solução *brute force*, ingênua ou não escalável:
     1. **Explicação Didática:** Explique por que a solução do dev não é uma boa prática para dados de bioinformática/scRNA-seq e apresente a intuição algorítmica de fundo de forma clara e acessível, para que o dev compreenda mesmo sem domínio prévio do ferramental.
     2. **Proposta Escalável:** Apresente a abordagem escalável alternativa demonstrando seus ganhos de tempo e espaço.
     3. **Decisão Interativa:** Utilize a ferramenta nativa `ask_question` para permitir ao dev escolher conscientemente:
        - `(Recomendado) Adotar a abordagem otimizada e escalável sugerida`
        - `Manter a abordagem original conforme solicitado`
     4. **Fluxo Pós-Escolha:**
        - **Se o dev escolher a abordagem otimizada sugerida:** O agente DEVE obrigatoriamente apresentar um plano formal como artefato interativo com `RequestFeedback: true` (renderizando o botão "Proceed" no Antigravity IDE) antes de iniciar qualquer implementação no código.
        - **Se o dev escolher manter a abordagem original:** Respeite imediatamente a escolha do dev e implemente a abordagem solicitada sem insistência, adicionando apenas comentários técnicos explicativos no código caso pertinente.


