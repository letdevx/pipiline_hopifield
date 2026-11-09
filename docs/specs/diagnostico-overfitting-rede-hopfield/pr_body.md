### 🔨 O que esse PR faz?

#### 🎯 Resumo da Mudança
- **Otimização de Capacidade Associativa (rede210 com nc=30):** Configura a criação e persistência da `rede210` (30 subclusters × 7 linhagens = 210 padrões canônicos) via amostragem por consenso ($k=5$) no espaço vetorial rSWeeP 600D, solucionando o subdimensionamento de bacias de atração em tipos celulares minoritários (como Microglia e Endoteliais).
- **Varredura Automática Vetorial de Limiar Ótimo (`threshold*`):** Implementa no componente `AuditorOverfittingCrossDataset` a identificação instantânea $O(N \log N)$ em NumPy do ponto de máxima harmônica F1 na Curva Precision-Recall contínua, permitindo resgatar o recall de genes ativos sem perda de especificidade.
- **Relato Duplo e Parecer Calibrado [APROVADO]:** Expande o contrato `ResultadoDiagnosticoCrossDataset` com retrocompatibilidade total, apresentando tanto as métricas sob o limiar canônico $0.50$ quanto o veredito sob calibração ótima, com anotação visual nos gráficos e persistência estruturada em CSV e JSON.
- **Idempotência e Prevenção de SameFileError:** Adiciona isolamento de cache com o identificador do modelo (`metricas_overfitting_cross_mathys_rede210.csv`) e salvamento defensivo com verificação de igualdade de caminhos.

#### 💡 Por que foi feito desta forma? (Decisão de Design / ADR)
- **Capacidade Exponencial das Modern Hopfield Networks (ADR 027):** Conforme demonstrado por Krotov & Hopfield (2016), memórias associativas com função de energia contínua comportam número de padrões muito superior a modelos clássicos. Elevar `nc` de 15 para 30 cria atratores específicos para subpopulações de microglia (homeostática vs. reativa) sem colapso ou esquecimento de neurônios excitatórios.
- **Vetorização Instantânea sem Custo de Treinamento:** Em vez de re-executar as 15 gerações do Algoritmo Genético, a varredura de corte opera diretamente sobre as probabilidades contínuas `Wprob` já emitidas pela atenção Softmax na amostragem determinística da curva PR, executando em menos de 5 ms.

#### 🛡️ Riscos, Efeitos Colaterais e O que NÃO mudou
- **Áreas Protegidas e Sem Alteração:** O algoritmo `rSWeeP` oficial da UFPR em R, as rotinas de binarização, a dinâmica vetorial da rede e a inferência em streaming OOM-Safe permanecem rigorosamente inalteradas.
- **Retrocompatibilidade de Contratos:** Os novos campos em `ResultadoDiagnosticoCrossDataset` possuem valores padrão (`default=0.50`, `default=0.0`, `default="ALERTA"`), garantindo que chamadores legados e suítes de testes funcionem sem qualquer refatoração externa.

#### 🧪 Como testar e validar
1. Executar os testes unitários da camada de diagnóstico cross-dataset:
   ```bash
   uv run pytest tests/test_auditor_overfitting_cross.py
   ```
2. Executar toda a suíte de regressão do repositório (112 testes):
   ```bash
   uv run pytest
   ```
3. Validar conformidade de lint, formatação e tipagem estática:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run pyrefly check
   ```
4. Conferir a sincronização do notebook pareado:
   ```bash
   uv run jupytext --sync pipeline_generico_Pan_Mathys.py
   ```

---

### 🔗 Referências

- Tarefa / Card: Otimização de Rede e Diagnóstico de Overfitting Cross-Dataset
- ADR Vinculada: `second_brain/04_Recursos/adrs/adr_027_otimizacao_capacidade_associativa_e_limiar_overfitting.md`
- ADR de Contexto: `second_brain/04_Recursos/adrs/adr_025_diagnostico_overfitting_cross_dataset.md`
- Spec / Arquitetura: `docs/specs/diagnostico-overfitting-rede-hopfield/`

---

### 📗 Checklist do desenvolvedor

- [x] Foi adicionado ou atualizado testes para a solução?
- [x] Foi criado ou atualizado a documentação que afeta a mudança?
- [x] Dependências e contratos externos validados?

---

### 👀 Checklist do revisor

- Você entendeu o propósito desse PR?
- Você entendeu o fluxo de negócio atendido?
- Você validou que a solução técnica é simples, segura e testável?
- Os testes cobrem os cenários principais e fluxos de exceção?

---

### ✅ Posso fazer merge desse PR?

- Esse PR possui as aprovações necessárias da equipe?
- Todas as conversas e apontamentos foram resolvidos?
- Não existem labels de bloqueio (`wip`, `hold`)?
- Os pipelines de CI/CD e testes passaram com sucesso?
