# 🧬 Análise: Teste red hopifield 

---

**📅 Data:** 22/05/2026
**🎯 Objetivo:** [Ex: Harmonização de dados scRNA-seq e correção de batch effects]
**🏷️ Tags:** <span style="color:#007acc; font-weight:bold;">#Transcriptômica</span> | <span style="color:#e65100; font-weight:bold;">#PyTorch</span> | <span style="color:#2e7d32; font-weight:bold;">#Autoencoder</span>

---

## 🛠️ Ambiente e Dependências

| Ferramenta / Bib | Versão | Ambiente | Notas |
| :--- | :--- | :--- | :--- |
| `Scanpy` | 1.9.3 | Python | Manipulação dos objetos h5ad |
| `Seurat` | 4.3.0 | R | Validação cruzada e plots de base |
| `SWeeP` | 1.0.0 | Python | Vetorização das sequências |

---

## 📂 Dados e Pré-processamento

> ⚠️ **Aviso Importante:** Os dados brutos estão no diretório `/data/raw`. Qualquer filtragem ou anotação deve ser salva no formato h5ad em `/data/processed`.

* **Dataset de Origem:** `experimento_1_raw.csv`
* **Limpeza Inicial:** Remoção de células com alta taxa de RNA mitocondrial.
* **Normalização:** Log-normalization aplicada com target sum de 10k.

---

## 🧠 Metodologia e Treinamento

Detalhes da arquitetura ou pipeline utilizado para extração de features e integração.

* **Arquitetura:** Modern Hopfield Network acoplada a um Variational Autoencoder.
* **Hiperparâmetros:** `learning_rate=0.001`, `batch_size=64`, `epochs=150`.
* **Hardware:** GPU T4 (Keras/TensorFlow backend).

---

## 📊 Resultados e Observações

Resumo das métricas de treinamento, validação e observações qualitativas (ex: clusters no UMAP).

<span style="color:#388e3c; font-weight:bold;">✅ Sucesso:</span> O modelo convergiu na época 80. A preservação do sinal biológico ficou acima do baseline nos tipos celulares raros.

<span style="color:#d32f2f; font-weight:bold;">🛑 Problema Encontrado:</span> Ocorreu erro de *Out of Memory* ao tentar projetar os tensores completos na memória. 

<span style="color:#fbc02d; font-weight:bold;">💡 Solução Aplicada:</span> Redução da dimensionalidade inicial usando PCA antes de passar para a camada de atenção.

---

## 🚀 Próximos Passos

* Extrair os *embeddings* finais e rodar o pipeline de clustering.
* Gerar os plots de UMAP comparando o *antes* e *depois* da correção de batch.
* Discutir a matriz de confusão dos tipos celulares com a Jeroniza na próxima reunião.# 🧬 Análise: [Título do Experimento]

---

**📅 Data:** 22/05/2026
**🎯 Objetivo:** [Ex: Harmonização de dados scRNA-seq e correção de batch effects]
**🏷️ Tags:** <span style="color:#007acc; font-weight:bold;">#Transcriptômica</span> | <span style="color:#e65100; font-weight:bold;">#PyTorch</span> | <span style="color:#2e7d32; font-weight:bold;">#Autoencoder</span>

---

## 🛠️ Ambiente e Dependências

| Ferramenta / Bib | Versão | Ambiente | Notas |
| :--- | :--- | :--- | :--- |
| `Scanpy` | 1.9.3 | Python | Manipulação dos objetos h5ad |
| `Seurat` | 4.3.0 | R | Validação cruzada e plots de base |
| `SWeeP` | 1.0.0 | Python | Vetorização das sequências |

---

## 📂 Dados e Pré-processamento

> ⚠️ **Aviso Importante:** Os dados brutos estão no diretório `/data/raw`. Qualquer filtragem ou anotação deve ser salva no formato h5ad em `/data/processed`.

* **Dataset de Origem:** `experimento_1_raw.csv`
* **Limpeza Inicial:** Remoção de células com alta taxa de RNA mitocondrial.
* **Normalização:** Log-normalization aplicada com target sum de 10k.

---

## 🧠 Metodologia e Treinamento

Detalhes da arquitetura ou pipelV# 🧬 Análise: [Título do Experimento]

---

**📅 Data:** 22/05/2026
**🎯 Objetivo:** [Ex: Harmonização de dados scRNA-seq e correção de batch effects]
**🏷️ Tags:** <span style="color:#007acc; font-weight:bold;">#Transcriptômica</span> | <span style="color:#e65100; font-weight:bold;">#PyTorch</span> | <span style="color:#2e7d32; font-weight:bold;">#Autoencoder</span>

---

## 🛠️ Ambiente e Dependências

| Ferramenta / Bib | Versão | Ambiente | Notas |
| :--- | :--- | :--- | :--- |
| `Scanpy` | 1.9.3 | Python | Manipulação dos objetos h5ad |
| `Seurat` | 4.3.0 | R | Validação cruzada e plots de base |
| `SWeeP` | 1.0.0 | Python | Vetorização das sequências |

---

## 📂 Dados e Pré-processamento

> ⚠️ **Aviso Importante:** Os dados brutos estão no diretório `/data/raw`. Qualquer filtragem ou anotação deve ser salva no formato h5ad em `/data/processed`.

* **Dataset de Origem:** `experimento_1_raw.csv`
* **Limpeza Inicial:** Remoção de células com alta taxa de RNA mitocondrial.
* **Normalização:** Log-normalization aplicada com target sum de 10k.

---

## 🧠 Metodologia e Treinamento

Detalhes da arquitetura ou pipeline utilizado para extração de features e integração.

* **Arquitetura:** Modern Hopfield Network acoplada a um Variational Autoencoder.
* **Hiperparâmetros:** `learning_rate=0.001`, `batch_size=64`, `epochs=150`.
* **Hardware:** GPU T4 (Keras/TensorFlow backend).

---

## 📊 Resultados e Observações

Resumo das métricas de treinamento, validação e observações qualitativas (ex: clusters no UMAP).

<span style="color:#388e3c; font-weight:bold;">✅ Sucesso:</span> O modelo convergiu na época 80. A preservação do sinal biológico ficou acima do baseline nos tipos celulares raros.

<span style="color:#d32f2f; font-weight:bold;">🛑 Problema Encontrado:</span> Ocorreu erro de *Out of Memory* ao tentar projetar os tensores completos na memória. 

<span style="color:#fbc02d; font-weight:bold;">💡 Solução Aplicada:</span> Redução da dimensionalidade inicial usando PCA antes de passar para a camada de atenção.

---

## 🚀 Próximos Passos

* Extrair os *embeddings* finais e rodar o pipeline de clustering.
* Gerar os plots de UMAP comparando o *antes* e *depois* da correção de batch.
* Discutir a matriz de confusão dos tipos celulares com a Jeroniza na próxima reunião.ine utilizado para extração de features e integração.

* **Arquitetura:** Modern Hopfield Network acoplada a um Variational Autoencoder.
* **Hiperparâmetros:** `learning_rate=0.001`, `batch_size=64`, `epochs=150`.
* **Hardware:** GPU T4 (Keras/TensorFlow backend).

---

## 📊 Resultados e Observações

Resumo das métricas de treinamento, validação e observações qualitativas (ex: clusters no UMAP).

<span style="color:#388e3c; font-weight:bold;">✅ Sucesso:</span> O modelo convergiu na época 80. A preservação do sinal biológico ficou acima do baseline nos tipos celulares raros.

<span style="color:#d32f2f; font-weight:bold;">🛑 Problema Encontrado:</span> Ocorreu erro de *Out of Memory* ao tentar projetar os tensores completos na memória. 

<span style="color:#fbc02d; font-weight:bold;">💡 Solução Aplicada:</span> Redução da dimensionalidade inicial usando PCA antes de passar para a camada de atenção.

---

## 🚀 Próximos Passos

* Extrair os *embeddings* finais e rodar o pipeline de clustering.
* Gerar os plots de UMAP comparando o *antes* e *depois* da correção de batch.
* Discutir a matriz de confusão dos tipos celulares com a Jeroniza na próxima reunião.