# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Treinamento da Rede Hopfield com Particionamento (Treino/Validação/Teste) e KerasTuner
#
# Este notebook implementa o fluxo metodológico completo para treinar uma **Modern Hopfield Network** (Ramsauer et al., 2020) em dados de scRNA-seq, com suporte nativo à **infraestrutura do Google Colab e integração com o Google Drive**:
#
# - **0. Ambiente Google Colab & Google Drive:** Detecção automática, instalação de dependências, sincronização Git e montagem do Drive.
# - **1. Carga e Preparação dos Dados:** Resolução dinâmica de caminhos no Google Drive / local e projeção no subespaço latente rSWeeP 600D.
# - **2. Particionamento Estratificado Tríplice:** Divisão em **60% Treino**, **20% Validação** e **20% Teste Holdout**.
# - **3. Otimização de Hiperparâmetros via KerasTuner:** Busca bayesiana guiada pelo F1-Score Macro com Cache LRU de protótipos SWeeP.
# - **4. Treinamento da Configuração Campeã:** Armazenamento analítico dos protótipos consolidados na memória associativa Hopfield.
# - **5. Avaliação Imparcial Cega no Teste:** Diagnóstico de generalização, matriz de confusão e teste de robustez a ruído sintético (dropout 15%).
# - **6. Persistência Permanente no Google Drive:** Salvamento dos modelos (`.pt`), metadados (`.json`) e gráficos diretamente no Drive.
# - **7. Análise Comparativa Metodológica:** KerasTuner vs. Algoritmo Genético Multiobjetivo (ADR 022).

# %% [markdown]
# ## 0. Configuração de Ambiente, Dependências e Google Drive
#
# Detectamos automaticamente se a execução ocorre no **Google Colab** ou em **Ambiente Local (Windows/Linux)**.
# Caso esteja no Colab, as dependências são instaladas e o repositório é sincronizado no `/content/pipiline_hopifield`.

# %%
from __future__ import annotations

import os
import shutil
import subprocess
import sys

# Configuração prévia obrigatória para Keras 3 operar com backend PyTorch
os.environ.setdefault("KERAS_BACKEND", "torch")


def _is_google_colab() -> bool:
    """Detecta se a execução atual está ocorrendo no Google Colab."""
    return (
        "google.colab" in sys.modules
        or os.path.exists("/content")
        or os.environ.get("COLAB_GPU") is not None
        or os.environ.get("COLAB_RELEASE_TAG") is not None
    )


IN_COLAB = _is_google_colab()

if IN_COLAB:
    print("=" * 70)
    print("  [GOOGLE COLAB] DETECTADO AMBIENTE EM NUVEM")
    print("=" * 70)

    # 1. Instalação de dependências essenciais caso ausentes
    print("[Colab] Verificando e instalando dependências necessárias...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "keras-tuner>=1.4.8",
            "anndata>=0.10.0",
            "scanpy>=1.10.0",
            "polars>=1.0.0",
            "jupytext>=1.16.0",
        ],
        check=False,
    )

    # 2. Sincronização do repositório no Colab
    REPO_URL = "https://github.com/letdevx/pipiline_hopifield.git"
    REPO_BRANCH = "reconstrução_Fujita_com_mlp_funcionando_corretamente"
    DEST_PATH = "/content/pipiline_hopifield"

    if not os.path.exists(DEST_PATH):
        print(f"[Colab] Clonando repositório '{REPO_URL}' em '{DEST_PATH}'...")
        subprocess.run(
            ["git", "clone", "-b", REPO_BRANCH, REPO_URL, DEST_PATH], check=False
        )
    else:
        print(f"[Colab] Atualizando repositório em '{DEST_PATH}'...")
        subprocess.run(["git", "-C", DEST_PATH, "checkout", REPO_BRANCH], check=False)
        subprocess.run(
            ["git", "-C", DEST_PATH, "pull", "origin", REPO_BRANCH], check=False
        )

    # Adiciona repositório ao sys.path
    for p in (DEST_PATH, os.path.join(DEST_PATH, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)

    # 3. Montagem do Google Drive
    try:
        from google.colab import drive  # type: ignore

        if not os.path.exists("/content/drive"):
            print("[Colab] Montando Google Drive em /content/drive...")
            drive.mount("/content/drive")
        else:
            print("[Colab] Google Drive já montado em /content/drive.")
    except Exception as e:
        print(f"[Colab] Aviso durante montagem do Drive: {e}")
else:
    print("[Ambiente] Executando em ambiente local (não-Colab).")

# %% [markdown]
# ### Importações do Pipeline e Definição de Sementes

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

# Importações dos módulos do pipeline local
from src.config import PATH_REFERENCIA
from src.treinamento import (
    AvaliadorHopfield,
    CarregadorDadosFujita,
    HopfieldKerasTuner,
    InjetorPerturbacaoTranscritica,
    ProjetorSWeP,
    remapear_labels_canonicos,
)

# Fixação de semente determinística para reprodutibilidade estrita
SEED: int = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Diretório de persistência local da VM
OUT_DIR = Path("outputs/treinamento_kerastuner")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Diretório de backup permanente no Google Drive (se montado)
OUT_DIR_DRIVE: Path | None = None
if IN_COLAB and os.path.exists("/content/drive/MyDrive"):
    OUT_DIR_DRIVE = Path("/content/drive/MyDrive/outputs_hopfield_kerastuner")
    OUT_DIR_DRIVE.mkdir(parents=True, exist_ok=True)
    print(f"[Google Drive] Pasta de persistência permanente: {OUT_DIR_DRIVE}")

print(f"[Ambiente] Python {sys.version.split()[0]} | PyTorch {torch.__version__}")
print(f"[Ambiente] Diretório de saída local: {OUT_DIR.resolve()}")

# %% [markdown]
# ## 1. Resolução dos Arquivos de Dados no Google Drive / Local
#
# Implementamos um mecanismo de busca multi-caminho para localizar o arquivo de expressão (`pan_anotado.h5ad`) nas estruturas de pastas típicas do Google Drive ou localmente. Caso você possua um arquivo com nome ou caminho específico, informe em `CAMINHO_DRIVE_CUSTOMIZADO`.

# %%
CLASSES_CANONICAS = [1, 2, 3, 4, 5, 6, 7]
NOMES_CLASSES = [
    "Astrocitos (1)",
    "Microglia (2)",
    "Endoteliais (3)",
    "Oligodendrocitos (4)",
    "OPCs (5)",
    "Neuronio Exc (6)",
    "Neuronio Inib (7)",
]

# Variável de personalização rápida: defina aqui caso seu arquivo esteja em outro local do Drive
CAMINHO_DRIVE_CUSTOMIZADO = ""


def resolver_caminho_dados(customizado: str = "") -> str | None:
    """Busca o arquivo de dados em múltiplos caminhos possíveis do Google Drive e local."""
    if customizado and os.path.exists(customizado):
        return customizado

    candidatos = [
        # 1. Caminho configurado no src.config
        PATH_REFERENCIA,
        # 2. Caminhos clássicos sincronizados com o Google Drive ("Meu laptop" / "Teste hop")
        "/content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/pan_anotado.h5ad",
        r"C:\Users\Leticia\Documents\Letworkspace\Teste hop\imputs\pan_anotado.h5ad",
        # 3. Caminhos padrão na raiz do Google Drive ("MyDrive")
        "/content/drive/MyDrive/Teste hop/imputs/pan_anotado.h5ad",
        "/content/drive/MyDrive/imputs/pan_anotado.h5ad",
        "/content/drive/MyDrive/pan_anotado.h5ad",
        # 4. Caminho interno clonado no repositório
        "/content/pipiline_hopifield/imputs/pan_anotado.h5ad",
        os.path.join(os.getcwd(), "imputs", "pan_anotado.h5ad"),
    ]

    for cand in candidatos:
        if cand and os.path.exists(cand):
            return cand

    # Se estiver no Colab e Drive montado, tenta uma varredura recursiva rápida na pasta do Drive
    if IN_COLAB and os.path.exists("/content/drive/MyDrive"):
        import glob

        achados = glob.glob("/content/drive/**/pan_anotado*.h5ad", recursive=True)
        if achados:
            return achados[0]

    return None


caminho_dados_resolvido = resolver_caminho_dados(CAMINHO_DRIVE_CUSTOMIZADO)

if caminho_dados_resolvido:
    print("=" * 70)
    print(f"  [DADOS CARREGADOS] ARQUIVO LOCALIZADO: {caminho_dados_resolvido}")
    print("=" * 70)
    tamanho_mb = os.path.getsize(caminho_dados_resolvido) / (1024 * 1024)
    print(f"  • Tamanho do arquivo: {tamanho_mb:.2f} MB")

    carregador = CarregadorDadosFujita(caminho_dados_resolvido, formato="h5ad")
    carregador.carregar()
    carregador.binarizar(limiar=0.0)

    W0 = carregador.W0
    assert W0 is not None
    assert carregador.adata is not None

    labels_raw = carregador.adata.obs.get("cell_type_code", None)
    if labels_raw is not None:
        labels_arr, _ = remapear_labels_canonicos(labels_raw.to_numpy())
    else:
        labels_arr = np.ones(W0.shape[0], dtype=int)

    # Projeção no espaço latente SWeeP 600D
    print("[SWeeP] Projetando matriz de referência para 600 dimensões...")
    projetor = ProjetorSWeP(d_target=600, seed=SEED)
    Wswp = projetor.projetar(W0)
    print(f"[Sucesso] W0: {W0.shape} | Wswp: {Wswp.shape}")
else:
    print("=" * 70)
    print(
        "  [AVISO] ARQUIVO 'pan_anotado.h5ad' NÃO ENCONTRADO NO GOOGLE DRIVE OU LOCAL"
    )
    print("=" * 70)
    print("Caminhos verificados:")
    print(
        "  - /content/drive/Othercomputers/Meu laptop/Documents/Letworkspace/Teste hop/imputs/pan_anotado.h5ad"
    )
    print("  - /content/drive/MyDrive/pan_anotado.h5ad")
    print("  - inputs locais do repositório")
    print(
        "\n[Contingência] Gerando matriz sintética de alta fidelidade para execução imediata..."
    )

    n_celulas_sim = 2000
    n_genes_sim = 500
    n_dims_latente = 600
    rng = np.random.default_rng(SEED)

    pesos_classes = [0.12, 0.04, 0.03, 0.25, 0.06, 0.35, 0.15]
    labels_arr = rng.choice(CLASSES_CANONICAS, size=n_celulas_sim, p=pesos_classes)

    assinaturas = rng.binomial(
        1, 0.35, size=(len(CLASSES_CANONICAS), n_genes_sim)
    ).astype(np.float32)
    matriz_expressao = np.zeros((n_celulas_sim, n_genes_sim), dtype=np.float32)
    for idx_c, cls in enumerate(CLASSES_CANONICAS):
        masc = labels_arr == cls
        ruido = rng.binomial(1, 0.08, size=(int(masc.sum()), n_genes_sim))
        matriz_expressao[masc] = np.bitwise_xor(
            assinaturas[idx_c].astype(int), ruido
        ).astype(np.float32)

    W0 = sp.csr_matrix(matriz_expressao)
    Wswp = rng.normal(0.0, 1.0, size=(n_celulas_sim, n_dims_latente)).astype(np.float32)
    for cls in CLASSES_CANONICAS:
        Wswp[labels_arr == cls] += rng.normal(
            float(cls), 0.5, size=(int((labels_arr == cls).sum()), n_dims_latente)
        )

    print(
        f"[Dados Sintéticos] Células: {n_celulas_sim} | Genes: {n_genes_sim} | Dimensão SWeeP: {n_dims_latente}"
    )

# %% [markdown]
# ## 2. Particionamento Estratificado Tríplice (Treino / Validação / Teste)
#
# Para scRNA-seq, o particionamento DEVE ser estritamente estratificado por linhagem celular:
# - **Treino (60%):** Usado para extrair centróides representativos no espaço rSWeeP e memorizar na Hopfield.
# - **Validação (20%):** Usado pelo KerasTuner para computar o F1-Score Macro e guiar a busca de hiperparâmetros.
# - **Teste Holdout (20%):** Conjunto cego e isolado, avaliado uma única vez ao final.

# %%
# Divisão Etapa 1: 60% Treino e 40% Temporário
split_1 = StratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=SEED)
idx_train, idx_temp = next(split_1.split(np.zeros(len(labels_arr)), labels_arr))

# Divisão Etapa 2: Divide os 40% temporários em 50% Validação (20% total) e 50% Teste (20% total)
labels_temp = labels_arr[idx_temp]
split_2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
idx_val_rel, idx_test_rel = next(split_2.split(np.zeros(len(labels_temp)), labels_temp))

idx_val = idx_temp[idx_val_rel]
idx_test = idx_temp[idx_test_rel]

print("=" * 65)
print("  DISTRIBUIÇÃO DAS PARTIÇÕES ESTRATIFICADAS")
print("=" * 65)
print(f"  • Treino    (60%): {len(idx_train):>6,d} células")
print(f"  • Validação (20%): {len(idx_val):>6,d} células")
print(f"  • Teste     (20%): {len(idx_test):>6,d} células")
print(f"  • Total          : {len(labels_arr):>6,d} células")

# Separação das matrizes
W0_train = W0[idx_train]
Wswp_train = Wswp[idx_train]
y_train = labels_arr[idx_train]

W0_val = W0[idx_val]
Wswp_val = Wswp[idx_val]
y_val = labels_arr[idx_val]

W0_test = W0[idx_test]
Wswp_test = Wswp[idx_test]
y_test = labels_arr[idx_test]

# %% [markdown]
# ### Inspeção Visual da Estratificação
#
# Confirmamos graficamente que a proporção das 7 classes (incluindo classes raras como a Microglia e Endoteliais) é rigorosamente preservada nas três partições.

# %%
contagens = []
for nome_part, y_part in [
    ("Treino (60%)", y_train),
    ("Validação (20%)", y_val),
    ("Teste (20%)", y_test),
]:
    for cls in CLASSES_CANONICAS:
        pct = float((y_part == cls).mean() * 100.0)
        contagens.append(
            {"Partição": nome_part, "Classe": f"Classe {cls}", "Percentual": pct}
        )

df_particoes = pd.DataFrame(contagens)

fig, ax = plt.subplots(figsize=(10, 4.5))
sns.barplot(
    data=df_particoes,
    x="Classe",
    y="Percentual",
    hue="Partição",
    palette="Blues_d",
    ax=ax,
)
ax.set_title(
    "Preservação da Proporção de Linhagens Celulares nas Partições",
    fontsize=12,
    fontweight="bold",
)
ax.set_ylabel("Frequência Relativa (%)")
ax.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
fig.savefig(OUT_DIR / "distribuicao_estratificada_classes.png", dpi=300)
if OUT_DIR_DRIVE:
    fig.savefig(OUT_DIR_DRIVE / "distribuicao_estratificada_classes.png", dpi=300)
plt.show()

# %% [markdown]
# ## 3. Configuração e Execução do KerasTuner
#
# Instanciamos o `HopfieldKerasTuner` com o algoritmo **Bayesian Optimization** (Processos Gaussianos).
# O tuner pesquisa o espaço não-linear:
# - `nc`: número de protótipos/centróides por linhagem biológica [10 .. 35];
# - `k_vizinhos`: vizinhança de consenso local [1 .. 7];
# - `beta`: temperatura inversa da atenção Softmax [0.2 .. 4.0];
# - `threshold`: limiar de ativação binarizada [0.1 .. 0.7];
# - `normalize`: similaridade cosseno esférica unitária (True/False);
# - `scale_by_dim`: escalonamento dimensional por raiz quadrada de D (True/False);
# - `estrategia`: K-Means Fixo vs Dinâmico.
#
# Graças ao **Cache LRU de Protótipos**, o agrupamento K-Means só é executado quando a combinação `(nc, k_vizinhos, estrategia)` é inédita!

# %%
NUM_TRIALS = 15
DIR_TUNER = OUT_DIR / "kt_hopfield_trials"

tuner = HopfieldKerasTuner(
    w0=W0_train,
    wswp=Wswp_train,
    labels=y_train,
    x_val=W0_val,
    y_val=y_val,
    classes=CLASSES_CANONICAS,
    algorithm="bayesian",
    objective="val_f1_macro",
    max_trials=NUM_TRIALS,
    directory=str(DIR_TUNER),
    project_name="estudo_hopfield_scrna",
    overwrite=True,
    seed=SEED,
)

print(f"[KerasTuner] Iniciando busca bayesiana com {NUM_TRIALS} trials...")
tuner.search()

# %% [markdown]
# ### Painel Gráfico de Convergência da Otimização
#
# Inspecionamos a evolução do F1-Score Macro acumulado e a estabilidade da acurácia ao longo dos trials.

# %%
fig_conv = tuner.gerar_painel_convergencia(
    path_png=OUT_DIR / "convergencia_kerastuner.png"
)
if OUT_DIR_DRIVE:
    fig_conv.savefig(OUT_DIR_DRIVE / "convergencia_kerastuner.png", dpi=300)
plt.show()

# Salva histórico completo em JSON (Local e no Google Drive)
tuner.exportar_historico_json(OUT_DIR / "historico_trials_kerastuner.json")
if OUT_DIR_DRIVE:
    tuner.exportar_historico_json(OUT_DIR_DRIVE / "historico_trials_kerastuner.json")

# %% [markdown]
# ## 4. Treinamento da Rede Hopfield com a Configuração Campeã
#
# Recuperamos os hiperparâmetros de máxima aptidão encontrados pelo KerasTuner e construímos o modelo final memorizando os protótipos do conjunto de treino.

# %%
melhor_cfg = tuner.obter_melhor_configuracao()

print("=" * 65)
print("  CONFIGURAÇÃO CAMPEÃ ENCONTRADA PELO KERASTUNER")
print("=" * 65)
for k, v in melhor_cfg.items():
    print(f"  • {k:<15s}: {v}")

# Materializa a rede campeã
rede_campea, padroes_campeoes, meta_campeao, _ = tuner.construir_melhor_modelo()

# Persistência estruturada do modelo treinado (Local)
path_pt = OUT_DIR / "modelo_hopfield_campeao.pt"
path_meta = OUT_DIR / "metadados_hopfield_campeao.json"

rede_campea.salvar_com_metadados(
    path_pt=path_pt,
    path_meta=path_meta,
    meta=meta_campeao,
    classes=CLASSES_CANONICAS,
    nc=melhor_cfg["nc"],
)

# Cópia permanente para o Google Drive
if OUT_DIR_DRIVE:
    shutil.copy2(path_pt, OUT_DIR_DRIVE / "modelo_hopfield_campeao.pt")
    shutil.copy2(path_meta, OUT_DIR_DRIVE / "metadados_hopfield_campeao.json")
    print(f"[Google Drive] Modelo salvo com sucesso em: {OUT_DIR_DRIVE}")

# %% [markdown]
# ## 5. Avaliação Imparcial Cega no Conjunto de Teste (Holdout)
#
# Avaliamos o modelo final treinado exclusivamente no conjunto de teste cego (não-visto nem pelo treino nem pelo tuner), garantindo total ausência de vazamento de dados.

# %%
print(
    "[Avaliação Teste] Executando inferência Softmax Class Pooling em lote OOM-Safe..."
)
att_test = rede_campea.compute_attention_weights(W0_test, batch_size=2048)

avaliador_test = AvaliadorHopfield(
    padroes=padroes_campeoes,
    classes=CLASSES_CANONICAS,
    nc=melhor_cfg["nc"],
    meta=meta_campeao,
)
avaliador_test.avaliar_por_atencao(att_test, y_test)

y_pred_test = avaliador_test.y_pred
assert y_pred_test is not None

print("\n" + "=" * 65)
print("  RELATÓRIO DE CLASSIFICAÇÃO NO CONJUNTO DE TESTE (HOLDOUT CEGO)")
print("=" * 65)
print(classification_report(y_test, y_pred_test, target_names=NOMES_CLASSES, digits=4))

# %% [markdown]
# ### Matriz de Confusão no Teste

# %%
cm = confusion_matrix(y_test, y_pred_test, labels=CLASSES_CANONICAS)
cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)

fig_cm, ax_cm = plt.subplots(figsize=(8.5, 7))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    xticklabels=NOMES_CLASSES,
    yticklabels=NOMES_CLASSES,
    cbar=True,
    ax=ax_cm,
)
ax_cm.set_title(
    f"Matriz de Confusão Normalizada (Teste Holdout - F1={avaliador_test.f1_macro:.4f})",
    fontsize=11,
    fontweight="bold",
)
ax_cm.set_ylabel("Classe Verdadeira")
ax_cm.set_xlabel("Classe Predita (Softmax Class Pooling)")
plt.xticks(rotation=40, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

fig_cm.savefig(OUT_DIR / "matriz_confusao_teste.png", dpi=300)
if OUT_DIR_DRIVE:
    fig_cm.savefig(OUT_DIR_DRIVE / "matriz_confusao_teste.png", dpi=300)
plt.show()

# %% [markdown]
# ## 6. Teste de Estresse sob Ruído de Dropout Sintético (15%)
#
# Para certificar a profundidade das bacias de atração da rede Hopfield frente a limitações técnicas de scRNA-seq (perda estocástica de transcritos / dropouts), injetamos 15% de perturbação no conjunto de teste.

# %%
injetor = InjetorPerturbacaoTranscritica()
W0_test_ruido = injetor.aplicar_dropout(W0_test, taxa=0.15, seed=SEED)

att_ruido = rede_campea.compute_attention_weights(W0_test_ruido, batch_size=2048)
avaliador_ruido = AvaliadorHopfield(
    padroes=padroes_campeoes,
    classes=CLASSES_CANONICAS,
    nc=melhor_cfg["nc"],
    meta=meta_campeao,
)
avaliador_ruido.avaliar_por_atencao(att_ruido, y_test)

f1_base = float(avaliador_test.f1_macro if avaliador_test.f1_macro is not None else 0.0)
f1_ruido = float(
    avaliador_ruido.f1_macro if avaliador_ruido.f1_macro is not None else 0.0
)
queda_pct = ((f1_base - f1_ruido) / max(f1_base, 1e-6)) * 100.0

print("=" * 65)
print("  TESTE DE ESTRESSE: ROBUSTEZ A RUÍDO DE DROPOUT (15%)")
print("=" * 65)
print(f"  • F1 Macro Holdout (Original) : {f1_base:.4f}")
print(f"  • F1 Macro Holdout (Ruído 15%): {f1_ruido:.4f}")
print(f"  • Degradação Relativa         : {queda_pct:.2f}%")
if queda_pct <= 10.0:
    print(
        "  • Parecer de Robustez         : [APROVADO] - Bacias de atração profundas e resilientes."
    )
else:
    print(
        "  • Parecer de Robustez         : [ALERTA] - Sensibilidade moderada ao ruído técnico."
    )

# %% [markdown]
# ## 7. Análise Comparativa Metodológica
#
# | Característica | Otimização Bayesiana (KerasTuner) | Algoritmo Genético Multiobjetivo (ADR 022) |
# | :--- | :--- | :--- |
# | **Paradigma de Busca** | Processos Gaussianos / Exploração baseada em Incerteza | Busca Evolutiva Estocástica (Crossover, Mutação e Elitismo) |
# | **Função Objetivo** | Escalar direta (`val_f1_macro`) | Multiobjetivo com Barreira (F1 + Ruído - Gap - Quimeras - Saturação) |
# | **Blindagem de Quimeras** | Indireta (via F1 na validação) | Explícita (penalização severa de coativação de marcadores) |
# | **Curva de Aprendizado** | Rápida convergência com poucos trials (15-25) | Ampla exploração combinatória (população de 16-24 indivíduos por 8-15 gerações) |
# | **Reuso de Padrões** | Cache LRU de Centróides SWeeP | Cache LRU de Centróides SWeeP |
# | **Casos de Uso Ideais** | Calibração rápida em notebooks interativos e grids limpos | Otimização científica profunda em pipelines de produção |
