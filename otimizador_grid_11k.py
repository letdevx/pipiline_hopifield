import os, sys, gc
import numpy as np, pandas as pd, polars as pl
import anndata as ad
from sklearn.metrics import accuracy_score, f1_score
import time

SRC_DIR = os.path.abspath(os.path.join(os.getcwd(), 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import (
    PATH_FEATURES_F, PATH_FEATURES_M, PATH_LABELS_F, PATH_LABELS_M, PATH_SWEEP_F,
    OUT_ALINHAMENTO, OUT_BINARIZACAO, OUT_TOP_GENES
)
from alinhamento.alinhador import Alinhador
from alinhamento.leitor_features import LeitorFeatures
from treinamento.extrator_padroes import EstrategiaKMeansDinamico, ExtratorPadroesSubcluster
from treinamento.hopfield import ModernHopfieldNetwork
from treinamento.avaliador_hopfield import AvaliadorHopfield

def main():
    print("=== GRID SEARCH 11k GENES ===")
    
    # 1. Obter índices 11k
    print("Lendo Features...")
    leitor = LeitorFeatures(PATH_FEATURES_F, PATH_FEATURES_M)
    leitor.ler()
    path_genes_target = os.path.join(OUT_TOP_GENES, 'genes_expandidos_frequentes.csv')
    genes_ordenados = pd.read_csv(path_genes_target)['gene'].tolist()
    gene_alvo_idx = {eid: i for i, eid in enumerate(genes_ordenados)}
    
    # 2. Carregar e Alinhar Fujita (como no notebook)
    print("Recriando Matriz Fujita...")
    path_bin_f = os.path.join(OUT_BINARIZACAO, 'matrizFiltradaeNormalizadaF', 'matrizBinarizadaM.h5ad')
    adataf = ad.read_h5ad(path_bin_f)
    alinhador = Alinhador("fake", "fake", OUT_ALINHAMENTO, leitor.map_f, leitor.map_m, gene_alvo_idx, genes_ordenados)
    adata_fujita_exp = alinhador._alinhar_direto(adataf, leitor.map_f, fill_value=0.0)
    W0_fujita = adata_fujita_exp.X
    import scipy.sparse as sp
    if sp.issparse(W0_fujita): W0_fujita = W0_fujita.toarray()
    del adataf, adata_fujita_exp; gc.collect()
    
    Wswp_f = pd.read_csv(PATH_SWEEP_F).to_numpy(dtype=np.float32)
    labels_f = np.loadtxt(PATH_LABELS_F, dtype=int).ravel()
    
    # Remapeamento Padrão de Classes do Fujita
    clo_f = labels_f.copy()
    clo_f[~np.isin(clo_f, [1, 3, 4, 5, 6, 7, 0])] = 2
    
    # 3. Carregar subset do Mathys para testar a Acurácia
    print("Carregando Subset do Mathys (Ground Truth e Features)...")
    path_mathys_11k = os.path.join(OUT_ALINHAMENTO, 'Mathys_Binarizado_Alinhado_Expandido.txt')
    # Ler apenas primeiras 2000 linhas para teste rápido
    Mathys_05_Teste = pd.read_csv(path_mathys_11k, nrows=2000).to_numpy(dtype=np.float32)
    
    labels_m = np.loadtxt(PATH_LABELS_M, dtype=int).ravel()[:2000]
    clo_m = labels_m.copy()
    clo_m[~np.isin(clo_m, [1, 3, 4, 5, 6, 7, 0])] = 2
    
    CLASSES = [1, 2, 3, 4, 5, 6, 7]
    
    # === GRID SEARCH PARAMS ===
    NC_TESTES = [10, 30]           # Variações de Memória (Densidade)
    BETA_TESTES = [10.0, 30.0, 50.0]     # Variações de Foco (Winner Takes all)
    ITERS_TESTES = [1, 3]            # Tempo de Convergência da Memória
    
    resultados = []
    
    for nc in NC_TESTES:
        print(f"\n>>>> TESTANDO DENSIDADE DE MEMÓRIA: nc={nc} <<<<")
        estr_kmeans = EstrategiaKMeansDinamico(k_range=[nc], seed=42)
        extrator = ExtratorPadroesSubcluster(W0=W0_fujita, labels=clo_f, classes=CLASSES, k=1, estrategia=estr_kmeans)
        extrator.extrair(Wswp_f)
        
        perf35_11k = extrator.padroes
        print(f"Extraídos {perf35_11k.shape[0]} protótipos de memória.")
        
        for beta in BETA_TESTES:
            for iters in ITERS_TESTES:
                try:
                    rede = ModernHopfieldNetwork(beta=beta, n_iters=iters, binary=True, threshold=0.0)
                    rede.store(perf35_11k)
                    
                    t0 = time.time()
                    Mathys_Output = rede.retrieve(Mathys_05_Teste, batch_size=1024)
                    t_inferencia = time.time() - t0
                    
                    # Avaliar (fazendo manual para extrair os números brutos mais rápido)
                    avaliador = AvaliadorHopfield(padroes=perf35_11k, classes=CLASSES, nc=nc, meta=extrator.meta)
                    
                    # O avaliador usa Mathys_Output e clo_m (true)
                    avaliador.avaliar(Mathys_Output, clo_m)
                    
                    acc = avaliador.acuracia
                    f1 = avaliador.f1_macro
                    rec = avaliador.taxa_reconstrucao
                    
                    res = {
                        "NC": nc,
                        "Beta": beta,
                        "Iters": iters,
                        "Acc": acc,
                        "F1_Macro": f1,
                        "Recon_Exata_Proto": rec,
                        "Time_Secs": round(t_inferencia, 2)
                    }
                    print(f"  [nc={nc:2d} | beta={beta:4.1f} | iters={iters}] Acc: {acc:.4f}  F1: {f1:.4f}  Time: {t_inferencia:.1f}s")
                    resultados.append(res)
                except Exception as e:
                    print(f"Erro na combinacao nc={nc}, beta={beta}, iters={iters}: {e}")
                    
    print("\n\n==== RESULTADO FINAL DO GRID SEARCH ====")
    df = pd.DataFrame(resultados).sort_values("Acc", ascending=False)
    print(df.to_string(index=False))
    
    df.to_csv("resultados_grid_search_11k.csv", index=False)
    print("Salvo em resultados_grid_search_11k.csv")

if __name__ == "__main__":
    main()
