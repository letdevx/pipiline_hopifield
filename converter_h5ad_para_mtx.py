"""
Script para conversão da matriz alinhada e binarizada (.h5ad) para o formato Matrix Market (.mtx).
Exporta adicionalmente os metadados de células (obs) e genes (var) em CSV para preservar a identificação das linhas e colunas.
"""

import os
import sys
import time
import anndata as ad
from scipy.io import mmwrite
from scipy.sparse import csr_matrix, issparse

# Configuração para evitar erros de caractere especial/encoding no console Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def converter_h5ad_para_mtx(path_h5ad: str, dir_saida: str = None, sobrescrever: bool = False):
    """
    Converte um arquivo AnnData (.h5ad) para o formato Matrix Market (.mtx).
    
    Args:
        path_h5ad (str): Caminho absoluto para o arquivo .h5ad de entrada.
        dir_saida (str, optional): Pasta onde os arquivos serão salvos. Se None, salva na mesma pasta do arquivo de entrada.
        sobrescrever (bool, optional): Se True, recria os arquivos mesmo se já existirem.
    """
    print(f"[{time.strftime('%X')}] [INFO] Carregando arquivo H5AD: {path_h5ad}")
    if not os.path.exists(path_h5ad):
        raise FileNotFoundError(f"Arquivo não encontrado: {path_h5ad}")

    # Se o diretório de saída não for especificado, utiliza o mesmo diretório de origem
    if dir_saida is None:
        dir_saida = os.path.dirname(path_h5ad)
    os.makedirs(dir_saida, exist_ok=True)

    # Definir caminhos dos arquivos de saída
    nome_base = os.path.splitext(os.path.basename(path_h5ad))[0]
    path_mtx = os.path.join(dir_saida, f"{nome_base}.mtx")
    path_genes = os.path.join(dir_saida, f"{nome_base}_genes.csv")
    path_celulas = os.path.join(dir_saida, f"{nome_base}_celulas.csv")

    # Verifica se a conversão já foi feita anteriormente para economizar tempo
    if not sobrescrever and os.path.exists(path_mtx) and os.path.exists(path_genes) and os.path.exists(path_celulas):
        print(f"[{time.strftime('%X')}] [SKIP] Arquivo já convertido em: {path_mtx} (pulando para economizar tempo).")
        return

    # Carregar o arquivo AnnData apenas após confirmar que a conversão será realizada
    adata = ad.read_h5ad(path_h5ad)
    print(f"[{time.strftime('%X')}] [OK] Matriz carregada com sucesso: {adata.shape[0]} células x {adata.shape[1]} genes")

    # Extrair e preparar a matriz principal
    matriz = adata.X
    print(f"[{time.strftime('%X')}] [INFO] Salvando matriz no formato Matrix Market (.mtx)...")
    
    # É altamente recomendável salvar em formato esparso no mmwrite por economia de espaço e tempo
    if not issparse(matriz):
        print(f"[{time.strftime('%X')}] [INFO] Convertendo matriz densa para esparsa (CSR) para otimizar gravação e tamanho do arquivo...")
        matriz = csr_matrix(matriz)

    # Salvar arquivo .mtx
    mmwrite(path_mtx, matriz)
    print(f"[{time.strftime('%X')}] [OK] Matriz salva com sucesso em: {path_mtx}")

    # Salvar metadados das colunas (genes/features) e linhas (células/barcodes)
    adata.var.to_csv(path_genes)
    adata.obs.to_csv(path_celulas)
    print(f"[{time.strftime('%X')}] [OK] Lista/Metadados de genes salvos em: {path_genes}")
    print(f"[{time.strftime('%X')}] [OK] Lista/Metadados de células salvos em: {path_celulas}")
    print(f"[{time.strftime('%X')}] [SUCESSO] Conversão concluída perfeitamente!")


if __name__ == '__main__':
    # Lista dos arquivos do pipeline para conversão (Mathys e Fujita)
    ARQUIVOS_PARA_CONVERTER = [
        r"c:\Users\Leticia\Documents\Letworkspace\pipiline_hopifield\outputs\alinhamento\adataM_binarizado_alinhado\adataM_binarizado_alinhado.h5ad",
        r"c:\Users\Leticia\Documents\Letworkspace\pipiline_hopifield\outputs\alinhamento\adataF_binarizado_alinhado\adataF_binarizado_alinhado.h5ad",
        r"c:\Users\Leticia\Documents\Letworkspace\pipiline_hopifield\outputs\top_genes\X_mathys_IMPUTADO_completo_36k_rede35.h5ad"
    ]
    
    print("=== Iniciando verificação e conversão de matrizes para .mtx ===")
    for caminho in ARQUIVOS_PARA_CONVERTER:
        print(f"\n--- Verificando: {os.path.basename(caminho)} ---")
        try:
            converter_h5ad_para_mtx(caminho)
        except Exception as e:
            print(f"[ERRO] Erro ao converter {caminho}: {e}")


