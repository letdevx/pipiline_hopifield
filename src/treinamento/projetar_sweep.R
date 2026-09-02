# projetar_sweep.R
# Executa a projeção rSWeeP canônica (AIBIALab/UFPR) sobre matriz esparsa.
#
# Uso:
#   Rscript projetar_sweep.R <path_entrada> <path_saida> [dim_proj] [seed] [path_orthbase]

suppressPackageStartupMessages({
  if (!requireNamespace("Matrix", quietly = TRUE)) {
    stop("[rSWeeP] Pacote 'Matrix' não encontrado.")
  }
  if (!requireNamespace("rSWeeP", quietly = TRUE)) {
    stop("[rSWeeP] Pacote 'rSWeeP' não encontrado. Instale com:\n",
         "  devtools::install_github('aibialab/rSWeeP') ou BiocManager::install('rSWeeP')")
  }
  library(Matrix)
  library(rSWeeP)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("[rSWeeP] Uso obrigatório: Rscript projetar_sweep.R <path_entrada> <path_saida> [dim_proj] [seed] [path_orthbase]")
}

path_entrada  <- args[1]
path_saida    <- args[2]
dim_proj      <- if (length(args) >= 3) as.integer(args[3]) else 600L
seed          <- if (length(args) >= 4) as.integer(args[4]) else 42L
path_orthbase <- if (length(args) >= 5 && nzchar(args[5])) args[5] else NULL

cat("=================================================================\n")
cat("          PROJEÇÃO CANÔNICA rSWeeP (UFPR / AIBIALab)             \n")
cat("=================================================================\n")
cat("[rSWeeP] Entrada         :", path_entrada, "\n")
cat("[rSWeeP] Saída           :", path_saida, "\n")
cat("[rSWeeP] Dimensão Alvo   :", dim_proj, "\n")
cat("[rSWeeP] Semente         :", seed, "\n")
cat("[rSWeeP] Base Congelada  :", ifelse(is.null(path_orthbase), "Não informada", path_orthbase), "\n")

# 1. Leitura OOM-Safe da Matriz de Entrada
cat("[rSWeeP] Carregando matriz de entrada...\n")
t_start <- proc.time()

mat <- if (grepl("\\.mtx(\\.gz)?$", path_entrada, ignore.case = TRUE)) {
  m <- Matrix::readMM(path_entrada)
  as(m, "CsparseMatrix")
} else if (grepl("\\.(csv|txt)$", path_entrada, ignore.case = TRUE)) {
  # Fallback de leitura para matrizes de texto via data.table se disponível
  if (requireNamespace("data.table", quietly = TRUE)) {
    m_dense <- as.matrix(data.table::fread(path_entrada, header = FALSE))
    Matrix(m_dense, sparse = TRUE)
  } else {
    m_dense <- as.matrix(read.table(path_entrada, header = FALSE))
    Matrix(m_dense, sparse = TRUE)
  }
} else {
  stop("[rSWeeP] Formato de entrada não suportado: ", path_entrada, ". Use .mtx ou .txt/.csv.")
}

n_amostras <- nrow(mat)
n_genes    <- ncol(mat)
cat("[rSWeeP] Matriz carregada com sucesso!\n")
cat("[rSWeeP]   Linhas (células) :", n_amostras, "\n")
cat("[rSWeeP]   Colunas (genes)  :", n_genes, "\n")
cat("[rSWeeP]   Não-zeros (nnz)  :", length(mat@x), "\n")

# 2. Obtenção / Congelamento da Base Ortonormal Canônica (orthBase)
base <- NULL
if (!is.null(path_orthbase) && file.exists(path_orthbase)) {
  cat("[rSWeeP] Base congelada encontrada. Carregando:", path_orthbase, "\n")
  base <- readRDS(path_orthbase)
  if (nrow(base$mat) != n_genes || ncol(base$mat) != dim_proj) {
    stop(sprintf("[rSWeeP] Incompatibilidade dimensional da base congelada: esperava (%d x %d), mas arquivo tem (%d x %d).",
                 n_genes, dim_proj, nrow(base$mat), ncol(base$mat)))
  }
} else {
  cat(sprintf("[rSWeeP] Gerando base ortonormal canônica via orthBase(lin=%d, col=%d, seed=%d)...\n",
              n_genes, dim_proj, seed))
  t_base <- proc.time()
  base <- orthBase(lin = n_genes, col = dim_proj, seed = seed)
  cat(sprintf("[rSWeeP] Base gerada em %.2f s. Dimensões: %d x %d\n",
              (proc.time() - t_base)[[3]], nrow(base$mat), ncol(base$mat)))
  
  if (!is.null(path_orthbase)) {
    dir.create(dirname(path_orthbase), showWarnings = FALSE, recursive = TRUE)
    cat("[rSWeeP] Salvando base congelada para reutilização obrigatória:", path_orthbase, "\n")
    saveRDS(base, file = path_orthbase)
  }
}

# 3. Execução da Projeção SWeeP Oficial
cat("[rSWeeP] Projetando amostras no espaço latente 600D...\n")
t_proj <- proc.time()

# Executa o método SWeeP oficial para dgCMatrix
resultado <- SWeeP(mat, orthbase = base, transpose = FALSE)
matriz_projetada <- resultado$proj

cat(sprintf("[rSWeeP] Projeção concluída em %.2f s! Dimensões da saída: %d x %d\n",
            (proc.time() - t_proj)[[3]], nrow(matriz_projetada), ncol(matriz_projetada)))

# Verificação de integridade
n_nas <- sum(is.na(matriz_projetada))
if (n_nas > 0) {
  stop(sprintf("[rSWeeP] ERRO: Foram detectados %d valores NaN/NA na matriz projetada!", n_nas))
}

# 4. Gravação da Matriz Projetada em Formato Tabulado (.txt)
dir.create(dirname(path_saida), showWarnings = FALSE, recursive = TRUE)
cat("[rSWeeP] Gravando matriz projetada em formato tabulado (.txt):", path_saida, "\n")
write.table(as.matrix(matriz_projetada), file = path_saida, sep = "\t", row.names = FALSE, col.names = FALSE)

t_total <- (proc.time() - t_start)[[3]]
cat(sprintf("[rSWeeP] Processo finalizado com sucesso absoluto em %.2f s!\n", t_total))
