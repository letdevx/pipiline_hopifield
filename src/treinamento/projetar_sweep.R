# projetar_sweep.R
# Aplica projeção rSWeeP sobre a matriz binária de entrada e salva o resultado.
#
# Uso:
#   Rscript projetar_sweep.R <path_entrada> <path_saida> [dim_proj] [seed]
#
# Argumentos:
#   path_entrada : CSV sem cabeçalho (células × genes)
#   path_saida   : CSV de saída (células × dim_proj)
#   dim_proj     : dimensão da projeção (padrão: 600)
#   seed         : semente aleatória (padrão: 42)

suppressPackageStartupMessages({
  if (!requireNamespace("rSWeeP", quietly = TRUE)) {
    stop("[SWeeP] Pacote rSWeeP não encontrado. Instale com:\n",
         "  devtools::install_github('aibialab/rSWeeP')")
  }
  library(rSWeeP)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("[SWeeP] Uso: Rscript projetar_sweep.R <path_entrada> <path_saida> [dim_proj] [seed]")
}

path_entrada <- args[1]
path_saida   <- args[2]
dim_proj     <- if (length(args) >= 3) as.integer(args[3]) else 600L
seed         <- if (length(args) >= 4) as.integer(args[4]) else 42L

set.seed(seed)

cat("[SWeeP] Lendo matriz:", path_entrada, "\n")
mat <- as.matrix(fread(path_entrada, header = FALSE))
cat("[SWeeP] Shape entrada:", nrow(mat), "x", ncol(mat), "\n")

cat("[SWeeP] Projetando para", dim_proj, "dimensões (seed=", seed, ")...\n")
resultado <- rSWeeP(mat, dimproj = dim_proj, seed = seed)

# rSWeeP retorna lista com $data (projeção) ou matrix direta
proj <- if (is.list(resultado)) resultado$data else resultado
cat("[SWeeP] Shape saída:", nrow(proj), "x", ncol(proj), "\n")

dir.create(dirname(path_saida), showWarnings = FALSE, recursive = TRUE)
cat("[SWeeP] Salvando:", path_saida, "\n")
fwrite(as.data.frame(proj), path_saida)
cat("[SWeeP] Concluído.\n")
