# projetar_sweep.R
# Aplica projeção rSWeeP sobre a matriz binária de entrada e salva o resultado.
#
# Uso:
#   Rscript projetar_sweep.R <path_entrada> <path_saida> [dim_proj] [seed]

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

# Detectar nome correto da função de projeção no pacote
fns <- ls("package:rSWeeP")
cat("[SWeeP] Funções disponíveis no pacote rSWeeP:", paste(fns, collapse=", "), "\n")

# Candidatos comuns para o nome da função
candidatos <- c("rSWeeP", "sweepFun", "sweep_projection", "projSWeeP",
                "sweepVector", "rSweep", "SWeeP")
fn_nome <- NULL
for (cand in candidatos) {
  if (cand %in% fns) { fn_nome <- cand; break }
}
if (is.null(fn_nome) && length(fns) > 0) {
  fn_nome <- fns[1]  # usa a primeira função disponível
}
if (is.null(fn_nome)) {
  stop("[SWeeP] Nenhuma função de projeção encontrada no pacote rSWeeP.")
}
cat("[SWeeP] Usando função:", fn_nome, "\n")
fn <- get(fn_nome, envir = asNamespace("rSWeeP"))

cat("[SWeeP] Lendo matriz:", path_entrada, "\n")
mat <- as.matrix(fread(path_entrada, header = FALSE))
cat("[SWeeP] Shape entrada:", nrow(mat), "x", ncol(mat), "\n")

cat("[SWeeP] Projetando para", dim_proj, "dimensões...\n")
resultado <- tryCatch(
  fn(mat, dimproj = dim_proj, seed = seed),
  error = function(e) fn(mat, dim_proj)   # tenta sem argumento nomeado
)

proj <- if (is.list(resultado)) {
  if ("data" %in% names(resultado)) resultado$data
  else if ("proj" %in% names(resultado)) resultado$proj
  else resultado[[1]]
} else {
  resultado
}

cat("[SWeeP] Shape saída:", nrow(proj), "x", ncol(proj), "\n")
dir.create(dirname(path_saida), showWarnings = FALSE, recursive = TRUE)
cat("[SWeeP] Salvando:", path_saida, "\n")
fwrite(as.data.frame(proj), path_saida)
cat("[SWeeP] Concluído.\n")
