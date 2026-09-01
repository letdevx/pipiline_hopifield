# -*- coding: utf-8 -*-
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
#     display_name: R (IRkernel)
#     language: R
#     name: ir
# ---

# %% vscode={"languageId": "r"}
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("Biostrings")


# %% vscode={"languageId": "r"}
# × NOVA ETAPA OTIMIZADA OOM-SAFE: SWeePlite com paralelismo controlado, pré-alocação e Garbage Collection agressivo!
cat('Injetando SWeePlite OTIMIZADO (OOM-Safe para proteção de RAM e estabilidade no Windows)...\n')

suppressMessages(library(rSWeeP))
suppressMessages(library(doParallel))
suppressMessages(library(foreach))

rSWeeP_SWeePlite_dgCMatrix <- function(input, psz = 1369, bin = FALSE, ncores = NULL, ...) 
{
    .local <- function(input, psz = 1369, bin = FALSE, ncores = NULL, 
        transpose = FALSE, RNAseqdata = FALSE, norm = "none", 
        nk = 15000, verbose = TRUE, batch_size = 250, seed = 42,
        path_orthbase = NULL) 
    {
        start_time = proc.time()
        if (transpose == FALSE & sum(inherits(input, "dgCMatrix")) == 1) {
            cat("Caution! \nIf your input is of the RNAseq type, probably each column contains a sample, and each row a gene. \nIn this case, use the 'transpose=TRUE' option.\n\n")
            Sys.sleep(2)
        }
        rSWeeP:::SW.checks("ncores", ncores)
        rSWeeP:::SW.checks("psz", psz)
        rSWeeP:::SW.checks("norm", norm)
        rSWeeP:::SW.checks("bin", bin)
        rSWeeP:::SW.checks("RNAseqdata", RNAseqdata)
        rSWeeP:::SW.checks("transpose", transpose)
        
        output = list()
        output$info = list()
        
        # OTIMIZAÇÃO 1: Transpor fora do loop se necessário para acesso O(1) aos vetores esparsos em colunas (@p, @i, @x)
        if (transpose | RNAseqdata) {
            N = dim(input)[2]
            lenmax = dim(input)[1]
            output$info$features = rownames(input)
            output$info$samples = colnames(input)
            mat_col <- input
        } else {
            N = dim(input)[1]
            lenmax = dim(input)[2]
            output$info$features = colnames(input)
            output$info$samples = rownames(input)
            mat_col <- Matrix::t(input)
        }
        if (!inherits(mat_col, "dgCMatrix")) {
            mat_col <- as(mat_col, "CsparseMatrix")
        }
        
        output$info$ProjectionSize = psz
        output$info$bin = ifelse(bin, "binary (TRUE)", "counting (FALSE)")
        output$info$RNAseqdata = RNAseqdata
        output$info$transpose = transpose
        output$info$norm = norm
        
        # FIX DE REPRODUTIBILIDADE: Fixação determinística de semente antes de liteParam
        if (!is.null(seed)) {
            set.seed(seed)
        }
        par = rSWeeP:::liteParam(NULL, input, NULL, N, psz, lenmax)
        
        # CONGELAMENTO E REUTILIZAÇÃO DA BASE ORTONORMAL (ADR 018)
        if (!is.null(path_orthbase) && file.exists(path_orthbase)) {
            cat(paste0(">>> Carregando base de projeção congelada existente: ", path_orthbase, "\n"))
            par$Mproj <- readRDS(path_orthbase)
        } else if (!is.null(path_orthbase)) {
            cat(paste0(">>> Salvando base de projeção para reutilização obrigatória: ", path_orthbase, "\n"))
            saveRDS(par$Mproj, path_orthbase)
        }
        
        # OTIMIZAÇÃO 2 (OOM-Safe): Configuração inteligente e conservadora do cluster na CPU
        if (is.null(ncores) || ncores <= 0) {
            ncores_uso <- min(4, max(1, parallel::detectCores() - 2))
        } else {
            ncores_uso <- ncores
        }
        ncores_def = rSWeeP:::NCoresDef(ncores_uso)
        sw.cluster <- parallel::makeCluster(ncores_def, type = "PSOCK")
        doParallel::registerDoParallel(cl = sw.cluster)
        
        # Exportar parâmetros fixos uma única vez aos trabalhadores (evita serialização redundante)
        parallel::clusterExport(cl = sw.cluster, varlist = c("par", "psz", "nk", "bin", "norm"), envir = environment())
        
        log_con <- file("progresso_projecao.txt", open="w")
        
        # OTIMIZAÇÃO 3 (OOM-Safe): Pré-alocar a matriz final na memória (Evita acúmulo de listas e estouro no do.call)
        output$proj <- matrix(0, nrow = N, ncol = psz)
        batches <- seq(1, N, by = batch_size)
        
        cat(paste0("Iniciando projeção OOM-Safe de ", N, " amostras utilizando ", ncores_def, " núcleos em lotes limpos de ", batch_size, " ...\n\n"))
        
        for (b in batches) {
            idx_end <- min(b + batch_size - 1, N)
            batch_idx <- b:idx_end
            
            if (verbose) {
                pct <- formatC(idx_end / N * 100, digits=2, format="f")
                msg <- paste0("Projetando amostras até ", idx_end, " de ", N, " (", pct, "%) ...\n")
                cat(msg)
                flush.console()
                writeLines(msg, log_con)
                flush(log_con)
            }
            
            # Extração ultrarrápida na thread principal (impede que a matriz gigante trafegue em sockets)
            batch_hdv <- vector("list", length(batch_idx))
            for (idx in seq_along(batch_idx)) {
                k <- batch_idx[idx]
                p_start <- mat_col@p[k] + 1L
                p_end <- mat_col@p[k + 1L]
                
                hdv <- list()
                if (p_start <= p_end) {
                    hdv$idx <- mat_col@i[p_start:p_end] + 1L
                    val <- mat_col@x[p_start:p_end]
                    if (norm != "none") {
                        val <- rSWeeP:::MakeLOG(val, bin, norm)
                    }
                    if (!bin) {
                        hdv$count <- val
                    }
                } else {
                    hdv$idx <- integer(0)
                    if (!bin) hdv$count <- numeric(0)
                }
                batch_hdv[[idx]] <- hdv
            }
            
            # Paralelismo enviando SOMENTE pacotes leves (hdv_item) para cada worker
            batch_res <- foreach::foreach(hdv_item = batch_hdv, .combine = "rbind", 
                .packages = c("rSWeeP", "Matrix"), 
                .noexport = c("input", "mat_col", "output", "batches", "batch_hdv")) %dopar% {
                foreach::registerDoSEQ()
                rSWeeP:::COREloop(par$xnorm, psz, par$Mproj, par$pslist, par$nps, hdv_item, nk, bin, norm)
            }
            
            # Grava o resultado na matriz principal por fatia
            output$proj[batch_idx, ] <- as.matrix(batch_res)
            
            # OTIMIZAÇÃO 4 (OOM-Safe): Limpeza imediata dos temporários e acionamento do Garbage Collector
            rm(batch_hdv, batch_res)
            gc(verbose = FALSE)
        }
        
        close(log_con)
        parallel::stopCluster(cl = sw.cluster)
        
        dimnames(output$proj) <- NULL
        if (transpose | RNAseqdata) {
            rownames(output$proj) <- colnames(input)
        } else {
            rownames(output$proj) <- rownames(input)
        }
        
        output$info$version = utils::packageVersion("rSWeeP")
        end_time = proc.time()
        output$info$timeElapsed = (end_time - start_time)[[3]]
        return(output)
    }
    .local(input, psz, bin, ncores, ...)
}

# Sobrescreve o método dgCMatrix da função SWeePlite com a versão otimizada
suppressMessages(setMethod("SWeePlite", "dgCMatrix", rSWeeP_SWeePlite_dgCMatrix))

cat("✅ Método SWeePlite OOM-Safe (Proteção de RAM + Multi-core Controlado + Pré-alocação) configurado com sucesso!\n")

# %% vscode={"languageId": "r"}
# 1. Carregar os pacotes necessários
cat('Etapa 1: Carregando pacotes...\n\n')
library(rSWeeP)
library(Matrix) # Pacote nativo do R para leitura de matrizes esparsas (.mtx)
cat('Pacotes carregados com sucesso!\n\n\n')

# 2. Carregar os dados do arquivo .mtx
cat('Etapa 2: Carregando arquivo .mtx... Isto pode levar alguns minutos.\n\n')
caminho_mtx <-"C:\\Users\\Leticia\\Documents\\Letworkspace\\Todos-os-sweep-do-meu-projeto\\matariz_fujita_binaria.mtx"
matriz_mtx <- readMM(caminho_mtx)
matriz_mtx <- as(matriz_mtx, 'CsparseMatrix')
cat(paste('Convertido para classe:', class(matriz_mtx)[1], '\n\n'))

# Etapa 2.5 (Adição de Nomes) removida: A matriz será processada estritamente sem nomes

cat(paste('Matriz carregada! Dimensões:', nrow(matriz_mtx), 'linhas x', ncol(matriz_mtx), 'colunas\n\n'))
cat(paste('A quantidade de entradas não nulas é:', length(matriz_mtx@x), '\n\n'))
cat(paste('Soma total de genes (expressão bruta):', sum(matriz_mtx@x), '\n\n\n'))

# 3. Executar a projeção com a função SWeePlite otimizada
cat('Etapa 3: Executando SWeePlite... \nAcompanhe o progresso no arquivo "progresso_projecao.txt" !\n\n')
ncores_disponiveis <- min(4, max(1, parallel::detectCores() - 2)) # Limite seguro de núcleos para evitar falta de RAM!
cat(paste("Utilizando", ncores_disponiveis, "núcleos do processador em paralelo...\n\n"))

# Define o caminho da base ortonormal congelada (Garante congruência Referência ↔ Alvo)
caminho_orthbase <- "orthbase_mproj_600d.rds"

resultado <- SWeePlite(
  input = matriz_mtx, 
  psz = 600,             
  transpose = FALSE,     
  ncores = ncores_disponiveis,
  batch_size = 250,
  seed = 42,
  path_orthbase = caminho_orthbase
)
cat(paste('SWeePlite concluído com sucesso! Tempo total:', round(resultado$info$timeElapsed, 2), 'segundos.\n'))
cat(paste('A nova matriz projetada tem:', nrow(resultado$proj), 'linhas x', ncol(resultado$proj), 'colunas\n\n\n'))

# 4. Acessar a matriz projetada final
cat('Etapa 4: Extraindo e verificando a matriz final...\n\n')
matriz_projetada <- resultado$proj
cat('Verificação de valores ausentes (NAs):', sum(is.na(matriz_projetada)), '\n\n')
cat('Processo finalizado com sucesso!\n\n')

# %% vscode={"languageId": "r"}
arquivo_saida <- "sweep_fujita_binario.txt"
write.table(matriz_projetada, file = arquivo_saida, sep = "\t", row.names = FALSE)
cat(paste("Matriz projetada salva com sucesso no arquivo:", arquivo_saida, "\n"))
