## A2.1 - DIABLO (mixOmics::block.splsda) on Metabolomics + Blood Biochemistry.
## Supervised multi-block sparse PLS-DA. Primary intermediate-integration method.
## Outputs:
##   analysis/02_intermediate/diablo_perf.csv
##   analysis/02_intermediate/diablo_loadings_<block>.csv
##   analysis/02_intermediate/diablo_top20_<block>.csv
##   analysis/figures/diablo_circos.png
##   analysis/figures/diablo_corrcircle.png
##   analysis/figures/diablo_performance.png
## Seed: 2026.

suppressPackageStartupMessages({
  library(mixOmics)
})
set.seed(2026)

## Resolve project root: prefer command-line arg, else assume Rscript was launched
## from project root.
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) normalizePath(args[1]) else normalizePath(getwd())
pre  <- file.path(root, "analysis", "01_preprocessing")
out  <- file.path(root, "analysis", "02_intermediate")
fig  <- file.path(root, "analysis", "figures")
dir.create(out, showWarnings = FALSE, recursive = TRUE)
dir.create(fig, showWarnings = FALSE, recursive = TRUE)

read_block <- function(name) {
  df <- read.csv(file.path(pre, paste0(name, "_z.csv")), check.names = FALSE)
  list(
    X = as.matrix(df[, !(colnames(df) %in% c("Sample","casecontrol","y"))]),
    y = factor(df$casecontrol, levels = c("Control","Colon Cancer")),
    sample = df$Sample
  )
}

metab   <- read_block("metabolomics")
biochem <- read_block("biochem")

stopifnot(all(metab$sample == biochem$sample))
Y <- metab$y
cat(sprintf("[A2.1] n = %d   Control = %d   CRC = %d\n",
            length(Y), sum(Y=="Control"), sum(Y=="Colon Cancer")))

X <- list(metab = metab$X, biochem = biochem$X)

## Design matrix - moderate connectivity 0.5 (mixOmics recommended default)
design <- matrix(0.5, ncol = length(X), nrow = length(X),
                 dimnames = list(names(X), names(X)))
diag(design) <- 0

## Tune number of components on a small grid via 5-fold CV (3 repeats)
ncomp_grid <- 3
basic.diablo <- block.splsda(X = X, Y = Y, ncomp = ncomp_grid, design = design)

perf.diablo <- perf(basic.diablo, validation = "Mfold", folds = 5, nrepeat = 3,
                    progressBar = FALSE)
png(file.path(fig, "diablo_performance.png"), width = 1400, height = 900, res = 150)
plot(perf.diablo)
dev.off()
ncomp.best <- perf.diablo$choice.ncomp$WeightedVote["Overall.BER","centroids.dist"]
if (is.na(ncomp.best) || ncomp.best < 1) ncomp.best <- 2
cat(sprintf("[A2.1] chosen ncomp = %d\n", ncomp.best))

## Tune keepX per component per block
test.keepX <- list(
  metab   = c(5, 10, 15, 20, 25),
  biochem = c(2, 4, 6, 8)
)
tune.diablo <- tune.block.splsda(
  X = X, Y = Y, ncomp = ncomp.best, test.keepX = test.keepX,
  design = design, validation = "Mfold", folds = 5, nrepeat = 3,
  dist = "centroids.dist", progressBar = FALSE
)
keepX <- tune.diablo$choice.keepX
cat("[A2.1] tuned keepX:\n"); print(keepX)

## Final model
final.diablo <- block.splsda(X = X, Y = Y, ncomp = ncomp.best, keepX = keepX, design = design)

## Final 5-fold x 3-repeat performance (BER, AUC)
final.perf <- perf(final.diablo, validation = "Mfold", folds = 5, nrepeat = 3,
                   dist = "centroids.dist", auc = TRUE, progressBar = FALSE)

auc_list <- final.perf$auc
auc_rows <- do.call(rbind, lapply(names(auc_list), function(block) {
  tab <- auc_list[[block]]
  data.frame(block = block,
             comp = rownames(tab),
             auc_mean = tab[,1],
             pvalue   = tab[,2],
             row.names = NULL)
}))
write.csv(auc_rows, file.path(out, "diablo_auc.csv"), row.names = FALSE)
cat("[A2.1] AUC table written.\n"); print(auc_rows)

ber <- final.perf$WeightedVote.error.rate[["centroids.dist"]]
write.csv(data.frame(comp = colnames(ber), BER_overall = ber["Overall.BER",]),
          file.path(out, "diablo_perf.csv"), row.names = FALSE)

## Loadings and top-20 features per block per component
for (b in names(X)) {
  load_mat <- final.diablo$loadings[[b]]
  write.csv(cbind(feature = rownames(load_mat), as.data.frame(load_mat)),
            file.path(out, paste0("diablo_loadings_", b, ".csv")), row.names = FALSE)
  top20 <- data.frame()
  for (k in seq_len(ncol(load_mat))) {
    s <- load_mat[, k]
    keep <- order(abs(s), decreasing = TRUE)[1:min(20, sum(abs(s) > 0))]
    if (length(keep) > 0) {
      top20 <- rbind(top20, data.frame(
        feature = rownames(load_mat)[keep],
        comp = k,
        loading = s[keep],
        rank = seq_along(keep)
      ))
    }
  }
  write.csv(top20, file.path(out, paste0("diablo_top20_", b, ".csv")), row.names = FALSE)
}

## Plots: circos, correlation circle
png(file.path(fig, "diablo_circos.png"), width = 1500, height = 1500, res = 180)
tryCatch(
  circosPlot(final.diablo, cutoff = 0.5, line = TRUE, size.labels = 1.0),
  error = function(e) message("circosPlot failed: ", conditionMessage(e))
)
dev.off()

png(file.path(fig, "diablo_corrcircle.png"), width = 1400, height = 1400, res = 180)
tryCatch(
  plotVar(final.diablo, var.names = FALSE, style = "graphics", legend = TRUE),
  error = function(e) message("plotVar failed: ", conditionMessage(e))
)
dev.off()

saveRDS(final.diablo, file.path(out, "diablo_model.rds"))
cat("[A2.1] done.\n")
