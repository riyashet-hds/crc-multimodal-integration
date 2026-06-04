## A2.2 - Regularised CCA (mixOmics::rcc) on Metabolomics + Blood Biochemistry.
## Unsupervised - identifies maximally correlated linear combinations across two blocks.
## Replaces PMA::CCA (not installed) using the equivalent mixOmics function.
## Outputs:
##   analysis/02_intermediate/rcc_canonical_correlations.csv
##   analysis/02_intermediate/rcc_loadings_<block>.csv
##   analysis/figures/rcc_var_plot.png
##   analysis/figures/rcc_indiv_plot.png
## Seed: 2026.

suppressPackageStartupMessages({ library(mixOmics) })
set.seed(2026)

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
    y = factor(df$casecontrol),
    sample = df$Sample
  )
}

metab   <- read_block("metabolomics")
biochem <- read_block("biochem")
stopifnot(all(metab$sample == biochem$sample))

## Tune regularisation lambdas via shrinkage method (fast, recommended on large n)
ncomp <- 3
rcc.fit <- rcc(metab$X, biochem$X, ncomp = ncomp, method = "shrinkage")
cat("[A2.2] regularisation parameters (shrinkage):\n"); print(rcc.fit$lambda)
cat("[A2.2] canonical correlations:\n"); print(rcc.fit$cor)

cor_vals <- as.numeric(rcc.fit$cor)[1:ncomp]
write.csv(
  data.frame(component = seq_len(ncomp),
             canonical_correlation = cor_vals,
             lambda_X1 = as.numeric(rcc.fit$lambda[1]),
             lambda_X2 = as.numeric(rcc.fit$lambda[2])),
  file.path(out, "rcc_canonical_correlations.csv"), row.names = FALSE
)

for (b in c("metab","biochem")) {
  load_mat <- rcc.fit$loadings[[ if (b=="metab") "X" else "Y" ]]
  write.csv(cbind(feature = rownames(load_mat), as.data.frame(load_mat)),
            file.path(out, paste0("rcc_loadings_", b, ".csv")), row.names = FALSE)
}

## Variable plot (correlation circle) and individual plot
png(file.path(fig, "rcc_var_plot.png"), width = 1400, height = 1400, res = 180)
tryCatch(
  plotVar(rcc.fit, var.names = FALSE, cex = c(3, 3), title = "rCCA correlation circle"),
  error = function(e) message("rcc plotVar failed: ", conditionMessage(e))
)
dev.off()

png(file.path(fig, "rcc_indiv_plot.png"), width = 1400, height = 1100, res = 180)
tryCatch(
  plotIndiv(rcc.fit, group = metab$y, legend = TRUE,
            title = "rCCA individuals - Control vs Colon Cancer"),
  error = function(e) message("rcc plotIndiv failed: ", conditionMessage(e))
)
dev.off()

saveRDS(rcc.fit, file.path(out, "rcc_model.rds"))
cat("[A2.2] done.\n")
