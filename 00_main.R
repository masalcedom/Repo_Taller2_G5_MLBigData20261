#──────────────────────────────────────────────────────────────────────────────
#  Problem Set 2 · Predicting Poverty in Colombia        MECA 4107 · 2026-10
#──────────────────────────────────────────────────────────────────────────────

# 0. Limpiar memoria y cargar paquetes ────────────────────────────────────────

cat("Working directory:\n")

print(getwd())

#install.packages("doParallel")
#library(doParallel)
library(xgboost)
library(pacman)
install.packages("naivebayes")
library(naivebayes)

p_load(
  rio, tidyverse, janitor, skimr,
  visdat, corrplot, ggpubr, patchwork,
  caret, glmnet, ranger, xgboost,
  smotefamily, ROSE,
  Metrics, pROC,
  stargazer, writexl,
  sf, geodata
)

# 1. Directorio ───────────────────────────────────────────────────────────────
# user_paths <- list(
#   "danny"  = "C:/Users/danny/OneDrive/Documentos/Maestria/BD-ML/Taller 2")
# 
# current_user <- Filter(
#   function(u) grepl(u, getwd(), ignore.case = TRUE),
#   names(user_paths)
# )
# 
# if (length(current_user) == 1) {
#   setwd(user_paths[[current_user]])
#   cat("✔ Directorio:", getwd(), "\n")
# } else {
#   stop("Usuario no reconocido. Agrega tu ruta en user_paths.")
# }
cat("Working directory:\n")
print(getwd())

# Crear carpetas de output si no existen
for (path in c("02_output/01_figures",
               "02_output/02_tables",
               "02_output/03_models",
               "02_output/04_submissions"
               )) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}



# 2. Correr scripts ───────────────────────────────────────────────────────────
source("01_code/01_data.R")
source("01_code/02_models.R")
#source("scripts/03_best_model.R")
#source("scripts/04_predict.R")
#source("scripts/05_mapas.R")