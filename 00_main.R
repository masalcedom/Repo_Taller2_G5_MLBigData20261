#──────────────────────────────────────────────────────────────────────────────
#  Problem Set 2 · Predicting Poverty in Colombia        MECA 4107 · 2026-10
#──────────────────────────────────────────────────────────────────────────────

# 0. Limpiar memoria y cargar paquetes ────────────────────────────────────────
rm(list = ls())

library(pacman)
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
user_paths <- list(
  "danny"  = "C:/Users/danny/OneDrive/Documentos/Maestria/BD-ML/Taller 2")

current_user <- Filter(
  function(u) grepl(u, getwd(), ignore.case = TRUE),
  names(user_paths)
)

if (length(current_user) == 1) {
  setwd(user_paths[[current_user]])
  cat("✔ Directorio:", getwd(), "\n")
} else {
  stop("Usuario no reconocido. Agrega tu ruta en user_paths.")
}

# 2. Correr scripts ───────────────────────────────────────────────────────────
source("scripts/01_data.R")
source("scripts/02_models.R")
source("scripts/03_best_model.R")
source("scripts/04_predict.R")
source("scripts/05_mapas.R")