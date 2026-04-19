#──────────────────────────────────────────────────────────────────────────────
#  Stage 4: Generar predicciones para Kaggle
#──────────────────────────────────────────────────────────────────────────────

cat("\n── Stage 4: Predicciones ────────────────────────────────────\n")

# 0. Cargar datos y modelo ────────────────────────────────────────────────────
test      <- read_rds("00_data/test_clean.rds")
# xgb_model <- xgb.load("01_output/03_models/xgb_model.ubj")
xgb_model <- readRDS("02_output/03_models/xgb_model.rds")
# 1. Preparar test igual que train ────────────────────────────────────────────
drop_cols <- c("id", "fex_c", "fex_dpto")
features  <- setdiff(names(test), drop_cols)

test_model <- test |>
  select(all_of(features)) |>
  mutate(across(where(is.character), ~as.integer(as.factor(.))))

# Asegurar mismo orden de columnas que en entrenamiento
train_features <- read_rds("00_data/train_clean.rds") |>
  select(-any_of(c("id", "pobre", "fex_c", "fex_dpto"))) |>
  names()

test_matrix <- test_model |>
  select(all_of(train_features)) |>
  as.matrix()

dtest <- xgb.DMatrix(data = test_matrix)

# 2. Predecir ─────────────────────────────────────────────────────────────────
probs_test <- predict(xgb_model, dtest)

# Umbral 0.5
preds_05 <- as.integer(probs_test > 0.5)
# Umbral 0.4 (mas recall, menos undercoverage)
preds_04 <- as.integer(probs_test > 0.4)

cat("✔ Predicciones generadas\n")
cat("  Umbral 0.5 — pobres:", sum(preds_05), "(", round(mean(preds_05)*100,1), "%)\n")
cat("  Umbral 0.4 — pobres:", sum(preds_04), "(", round(mean(preds_04)*100,1), "%)\n")

# 3. Generar submissions ──────────────────────────────────────────────────────
sample_sub <- read_csv("00_data/sample_submission.csv", show_col_types = FALSE)
# dir.create("submissions", showWarnings = FALSE)

# Submission umbral 0.5
sub_05 <- tibble(id = test$id, pobre = preds_05)
stopifnot(nrow(sub_05) == nrow(sample_sub))
write_csv(sub_05, "02_output/04_submissions/XGB_depth6_eta01_nrounds100_thresh05.csv")

# Submission umbral 0.4
sub_04 <- tibble(id = test$id, pobre = preds_04)
stopifnot(nrow(sub_04) == nrow(sample_sub))
write_csv(sub_04, "02_output/04_submissions/XGB_depth6_eta01_nrounds100_thresh04.csv")

cat("\n✔ Stage 4 completo.\n")
cat("  submissions/XGB_depth6_eta01_nrounds100_thresh05.csv\n")
cat("  submissions/XGB_depth6_eta01_nrounds100_thresh04.csv\n")
cat("  Sube ambos a Kaggle y compara el F1\n")

