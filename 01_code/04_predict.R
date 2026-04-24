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

# 2. Predecir xgb_mobel ─────────────────────────────────────────────────────────────────
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

## Submission umbral threshold optimo
preds_12 <- as.integer(probs_test > best_threshold_XBOOST)
sub_12 <- tibble(id = test$id, pobre = preds_12)
stopifnot(nrow(sub_12) == nrow(sample_sub))
write_csv(sub_04, "02_output/04_submissions/XGB_depth6_eta01_nrounds100_thresh_PR.csv")
cat("  submissions/XGB_depth6_eta01_nrounds100_thresh_PR.csv\n")

cat("\n✔ Stage 4 completo.\n")
cat("  submissions/XGB_depth6_eta01_nrounds100_thresh05.csv\n")
cat("  submissions/XGB_depth6_eta01_nrounds100_thresh04.csv\n")
cat("  submissions/XGB_depth6_eta01_nrounds100_thresh_PR.csv\n")
cat("  Sube ambos a Kaggle y compara el F1\n")

# 2. Predecir CART ─────────────────────────────────────────────────────────────────

test_CART <- test_model

cart_model <- readRDS("02_output/03_models/cart.rds")
preds_06 <- predict(cart_model,test_CART)
sub_06 <- tibble(id = test$id, pobre = preds_06)  
sub_06$pobre <- ifelse(sub_06$pobre == "yes", 1, 0) 
cp_optimo <- cart$bestTune$cp
# Submission CART
write_csv(sub_06, "02_output/04_submissions/CART_CP_PR.csv")
cat("  submissions/02_output/04_submissions/CART_CP_PR.csv\n")

# 3. Predecir Logit ─────────────────────────────────────────────────────────────────
Logit_model <- readRDS("02_output/03_models/logit.rds")
preds_07 <- predict(Logit_model,test_model)
sub_07 <- tibble(id = test$id, pobre = preds_07)
sub_07$pobre <- ifelse(sub_07$pobre == "yes", 1, 0)
write_csv(sub_07, "02_output/04_submissions/LP_ROSE.csv")
cat("  submissions/02_output/04_submissions/LP_ROSE.csv\n")


# 4. Predecir Ramdom Forest ─────────────────────────────────────────────────────────────────
RF_model <- readRDS("02_output/03_models/rf.rds")
preds_08 <- predict(RF_model,test_model)
sub_08 <- tibble(id = test$id, pobre = preds_08)
sub_08$pobre <- ifelse(sub_08$pobre == "yes", 1, 0)
write_csv(sub_08, "02_output/04_submissions/RF_mtry4_size10.csv")
  cat("  submissions/02_output/04_submissions/RF_mtry4_size10.csv\n")

  
# 4. Predecir Naive Bayes ─────────────────────────────────────────────────────────────────
RF_model <- readRDS("02_output/03_models/rf.rds")
preds_08 <- predict(RF_model,test_model)
sub_08 <- tibble(id = test$id, pobre = preds_08)
sub_08$pobre <- ifelse(sub_08$pobre == "yes", 1, 0)
write_csv(sub_08, "02_output/04_submissions/RF_mtry4_size10.csv")
cat("  submissions/02_output/04_submissions/RF_mtry4_size10.csv\n")
  
# 5. Predecir Naive Bayes ─────────────────────────────────────────────────────────────────
nb_model <- readRDS("02_output/03_models/nb.rds")
preds_09 <- predict(nb_model,test_model)
sub_09 <- tibble(id = test$id, pobre = preds_09)
sub_09$pobre <- ifelse(sub_09$pobre == "yes", 1, 0)
write_csv(sub_09, "02_output/04_submissions/NB.csv")
cat("  submissions/02_output/04_submissions/NB.csv\n")

# 6. Predecir LPM ─────────────────────────────────────────────────────────────────
lpm_model <- readRDS("02_output/03_models/LPM.rds")
test_lpm <- test_model
test_lpm$pobre <- "yes"
test_lmp  <- test_lpm |> mutate(pobre_num = as.integer(pobre == "yes"))
preds_10 <- predict(lpm_model,test_lpm)
sub_10 <- tibble(id = test$id, pobre = preds_10)
sub_10$pobre <- ifelse(sub_10$pobre > 0.5, 1, 0)
write_csv(sub_10, "02_output/04_submissions/LPM_05.csv")
cat("  submissions/02_output/04_submissions/LPM_05\n")

# 7. Elastic NET ─────────────────────────────────────────────────────────────────
lpm_model <- readRDS("02_output/03_models/enet.rds")
preds_11 <- predict(lpm_model,test_lpm)
sub_11 <- tibble(id = test$id, pobre = preds_11)
sub_11$pobre <- ifelse(sub_11$pobre == "yes", 1, 0)
write_csv(sub_11, "02_output/04_submissions/enet_alpha_05_lamda_00021.csv")
cat("  submissions/02_output/04_submissions/enet_alpha_05_lamda_00021\n")
  