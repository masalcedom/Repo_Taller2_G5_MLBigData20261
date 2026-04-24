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





# ──────────────────────────────────────────────────────────────────────────────
# Submissions adicionales
# ──────────────────────────────────────────────────────────────────────────────

# Reproducir el split de validacion usado en 02_models.R ────────────────────
train_full <- read_rds("00_data/train_clean.rds")
set.seed(2005)
drop_cols_v <- c("id", "pobre", "fex_c", "fex_dpto")
features_v  <- setdiff(names(train_full), drop_cols_v)
train_mod   <- train_full |>
  select(all_of(c(features_v, "pobre"))) |>
  mutate(across(where(is.character), ~as.integer(as.factor(.))))
idx_v   <- createDataPartition(train_mod$pobre, p = 0.8, list = FALSE)
val     <- train_mod[-idx_v, ]
val_mat <- val |> select(-pobre) |> as.matrix()
dval    <- xgb.DMatrix(data = val_mat)

# Helper: barrer threshold optimo en validacion para maximizar F1 ───────────
best_threshold_f1 <- function(probs_val, y_val) {
  thrs <- seq(0.10, 0.70, by = 0.01)
  f1s  <- sapply(thrs, function(t) {
    pred <- factor(ifelse(probs_val > t, "yes", "no"),
                   levels = c("no", "yes"))
    cm   <- confusionMatrix(pred, y_val, positive = "yes")
    cm$byClass["F1"]
  })
  list(t = thrs[which.max(f1s)], f1 = max(f1s))
}

# Cargar modelos adicionales y existentes ────────────────────────────────────
xgb_spw    <- readRDS("02_output/03_models/xgb_spw.rds")
rf_wide    <- readRDS("02_output/03_models/rf_wide.rds")
enet_model <- readRDS("02_output/03_models/enet.rds")
cart_model <- readRDS("02_output/03_models/cart.rds")

# 8. Predecir XGB original (threshold optimo) ───────────────────────────────────
cat("[+1] XGB original con threshold optimo...\n")
probs_val_xgb  <- predict(xgb_model, dval)
opt_xgb        <- best_threshold_f1(probs_val_xgb, val$pobre)
cat(sprintf("  Threshold optimo: %.2f | F1 val: %.4f\n",
            opt_xgb$t, opt_xgb$f1))
probs_xgb_test <- predict(xgb_model, dtest)
sub_xgb_opt <- tibble(id = test$id,
                      pobre = as.integer(probs_xgb_test > opt_xgb$t))
write_csv(sub_xgb_opt, "02_output/04_submissions/XGB_orig_F1opt.csv")
cat("  submissions/02_output/04_submissions/XGB_orig_F1opt.csv\n")

# 9. Predecir XGB scale_pos_weight (threshold 0.5) ──────────────────────────────
cat("[+2] XGB scale_pos_weight thresh 0.5...\n")
probs_spw_test <- predict(xgb_spw, dtest)
sub_spw_05 <- tibble(id = test$id,
                     pobre = as.integer(probs_spw_test > 0.5))
write_csv(sub_spw_05,
          "02_output/04_submissions/XGB_spw_depth8_eta005_thresh05.csv")
cat("  submissions/02_output/04_submissions/XGB_spw_depth8_eta005_thresh05.csv\n")

# 10. Predecir XGB scale_pos_weight (threshold optimo) ──────────────────────────
cat("[+3] XGB scale_pos_weight con threshold optimo...\n")
probs_val_spw <- predict(xgb_spw, dval)
opt_spw       <- best_threshold_f1(probs_val_spw, val$pobre)
cat(sprintf("  Threshold optimo: %.2f | F1 val: %.4f\n",
            opt_spw$t, opt_spw$f1))
sub_spw_opt <- tibble(id = test$id,
                      pobre = as.integer(probs_spw_test > opt_spw$t))
write_csv(sub_spw_opt,
          "02_output/04_submissions/XGB_spw_depth8_eta005_F1opt.csv")
cat("  submissions/02_output/04_submissions/XGB_spw_depth8_eta005_F1opt.csv\n")

# 11. Predecir Random Forest grilla amplia (threshold optimo) ───────────────────
cat("[+4] RF grilla amplia con threshold optimo...\n")
probs_val_rf <- predict(rf_wide, val, type = "prob")[, "yes"]
opt_rf       <- best_threshold_f1(probs_val_rf, val$pobre)
cat(sprintf("  Threshold optimo: %.2f | F1 val: %.4f\n",
            opt_rf$t, opt_rf$f1))
probs_rf_test <- predict(rf_wide, test_model, type = "prob")[, "yes"]
sub_rf_opt <- tibble(id = test$id,
                     pobre = as.integer(probs_rf_test > opt_rf$t))
write_csv(sub_rf_opt,
          sprintf("02_output/04_submissions/RF_wide_mtry%d_node%d_F1opt.csv",
                  rf_wide$bestTune$mtry,
                  rf_wide$bestTune$min.node.size))
cat("  submissions/02_output/04_submissions/RF_wide_*_F1opt.csv\n")

# 12. Predecir ElasticNet (threshold optimo) ────────────────────────────────────
cat("[+5] ElasticNet con threshold optimo...\n")
probs_val_enet <- predict(enet_model, val, type = "prob")[, "yes"]
opt_enet       <- best_threshold_f1(probs_val_enet, val$pobre)
cat(sprintf("  Threshold optimo: %.2f | F1 val: %.4f\n",
            opt_enet$t, opt_enet$f1))
probs_enet_test <- predict(enet_model, test_model, type = "prob")[, "yes"]
sub_enet_opt <- tibble(id = test$id,
                       pobre = as.integer(probs_enet_test > opt_enet$t))
write_csv(sub_enet_opt, "02_output/04_submissions/enet_F1opt.csv")
cat("  submissions/02_output/04_submissions/enet_F1opt.csv\n")

# 13. Predecir CART (threshold optimo) ──────────────────────────────────────────
cat("[+6] CART con threshold optimo...\n")
probs_val_cart <- predict(cart_model, val, type = "prob")[, "yes"]
opt_cart       <- best_threshold_f1(probs_val_cart, val$pobre)
cat(sprintf("  Threshold optimo: %.2f | F1 val: %.4f\n",
            opt_cart$t, opt_cart$f1))
probs_cart_test <- predict(cart_model, test_model, type = "prob")[, "yes"]
sub_cart_opt <- tibble(id = test$id,
                       pobre = as.integer(probs_cart_test > opt_cart$t))
write_csv(sub_cart_opt, "02_output/04_submissions/CART_F1opt.csv")
cat("  submissions/02_output/04_submissions/CART_F1opt.csv\n")

# 14. Predecir Ensemble (XGB-SPW + RF wide + ElasticNet, threshold optimo) ─────
cat("[+7] Ensemble (XGB-SPW + RF wide + ElasticNet)...\n")
probs_val_ens  <- (probs_val_spw + probs_val_rf + probs_val_enet) / 3
opt_ens        <- best_threshold_f1(probs_val_ens, val$pobre)
cat(sprintf("  Threshold optimo: %.2f | F1 val: %.4f\n",
            opt_ens$t, opt_ens$f1))
probs_ens_test <- (probs_spw_test + probs_rf_test + probs_enet_test) / 3
sub_ens <- tibble(id = test$id,
                  pobre = as.integer(probs_ens_test > opt_ens$t))
write_csv(sub_ens,
          "02_output/04_submissions/ensemble_xgbspw_rfwide_enet_F1opt.csv")
cat("  submissions/02_output/04_submissions/ensemble_xgbspw_rfwide_enet_F1opt.csv\n")

cat("\submissions adicionales guardadas.") 

  