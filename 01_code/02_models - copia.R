#──────────────────────────────────────────────────────────────────────────────
#  Stage 2: Entrenamiento y comparacion de algoritmos
#  LPM · Logit · Elastic Net · CART · Random Forest · XGBoost · Naive Bayes
#──────────────────────────────────────────────────────────────────────────────

cat("\n── Stage 2: Modelos ─────────────────────────────────────────\n")

# 0. Cargar datos ─────────────────────────────────────────────────────────────
train <- read_rds("00_data/train_clean.rds")
test  <- read_rds("00_data/test_clean.rds")

# 1. Preparar features ────────────────────────────────────────────────────────
set.seed(2005)

drop_cols <- c("id", "pobre", "fex_c", "fex_dpto")
features  <- setdiff(names(train), drop_cols)

train_model <- train |>
  select(all_of(c(features, "pobre"))) |>
  mutate(across(where(is.character), ~as.integer(as.factor(.))))


idx <- createDataPartition(train_model$pobre, p = 0.8, list = FALSE)
tr  <- train_model[ idx, ]
val <- train_model[-idx, ]

cat("✔ Split: train =", nrow(tr), "| val =", nrow(val), "\n")

# 2. Control de CV + ROSE ─────────────────────────────────────────────────────

ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  sampling        = "rose",
  savePredictions = "final",
  verboseIter     = FALSE
)

# 3. Funcion de evaluacion ─────────────────────────────────────────────────────
eval_model <- function(model, val_data, nombre) {
  preds <- predict(model, val_data)
  probs <- predict(model, val_data, type = "prob")[, "yes"]
  cm    <- confusionMatrix(preds, val_data$pobre, positive = "yes")
  f1    <- cm$byClass["F1"]
  prec  <- cm$byClass["Precision"]
  rec   <- cm$byClass["Recall"]
  auc   <- as.numeric(auc(roc(val_data$pobre, probs, quiet = TRUE)))
  cat(sprintf("  %-20s F1: %.4f | Prec: %.4f | Rec: %.4f | AUC: %.4f\n",
              nombre, f1, prec, rec, auc))
  tibble(modelo = nombre, f1 = f1, precision = prec, recall = rec, auc = auc)
}

resultados <- list()

# ── 4. LPM ────────────────────────────────────────────────────────────────────
cat("\n[1/7] LPM...\n")
tr_lpm    <- tr |> mutate(pobre_num = as.integer(pobre == "yes"))
test_lmp  <- val |> mutate(pobre_num = as.integer(pobre == "yes"))
lpm       <- lm(pobre_num ~ . - pobre, data = tr_lpm)
lpm_preds <- factor(
  ifelse(predict(lpm, val) > 0.5, "yes", "no"),
  levels = c("no", "yes")
)
lpm_probs <- pmin(pmax(predict(lpm, val), 0), 1)
cm_lpm    <- confusionMatrix(lpm_preds, val$pobre, positive = "yes")
resultados[["LPM"]] <- tibble(
  modelo    = "LPM",
  f1        = cm_lpm$byClass["F1"],
  precision = cm_lpm$byClass["Precision"],
  recall    = cm_lpm$byClass["Recall"],
  auc       = as.numeric(auc(roc(val$pobre, lpm_probs, quiet = TRUE)))
)
cat(sprintf("  %-20s F1: %.4f | AUC: %.4f\n",
            "LPM", resultados[["LPM"]]$f1, resultados[["LPM"]]$auc))

# ── 5. Logit ──────────────────────────────────────────────────────────────────
cat("[2/7] Logit...\n")
logit <- train(
  pobre ~ ., data = tr,
  method    = "glm",
  family    = "binomial",
  metric    = "ROC",
  trControl = ctrl
)
resultados[["Logit"]] <- eval_model(logit, val, "Logit")

# ── 6. Elastic Net ────────────────────────────────────────────────────────────

## Funcion para r1
f1Summary <- function(data, lev = NULL, model = NULL) {
  # Matriz de confusión
  cm <- caret::confusionMatrix(data$pred, data$obs, positive = lev[2])
  precision <- cm$byClass["Precision"]
  recall    <- cm$byClass["Recall"]
  f1 <- 2 * (precision * recall) / (precision + recall)
  c(F1 = f1)
}


ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = f1Summary,
  sampling        = "rose",
  savePredictions = "final",
  verboseIter     = FALSE
)

# 
# ctrl <- trainControl(
#   method          = "cv",
#   number          = 5,
#   classProbs      = TRUE,
#   summaryFunction = twoClassSummary,
#   sampling        = "rose",
#   savePredictions = "final",
#   verboseIter     = FALSE
# )


cat("[3/7] Elastic Net...\n")
enet <- train(
  pobre ~ ., data = tr,
  method    = "glmnet",
  metric    = "F1",
  trControl = ctrl,
  tuneGrid  = expand.grid(
    alpha  = c(0, 0.5, 1),
    lambda = 10^seq(-4, 0, length = 10)
  )
)

enet$bestTune
resultados[["ElasticNet"]] <- eval_model(enet, val, "Elastic Net")


# ── 7. CART ───────────────────────────────────────────────────────────────────
cat("[4/7] CART...\n")
cart <- train(
  pobre ~ ., data = tr,
  method    = "rpart",
  metric    = "F1",
  trControl = ctrl,
  tuneGrid  = data.frame(cp = 10^seq(-5, -1, length = 10))
)
cart$bestTune
resultados[["CART"]] <- eval_model(cart, val, "CART")

# ── 8. Random Forest ──────────────────────────────────────────────────────────
cat("[5/7] Random Forest...\n")
rf <- train(
  pobre ~ ., data = tr,
  method    = "ranger",
  metric    = "F1",
  trControl = ctrl,
  tuneGrid  = expand.grid(
    mtry          = c(4, 6),
    splitrule     = "gini",
    min.node.size = 10
  ),
  num.trees = 100
)
resultados[["RF"]] <- eval_model(rf, val, "Random Forest")
rf$bestTune
# ── 9. XGBoost ────────────────────────────────────────────────────────────────
cat("[6/7] XGBoost...\n")
# library(xgboost)
#cl <- makeCluster(4)
#registerDoParallel(cl)

tr_matrix  <- tr  |> select(-pobre) |> as.matrix()
val_matrix <- val |> select(-pobre) |> as.matrix()
tr_label   <- as.integer(tr$pobre  == "yes")
val_label  <- as.integer(val$pobre == "yes")

dtrain <- xgb.DMatrix(data = tr_matrix,  label = tr_label)
dval   <- xgb.DMatrix(data = val_matrix, label = val_label)

cv_xgb <- xgb.cv(
  params = list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    max_depth        = 6,
    eta              = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8
  ),
  data    = dtrain,
  nrounds = 200,
  nfold   = 5,
  verbose = 0,
  early_stopping_rounds = 20
)

best_nrounds <- cv_xgb$best_iteration
if (length(best_nrounds) == 0 || is.na(best_nrounds)) best_nrounds <- 100
cat("  Mejor nrounds:", best_nrounds, "\n")

xgb_model <- xgb.train(
  params = list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    max_depth        = 6,
    eta              = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8
  ),
  data    = dtrain,
  nrounds = best_nrounds,
  verbose = 0
)


# Guardar INMEDIATAMENTE con xgb.save
dir.create("models", showWarnings = FALSE)
xgb.save(xgb_model, "02_output/03_models/xgb_model.ubj")
saveRDS(xgb_model, "02_output/03_models/xgb_model.rds")

xgb_probs <- predict(xgb_model, dval)
xgb_preds <- factor(ifelse(xgb_probs > 0.5, "yes", "no"),
                    levels = c("no", "yes"))
cm_xgb <- confusionMatrix(xgb_preds, val$pobre, positive = "yes")
resultados[["XGBoost"]] <- tibble(
  modelo    = "XGBoost",
  f1        = cm_xgb$byClass["F1"],
  precision = cm_xgb$byClass["Precision"],
  recall    = cm_xgb$byClass["Recall"],
  auc       = as.numeric(auc(roc(val$pobre, xgb_probs, quiet = TRUE)))
)
cat(sprintf("  %-20s F1: %.4f | Prec: %.4f | Rec: %.4f | AUC: %.4f\n",
            "XGBoost",
            resultados[["XGBoost"]]$f1,
            resultados[["XGBoost"]]$precision,
            resultados[["XGBoost"]]$recall,
            resultados[["XGBoost"]]$auc))

# ── 10. Naive Bayes ───────────────────────────────────────────────────────────
cat("[7/7] Naive Bayes...\n")
library(MLmetrics)

# f1Summary <- function(data, lev = NULL, model = NULL) {
#   f1 <- F1_Score(y_pred = data$pred, y_true = data$obs, positive = lev[1])
#   c(F1 = f1)
#}
ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  sampling        = "rose",
  savePredictions = "final",
  verboseIter     = FALSE
)

# ctrl <- trainControl(
#   method = "cv",
#   number = 5,
#   summaryFunction = f1Summary,
#   classProbs = TRUE
# )

nb <- train(
  pobre ~ .,
  data = tr,
  method    = "naive_bayes",
  metric    = "ROC",
  trControl = ctrl
)
resultados[["NB"]] <- eval_model(nb, val, "Naive Bayes")

####__________________________________________________________
# ##  Probar threshold
# 
# probs <- predict(nb, newdata = val, type = "prob")
# p <- probs$yes
# y_true <- val$pobre
# thresholds <- seq(0, 1, by = 0.01)
# 
# f1_scores <- sapply(thresholds, function(t) {
#   y_pred <- ifelse(p > t, "yes", "no")
#   F1_Score(y_pred = y_pred, y_true = y_true, positive = "yes")
# })
# best_t <- thresholds[which.max(f1_scores)]
# best_f1 <- max(f1_scores)
# 
# best_t
# best_f1
# y_pred_opt <- ifelse(p > best_t, "yes", "no")
# 
# nb_PR <- train(
#   pobre ~ .,
#   data = tr,
#   method    = "naive_bayes",
#   metric    = "ROC",
#   trControl = ctrl
# )

####__________________________________________________________

# 11. Tabla comparativa ───────────────────────────────────────────────────────
tabla_modelos <- bind_rows(resultados) |>
  arrange(desc(f1))

cat("\n── Resumen comparativo ──────────────────────────────────────\n")
print(tabla_modelos)

tabla_modelos |>
  pivot_longer(c(f1, precision, recall, auc), names_to = "metrica") |>
  ggplot(aes(x = reorder(modelo, value), y = value, fill = metrica)) +
  geom_col(position = "dodge", width = 0.7) +
  coord_flip() +
  scale_fill_manual(
    values = c(f1 = "#378ADD", precision = "#1D9E75",
               recall = "#D85A30", auc = "#7F77DD")
  ) +
  labs(title = "Comparacion de algoritmos en validacion",
       x = NULL, y = "Score", fill = "Metrica") +
  theme_minimal(base_size = 12)

ggsave("02_output/01_figures/comparacion_modelos.png", width = 10, height = 6, dpi = 200)

# 12. Guardar todo ─────────────────────────────────────────────────────────────
write_rds(lpm,           "02_output/03_models/lpm.rds")
write_rds(logit,         "02_output/03_models/logit.rds")
write_rds(enet,          "02_output/03_models/enet.rds")
write_rds(cart,          "02_output/03_models/cart.rds")
write_rds(rf,            "02_output/03_models/rf.rds")
write_rds(nb,            "02_output/03_models/nb.rds")
write_rds(tabla_modelos, "02_output/02_tables/tabla_comparativa.rds")

cat("\n✔ Stage 2 completo. Mejor modelo:",
    tabla_modelos$modelo[1], "| F1 =",
    round(tabla_modelos$f1[1], 4), "\n")

