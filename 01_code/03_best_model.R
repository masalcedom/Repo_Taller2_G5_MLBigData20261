#──────────────────────────────────────────────────────────────────────────────
#  Stage 3: Deep dive del mejor modelo (XGBoost)
#──────────────────────────────────────────────────────────────────────────────

cat("\n── Stage 3: Mejor modelo ────────────────────────────────────\n")

0# 0. Cargar datos y modelos ───────────────────────────────────────────────────
train         <- read_rds("00_data/train_clean.rds")
test          <- read_rds("00_data/test_clean.rds")
tabla_modelos <- read_rds("02_output/03_models/tabla_comparativa.rds")
# xgb_model     <- xgb.load("02_output/03_models/xgb_model.ubj")
xgb_model <- readRDS("02_output/03_models/xgb_model.rds")

cat("✔ Mejor modelo: XGBoost | F1 =",
    round(tabla_modelos$f1[1], 4), "\n")

# 1. Preparar datos igual que en 02_models ────────────────────────────────────
set.seed(42)
drop_cols <- c("id", "pobre", "fex_c", "fex_dpto")
features  <- setdiff(names(train), drop_cols)

train_model <- train |>
  select(all_of(c(features, "pobre"))) |>
  mutate(across(where(is.character), ~as.integer(as.factor(.))))

idx       <- createDataPartition(train_model$pobre, p = 0.8, list = FALSE)
val       <- train_model[-idx, ]
val_mat   <- val |> select(-pobre) |> as.matrix()
val_label <- as.integer(val$pobre == "yes")
dval      <- xgb.DMatrix(data = val_mat, label = val_label)

# 2. Predicciones en validacion ───────────────────────────────────────────────
probs <- predict(xgb_model, dval)
preds <- factor(ifelse(probs > 0.5, "yes", "no"),
                levels = c("no", "yes"))
cm    <- confusionMatrix(preds, val$pobre, positive = "yes")

cat("\nMatriz de confusion:\n")
print(cm$table)
cat(sprintf("\nF1: %.4f | Precision: %.4f | Recall: %.4f\n",
            cm$byClass["F1"], cm$byClass["Precision"], cm$byClass["Recall"]))

# 3. Grafico: matriz de confusion ─────────────────────────────────────────────
cm$table |>
  as.data.frame() |>
  ggplot(aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = scales::comma(Freq)),
            size = 5, fontface = "bold", color = "white") +
  scale_fill_gradient(low = "#85B7EB", high = "#185FA5") +
  labs(title = "Matriz de confusion — XGBoost",
       x = "Real", y = "Predicho") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

ggsave("02_output/01_figures/confusion_matrix.png", width = 5, height = 4, dpi = 200)

# 4. Grafico: curva ROC ───────────────────────────────────────────────────────
roc_obj <- roc(val$pobre, probs, quiet = TRUE)
roc_df  <- data.frame(
  fpr = 1 - roc_obj$specificities,
  tpr = roc_obj$sensitivities
)

ggplot(roc_df, aes(fpr, tpr)) +
  geom_line(color = "#185FA5", linewidth = 1.2) +
  geom_abline(linetype = "dashed", color = "#888780") +
  annotate("text", x = 0.75, y = 0.25,
           label = paste0("AUC = ", round(auc(roc_obj), 3)),
           size = 5, color = "#185FA5") +
  labs(title = "Curva ROC — XGBoost",
       x = "Tasa de falsos positivos",
       y = "Tasa de verdaderos positivos") +
  theme_minimal(base_size = 13)

ggsave("02_output/01_figures/curva_roc.png", width = 6, height = 5, dpi = 200)

# 5. Grafico: feature importance ──────────────────────────────────────────────
imp_matrix <- xgb.importance(model = xgb_model)

ggplot(imp_matrix[1:min(15, nrow(imp_matrix)), ],
       aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#1D9E75", width = 0.7) +
  coord_flip() +
  labs(title = "Importancia de variables — XGBoost",
       subtitle = "Gain: reduccion promedio de impureza por variable",
       x = NULL, y = "Gain") +
  theme_minimal(base_size = 13)

ggsave("02_output/01_figures/feature_importance.png", width = 8, height = 6, dpi = 200)

cat("\nTop 10 variables mas importantes:\n")
print(imp_matrix[1:min(10, nrow(imp_matrix)), c("Feature", "Gain")])

# 6. Grafico: precision-recall por umbral ─────────────────────────────────────
pr_df <- data.frame(threshold = seq(0.1, 0.9, by = 0.02)) |>
  rowwise() |>
  mutate(
    pred = list(factor(ifelse(probs > threshold, "yes", "no"),
                       levels = c("no", "yes"))),
    cm_t = list(confusionMatrix(pred, val$pobre, positive = "yes")),
    prec = cm_t$byClass["Precision"],
    rec  = cm_t$byClass["Recall"],
    f1   = cm_t$byClass["F1"]
  ) |>
  select(threshold, prec, rec, f1)

ggplot(pr_df, aes(x = threshold)) +
  geom_line(aes(y = prec, color = "Precision"), linewidth = 1) +
  geom_line(aes(y = rec,  color = "Recall"),    linewidth = 1) +
  geom_line(aes(y = f1,   color = "F1"),        linewidth = 1, linetype = "dashed") +
  scale_color_manual(
    values = c(Precision = "#1D9E75", Recall = "#D85A30", F1 = "#185FA5")
  ) +
  labs(title = "Precision, Recall y F1 por umbral — XGBoost",
       subtitle = "Umbral bajo = menos undercoverage | Umbral alto = menos leakage",
       x = "Umbral de clasificacion", y = "Score", color = NULL) +
  theme_minimal(base_size = 13)

ggsave("02_output/01_figures/precision_recall_threshold.png", width = 8, height = 5, dpi = 200)

best_threshold_XBOOST <- pr_df[which.max(pr_df$f1), ]
best_threshold_XBOOST <- best_threshold_XBOOST$threshold
best_threshold_XBOOST
cat("\n✔ Stage 3 completo. Figuras guardadas en figures/\n")
cat("   confusion_matrix.png\n")
cat("   curva_roc.png\n")
cat("   feature_importance.png\n")
cat("   precision_recall_threshold.png\n")

