#──────────────────────────────────────────────────────────────────────────────
#  Script 06: Todas las figuras, tablas y mapas para las slides
#──────────────────────────────────────────────────────────────────────────────

cat("\n── Generando figuras para slides ────────────────────────────\n")

library(tidyverse)
library(pROC)
library(caret)
library(xgboost)
library(sf)
library(geodata)
library(scales)
library(patchwork)

dir.create("figures", showWarnings = FALSE)

col_poor   <- "#D85A30"
col_nopoor <- "#378ADD"
col_teal   <- "#0D9488"
col_navy   <- "#1B2A4A"

# 0. Cargar datos y modelo
train     <- read_rds("data/train_clean.rds")
xgb_model <- xgb.load("models/xgb_model.ubj")
tabla_mod <- read_rds("models/tabla_comparativa.rds")

set.seed(42)
drop_cols <- c("id", "pobre", "fex_c", "fex_dpto")
features  <- setdiff(names(train), drop_cols)

train_model <- train |>
  select(all_of(c(features, "pobre"))) |>
  mutate(across(where(is.character), ~as.integer(as.factor(.))))

idx <- createDataPartition(train_model$pobre, p = 0.8, list = FALSE)
val <- train_model[-idx, ]

val_mat   <- val |> select(-pobre) |> as.matrix()
val_label <- as.integer(val$pobre == "yes")
dval      <- xgb.DMatrix(data = val_mat, label = val_label)
probs     <- predict(xgb_model, dval)
preds     <- factor(ifelse(probs > 0.4, "yes", "no"), levels = c("no", "yes"))
cm        <- confusionMatrix(preds, val$pobre, positive = "yes")

cat("✔ Datos y modelo cargados\n")

# 1. Desbalance de clases
p_imbalance <- train |>
  count(pobre) |>
  mutate(label = ifelse(pobre == "yes", "Poor (20%)", "Non-poor (80%)"),
         pct   = n / sum(n)) |>
  ggplot(aes(x = label, y = n, fill = pobre)) +
  geom_col(width = 0.5, show.legend = FALSE) +
  geom_text(aes(label = paste0(comma(n), "\n(", percent(pct, accuracy = 1), ")")),
            vjust = -0.3, size = 4, fontface = "bold") +
  scale_fill_manual(values = c(no = col_nopoor, yes = col_poor)) +
  scale_y_continuous(labels = comma, expand = expansion(c(0, 0.15))) +
  labs(title = "Class imbalance: 80/20 split",
       subtitle = "164,960 households in training set",
       x = NULL, y = "Number of households") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figures/class_imbalance.png", p_imbalance, width = 6, height = 4, dpi = 200)

# 2. Pobreza por educacion
p_educ <- train |>
  mutate(educ_bin = cut(max_educ,
                        breaks = c(-Inf, 1, 3, 5, 7, Inf),
                        labels = c("None", "Primary", "Secondary", "High school", "University")
  )) |>
  group_by(educ_bin) |>
  summarise(tasa = mean(pobre == "yes"), n = n()) |>
  ggplot(aes(x = educ_bin, y = tasa, fill = educ_bin)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = percent(tasa, accuracy = 1)),
            vjust = -0.4, size = 4, fontface = "bold") +
  scale_y_continuous(labels = percent, expand = expansion(c(0, 0.12))) +
  scale_fill_manual(values = c("#DC2626","#D97706","#F59E0B","#16A34A","#0D9488")) +
  labs(title = "Poverty rate by maximum education level in household",
       subtitle = "Households with no schooling are 8x more likely to be poor",
       x = "Maximum education level", y = "Poverty rate") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figures/poverty_by_education.png", p_educ, width = 8, height = 5, dpi = 200)

# 3. Distribuciones
p_dist <- train |>
  select(pobre, hh_size, dependency_ratio, share_employed, head_age) |>
  pivot_longer(-pobre) |>
  mutate(name = recode(name,
                       hh_size = "Household size", dependency_ratio = "Dependency ratio",
                       share_employed = "Share employed", head_age = "Age of household head"
  )) |>
  ggplot(aes(x = value, fill = pobre)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~name, scales = "free", ncol = 2) +
  scale_fill_manual(values = c(no = col_nopoor, yes = col_poor),
                    labels = c("Non-poor", "Poor")) +
  labs(title = "Feature distributions by poverty status",
       fill = NULL, x = NULL, y = "Density") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom")

ggsave("figures/distributions.png", p_dist, width = 9, height = 6, dpi = 200)

# 4. Confusion matrix
p_cm <- cm$table |>
  as.data.frame() |>
  mutate(label = paste0(comma(Freq), "\n",
                        case_when(
                          Prediction == "yes" & Reference == "yes" ~ "True Positive",
                          Prediction == "no"  & Reference == "no"  ~ "True Negative",
                          Prediction == "yes" & Reference == "no"  ~ "False Positive\n(Leakage)",
                          Prediction == "no"  & Reference == "yes" ~ "False Negative\n(Undercoverage)"
                        ))) |>
  ggplot(aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white", linewidth = 2) +
  geom_text(aes(label = label), size = 4, fontface = "bold", color = "white") +
  scale_fill_gradient(low = "#85B7EB", high = col_navy) +
  scale_x_discrete(labels = c(no = "Non-poor", yes = "Poor")) +
  scale_y_discrete(labels = c(no = "Non-poor", yes = "Poor")) +
  labs(title = "Confusion Matrix — XGBoost (threshold = 0.4)",
       x = "Actual", y = "Predicted") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none", plot.title = element_text(face = "bold"))

ggsave("figures/confusion_matrix.png", p_cm, width = 6, height = 5, dpi = 200)

# 5. ROC
roc_obj <- roc(val$pobre, probs, quiet = TRUE)
auc_val <- round(auc(roc_obj), 3)

p_roc <- data.frame(fpr = 1 - roc_obj$specificities, tpr = roc_obj$sensitivities) |>
  ggplot(aes(fpr, tpr)) +
  geom_line(color = col_teal, linewidth = 1.4) +
  geom_abline(linetype = "dashed", color = "#888780") +
  geom_area(alpha = 0.08, fill = col_teal) +
  annotate("label", x = 0.75, y = 0.25,
           label = paste0("AUC = ", auc_val),
           size = 5, color = col_teal, fontface = "bold",
           fill = "white", label.size = 0.5) +
  labs(title = "ROC Curve — XGBoost",
       subtitle = "Area Under Curve measures overall discriminatory power",
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figures/curva_roc.png", p_roc, width = 6, height = 5, dpi = 200)

# 6. Precision-recall
pr_df <- data.frame(threshold = seq(0.1, 0.9, by = 0.05)) |>
  rowwise() |>
  mutate(
    pred = list(factor(ifelse(probs > threshold, "yes", "no"), levels = c("no", "yes"))),
    cm_t = list(confusionMatrix(pred, val$pobre, positive = "yes")),
    prec = cm_t$byClass["Precision"],
    rec  = cm_t$byClass["Recall"],
    f1   = cm_t$byClass["F1"]
  ) |>
  select(threshold, prec, rec, f1)

p_pr <- pr_df |>
  pivot_longer(c(prec, rec, f1), names_to = "metric") |>
  mutate(metric = recode(metric, prec="Precision", rec="Recall", f1="F1")) |>
  ggplot(aes(x = threshold, y = value, color = metric, linetype = metric)) +
  geom_line(linewidth = 1.2) +
  geom_vline(xintercept = 0.4, linetype = "dashed", color = col_navy, alpha = 0.5) +
  annotate("text", x = 0.42, y = 0.2, label = "Optimal\nthreshold = 0.4",
           size = 3.5, color = col_navy) +
  scale_color_manual(values = c(Precision = col_teal, Recall = col_poor, F1 = col_navy)) +
  scale_linetype_manual(values = c(Precision = "solid", Recall = "solid", F1 = "dashed")) +
  labs(title = "Precision, Recall & F1 by Classification Threshold",
       subtitle = "Low threshold = less undercoverage | High threshold = less leakage",
       x = "Classification threshold", y = "Score", color = NULL, linetype = NULL) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), legend.position = "bottom")

ggsave("figures/precision_recall_threshold.png", p_pr, width = 8, height = 5, dpi = 200)

# 7. Feature importance
imp <- xgb.importance(model = xgb_model)

p_imp <- imp[1:min(15, nrow(imp)), ] |>
  ggplot(aes(x = reorder(Feature, Gain), y = Gain,
             fill = Gain > median(imp$Gain[1:15]))) +
  geom_col(width = 0.7, show.legend = FALSE) +
  geom_text(aes(label = round(Gain, 3)), hjust = -0.1, size = 3.2) +
  coord_flip() +
  scale_fill_manual(values = c("FALSE"="#85B7EB", "TRUE"=col_teal)) +
  labs(title = "Feature Importance — XGBoost (Gain)",
       subtitle = "Gain: average reduction in impurity contributed by each variable",
       x = NULL, y = "Gain") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figures/feature_importance.png", p_imp, width = 9, height = 6, dpi = 200)

# 8. Comparacion de modelos
p_comp <- tabla_mod |>
  mutate(modelo = fct_reorder(modelo, f1)) |>
  ggplot(aes(x = modelo, y = f1, fill = modelo == tabla_mod$modelo[1])) +
  geom_col(width = 0.6, show.legend = FALSE) +
  geom_text(aes(label = round(f1, 3)), hjust = -0.1, size = 4, fontface = "bold") +
  coord_flip() +
  scale_fill_manual(values = c("FALSE"="#85B7EB", "TRUE"=col_teal)) +
  scale_y_continuous(limits = c(0, 0.75)) +
  labs(title = "Algorithm Comparison — F1 Score on Validation Set",
       subtitle = "XGBoost achieves the highest F1, balancing precision and recall",
       x = NULL, y = "F1 Score") +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave("figures/model_comparison.png", p_comp, width = 8, height = 5, dpi = 200)

# 9. Tabla CSV
tabla_mod |>
  mutate(across(c(f1, precision, recall, auc), ~round(., 3))) |>
  arrange(desc(f1)) |>
  rename(Model=modelo, F1=f1, Precision=precision, Recall=recall, AUC=auc) |>
  write_csv("figures/tabla_modelos.csv")

# 10. Mapa
col_map <- gadm(country = "COL", level = 1, path = "data/") |> st_as_sf()

dane_codes <- tibble(
  depto  = c("05","08","11","13","15","17","18","19","20","23",
             "25","27","41","44","47","50","52","54","63","66","68","70","73","76"),
  NAME_1 = c("Antioquia","Atlántico","Bogotá D.C.","Bolívar","Boyacá","Caldas",
             "Caquetá","Cauca","Cesar","Córdoba","Cundinamarca","Chocó","Huila",
             "La Guajira","Magdalena","Meta","Nariño","Norte de Santander",
             "Quindío","Risaralda","Santander","Sucre","Tolima","Valle del Cauca")
)

pobreza_depto <- train |>
  group_by(depto) |>
  summarise(tasa_pobreza = mean(pobre == "yes"), n_hogares = n()) |>
  left_join(dane_codes, by = "depto")

mapa_data <- col_map |> left_join(pobreza_depto, by = "NAME_1")

ggplot(mapa_data) +
  geom_sf(aes(fill = tasa_pobreza), color = "white", linewidth = 0.3) +
  scale_fill_gradient(low = "#E6F1FB", high = col_poor,
                      na.value = "#D3D1C7", labels = percent, name = "Poverty\nrate") +
  labs(title = "Poverty rate by department", subtitle = "Colombia — GEIH 2024",
       caption = "Source: DANE - GEIH | Grey = not in sample") +
  theme_void(base_size = 12) +
  theme(plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5, color = "gray50"),
        plot.caption = element_text(hjust = 0.5, color = "gray60", size = 9))

ggsave("figures/mapa_pobreza.png", width = 7, height = 9, dpi = 200)

cat("\n✔ Script 06 completo. 10 figuras generadas en figures/\n")