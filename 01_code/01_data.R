# scripts/01_data.R
# Stage 1: Carga, limpieza, feature engineering y EDA
# Datos reales: GEIH Colombia — MECA 4107

library(pacman)
p_load(tidyverse, janitor, skimr, visdat, ggpubr, patchwork)

# ── 1. Cargar datos ───────────────────────────────────────────────────────────
train_hh  <- read_csv("data/train_hogares.csv")  |> clean_names()
train_ind <- read_csv("data/train_personas.csv") |> clean_names()
test_hh   <- read_csv("data/test_hogares.csv")   |> clean_names()
test_ind  <- read_csv("data/test_personas.csv")  |> clean_names()

# ── 2. Feature engineering desde personas ────────────────────────────────────
# Colapsar al nivel hogar usando las variables disponibles en AMBOS sets
build_ind_features <- function(ind) {
  ind |>
    group_by(id) |>
    summarise(
      hh_size          = n(),
      max_educ         = max(p6210, na.rm = TRUE),
      mean_educ        = mean(p6210, na.rm = TRUE),
      head_sex         = first(p6020[orden == 1]),     # 1=hombre, 2=mujer
      head_age         = first(p6040[orden == 1]),
      head_educ        = first(p6210[orden == 1]),
      head_employed    = first(oc[orden == 1]),        # 1=ocupado
      n_employed       = sum(oc == 1, na.rm = TRUE),
      n_unemployed     = sum(des == 1, na.rm = TRUE),
      n_inactive       = sum(ina == 1, na.rm = TRUE),
      n_children       = sum(p6040 < 15, na.rm = TRUE),
      n_elderly        = sum(p6040 >= 65, na.rm = TRUE),
      share_employed   = n_employed / hh_size,
      dependency_ratio = (n_children + n_elderly) /
        pmax(hh_size - n_children - n_elderly, 1),
      head_female      = as.integer(first(p6020[orden == 1]) == 2),
      any_professional = as.integer(any(p6430 == 1, na.rm = TRUE)) # emp. particular
    )
}

ind_features_train <- build_ind_features(train_ind)
ind_features_test  <- build_ind_features(test_ind)

# ── 3. Merge ──────────────────────────────────────────────────────────────────
# Variables a descartar: solo existen en train (ingreso y derivados)
drop_train_only <- c("indigente", "ingtotug", "ingtotugarr",
                     "ingpcug", "npobres", "nindigentes")

train <- train_hh |>
  left_join(ind_features_train, by = "id") |>
  mutate(pobre = factor(pobre, levels = c(0, 1),
                        labels = c("no", "yes"))) |>
  select(-all_of(drop_train_only))

test <- test_hh |>
  left_join(ind_features_test, by = "id")

# ── 4. Limpiar NAs ────────────────────────────────────────────────────────────
# Imputar medianas en numéricas, "Desconocido" en categóricas
train <- train |>
  mutate(across(where(is.numeric) & !c(id),
                ~replace_na(., median(., na.rm = TRUE)))) |>
  mutate(across(where(is.character),
                ~replace_na(., "Desconocido")))

test <- test |>
  mutate(across(where(is.numeric) & !c(id),
                ~replace_na(., median(., na.rm = TRUE)))) |>
  mutate(across(where(is.character),
                ~replace_na(., "Desconocido")))

# ── 5. EDA ────────────────────────────────────────────────────────────────────

# 5a. Desbalance de clases
train |> count(pobre) |>
  mutate(pct = scales::percent(n / sum(n)))

# 5b. Tasa de pobreza por educación máxima del hogar
train |>
  mutate(educ_bin = cut(max_educ, breaks = c(-Inf, 1, 3, 5, 7, Inf),
                        labels = c("Ninguna","Primaria","Secundaria","Media","Superior"))) |>
  group_by(educ_bin) |>
  summarise(tasa_pobreza = mean(pobre == "yes"),
            n = n()) |>
  ggplot(aes(x = educ_bin, y = tasa_pobreza, fill = educ_bin)) +
  geom_col(width = 0.6, show.legend = FALSE) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("#F09595","#EF9F27","#97C459","#5DCAA5","#85B7EB")) +
  labs(title = "Tasa de pobreza según nivel educativo del hogar",
       x = "Nivel educativo máximo", y = "% de hogares pobres") +
  theme_minimal(base_size = 13)

ggsave("figures/pobreza_por_educacion.png", width = 8, height = 5, dpi = 200)

# 5c. Distribución de variables clave por estatus
train |>
  select(pobre, hh_size, dependency_ratio, share_employed, head_age) |>
  pivot_longer(-pobre) |>
  ggplot(aes(x = value, fill = pobre)) +
  geom_density(alpha = 0.45) +
  facet_wrap(~name, scales = "free") +
  scale_fill_manual(values = c("no" = "#378ADD", "yes" = "#D85A30")) +
  labs(title = "Distribución de variables por condición de pobreza",
       fill = "Pobre", x = NULL) +
  theme_minimal(base_size = 12)

ggsave("figures/distribuciones.png", width = 10, height = 6, dpi = 200)

# ── 6. Guardar ────────────────────────────────────────────────────────────────
dir.create("data", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

write_rds(train, "data/train_clean.rds")
write_rds(test,  "data/test_clean.rds")

cat("\n✔ Stage 1 completo.\n",
    "  Train:", nrow(train), "hogares |",
    "  Test:", nrow(test), "hogares\n",
    "  Variables en modelo:", ncol(train) - 2, "\n")