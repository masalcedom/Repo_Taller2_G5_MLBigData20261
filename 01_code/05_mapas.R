# scripts/05_mapas.R
library(sf)
library(geodata)
library(tidyverse)

# 1. Cargar mapa y datos ───────────────────────────────────────────────────────
col_map <- gadm(country = "COL", level = 1, path = "data/") |>
  st_as_sf()

# Tabla de codigos DANE -> nombre departamento
dane_codes <- tibble(
  depto = c("05","08","11","13","15","17","18","19","20","23",
            "25","27","41","44","47","50","52","54","63","66",
            "68","70","73","76"),
  NAME_1 = c("Antioquia","Atlántico","Bogotá D.C.","Bolívar",
             "Boyacá","Caldas","Caquetá","Cauca","Cesar","Córdoba",
             "Cundinamarca","Chocó","Huila","La Guajira","Magdalena",
             "Meta","Nariño","Norte de Santander","Quindío","Risaralda",
             "Santander","Sucre","Tolima","Valle del Cauca")
)

# 2. Calcular tasa de pobreza por depto ───────────────────────────────────────
train <- read_rds("data/train_clean.rds")

pobreza_depto <- train |>
  group_by(depto) |>
  summarise(
    tasa_pobreza = mean(pobre == "yes"),
    n_hogares    = n()
  ) |>
  left_join(dane_codes, by = "depto")

# 3. Merge con shapefile ───────────────────────────────────────────────────────
mapa_data <- col_map |>
  left_join(pobreza_depto, by = "NAME_1")

# 4. Mapa de tasa de pobreza ───────────────────────────────────────────────────
ggplot(mapa_data) +
  geom_sf(aes(fill = tasa_pobreza), color = "white", linewidth = 0.3) +
  scale_fill_gradient(
    low      = "#E6F1FB",
    high     = "#D85A30",
    na.value = "#D3D1C7",
    labels   = scales::percent,
    name     = "Tasa de\npobreza"
  ) +
  labs(
    title    = "Tasa de pobreza por departamento",
    subtitle = "Colombia — GEIH 2024",
    caption  = "Fuente: DANE - GEIH | Departamentos en gris no están en la muestra"
  ) +
  theme_void(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, color = "gray50"),
    plot.caption  = element_text(hjust = 0.5, color = "gray60", size = 9),
    legend.position = "right"
  )

ggsave("02_output/01_figures/mapa_pobreza.png", width = 7, height = 9, dpi = 200)
cat("✔ Mapa guardado en figures/mapa_pobreza.png\n")
