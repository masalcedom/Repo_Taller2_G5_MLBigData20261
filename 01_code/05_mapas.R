
cat("\n── Stage 5: Mapas ───────────────────────────────────────────\n")

library(sf); library(geodata)

col_map <- gadm(country="COL", level=1, path="data/") |> st_as_sf()

dane_codes <- tibble(
  depto  = c("05","08","11","13","15","17","18","19","20","23",
             "25","27","41","44","47","50","52","54","63","66",
             "68","70","73","76"),
  NAME_1 = c("Antioquia","Atlántico","Bogotá D.C.","Bolívar",
             "Boyacá","Caldas","Caquetá","Cauca","Cesar","Córdoba",
             "Cundinamarca","Chocó","Huila","La Guajira","Magdalena",
             "Meta","Nariño","Norte de Santander","Quindío","Risaralda",
             "Santander","Sucre","Tolima","Valle del Cauca")
)

train <- read_rds("data/train_clean.rds")

pobreza_depto <- train |>
  group_by(depto) |>
  summarise(tasa_pobreza=mean(pobre=="yes"), n_hogares=n()) |>
  left_join(dane_codes, by="depto")

mapa_data <- col_map |> left_join(pobreza_depto, by="NAME_1")

extremos <- pobreza_depto |>
  arrange(desc(tasa_pobreza)) |>
  slice(c(1:3, (n()-2):n()))

mapa_extremos <- col_map |>
  left_join(extremos, by="NAME_1") |>
  filter(!is.na(tasa_pobreza)) |>
  st_centroid()

ggplot(mapa_data) +
  geom_sf(aes(fill=tasa_pobreza), color="white", linewidth=0.3) +
  geom_sf_label(data=mapa_extremos,
                aes(label=paste0(NAME_1,"\n",scales::percent(tasa_pobreza,accuracy=1))),
                size=2.5, fontface="bold", color="#1B2A4A",
                fill="white", alpha=0.8, label.size=0) +
  scale_fill_gradient(low="#E6F1FB", high="#D85A30", na.value="#C8D4E0",
                      labels=scales::percent, name="Tasa de\npobreza") +
  labs(title="Tasa de pobreza por departamento", subtitle="Colombia — GEIH 2024",
       caption="Fuente: DANE - GEIH | Azul = no cubiertos por la GEIH") +
  theme_void(base_size=12) +
  theme(plot.title=element_text(face="bold",hjust=0.5),
        plot.subtitle=element_text(hjust=0.5,color="gray50"),
        plot.caption=element_text(hjust=0.5,color="gray60",size=9),
        plot.background=element_rect(fill="white",color=NA))

ggsave("02_output/01_figures/mapa_pobreza.png", width=7, height=9, dpi=200)

