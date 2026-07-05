# Proyecto de Métodos Numéricos: modelos predictivos

Código para el artículo **Predicción del índice académico en estudiantes universitarios mediante regresión lineal múltiple y regresión exponencial**.

El proyecto analiza respuestas de una encuesta sobre gestión del tiempo libre, calidad del sueño y rendimiento académico en estudiantes de la Universidad Tecnológica de Panamá.

## Estructura

```text
.
├── src/
│   └── analisis_modelos.py
├── paper/
│   └── paper.tex
├── data/
│   └── raw/
│       └── .gitkeep
├── results/
│   └── .gitkeep
├── requirements.txt
└── .gitignore
```

## Uso

1. Exporta las respuestas de Google Forms como CSV.
2. Guarda el archivo como:

```text
data/raw/respuestas.csv
```

3. Instala dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecuta el análisis:

```bash
python src/analisis_modelos.py --csv data/raw/respuestas.csv --out results
```

El script genera coeficientes, métricas, descriptivos y gráficas en `results/`.

## Modelos

### Regresión lineal múltiple

```text
GPA = β0 + β1(GTL) + β2(CalidadSueño) + β3(HorasSueño)
    + β4(Trabajo) + β5(SexoMasculino) + β6(Edad) + β7(AñoAcadémico) + ε
```

### Regresión exponencial log-lineal

```text
GPA = exp(α + b1(GTL) + b2(CalidadSueño) + b3(HorasSueño)
          + b4(Trabajo) + b5(SexoMasculino) + b6(Edad) + b7(AñoAcadémico))
```

## Nota sobre datos

El CSV original de respuestas no se sube al repositorio porque contiene datos de encuesta. El código está preparado para ejecutarse localmente con el archivo exportado desde Google Forms.
