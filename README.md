---
title: Incidencias AIE
emoji: 📊
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: "1.38.0"
app_file: incidencias.py
pinned: false
---

# Incidencias AIE

Aplicación Streamlit para explorar y segmentar lotes por **Restricciones** y **Oportunidades** a partir de una geodatabase (GDB/GPKG). Optimizada para ejecutarse sin dependencias pesadas de GDAL/Fiona usando **PyOgrio** y **GeoPandas**.

## 🚀 Uso en Hugging Face Spaces

- Este Space está configurado para ejecutar `incidencias.py` (ver front‑matter arriba).
- Requisitos en `requirements.txt` y versión de Python en `runtime.txt`.

## ▶️ Ejecución local

```bash
pip install -r requirements.txt
streamlit run incidencias.py
```

## 📁 Estructura mínima del repo

```
.
├── incidencias.py         # archivo principal de la app
├── requirements.txt       # dependencias (sin GDAL/Fiona)
└── runtime.txt            # versión de Python (p.ej., 3.12)
```

## 🗂️ Entrada de datos

- Sube tu **.gdb** o especifica una **URL remota** a un GPKG/GeoJSON.
- La app limita vistas pesadas para evitar exceder el tamaño en navegador.

## ⚙️ Notas técnicas

- Lectura de capas con `engine="pyogrio"` para mayor compatibilidad en Spaces.
- Evita `fiona/gdal` salvo que sea imprescindible.
- Para muestras grandes, preferir resúmenes o descarga de resultados.

---

Consulta la referencia de configuración de Spaces:  
https://huggingface.co/docs/hub/spaces-config-reference
