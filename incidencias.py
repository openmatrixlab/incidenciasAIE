import streamlit as st
import os
import io
import warnings
import pandas as pd
import geopandas as gpd
import concurrent.futures
import tempfile
import zipfile
import shutil
import atexit
from functools import partial
from typing import Optional, List, Dict

# Configuración de página debe ser lo primero
st.set_page_config(page_title="Incidencia CBML", layout="wide")

# Función para limpiar recursos temporales al salir
def cleanup_temp_resources():
    if "temp_gdb_dir" in st.session_state and os.path.exists(st.session_state.temp_gdb_dir):
        try:
            shutil.rmtree(st.session_state.temp_gdb_dir)
        except Exception as e:
            print(f"Error al limpiar recursos temporales: {e}")
# Registrar la función para ejecutarse al salir
atexit.register(cleanup_temp_resources)

# ================== Estado inicial ==================
if "df_long" not in st.session_state:
    st.session_state.df_long = None
if "df_wide" not in st.session_state:
    st.session_state.df_wide = None
if "temp_gdb_dir" not in st.session_state:
    st.session_state.temp_gdb_dir = tempfile.mkdtemp()

# ============ Utils de lectura ============
def list_gdb_layers(gdb_path: str) -> List[str]:
    if not (os.path.isdir(gdb_path) and gdb_path.lower().endswith(".gdb")):
        return []
    try:
        import pyogrio
        info = pyogrio.list_layers(gdb_path)
        return list(info["name"])
    except Exception:
        import fiona
        with fiona.Env():
            return list(fiona.listlayers(gdb_path))

@st.cache_data(ttl=3600, show_spinner=False)
def read_layer_cached(gdb_path: str, layer: str, columns: Optional[List[str]] = None) -> gpd.GeoDataFrame:
    """Versión cacheada de read_layer para evitar lecturas repetidas"""
    read_kwargs = {}
    if columns:
        read_kwargs["columns"] = list(dict.fromkeys(columns + ["geometry"]))
    try:
        import pyogrio  # noqa
        return gpd.read_file(gdb_path, layer=layer, **read_kwargs)
    except Exception:
        return gpd.read_file(gdb_path, layer=layer)

def read_layer(gdb_path: str, layer: str, columns: Optional[List[str]] = None) -> gpd.GeoDataFrame:
    # Delegamos a la versión cacheada
    return read_layer_cached(gdb_path, layer, columns)

def is_polygonal(gdf: gpd.GeoDataFrame) -> bool:
    if gdf.empty:
        return False
    gtypes = set(gdf.geometry.geom_type.unique())
    return bool(gtypes & {"Polygon", "MultiPolygon"})

def clean_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
    gdf.geometry = gdf.geometry.buffer(0)  # corrige self-intersections
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    return gdf

def auto_project(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        warnings.warn("Capa sin CRS. Se asume EPSG:4326.")
        gdf = gdf.set_crs(4326)
    try:
        from pyproj import CRS
        crs = CRS.from_user_input(gdf.crs)
        if crs.is_projected:
            return gdf
        c = gdf.unary_union.centroid
        lon, lat = c.x, c.y
        zone = int((lon + 180) // 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return gdf.to_crs(epsg)
    except Exception:
        warnings.warn("No se pudo inferir UTM. Uso EPSG:3857 (verifica unidades).")
        return gdf.to_crs(3857)

# ============ Núcleo ============
def procesar_capa(lyr_info, cons_diss_local, cbml_field_local, gdb_path_local, sliver_min_area_local):
    """Función para procesar una capa en paralelo"""
    lyr, campos = lyr_info
    registros_lyr = []
    campos_a_procesar = campos if campos else [None]

    for campo in campos_a_procesar:
        try:
            gdf = read_layer(gdb_path_local, lyr, [campo] if campo else None)
            if gdf.empty or not is_polygonal(gdf):
                continue

            # Proyectar y limpiar
            if gdf.crs != cons_diss_local.crs:
                gdf = gdf.to_crs(cons_diss_local.crs) if gdf.crs else gdf.set_crs(cons_diss_local.crs)
            gdf = clean_polygons(gdf)

            # Intersección
            left = cons_diss_local[[cbml_field_local, "geometry"]]
            right = gdf[[campo, "geometry"]] if campo else gdf[["geometry"]]
            ix = gpd.overlay(left, right, how="intersection", keep_geom_type=False)
            if ix.empty:
                continue

            ix["ix_area"] = ix.geometry.area
            if sliver_min_area_local > 0:
                ix = ix[ix["ix_area"] >= sliver_min_area_local]
                if ix.empty:
                    continue

            ix["category_value"] = ix[campo].astype(str) if (campo and campo in ix.columns) else None
            ix["field_name"] = campo if campo else "(Sin categoría)"
            grp = ix.groupby(
                [cbml_field_local, "category_value", "field_name"], dropna=False, as_index=False
            )["ix_area"].sum()
            grp["layer_name"] = lyr
            registros_lyr.append(grp)
        except Exception as e:
            # Capturar errores en procesos paralelos
            print(f"Error procesando capa {lyr}, campo {campo}: {e}")

    return pd.concat(registros_lyr, ignore_index=True) if registros_lyr else None

def calcular_incidencia_paralelo(
        gdb_path: str,
        construcciones_layer: str,
        cbml_field: str,
        capas_y_campos: Dict[str, List[str]],
        sliver_min_area: float = 0.0,
        hacer_pivote: bool = True,
        seg_field: Optional[str] = None,
        seg_values: Optional[List[str]] = None,
        progress_callback=None,
        usar_muestra: bool = False,
        pct_muestra: int = 100
):
    # Leer construcciones una sola vez
    cons_cols = [cbml_field]
    if seg_field and seg_field not in cons_cols:
        cons_cols.append(seg_field)

    cons = read_layer(gdb_path, construcciones_layer, cons_cols)
    if cons.empty:
        raise ValueError("La capa de construcciones está vacía.")
    if cbml_field not in cons.columns:
        raise ValueError(f"El campo '{cbml_field}' no existe en {construcciones_layer}.")
    if not is_polygonal(cons):
        raise ValueError("La capa de construcciones no es poligonal.")

    # Aplicar muestra si se solicita
    if usar_muestra and pct_muestra < 100:
        sample_size = max(1, int(len(cons) * pct_muestra / 100))
        cons = cons.sample(n=sample_size, random_state=42)
        if cons.empty:
            raise ValueError("La muestra resultó vacía. Aumenta el porcentaje.")

    # Segmentación opcional
    if seg_field and seg_values:
        cons = cons[cons[seg_field].astype(str).isin([str(v) for v in seg_values])]
        if cons.empty:
            return (
                pd.DataFrame(columns=[cbml_field, "layer_name", "field_name", "category_value",
                                      "area_cbml", "area_intersect", "pct_incidence"]),
                pd.DataFrame()
            )

    # Preparar construcciones
    cons = clean_polygons(cons)
    cons = auto_project(cons)
    cons_diss = cons[[cbml_field, "geometry"]].dissolve(by=cbml_field, as_index=False)
    cons_diss["area_cbml"] = cons_diss.geometry.area

    # Notificar progreso si hay callback
    if progress_callback:
        progress_callback("Preparando construcciones completado", 0.1)

    # Crear una función parcial con los parámetros fijos
    func = partial(
        procesar_capa,
        cons_diss_local=cons_diss,
        cbml_field_local=cbml_field,
        gdb_path_local=gdb_path,
        sliver_min_area_local=sliver_min_area
    )

    # Procesar en paralelo
    registros = []
    total_capas = len(capas_y_campos)
    processed = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Iniciar todas las tareas
        future_to_layer = {
            executor.submit(func, item): item[0] for item in capas_y_campos.items()
        }

        # Procesar resultados a medida que completan
        for future in concurrent.futures.as_completed(future_to_layer):
            processed += 1
            layer_name = future_to_layer[future]

            if progress_callback:
                progress_callback(f"Procesando capa {layer_name} ({processed}/{total_capas})",
                                  0.1 + 0.8 * (processed / total_capas))

            try:
                resultado = future.result()
                if resultado is not None:
                    registros.append(resultado)
            except Exception as e:
                print(f"Error en capa {layer_name}: {e}")

    if not registros:
        df_largo = pd.DataFrame(
            columns=[cbml_field, "layer_name", "field_name", "category_value",
                     "area_cbml", "area_intersect", "pct_incidence"]
        )
        return df_largo, pd.DataFrame()


    if progress_callback:
        progress_callback("Consolidando resultados...", 0.9)

    inter_all = pd.concat(registros, ignore_index=True)
    base = cons_diss[[cbml_field, "area_cbml"]]
    df = inter_all.merge(base, on=cbml_field, how="left").rename(columns={"ix_area": "area_intersect"})
    df["pct_incidence"] = (df["area_intersect"] / df["area_cbml"].where(df["area_cbml"] > 0, pd.NA)) * 100.0
    df["pct_incidence"] = df["pct_incidence"].fillna(0.0)

    df_largo = df[[cbml_field, "layer_name", "field_name", "category_value",
                   "area_cbml", "area_intersect", "pct_incidence"]].copy()

    if hacer_pivote:
        df_largo["col_key"] = df_largo.apply(
            lambda r: f"{r['layer_name']}|{r['field_name']}|{r['category_value']}"
            if pd.notna(r["category_value"]) else f"{r['layer_name']}|{r['field_name']}",
            axis=1
        )
        df_ancho = df_largo.pivot_table(
            index=cbml_field, columns="col_key", values="pct_incidence", aggfunc="sum", fill_value=0.0
        ).reset_index()
        df_ancho.columns.name = None
    else:
        df_ancho = pd.DataFrame()

    if progress_callback:
        progress_callback("Cálculo completado", 1.0)

    return df_largo, df_ancho

def calcular_con_progreso(gdb_path, construcciones_layer, cbml_field, capas_y_campos, **kwargs):
    """Envuelve la función de cálculo con una barra de progreso"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Función para actualizar el progreso
    def update_progress(msg, value):
        progress_bar.progress(value)
        status_text.text(msg)

    try:
        # Llama a la función optimizada con callback de progreso
        df_long, df_wide = calcular_incidencia_paralelo(
            gdb_path=gdb_path,
            construcciones_layer=construcciones_layer,
            cbml_field=cbml_field,
            capas_y_campos=capas_y_campos,
            progress_callback=update_progress,
            **kwargs
        )
        update_progress("Cálculo completado", 1.0)
        return df_long, df_wide
    except Exception as e:
        update_progress(f"Error: {str(e)}", 0)
        raise e

def to_excel_bytes(df_long: pd.DataFrame, df_wide: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        df_long.to_excel(xw, sheet_name="largo", index=False)
        if not df_wide.empty:
            df_wide.to_excel(xw, sheet_name="ancho", index=False)
    return buf.getvalue()

# ============ UI Streamlit ============
st.set_page_config(page_title="Incidencia CBML", layout="wide")
st.title("Análisis de Incidencias — Equipo AIE-DAP")

gdb_path = ""  # Valor por defecto inicial

with st.sidebar:
    st.header("Entrada")

    # Opción de carga directa o ruta
    input_method = st.radio(
        "Método de entrada",
        options=["Cargar archivo .zip con GDB", "Ingresar ruta a GDB"],
        index=0
    )

    if input_method == "Cargar archivo .zip con GDB":
        uploaded_file = st.file_uploader(
            "Cargar archivo .zip con File Geodatabase (.gdb)",
            type=["zip"],
            help="El archivo .zip debe contener una carpeta .gdb en su raíz"
        )

        # Inicializar layers como lista vacía por defecto
        layers = []

        if uploaded_file is not None:
            # Crear directorio temporal para extraer el zip
            extract_dir = os.path.join(st.session_state.temp_gdb_dir, "extracted")
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)

            # Guardar el archivo zip
            zip_path = os.path.join(st.session_state.temp_gdb_dir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Extraer el zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            # Buscar la carpeta .gdb dentro del directorio extraído
            gdb_paths = []
            for root, dirs, files in os.walk(extract_dir):
                for dir_name in dirs:
                    if dir_name.lower().endswith(".gdb"):
                        gdb_paths.append(os.path.join(root, dir_name))

            if not gdb_paths:
                st.error("No se encontró ningún archivo .gdb en el ZIP cargado.")
                gdb_path = ""
                layers = []
            else:
                # Usar la primera GDB encontrada
                gdb_path = gdb_paths[0]
                st.success(f"GDB cargada: {os.path.basename(gdb_path)}")
                layers = list_gdb_layers(gdb_path)
    else:
        # Método original por ruta
        gdb_path = st.text_input("Ruta a la File Geodatabase (.gdb)", value="", placeholder="/ruta/a/tu.gdb")
        if gdb_path:
            layers = list_gdb_layers(gdb_path)
        else:
            layers = []

    if layers:
        construcciones_layer = st.selectbox("Capa de construcciones (polígonos)", options=layers, index=0)
        cbml_field = st.text_input("Campo código en construcciones", value="cbml")

        # -------- Segmentación por ámbito (opcional) --------
        st.markdown("**Segmentación por ámbito (opcional, en construcciones):**")
        try:
            cons_head = read_layer(gdb_path, construcciones_layer, columns=None)[:1]
            cons_fields = [c for c in cons_head.columns if c != "geometry"]
        except Exception:
            cons_fields = []
        usar_seg = st.checkbox("Activar segmentación", value=False)
        seg_field: Optional[str] = None
        seg_values: Optional[List[str]] = None
        if usar_seg and cons_fields:
            seg_field = st.selectbox("Campo de segmentación", options=cons_fields)
            try:
                cons_all_for_vals = read_layer(gdb_path, construcciones_layer, columns=[seg_field])
                uniq_vals = sorted(map(str, pd.unique(cons_all_for_vals[seg_field].astype(str).fillna(""))))[:500]
            except Exception:
                uniq_vals = []
            seg_values = st.multiselect("Valores a incluir", options=uniq_vals,
                                        help="Solo se procesarán estos valores.")
        # ----------------------------------------------------

        # Filtrar solo poligonales para selección
        poly_layers = []
        for lyr in layers:
            try:
                gdf_head = read_layer(gdb_path, lyr, columns=[])[:50]
                if not gdf_head.empty and is_polygonal(gdf_head):
                    poly_layers.append(lyr)
            except Exception:
                pass

        target_layers = st.multiselect(
            "Capas a evaluar (poligonales)",
            options=[l for l in poly_layers if l != construcciones_layer],
        )
        sliver_min_area = st.number_input(
            "Umbral mínimo de área de intersección (slivers) – unidades del CRS (p.ej., m²)",
            min_value=0.0, value=0.0, step=1.0
        )
        hacer_pivote = st.checkbox("Generar tabla pivote (ancha)", value=True)

        with st.expander("Opciones avanzadas"):
            usar_muestra = st.checkbox("Usar muestra para prueba rápida", value=False)
            pct_muestra = 10  # Valor por defecto
            if usar_muestra:
                pct_muestra = st.slider(
                    "Porcentaje de la muestra",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="Reduce el tamaño de los datos para pruebas rápidas"
                )
    else:
        construcciones_layer = None
        cbml_field = "CBML"
        target_layers = []
        sliver_min_area = 0.0
        hacer_pivote = True
        usar_seg = False
        seg_field = None
        seg_values = None
        usar_muestra = False
        pct_muestra = 10

    # Logo al final de la barra lateral (fuera de las condiciones)
    st.markdown("---")
    try:
        st.image("Logo.png", width=200)
    except FileNotFoundError:
        st.warning("Logo.png no encontrado")

if not gdb_path:
    st.info("👉 Ingresa la ruta a tu carpeta .gdb o carga un archivo .zip con una GDB.")
    st.stop()
if not layers:
    st.error("No se pudieron listar capas. Verifica la ruta o el archivo cargado.")
    st.stop()
if construcciones_layer is None:
    st.warning("Selecciona la capa de construcciones.")
    st.stop()

# Atributos de interés por capa
capas_y_campos: Dict[str, List[str]] = {}
if target_layers:
    st.subheader("Atributos de interés por capa")
    for lyr in target_layers:
        try:
            gdf_head = read_layer(gdb_path, lyr, columns=None)[:1]
            field_options = [c for c in gdf_head.columns if c != "geometry"]
        except Exception:
            field_options = []
        field_options = ["(Todos)"] + field_options
        sel = st.multiselect(
            f"Campos para '{lyr}'",
            options=field_options,
            default=["(Todos)"],
            key=f"fld_{lyr}"
        )
        capas_y_campos[lyr] = [] if "(Todos)" in sel else [c for c in sel if c != "(Todos)"]
else:
    st.info("Selecciona al menos una capa poligonal a evaluar.")
    st.stop()

# -------- Botón de cálculo (antes del umbral) --------
run = st.button("Calcular incidencia")

if run:
    try:
        # Usar la función con barra de progreso
        df_long, df_wide = calcular_con_progreso(
            gdb_path=gdb_path,
            construcciones_layer=construcciones_layer,
            cbml_field=cbml_field,
            capas_y_campos=capas_y_campos,
            sliver_min_area=sliver_min_area,
            hacer_pivote=hacer_pivote,
            seg_field=(seg_field if usar_seg else None),
            seg_values=(seg_values if usar_seg else None),
            usar_muestra=usar_muestra,
            pct_muestra=pct_muestra
        )
        # Guardar en session_state para no perderlos en cada rerun
        st.session_state.df_long = df_long
        st.session_state.df_wide = df_wide
        st.success("¡Cálculo completado!")
    except Exception as e:
        st.session_state.df_long = None
        st.session_state.df_wide = None
        st.error(f"Error: {e}")

# ===== Mostrar resultados si existen en session_state =====
if st.session_state.df_long is not None:
    st.subheader("Resultado (largo)")
    st.dataframe(st.session_state.df_long, use_container_width=True, height=350)
    if st.session_state.df_wide is not None and not st.session_state.df_wide.empty:
        st.subheader("Resultado (ancho / pivote)")
        st.dataframe(st.session_state.df_wide, use_container_width=True, height=350)

    # ======= Configuración y Resumen por umbral (usa df_long guardado) =======
    st.markdown("---")
    st.subheader("Resumen por umbral de % de incidencia (CBML) – configuración")

    # Campo numérico en construcciones (viviendas/personas/etc.)
    try:
        cons_sample = read_layer(gdb_path, construcciones_layer, columns=None)
        numeric_fields = [c for c in cons_sample.columns if
                          c != "geometry" and pd.api.types.is_numeric_dtype(cons_sample[c])]
    except Exception:
        numeric_fields = []

    medida_field_umbral = st.selectbox(
        "Campo numérico en construcciones a sumar (viviendas/personas/etc.)",
        options=(["(Ninguno)"] + numeric_fields) if numeric_fields else ["(Ninguno)"],
        index=0,
        key="umbral_field"
    )

    umbral_pct = st.slider(
        "Umbral mínimo de % de incidencia por CBML (≥)",
        min_value=0, max_value=100, value=50, step=1,
        key="umbral_pct"
    )

    st.subheader("Resumen por umbral de % de incidencia (a nivel CBML)")
    if medida_field_umbral and medida_field_umbral != "(Ninguno)":
        try:
            df_long = st.session_state.df_long

            # (A) Universo CBML
            cons_uni = read_layer(gdb_path, construcciones_layer, columns=[cbml_field])
            universe_cbml = cons_uni[[cbml_field]].dropna().drop_duplicates()

            # (B) Máximo % incidencia por CBML
            max_pct_cbml = (
                df_long.groupby(cbml_field, as_index=False)["pct_incidence"]
                .max()
                .rename(columns={"pct_incidence": "max_pct_cbml"})
            )

            # (C) Join universo + max% (faltantes => 0)
            tabla_umbral = universe_cbml.merge(max_pct_cbml, on=cbml_field, how="left")
            tabla_umbral["max_pct_cbml"] = pd.to_numeric(tabla_umbral["max_pct_cbml"], errors="coerce").fillna(0.0)

            # (D) Suma del campo elegido por CBML
            cons_med = read_layer(gdb_path, construcciones_layer, columns=[cbml_field, medida_field_umbral])
            cons_med = cons_med[cons_med[cbml_field].notna()].copy()
            medida_cbml = (
                cons_med.assign(__mf=pd.to_numeric(cons_med[medida_field_umbral], errors="coerce").fillna(0.0))
                .groupby(cbml_field, as_index=False)["__mf"].sum()
                .rename(columns={"__mf": "measure_total_cbml"})
            )

            # (E) Join medida (faltantes => 0)
            tabla_umbral = tabla_umbral.merge(medida_cbml, on=cbml_field, how="left")
            tabla_umbral["measure_total_cbml"] = pd.to_numeric(
                tabla_umbral["measure_total_cbml"], errors="coerce"
            ).fillna(0.0)

            # (F) Filtro por umbral (≥) — si 0, incluye sin intersección
            afectados = tabla_umbral[tabla_umbral["max_pct_cbml"] >= umbral_pct].copy()

            # (G) Totales
            total_cbml_cumplen = int(afectados[cbml_field].nunique())
            total_medida_cumplen = afectados["measure_total_cbml"].sum()

            colA, colB = st.columns(2)
            with colA:
                st.metric(f"CBML con ≥ {umbral_pct}%", value=f"{total_cbml_cumplen:,}".replace(",", "."))
            with colB:
                st.metric(
                    f"Suma de {medida_field_umbral} en esos CBML",
                    value=f"{total_medida_cumplen:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
                )

            st.markdown(f"**Detalle de CBML que cumplen ≥ {umbral_pct}%**")
            st.dataframe(
                afectados[[cbml_field, "max_pct_cbml", "measure_total_cbml"]]
                .sort_values(["max_pct_cbml", "measure_total_cbml"], ascending=[False, False]),
                use_container_width=True, height=320
            )
        except Exception as e:
            st.warning(f"No fue posible calcular el resumen por umbral: {e}")
    else:
        st.info("Selecciona un campo numérico en construcciones para sumar (viviendas/personas/etc.).")

    # Descargas (desde session_state)
    st.subheader("Descargar")
    xlsx_bytes = to_excel_bytes(st.session_state.df_long,
                                st.session_state.df_wide if st.session_state.df_wide is not None else pd.DataFrame())
    st.download_button(
        "Descargar Excel (largo + ancho)",
        data=xlsx_bytes,
        file_name="incidencias_cbml.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    st.download_button(
        "Descargar CSV (largo)",
        data=st.session_state.df_long.to_csv(index=False).encode("utf-8"),
        file_name="incidencias_cbml_largo.csv",
        mime="text/csv",
    )
    if st.session_state.df_wide is not None and not st.session_state.df_wide.empty:
        st.download_button(
            "Descargar CSV (ancho)",
            data=st.session_state.df_wide.to_csv(index=False).encode("utf-8"),
            file_name="incidencias_cbml_ancho.csv",
            mime="text/csv",
        )
else:
    st.info("Pulsa **Calcular incidencia** para ver resultados y el resumen por umbral.")
