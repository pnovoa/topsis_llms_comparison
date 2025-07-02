import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

st.set_page_config(page_title="Comparador de LLMs con TOPSIS", layout="wide")
st.title("🔍 Comparador de Modelos de Lenguaje Abiertos (LLMs) usando TOPSIS")

# Definición de nombres cortos para criterios en radar plot
nombre_corto = {
    "Accesibilidad": "Acceso",
    "Requisitos técnicos": "Técnicos",
    "Comunidad": "Comunidad",
    "Educativo": "Educ.",
    "Documentación": "Docs",
    "Seguridad": "Seguridad"
}

def exportar_radar_grid(promedio_df, nombre_corto):
    modelos = promedio_df.index.tolist()
    criterios = list(nombre_corto.keys())
    etiquetas = list(nombre_corto.values())
    num_modelos = len(modelos)

    fig = plt.figure(figsize=(18, 18))
    gs = gridspec.GridSpec(3, 3)

    N = len(criterios)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Cierra el círculo

    for i, modelo in enumerate(modelos):
        valores = promedio_df.loc[modelo, criterios].tolist()
        valores += valores[:1]  # Cierra el polígono

        ax = fig.add_subplot(gs[i], polar=True)
        ax.plot(angles, valores, color='black', linewidth=2)
        ax.fill(angles, valores, color='gray', alpha=0.3)
        ax.set_title(modelo, fontsize=14, color='black', pad=20)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(etiquetas, color='black', fontsize=10)
        ax.set_yticklabels([])
        ax.spines['polar'].set_color('black')
        ax.grid(color='black', linestyle='dotted')

    fig.tight_layout()
    return fig


tabs = st.tabs(["🧭 Instrucciones", "🔍 Análisis", "📘 Acerca de"])

with tabs[0]:
    st.header("🧭 Cómo usar esta aplicación")
    st.markdown("""
    - **Sube un archivo JSON** que contenga los datos de evaluación de modelos (estructura predefinida).
    - La app elimina automáticamente el modelo "Qwen-7B" para mantener 9 modelos.
    - Se muestran la matriz de evaluación, ponderaciones, ranking TOPSIS y gráficos.
    - Puedes descargar los resultados en Excel y la imagen combinada de radar en PNG.
    - Los gráficos individuales son interactivos y se muestran con Plotly.
    - El gráfico global en grid (3x3) se exporta con matplotlib para mejor calidad de imagen.
    """)

with tabs[1]:
    st.header("🔍 Análisis de Modelos")

    uploaded_file = st.file_uploader("📄 Sube un archivo JSON con los datos de entrada:", type="json")
    if uploaded_file:
        data = json.load(uploaded_file)

        # Eliminar uno de los modelos chinos (Qwen-7B)
        if "Qwen-7B" in data["modelos"]:
            idx = data["modelos"].index("Qwen-7B")
            for k in data["matriz"]:
                data["matriz"][k].pop(idx)
            data["modelos"].remove("Qwen-7B")

        modelos = data["modelos"]
        matriz_dict = data["matriz"]
        pesos = pd.Series(data["pesos"])
        criterios = data["criterios"]

        # DataFrame con subcriterios como filas y modelos como columnas
        matriz = pd.DataFrame(matriz_dict, index=modelos).T

        st.subheader("📊 Matriz de Evaluación (LLMs como columnas)")
        st.dataframe(matriz, use_container_width=True)

        st.subheader("⚖️ Ponderaciones de Subcriterios")
        st.dataframe(pesos.to_frame("Peso"), use_container_width=True)

        # --- Normalización y TOPSIS ---
        X = matriz.to_numpy().astype(float).T
        norm = np.linalg.norm(X, axis=0)
        X_norm = X / norm
        weights = pesos.loc[matriz.index].to_numpy()
        X_weighted = X_norm * weights

        ideal = np.max(X_weighted, axis=0)
        anti_ideal = np.min(X_weighted, axis=0)
        dist_ideal = np.linalg.norm(X_weighted - ideal, axis=1)
        dist_anti = np.linalg.norm(X_weighted - anti_ideal, axis=1)

        scores = dist_anti / (dist_ideal + dist_anti)
        ranking = pd.DataFrame({
            "Modelo": modelos,
            "Distancia al ideal": dist_ideal,
            "Distancia al anti-ideal": dist_anti,
            "Score TOPSIS": scores
        })
        ranking["Ranking"] = ranking["Score TOPSIS"].rank(ascending=False).astype(int)
        ranking = ranking.sort_values("Ranking")

        # --- Promedios por criterio global ---
        promedio_criterios = []
        for modelo in modelos:
            entrada = matriz[modelo]
            proms = {
                crit: round(np.mean([entrada[s] for s in subcs]), 2)
                for crit, subcs in criterios.items()
            }
            promedio_criterios.append(proms)

        promedio_df = pd.DataFrame(promedio_criterios, index=modelos)

        st.subheader("📈 Promedios por Criterio Global")
        st.dataframe(promedio_df, use_container_width=True)

        # --- Gráficos individuales interactivos (Plotly) ---
        st.subheader("📊 Gráficos Interactivos por Modelo")
        for modelo in modelos:
            df_plot = promedio_df.loc[modelo].reset_index()
            df_plot.columns = ["Criterio", "Valor"]
            fig = px.line_polar(df_plot, r="Valor", theta="Criterio", line_close=True,
                                title=modelo, markers=True)
            fig.update_traces(line=dict(color="black"))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 5]),
                    angularaxis=dict(tickfont=dict(color="black"))
                ),
                font_color="black",
                title_font_color="black"
            )
            st.plotly_chart(fig, use_container_width=True)

        # --- Radar plot grid (matplotlib) ---
        st.subheader("📊 Gráficos de Telaraña Global (Grid 3x3)")

        fig_radar_grid = exportar_radar_grid(promedio_df, nombre_corto)
        st.pyplot(fig_radar_grid)

        # --- Ranking bar chart ---
        st.subheader("🏆 Ranking de Modelos (Gráfico de Barras)")
        fig_bar = px.bar(ranking, x="Modelo", y="Score TOPSIS", color="Modelo",
                         hover_data=["Ranking"], text_auto=".2f",
                         category_orders={"Modelo": ranking["Modelo"].tolist()})
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- Exportar Excel ---
        st.subheader("📥 Exportar Resultados")
        buffer_excel = BytesIO()
        with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
            matriz.T.to_excel(writer, sheet_name="Matriz")
            pesos.to_frame("Peso").to_excel(writer, sheet_name="Pesos")
            promedio_df.to_excel(writer, sheet_name="Promedios")
            ranking.to_excel(writer, sheet_name="Ranking")
        st.download_button("📥 Descargar Excel con resultados", data=buffer_excel.getvalue(), file_name="resultados_llms.xlsx")

        # --- Exportar imagen radar grid ---
        buffer_img = BytesIO()
        fig_radar_grid.savefig(buffer_img, format="png", dpi=300, bbox_inches="tight")
        st.download_button("🖼️ Descargar imagen de radar global (PNG)", data=buffer_img.getvalue(),
                           file_name="radar_grid_llms.png", mime="image/png")

    else:
        st.info("Por favor, sube un archivo JSON con la estructura adecuada para ver el análisis.")

with tabs[2]:
    st.header("📘 Acerca de esta aplicación")
    st.markdown("""
    **Proyecto:** Comparador de Modelos de Lenguaje Abiertos (LLMs) usando TOPSIS  
    **Autor:** Pavel Novoa Hernández  
    **Descripción:**  
    Esta aplicación permite comparar diferentes modelos de lenguaje mediante un análisis multicriterio TOPSIS, mostrando puntuaciones, rankings y visualizaciones.  
    Se enfoca en modelos open source y permite descargar los resultados para uso académico o profesional.  
    """)
