import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

st.set_page_config(page_title="Comparador de LLMs con TOPSIS", layout="wide", initial_sidebar_state="expanded", 
                   page_icon="🧠")

st.title("🧠 LLMscore")
# st.subheader("Evaluador multicriterio de modelos de lenguaje abiertos (LLMs) mediante el método TOPSIS")

# Definición de nombres cortos para criterios en radar plot
nombre_corto = {
    "Accesibilidad": "Acceso",
    "Requisitos técnicos": "Técnicos",
    "Comunidad": "Comunidad",
    "Educativo": "Educ.",
    "Documentación": "Docs",
    "Seguridad": "Seguridad"
}


tabs = st.tabs(["Instrucciones", "Análisis", "Acerca de"])

with tabs[0]:
    st.markdown("""
    Esta aplicación permite comparar modelos de lenguaje abiertos (LLMs) mediante el método multicriterio TOPSIS, evaluando su rendimiento en criterios relevantes para entornos educativos.

    ### Formato del archivo JSON esperado

    El archivo debe contener cuatro claves principales:

    - `"modelos"`: lista con los nombres de los modelos a comparar.
    - `"matriz"`: diccionario de subcriterios, cada uno asociado a una lista de valores por modelo.
    - `"pesos"`: diccionario con los pesos asignados a cada subcriterio (la suma debe ser 1).
    - `"criterios"`: agrupación de subcriterios en criterios globales.

    #### Ejemplo de estructura:
    ```json
    {
      "modelos": ["Model A", "Model B"],
      "matriz": {
        "Criterio 1": [3, 4],
        "Criterio 2": [5, 2]
      },
      "pesos": {
        "Criterio 1": 0.5,
        "Criterio 2": 0.5
      },
      "criterios": {
        "Grupo 1": ["Criterio 1", "Criterio 2"]
      }
    }
    ```

    Puedes cargar tu propio archivo o descargar una plantilla editable a continuación.
    """)

    from io import BytesIO
    import json
    plantilla_json = {
        "modelos": ["Modelo A", "Modelo B"],
        "matriz": {
            "Facilidad de instalación": [4, 3],
            "Uso de RAM": [2, 5]
        },
        "pesos": {
            "Facilidad de instalación": 0.5,
            "Uso de RAM": 0.5
        },
        "criterios": {
            "Accesibilidad": ["Facilidad de instalación"],
            "Requisitos técnicos": ["Uso de RAM"]
        }
    }

    buffer_json = BytesIO()
    buffer_json.write(json.dumps(plantilla_json, indent=2).encode("utf-8"))
    buffer_json.seek(0)
    st.download_button("Descargar plantilla JSON", data=buffer_json, file_name="plantilla_llms.json", mime="application/json")

with tabs[1]:
    st.header("Análisis de Modelos")

    uploaded_file = st.file_uploader("Sube un archivo JSON con los datos de entrada:", type="json")
    if uploaded_file:
        data = json.load(uploaded_file)
        

        modelos = data["modelos"]
        matriz_dict = data["matriz"]
        pesos = pd.Series(data["pesos"])
        criterios = data["criterios"]

        # DataFrame con subcriterios como filas y modelos como columnas
        matriz = pd.DataFrame(matriz_dict, index=modelos).T

        st.subheader("Matriz de Evaluación (LLMs como columnas)")
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

        st.subheader("Promedios por Criterio Global")
        st.dataframe(promedio_df, use_container_width=True)

        st.subheader("Gráficos Interactivos por Modelo")
        cols = st.columns(3)
        for idx, modelo in enumerate(modelos):
            df_plot = promedio_df.loc[modelo].reset_index()
            df_plot.columns = ["Criterio", "Valor"]

            # Usar nombres cortos para etiquetas y agregar saltos de línea si quieres
            df_plot["Criterio"] = df_plot["Criterio"].map(nombre_corto)

            # O bien para saltos de línea:
            # df_plot["Criterio"] = df_plot["Criterio"].apply(lambda x: x.replace(" ", "<br>"))

            fig = px.line_polar(df_plot, r="Valor", theta="Criterio", line_close=True, title=modelo)
            
            fig.update_traces(
                line=dict(color='#E74C3C', width=3),
                fill='toself',
                fillcolor='rgba(236, 112, 99, 0.3)'
            )
            
            fig.update_layout(
                polar=dict(
                    bgcolor='#F8F9F9',
                    radialaxis=dict(
                        visible=True,
                        range=[0, 5],
                        gridcolor='#D7DBDD',
                        linecolor='#283747',
                        tickfont=dict(color='#283747')
                    ),
                    angularaxis=dict(
                        gridcolor='#D7DBDD',
                        tickfont=dict(color='#283747', size=16),
                        rotation=90
                    )
                ),
                title=modelo,
                margin=dict(l=20, r=20, t=50, b=80),
                height=350
            )
            cols[idx % 3].plotly_chart(fig, use_container_width=True)


        # --- Ranking bar chart ---
        st.subheader("🏆 Ranking de Modelos (Gráfico de Barras)")
        fig_bar = px.bar(ranking, x="Modelo", y="Score TOPSIS", color="Modelo",
                         hover_data=["Ranking"], text_auto=".2f",
                         category_orders={"Modelo": ranking["Modelo"].tolist()})
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- Exportar Excel ---
        st.subheader("Exportar Resultados")
        buffer_excel = BytesIO()
        with pd.ExcelWriter(buffer_excel, engine='xlsxwriter') as writer:
            matriz.T.to_excel(writer, sheet_name="Matriz")
            pesos.to_frame("Peso").to_excel(writer, sheet_name="Pesos")
            promedio_df.to_excel(writer, sheet_name="Promedios")
            ranking.to_excel(writer, sheet_name="Ranking")
        st.download_button("Descargar Excel con resultados", data=buffer_excel.getvalue(), file_name="resultados_llms.xlsx")

        

    else:
        st.info("Por favor, sube un archivo JSON con la estructura adecuada para ver el análisis.")

with tabs[2]:
    st.header("Acerca de esta aplicación")
    st.markdown("""
    **Proyecto:** LLMscore 
    **Descripción:** Esta aplicación permite comparar diferentes modelos de lenguaje mediante un **análisis multicriterio TOPSIS**, mostrando puntuaciones, rankings y visualizaciones. Se enfoca en modelos open source y y permite descargar los resultados para uso académico o profesional.  
    """)

    st.subheader("Autores")
    st.markdown("""
    * **[Pavel Novoa Hernández](https://portalciencia.ull.es/investigadores/1244723/detalle)**. Departamento: Ingeniería Informática y de Sistemas (pnovoahe@ull.edu.es).
    """)
    st.markdown("""
    * **[Fulgencio Sánchez Vera](https://portalciencia.ull.es/investigadores/120481/detalle)**. Departamento: Didáctica e Investigación Educativa (fsanchev@ull.edu.es).
    """)
    st.markdown("""
    * **[Betty Coromoto Estévez Cedeño](https://portalciencia.ull.es/investigadores/81945/detalle)**. Departamento: Sociología y Antropología (bestevec@ull.edu.es).
    """)

    st.subheader("Licencia")
    st.markdown("Este proyecto está bajo la **Licencia MIT**.")
