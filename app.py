import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import streamlit as st
from io import StringIO

st.sidebar.title("Clustering y Análisis de Datos de Clientes")
st.sidebar.header("Cargar Datos")
file_path = st.sidebar.file_uploader("Sube tu archivo CSV", type=['csv'])

if file_path is not None:
    df = pd.read_csv(file_path)
    st.write("Primeras filas del dataset:\n", df.head())

    # 1. Entender los datos
    st.subheader("Información del Dataset")
    buffer = StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.write("\nDescripción estadística del dataset:")
    st.write(df.describe())

    # 2. Limpiar y preparar los datos
    df_clean = df.dropna()
    df_clean = pd.get_dummies(df_clean, drop_first=True)
    X = df_clean.drop('ID', axis=1)

    st.sidebar.subheader("Selecciona opciones para visualización:")
    visualize_distribution = st.sidebar.checkbox("Visualización de Distribuciones")
    visualize_correlations = st.sidebar.checkbox("Correlaciones")
    perform_clustering = st.sidebar.checkbox("Clustering con K-Means")

    # Distribución de las variables numéricas
    if visualize_distribution:
        st.subheader("Distribución de Variables Numéricas")
        df.hist(bins=50, figsize=(20, 15))
        plt.suptitle("Distribución de Variables Numéricas", fontsize=16)
        st.pyplot()

    # Correlaciones entre las variables
    if visualize_correlations:
        plt.figure(figsize=(12, 8))
        sns.heatmap(X.corr(), cmap="coolwarm", annot=False)
        plt.title("Mapa de Correlaciones entre Variables")
        st.pyplot()

    # 3. Estandarización de los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Reducción de Dimensionalidad con PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 5. Clustering con K-Means
    if perform_clustering:
        # Aplicar el modelo K-Means para Clustering (3 clusters)
        kmeans_3 = KMeans(n_clusters=3, random_state=42)
        kmeans_3.fit(X_pca)

        # Evaluación del modelo usando el Silhouette Score
        silhouette_avg_3 = silhouette_score(X_pca, kmeans_3.labels_)
        st.write(f'Silhouette Score (con 3 clusters): {silhouette_avg_3:.4f}')

        # Visualización de los clusters formados con 3 clusters
        plt.figure(figsize=(10, 8))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_3.labels_, cmap='viridis', marker='o')
        plt.title('Clustering de Clientes en Componentes Principales (3 Clusters)')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.colorbar(label='Cluster')
        st.pyplot()

        for n_clusters in [4, 5]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_pca)
            silhouette_avg = silhouette_score(X_pca, kmeans.labels_)
            st.write(f'Silhouette Score (con {n_clusters} clusters): {silhouette_avg:.4f}')
            
            plt.figure(figsize=(10, 8))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
            plt.title(f'Clustering de Clientes en Componentes Principales ({n_clusters} Clusters)')
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            plt.colorbar(label='Cluster')
            st.pyplot()

        # Comparar visualmente los Silhouette Scores en un gráfico de barras
        scores = [silhouette_avg_3]
        for n_clusters in [4, 5]:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_pca)
            silhouette_avg = silhouette_score(X_pca, kmeans.labels_)
            scores.append(silhouette_avg)
        
        plt.figure(figsize=(8, 6))
        clusters = ['3 Clusters', '4 Clusters', '5 Clusters']
        plt.bar(clusters, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        plt.title('Comparación de Silhouette Scores para Diferentes Números de Clusters')
        plt.ylabel('Silhouette Score')
        plt.xlabel('Número de Clusters')
        plt.ylim(0, 1)
        st.pyplot()
else:
    st.warning("Por favor, sube un archivo CSV para continuar.")

