import time
import numpy as np
import pandas as pd
import os
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans as SKKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from km_hardcore import HardcoreKMeans


def run():
    iris = datasets.load_iris()
    X = iris.data.astype(np.float64)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    summary_rows = []
    best_k = None
    best_sil = -1
    best_labels = None
    best_centroids = None
    best_algo = None

    for k in (3, 5):
        hc = HardcoreKMeans(n_clusters=k, max_iter=300, tol=1e-6, random_state=42)
        t0 = time.perf_counter()
        hc.fit(X_std)
        hc_time = time.perf_counter() - t0
        hc_sil = silhouette_score(X_std, hc.labels_)

        t0 = time.perf_counter()
        sk = SKKMeans(n_clusters=k, n_init=10, random_state=42)
        sk_labels = sk.fit_predict(X_std)
        sk_time = time.perf_counter() - t0
        sk_sil = silhouette_score(X_std, sk_labels)

        summary_rows.append([k, "Hardcore", hc_sil, hc.inertia_, hc.n_iter_, hc_time])
        summary_rows.append([k, "Sklearn", sk_sil, sk.inertia_, getattr(sk, "n_iter_", -1), sk_time])

        # Guardar o melhor k e labels para plotar PCA
        if hc_sil > best_sil:
            best_k = k
            best_sil = hc_sil
            best_labels = hc.labels_
            best_centroids = hc.centroids_
            best_algo = 'hardcore'
        if sk_sil > best_sil:
            best_k = k
            best_sil = sk_sil
            best_labels = sk_labels
            best_centroids = sk.cluster_centers_
            best_algo = 'sklearn'

    df = pd.DataFrame(summary_rows, columns=["k", "Algoritmo", "Silhouette", "Inertia", "Iterações", "Tempo (s)"])
    print(df.to_string(index=False))

    # Gerar e sobrescrever imagens de PCA (1 e 2 componentes)
    figuras_dir = os.path.join(os.path.dirname(__file__), '..', 'figuras')
    os.makedirs(figuras_dir, exist_ok=True)
    for n_comp in [1, 2]:
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(X_std)
        centroids_pca = pca.transform(best_centroids)
        plt.figure(figsize=(7, 5))
        if n_comp == 1:
            plt.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), c=best_labels, cmap='viridis', s=40, alpha=0.7, label='Amostras')
            plt.scatter(centroids_pca[:, 0], np.zeros_like(centroids_pca[:, 0]), c='red', s=120, marker='X', label='Centróides')
            plt.xlabel('PCA 1')
            plt.yticks([])
        else:
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels, cmap='viridis', s=40, alpha=0.7, label='Amostras')
            plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', s=120, marker='X', label='Centróides')
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
        plt.title(f'Clusterização (k={best_k}, {best_algo}, PCA {n_comp}D)')
        plt.legend()
        fname = os.path.join(figuras_dir, f"pca{n_comp}_bestk_{best_k}.png")
        if os.path.exists(fname):
            os.remove(fname)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

if __name__ == "__main__":
    run()
