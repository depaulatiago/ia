import time
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans as SKKMeans
from km_hardcore import HardcoreKMeans

def run():
    iris = datasets.load_iris()
    X = iris.data.astype(np.float64)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    summary_rows = []
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

    df = pd.DataFrame(summary_rows, columns=["k", "Algoritmo", "Silhouette", "Inertia", "Iterações", "Tempo (s)"])
    print(df.to_string(index=False))

if __name__ == "__main__":
    run()
