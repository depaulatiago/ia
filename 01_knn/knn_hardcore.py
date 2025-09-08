# knn_hardcore.py
# Implementação "hardcore" do KNN (Iris)
import numpy as np

def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) ** 2)))

def knn_predict_single(x: np.ndarray, X_ref: np.ndarray, y_ref: np.ndarray, k: int) -> int:
    dists = [euclidean(x, X_ref[i]) for i in range(len(X_ref))]
    idx = np.argsort(dists)[:k]
    votes, counts = np.unique(y_ref[idx], return_counts=True)
    max_count = np.max(counts)
    winners = votes[counts == max_count]
    return int(np.min(winners))

def knn_predict(Xq: np.ndarray, X_ref: np.ndarray, y_ref: np.ndarray, k: int) -> np.ndarray:
    return np.array([knn_predict_single(x, X_ref, y_ref, k) for x in Xq], dtype=int)
