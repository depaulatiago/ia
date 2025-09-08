# knn_sklearn.py
from sklearn.neighbors import KNeighborsClassifier

def build_knn(k: int) -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=k)
