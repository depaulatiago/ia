import time
import numpy as np

class HardcoreKMeans:
    def __init__(self, n_clusters: int, max_iter: int = 300, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self._rng = np.random.default_rng(random_state)

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)

        idx0 = self._rng.integers(0, n_samples)
        centroids[0] = X[idx0]

        closest_dist_sq = np.full(n_samples, np.inf, dtype=X.dtype)
        for c in range(1, self.n_clusters):
            dist_sq = np.sum((X - centroids[c-1])**2, axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
            probs = closest_dist_sq / closest_dist_sq.sum()
            cumulative_probs = np.cumsum(probs)
            r = self._rng.random()
            idx = np.searchsorted(cumulative_probs, r)
            centroids[c] = X[idx]
        return centroids

    def fit(self, X: np.ndarray) -> "HardcoreKMeans":
        centroids = self._init_centroids(X)

        for it in range(self.max_iter):
            dists = np.sum((X[:, None, :] - centroids[None, :, :])**2, axis=2)
            labels = np.argmin(dists, axis=1)

            new_centroids = np.copy(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = X[self._rng.integers(0, X.shape[0])]

            shift = np.sqrt(np.sum((new_centroids - centroids)**2))
            centroids = new_centroids
            self.n_iter_ = it + 1
            if shift <= self.tol:
                break

        dists = np.sum((X[:, None, :] - centroids[None, :, :])**2, axis=2)
        labels = np.argmin(dists, axis=1)
        inertia = float(np.sum(np.min(dists, axis=1)))

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = np.sum((X[:, None, :] - self.centroids_[None, :, :])**2, axis=2)
        return np.argmin(dists, axis=1)
