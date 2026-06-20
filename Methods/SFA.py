import numpy as np
class SFA:
    def __init__(self, n_components=None, eps=1e-10):
        self.n_components = n_components
        self.eps = eps

    def fit(self, X):
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        C = (Xc.T @ Xc) / max(n_samples - 1, 1)

        evals, evecs = np.linalg.eigh(C)

        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]

        valid = evals > self.eps
        evals = evals[valid]
        evecs = evecs[:, valid]

        whitening = evecs @ np.diag(1.0 / np.sqrt(evals + self.eps))
        Z = Xc @ whitening

        dZ = np.diff(Z, axis=0)

        Cdot = (dZ.T @ dZ) / dZ.shape[0]
        Cdot = (Cdot + Cdot.T) / 2.0
        slow_values, slow_vecs = np.linalg.eigh(Cdot)

        slow_order = np.argsort(slow_values)
        slow_values = slow_values[slow_order]
        slow_vecs = slow_vecs[:, slow_order]

        k_max = slow_vecs.shape[1]

        if self.n_components is None:
            k = k_max
        else:
            k = int(self.n_components)
            k = min(k, k_max)

        self.n_components_ = k
        self.whitening_ = whitening
        self.slow_vectors_ = slow_vecs[:, :k]
        self.components_ = whitening @ self.slow_vectors_
        self.delta_values_ = np.maximum(slow_values[:k], self.eps)
        self.Y_train_ = Xc @ self.components_
        self.P_ = Xc.T @ self.Y_train_ @ np.linalg.pinv(
            self.Y_train_.T @ self.Y_train_
            + self.eps * np.eye(self.n_components_)
        )

        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_

    def inverse_transform(self, Y):
        X_rec = Y @ self.P_.T + self.mean_

        return X_rec

    def get_monitoring_statistics(self, X):
        Y = self.transform(X)
        T2 = np.sum(Y ** 2, axis=1)
        dY = np.diff(Y, axis=0)
        S2_valid = np.sum(
            (dY ** 2) / (self.delta_values_ + self.eps),
            axis=1,
        )

        S2 = np.full(Y.shape[0], np.nan)
        S2[1:] = S2_valid

        return T2, S2

    def fit_transform(self, X):
        return self.fit(X).transform(X)