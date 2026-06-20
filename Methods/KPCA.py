from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

class KPCA:
    def __init__(
        self,
        n_components=10,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=1,
        center=True,
        standardize=False,
        eps=1e-12,
        eigen_tol=1e-10,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.center = center
        self.standardize = standardize
        self.eps = eps
        self.eigen_tol = eigen_tol

    def fit(self, X):
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features

        if self.standardize:
            self.scaler_ = StandardScaler()
            Xn = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            Xn = X.copy()

        self.X_fit_ = Xn.copy()
        K_raw = self._kernel_matrix(Xn)
        self.K_raw_ = K_raw.copy()

        # Centered training kernel
        if self.center:
            K = self._center_kernel(K_raw)
        else:
            K = K_raw.copy()

        K = (K + K.T) / 2.0
        self.K_centered_ = K

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(K)

        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        valid = eigvals > self.eigen_tol

        eigvals = eigvals[valid]
        eigvecs = eigvecs[:, valid]
        k = self._resolve_n_components(eigvals)
        self.n_components_ = k

        self.alphas_ = eigvecs[:, :k]
        self.lambdas_ = eigvals[:k]
        self.sqrt_lambdas_ = np.sqrt(self.lambdas_ + self.eps)

        self.explained_variance_ = self.lambdas_
        self.explained_variance_ratio_ = self.lambdas_ / (
            eigvals.sum() + self.eps
        )

        self.T_train_ = K @ self.alphas_ / self.sqrt_lambdas_

        self.T2_train_, self.SPE_train_ = self.get_monitoring_statistics(
            X,
            already_original_scale=True,
        )

        return self

    def transform(self, X, already_original_scale=True):
        Xn, K_new_raw, K_new_c = self._prepare_new_kernel(
            X,
            already_original_scale=already_original_scale,
        )

        T = K_new_c @ self.alphas_ / self.sqrt_lambdas_

        return T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_monitoring_statistics(self, X, already_original_scale=True):
        Xn, K_new_raw, K_new_c = self._prepare_new_kernel(
            X,
            already_original_scale=already_original_scale,
        )

        T = K_new_c @ self.alphas_ / self.sqrt_lambdas_
        T2 = np.sum((T ** 2) / (self.lambdas_ + self.eps), axis=1)
        kxx_centered = self._center_new_self_kernel_diag(Xn, K_new_raw)
        SPE = kxx_centered - np.sum(T ** 2, axis=1)
        SPE = np.maximum(SPE, 0.0)
        return T2, SPE

    def inverse_transform(self, Z):
        X_rec_norm = Z @ (self.alphas_ / self.sqrt_lambdas_).T @ self.X_fit_

        if self.standardize and self.scaler_ is not None:
            X_rec = self.scaler_.inverse_transform(X_rec_norm)
        else:
            X_rec = X_rec_norm

        return X_rec

    def _prepare_new_kernel(self, X, already_original_scale=True):
        if self.standardize and already_original_scale:
            Xn = self.scaler_.transform(X)
        else:
            Xn = X.copy()

        K_new_raw = self._kernel_matrix_new(Xn, self.X_fit_)

        if self.center:
            K_new_c = self._center_new_kernel(K_new_raw)
        else:
            K_new_c = K_new_raw.copy()

        return Xn, K_new_raw, K_new_c

    def _kernel_matrix(self, X):
        if self.kernel == "linear":
            K = X @ X.T

        elif self.kernel == "rbf":
            gamma = self._get_gamma(X)
            sq_dists = cdist(X, X, "sqeuclidean")
            K = np.exp(-gamma * sq_dists)

        elif self.kernel == "poly":
            gamma = self._get_gamma(X)
            K = (gamma * (X @ X.T) + self.coef0) ** self.degree

        elif callable(self.kernel):
            n = X.shape[0]
            K = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    K[i, j] = self.kernel(X[i], X[j])

        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

        return K

    def _kernel_matrix_new(self, X_new, X_train):
        if self.kernel == "linear":
            K = X_new @ X_train.T

        elif self.kernel == "rbf":
            gamma = self._get_gamma(X_train)
            sq_dists = cdist(X_new, X_train, "sqeuclidean")
            K = np.exp(-gamma * sq_dists)

        elif self.kernel == "poly":
            gamma = self._get_gamma(X_train)
            K = (gamma * (X_new @ X_train.T) + self.coef0) ** self.degree

        elif callable(self.kernel):
            n_new = X_new.shape[0]
            n_train = X_train.shape[0]
            K = np.zeros((n_new, n_train))
            for i in range(n_new):
                for j in range(n_train):
                    K[i, j] = self.kernel(X_new[i], X_train[j])

        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

        return K

    def _kernel_diag(self, X):
        if self.kernel == "linear":
            return np.sum(X * X, axis=1)

        elif self.kernel == "rbf":
            return np.ones(X.shape[0])

        elif self.kernel == "poly":
            gamma = self._get_gamma(X)
            return (gamma * np.sum(X * X, axis=1) + self.coef0) ** self.degree

        elif callable(self.kernel):
            return np.array([self.kernel(x, x) for x in X], dtype=float)

        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    @staticmethod
    def _center_kernel(K):
        n = K.shape[0]
        one = np.ones((n, n)) / n

        Kc = K - one @ K - K @ one + one @ K @ one

        return (Kc + Kc.T) / 2.0

    def _center_new_kernel(self, K_new):
        mean_row = K_new.mean(axis=1, keepdims=True)
        mean_col = self.K_raw_.mean(axis=0, keepdims=True)
        total_mean = self.K_raw_.mean()

        K_new_c = K_new - mean_row - mean_col + total_mean

        return K_new_c

    def _center_new_self_kernel_diag(self, Xn, K_new_raw):
        if not self.center:
            return self._kernel_diag(Xn)

        diag_raw = self._kernel_diag(Xn)
        cross_mean = K_new_raw.mean(axis=1)
        total_mean = self.K_raw_.mean()

        diag_centered = diag_raw - 2.0 * cross_mean + total_mean

        return np.maximum(diag_centered, 0.0)

    def _resolve_n_components(self, eigenvalues):
        n_valid = len(eigenvalues)

        if self.n_components is None:
            return n_valid

        if isinstance(self.n_components, float):
            ratio = eigenvalues / (eigenvalues.sum() + self.eps)
            cumsum = np.cumsum(ratio)

            return int(np.searchsorted(cumsum, self.n_components) + 1)

        k = int(self.n_components)

        return k

    def _get_gamma(self, X):
        if self.gamma is None:
            return 1.0 / X.shape[1]

        return self.gamma