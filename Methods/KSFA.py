from __future__ import annotations
import numpy as np
from typing import Literal, Optional
from sklearn.preprocessing import StandardScaler
class _LinearSFA:
    def __init__(self, n_components=None, eps=1e-10):
        self.n_components = n_components
        self.eps = eps

    def fit(self, X):
        X = self._check_array(X, min_samples=2)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        C = (Xc.T @ Xc) / max(n_samples - 1, 1)
        C = (C + C.T) / 2.0

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

        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class KSFA:
    def __init__(
        self,
        n_components: Optional[int] = None,
        kernel: Literal["rbf", "poly", "linear"] = "rbf",
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
        n_landmarks: Optional[int] = None,
        landmark_indices=None,
        random_state=None,
        standardize: bool = True,
        eps: float = 1e-10,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.n_landmarks = n_landmarks
        self.landmark_indices = landmark_indices
        self.random_state = random_state
        self.standardize = standardize
        self.eps = eps

    def fit(self, X):
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.gamma_ = self.gamma if self.gamma is not None else 1.0 / n_features
        if self.standardize:
            self.scaler_ = StandardScaler()
            Xn = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            Xn = X.copy()

        self.X_train_norm_ = Xn

        self.landmarks_ = self._choose_landmarks(Xn)

        K_mm = self._kernel_matrix(self.landmarks_, self.landmarks_)
        K_mm = (K_mm + K_mm.T) / 2.0

        evals, evecs = np.linalg.eigh(K_mm)

        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        valid = evals > self.eps
        evals = evals[valid]
        evecs = evecs[:, valid]

        self.landmark_eigvals_ = evals
        self.landmark_eigvecs_ = evecs

        self.nystrom_normalizer_ = evecs @ np.diag(
            1.0 / np.sqrt(evals + self.eps)
        )

        Phi_train = self._nystrom_features(Xn)

        self.sfa_ = _LinearSFA(
            n_components=self.n_components,
            eps=self.eps,
        )

        self.sfa_.fit(Phi_train)

        self.n_components_ = self.sfa_.n_components_
        self.delta_values_ = self.sfa_.delta_values_

        self.Y_train_ = self.sfa_.transform(Phi_train)
        self.T2_train_, self.S2_train_ = self.get_monitoring_statistics(
            X,
            already_original_scale=True,
        )

        return self

    def transform(self, X, already_original_scale=True):
        Xn = self._prepare_X(X, already_original_scale=already_original_scale)
        Phi = self._nystrom_features(Xn)

        return self.sfa_.transform(Phi)

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def get_monitoring_statistics(self, X, already_original_scale=True):
        Y = self.transform(
            X,
            already_original_scale=already_original_scale,
        )

        T2 = np.sum(Y ** 2, axis=1)

        dY = np.diff(Y, axis=0)

        S2_valid = np.sum(
            (dY ** 2) / (self.delta_values_ + self.eps),
            axis=1,
        )

        S2 = np.full(Y.shape[0], np.nan)
        S2[1:] = S2_valid

        return T2, S2

    def _prepare_X(self, X, already_original_scale=True):
        if self.standardize and already_original_scale:
            Xn = self.scaler_.transform(X)
        else:
            Xn = X.copy()

        return Xn

    def _choose_landmarks(self, Xn):
        n_samples = Xn.shape[0]

        if self.landmark_indices is not None:
            idx = np.asarray(self.landmark_indices, dtype=int)
            self.landmark_indices_ = idx.copy()

            return Xn[idx].copy()

        if self.n_landmarks is None:
            m = n_samples
        else:
            m = int(self.n_landmarks)
            m = min(m, n_samples)

        rng = np.random.default_rng(self.random_state)

        idx = rng.choice(
            n_samples,
            size=m,
            replace=False,
        )

        idx.sort()

        self.landmark_indices_ = idx

        return Xn[idx].copy()

    def _nystrom_features(self, Xn):
        K_nm = self._kernel_matrix(Xn, self.landmarks_)
        Phi = K_nm @ self.nystrom_normalizer_

        return Phi

    def _kernel_matrix(self, A, B):
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)

        if self.kernel == "linear":
            return A @ B.T

        if self.kernel == "rbf":
            A2 = np.sum(A ** 2, axis=1, keepdims=True)
            B2 = np.sum(B ** 2, axis=1, keepdims=True).T
            dist2 = np.maximum(A2 + B2 - 2.0 * A @ B.T, 0.0)

            return np.exp(-self.gamma_ * dist2)

        if self.kernel == "poly":
            return (self.gamma_ * (A @ B.T) + self.coef0) ** self.degree

        raise ValueError(f"Unsupported kernel: {self.kernel!r}")