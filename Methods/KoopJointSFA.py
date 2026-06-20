from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]
class KoopJointSFA:
    def __init__(
        self,
        n_components: int = 2,
        dictionary: Literal["identity", "poly2", "rbf", "mixed"] = "mixed",
        n_lags: int = 2,
        lag_step: int = 1,
        n_centers: Optional[int] = 100,
        center_method: Literal["kmeans", "random", "uniform"] = "kmeans",
        gamma: Optional[Union[float, Sequence[float]]] = None,
        joint_alpha: float = 1.0,
        derivative_weight: float = 1.0,

        include_bias: bool = True,
        center_features: bool = True,
        whiten_features: bool = True,
        control_quantile: float = 0.99,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.dictionary = dictionary
        self.n_lags = n_lags
        self.lag_step = lag_step
        self.n_centers = n_centers
        self.center_method = center_method
        self.gamma = gamma
        self.include_bias = include_bias
        self.joint_alpha = joint_alpha
        self.derivative_weight = derivative_weight
        self.center_features = center_features
        self.whiten_features = whiten_features
        self.control_quantile = control_quantile
        # Default
        self.random_state = random_state
        self.covariance_ridge = 1e-8
        self.ridge = 1e-6
        self.kmeans_max_iter = 100
        self.kmeans_tol = 1e-5
        self.eps = 1e-10

    def fit(self, X: ArrayLike) -> "KoopJointSFA":
        self.n_input_features_ = X.shape[1]
        self._validate_parameters()

        X_delay = self._delay_embed(X)
        if X_delay.shape[0] < 3:
            raise ValueError("Error")
        self.n_delay_features_ = X_delay.shape[1]

        self._fit_dictionary(X_delay)
        Psi = self._raw_observables(X_delay)
        Z = self._fit_feature_normalizer(Psi)
        A = Z[:-1]
        B = Z[1:]
        n_pairs = A.shape[0]

        # Koopman least squares in normalized observable space
        G = (A.T @ A) / n_pairs
        C = (A.T @ B) / n_pairs
        I = np.eye(G.shape[0])
        self.K_ = np.linalg.solve(G + self.ridge * I, C)

        eigvals, eigvecs = np.linalg.eig(self.K_)
        self.eigenvalues_all_ = eigvals
        self.eigenvectors_all_ = eigvecs

        # Joint Koopman-SFA projection.
        D = B - A
        R = B - A @ self.K_
        C_delta = (D.T @ D) / n_pairs
        C_residual = (R.T @ R) / n_pairs
        C_state = (A.T @ A) / n_pairs

        objective = (
            self.derivative_weight * C_delta
            + self.joint_alpha * C_residual
        )
        objective = self._symmetrize(objective)
        C_state = self._symmetrize(C_state)

        eigvals_joint, W_all = self._solve_generalized_eigh(
            objective,
            C_state + self.covariance_ridge * I,
        )
        selected = []
        for idx in range(W_all.shape[1]):
            y = Z @ W_all[:, idx]
            if np.var(y) > self.eps:
                selected.append(idx)
            if len(selected) >= int(self.n_components):
                break

        self.selected_indices_ = np.asarray(selected, dtype=int)
        self.joint_eigenvalues_all_ = eigvals_joint
        self.joint_eigenvalues_ = eigvals_joint[self.selected_indices_]
        self.W_all_ = W_all
        self.W_ = W_all[:, self.selected_indices_]

        Y_raw = Z @ self.W_
        self.output_mean_ = Y_raw.mean(axis=0)
        self.output_std_ = Y_raw.std(axis=0) + self.eps
        Y = self._standardize_output(Y_raw)

        self.delta_values_ = np.mean(np.diff(Y, axis=0) ** 2, axis=0)
        self._fit_koopman_eigenfeature_metadata(Z)
        self._fit_monitoring_statistics(Z, Y)

        self.n_components_ = len(self.selected_indices_)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        Z = self._normalized_observables_from_X(X)
        Y_raw = Z @ self.W_
        return self._standardize_output(Y_raw)

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)

    def predict_observables(self, X: ArrayLike, steps: int = 1) -> np.ndarray:
        Z = self._normalized_observables_from_X(X)
        K_power = np.linalg.matrix_power(self.K_, int(steps))
        return Z @ K_power

    def predict_features(self, X: ArrayLike, steps: int = 1) -> np.ndarray:
        Z_future = self.predict_observables(X, steps=steps)
        Y_raw = Z_future @ self.W_
        return self._standardize_output(Y_raw)

    def transform_koopman_eigenfeatures(self, X: ArrayLike) -> np.ndarray:
        Z = self._normalized_observables_from_X(X)
        return self._complex_eigenfeatures_to_real(Z, self.eigenvectors_all_)[0]

    def monitor(self, X: ArrayLike, include_features: bool = False) -> Dict[str, np.ndarray]:
        Z = self._normalized_observables_from_X(X)
        Y = self._standardize_output(Z @ self.W_)
        A = Z[:-1]
        B = Z[1:]
        dY = np.diff(Y, axis=0)
        R = B - A @ self.K_

        T2 = np.sum(Y[1:] ** 2, axis=1)
        S2 = np.sum((dY ** 2) / self.delta_scale_, axis=1)
        Qkoop = np.sum((R ** 2) / self.residual_scale_, axis=1)
        combined = (
            T2 / self.T2_limit_
            + S2 / self.S2_limit_
            + Qkoop / self.Qkoop_limit_
        )

        lag_offset = self.n_lags_ * self.lag_step_
        time_index = np.arange(lag_offset + 1, lag_offset + 1 + len(T2))

        out: Dict[str, np.ndarray] = {
            "T2": T2,
            "S2": S2,
            "Qkoop": Qkoop,
            "combined": combined,
            "time_index": time_index,
            "T2_limit": np.full_like(T2, self.T2_limit_, dtype=float),
            "S2_limit": np.full_like(S2, self.S2_limit_, dtype=float),
            "Qkoop_limit": np.full_like(Qkoop, self.Qkoop_limit_, dtype=float),
            "combined_limit": np.full_like(combined, self.combined_limit_, dtype=float),
        }
        if include_features:
            out["Y"] = Y[1:]
            out["dY"] = dY
            out["koopman_residual"] = R
        return out


    def _validate_parameters(self) -> None:
        self.dictionary_ = str(self.dictionary).lower()
        if self.dictionary_ not in {"identity", "poly2", "rbf", "mixed"}:
            raise ValueError("Error")
        self.center_method_ = str(self.center_method).lower()
        if self.center_method_ not in {"kmeans", "random", "uniform"}:
            raise ValueError("Error")
        self.n_lags_ = int(self.n_lags)
        self.lag_step_ = int(self.lag_step)

    def _delay_embed(self, X: np.ndarray) -> np.ndarray:
        lag_offset = self.n_lags_ * self.lag_step_
        blocks = []
        for lag in range(self.n_lags_ + 1):
            start = lag_offset - lag * self.lag_step_
            stop = X.shape[0] - lag * self.lag_step_
            blocks.append(X[start:stop])
        return np.hstack(blocks)

    def _fit_dictionary(self, X_delay: np.ndarray) -> None:
        if self.dictionary_ in {"rbf", "mixed"}:
            n_samples = X_delay.shape[0]
            m = n_samples if self.n_centers is None else int(self.n_centers)

            if self.center_method_ == "kmeans":
                centers = self._kmeans_centers(X_delay, m)
            elif self.center_method_ == "uniform":
                idx = np.linspace(0, n_samples - 1, m).round().astype(int)
                centers = X_delay[idx].copy()
            else:
                rng = np.random.default_rng(self.random_state)
                idx = rng.choice(n_samples, size=m, replace=False)
                idx.sort()
                centers = X_delay[idx].copy()

            self.centers_ = centers
            self.gamma_ = self._prepare_gamma(centers)
        else:
            self.centers_ = None
            self.gamma_ = None

    def _raw_observables(self, X_delay: np.ndarray) -> np.ndarray:
        X_delay = np.asarray(X_delay, dtype=float)
        blocks = []

        if self.dictionary_ in {"identity", "mixed"}:
            blocks.append(X_delay)

        if self.dictionary_ == "poly2":
            blocks.append(X_delay)
            blocks.append(self._quadratic_terms(X_delay))

        if self.dictionary_ in {"rbf", "mixed"}:
            blocks.append(self._rbf_features(X_delay))

        if self.include_bias:
            blocks.append(np.ones((X_delay.shape[0], 1), dtype=float))

        return np.hstack(blocks)

    def _quadratic_terms(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape
        cols = []
        for i in range(n_features):
            for j in range(i, n_features):
                cols.append((X[:, i] * X[:, j])[:, None])
        return np.hstack(cols) if cols else np.empty((n_samples, 0))

    def _rbf_features(self, X: np.ndarray) -> np.ndarray:
        X2 = np.sum(X ** 2, axis=1, keepdims=True)
        C2 = np.sum(self.centers_ ** 2, axis=1, keepdims=True).T
        dist2 = np.maximum(X2 + C2 - 2 * X @ self.centers_.T, 0.0)

        gammas = np.atleast_1d(self.gamma_).astype(float)
        return np.hstack([np.exp(-g * dist2) for g in gammas])

    def _prepare_gamma(self, centers: np.ndarray) -> np.ndarray:
        if self.gamma is None:
            return np.asarray([self._median_gamma(centers)], dtype=float)
        gammas = np.atleast_1d(np.asarray(self.gamma, dtype=float))
        return gammas

    def _median_gamma(self, centers: np.ndarray) -> float:
        if len(centers) <= 1:
            return 1.0
        C2 = np.sum(centers ** 2, axis=1, keepdims=True)
        dist2 = np.maximum(C2 + C2.T - 2 * centers @ centers.T, 0.0)
        upper = dist2[np.triu_indices_from(dist2, k=1)]
        med = np.median(upper[upper > self.eps]) if np.any(upper > self.eps) else 1.0
        return 1.0 / (med + self.eps)

    def _kmeans_centers(self, X: np.ndarray, n_centers: int) -> np.ndarray:
        n_samples = X.shape[0]
        if n_centers == n_samples:
            return X.copy()

        rng = np.random.default_rng(self.random_state)
        centers = self._kmeans_plus_plus_init(X, n_centers, rng)

        prev_inertia = None
        for _ in range(int(self.kmeans_max_iter)):
            dist2 = self._squared_distances(X, centers)
            labels = np.argmin(dist2, axis=1)
            inertia = float(np.sum(dist2[np.arange(n_samples), labels]))

            new_centers = centers.copy()
            for j in range(n_centers):
                mask = labels == j
                if np.any(mask):
                    new_centers[j] = X[mask].mean(axis=0)
                else:
                    farthest = np.argmax(np.min(dist2, axis=1))
                    new_centers[j] = X[farthest]

            movement = np.linalg.norm(new_centers - centers) / (np.linalg.norm(centers) + self.eps)
            centers = new_centers
            if prev_inertia is not None:
                if abs(prev_inertia - inertia) <= self.kmeans_tol * max(prev_inertia, 1.0):
                    break
            if movement <= self.kmeans_tol:
                break
            prev_inertia = inertia

        return centers

    def _kmeans_plus_plus_init(
        self,
        X: np.ndarray,
        n_centers: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_samples = X.shape[0]
        centers = np.empty((n_centers, X.shape[1]), dtype=float)
        first = int(rng.integers(0, n_samples))
        centers[0] = X[first]
        closest_dist2 = self._squared_distances(X, centers[:1]).ravel()

        for j in range(1, n_centers):
            total = float(np.sum(closest_dist2))
            if total <= self.eps:
                idx = int(rng.integers(0, n_samples))
            else:
                probs = closest_dist2 / total
                idx = int(rng.choice(n_samples, p=probs))
            centers[j] = X[idx]
            new_dist2 = self._squared_distances(X, centers[j:j + 1]).ravel()
            closest_dist2 = np.minimum(closest_dist2, new_dist2)
        return centers

    @staticmethod
    def _squared_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        X2 = np.sum(X ** 2, axis=1, keepdims=True)
        C2 = np.sum(C ** 2, axis=1, keepdims=True).T
        return np.maximum(X2 + C2 - 2 * X @ C.T, 0.0)

    def _fit_feature_normalizer(self, Psi: np.ndarray) -> np.ndarray:
        self.feature_mean_ = Psi.mean(axis=0) if self.center_features else np.zeros(Psi.shape[1])
        Z = Psi - self.feature_mean_

        if self.whiten_features:
            C = (Z.T @ Z) / max(Z.shape[0] - 1, 1)
            evals, evecs = np.linalg.eigh(self._symmetrize(C))
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]
            valid = evals > self.eps
            self.whitening_ = evecs[:, valid] @ np.diag(1.0 / np.sqrt(evals[valid] + self.eps))
            return Z @ self.whitening_

        self.whitening_ = None
        return Z

    def _normalize_features(self, Psi: np.ndarray) -> np.ndarray:
        Z = Psi - self.feature_mean_
        if self.whitening_ is not None:
            Z = Z @ self.whitening_
        return Z

    def _normalized_observables_from_X(self, X: ArrayLike) -> np.ndarray:
        X_delay = self._delay_embed(X)
        Psi = self._raw_observables(X_delay)
        return self._normalize_features(Psi)

    def _solve_generalized_eigh(
        self,
        M: np.ndarray,
        S: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        M = self._symmetrize(M)
        S = self._symmetrize(S)
        evals_s, evecs_s = np.linalg.eigh(S)
        valid = evals_s > self.eps
        Sinv_half = evecs_s[:, valid] @ np.diag(1.0 / np.sqrt(evals_s[valid]))
        H = self._symmetrize(Sinv_half.T @ M @ Sinv_half)
        evals_h, evecs_h = np.linalg.eigh(H)
        order = np.argsort(evals_h)
        evals_h = evals_h[order]
        evecs_h = evecs_h[:, order]
        W = Sinv_half @ evecs_h
        return evals_h, W

    def _fit_koopman_eigenfeature_metadata(self, Z: np.ndarray) -> None:
        Phi, meta = self._complex_eigenfeatures_to_real(Z, self.eigenvectors_all_)
        self.eigenfeature_metadata_ = meta
        if Phi.shape[1] > 0:
            self.eigenfeature_delta_scores_ = np.mean(np.diff(Phi, axis=0) ** 2, axis=0) / (
                np.var(Phi, axis=0) + self.eps
            )
        else:
            self.eigenfeature_delta_scores_ = np.empty(0)

    def _complex_eigenfeatures_to_real(
        self,
        Z: np.ndarray,
        eigvecs: np.ndarray,
    ) -> Tuple[np.ndarray, list]:
        channels = []
        metadata = []
        Phi_complex = Z @ eigvecs
        for i in range(Phi_complex.shape[1]):
            phi = Phi_complex[:, i]
            if np.max(np.abs(np.imag(phi))) > 100 * self.eps:
                channels.append(np.real(phi)[:, None])
                metadata.append({"eigen_index": i, "part": "real", "eigenvalue": self.eigenvalues_all_[i]})
                channels.append(np.imag(phi)[:, None])
                metadata.append({"eigen_index": i, "part": "imag", "eigenvalue": self.eigenvalues_all_[i]})
            else:
                channels.append(np.real(phi)[:, None])
                metadata.append({"eigen_index": i, "part": "real", "eigenvalue": self.eigenvalues_all_[i]})
        if not channels:
            return np.empty((Z.shape[0], 0)), metadata
        return np.hstack(channels).astype(float), metadata

    def _standardize_output(self, Y_raw: np.ndarray) -> np.ndarray:
        return (Y_raw - self.output_mean_) / self.output_std_

    @staticmethod
    def _symmetrize(A: np.ndarray) -> np.ndarray:
        return 0.5 * (A + A.T)

    def _fit_monitoring_statistics(self, Z: np.ndarray, Y: np.ndarray) -> None:
        A = Z[:-1]
        B = Z[1:]
        dY = np.diff(Y, axis=0)
        R = B - A @ self.K_

        self.delta_scale_ = np.mean(dY ** 2, axis=0) + self.eps
        self.residual_scale_ = np.mean(R ** 2, axis=0) + self.eps

        T2 = np.sum(Y[1:] ** 2, axis=1)
        S2 = np.sum((dY ** 2) / self.delta_scale_, axis=1)
        Qkoop = np.sum((R ** 2) / self.residual_scale_, axis=1)

        q = float(self.control_quantile)
        self.T2_limit_ = self._kde_limit(T2, q)
        self.S2_limit_ = self._kde_limit(S2, q)
        self.Qkoop_limit_ = self._kde_limit(Qkoop, q)

        combined = (
            T2 / self.T2_limit_
            + S2 / self.S2_limit_
            + Qkoop / self.Qkoop_limit_
        )
        self.combined_limit_ = self._kde_limit(combined, q)

        self.training_statistics_ = {
            "T2": T2,
            "S2": S2,
            "Qkoop": Qkoop,
            "combined": combined,
        }

    def _safe_quantile(self, values: np.ndarray, q: float) -> float:
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return self.eps
        value = float(np.quantile(values, q))
        return max(value, self.eps)
    
    def _kde_limit(self, stat, conf_level=0.99, grids=5000):
        """
        KDE-based control limit.
        """
        stat = np.asarray(stat, dtype=float).ravel()
        stat = stat[np.isfinite(stat)]
        if np.allclose(stat, stat[0]):
            return float(stat[0])

        std = np.std(stat)
        mean = np.mean(stat)

        lower = max(0.0, np.min(stat) - 3.0 * std)
        upper = max(np.max(stat), mean + 8.0 * std)

        if upper <= lower:
            upper = lower + 1.0

        grid = np.linspace(lower, upper, grids)

        kde = gaussian_kde(stat)
        pdf = kde(grid)

        cdf = np.r_[0.0, cumulative_trapezoid(pdf, grid)]
        cdf = cdf / cdf[-1]

        return float(np.interp(conf_level, cdf, grid))

