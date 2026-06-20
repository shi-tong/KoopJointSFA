import numpy as np
from scipy.linalg import eigh, pinv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
class DiCCA:
    def __init__(
        self,
        d=3,
        k=13,
        alpha=0.99,
        eps=1e-8,
        tol=1e-7,
        max_iter=200,
        cpv_threshold=0.85,
        random_state=None,
        use_pca_static=True,
    ):
        self.d = d
        self.k = k
        self.alpha = alpha
        self.eps = eps
        self.tol = tol
        self.max_iter = max_iter
        self.cpv_threshold = cpv_threshold
        self.random_state = random_state
        self.use_pca_static = use_pca_static

    def fit(self, X):
        n, m = X.shape
        self.n_features_in_ = m
        self.k_ = min(self.k, m)
        self.rng_ = np.random.default_rng(self.random_state)
        self.scaler_ = StandardScaler()
        Xn = self.scaler_.fit_transform(X)
        X_current = Xn[self.d:, :]
        W = []
        Beta = []
        lambdas = []
        for comp in range(self.k_):
            w = self._random_unit_vector(m, W)
            beta = np.zeros(self.d)
            eigval = np.nan

            for _ in range(self.max_iter):
                S_lag = self._lagged_projection_matrix(Xn, w)
                s_cur = X_current @ w

                beta_new = pinv(
                    S_lag.T @ S_lag + self.eps * np.eye(self.d)
                ) @ (S_lag.T @ s_cur)

                X_pred = self._lagged_linear_combination(Xn, beta_new)

                C = X_current.T @ X_pred + X_pred.T @ X_current
                H = X_current.T @ X_current + X_pred.T @ X_pred

                C = (C + C.T) / 2.0
                H = (H + H.T) / 2.0 + self.eps * np.eye(m)

                try:
                    vals, vecs = eigh(C, H, subset_by_index=[m - 1, m - 1])
                    w_new = vecs[:, -1]
                    eigval = vals[-1]
                except Exception:
                    vals, vecs = np.linalg.eig(pinv(H) @ C)
                    idx = np.argmax(np.real(vals))
                    w_new = np.real(vecs[:, idx])
                    eigval = np.real(vals[idx])
                
                w_new = self._orthogonalize(w_new, W)

                norm = np.linalg.norm(w_new)
                if norm < self.eps:
                    w_new = self._random_unit_vector(m, W)
                else:
                    w_new = w_new / norm

                if np.dot(w_new, w) < 0:
                    w_new = -w_new

                if (
                    np.linalg.norm(w_new - w) < self.tol
                    and np.linalg.norm(beta_new - beta) < self.tol
                ):
                    w = w_new
                    beta = beta_new
                    break

                w = w_new
                beta = beta_new

            beta = self._estimate_beta(Xn, w)

            W.append(w.copy())
            Beta.append(beta.copy())
            lambdas.append(eigval)

        self.W_ = np.column_stack(W)              # m × k
        self.Beta_ = np.column_stack(Beta)        # d × k
        self.lambda_ = np.asarray(lambdas)

        S = Xn @ self.W_
        self.P_ = Xn.T @ S @ pinv(S.T @ S + self.eps * np.eye(self.k_))
        X_dyn_hat = S @ self.P_.T

        # Residual space
        X_res = Xn - X_dyn_hat

        pca = PCA(n_components=self.cpv_threshold, svd_solver="full")
        pca.fit(X_res)

        self.V_ = pca.components_.T

        U = X_res @ self.V_
        E = X_res - U @ self.V_.T

        self.S_ = S
        self.U_ = U
        self.E_ = E
        self.X_res_ = X_res
        self.cov_d_ = self._cov2d(S)
        self.cov_s_ = self._cov2d(U)
        self.cov_e_ = self._cov2d(E)

        _, _, _, phi_train = self.get_monitoring_statistics(X)
        self.phi_limit_ = np.nanpercentile(phi_train, self.alpha * 100)

        return self

    def transform(self, X):
        Xn = self.scaler_.transform(X)
        S = Xn @ self.W_
        X_dyn_hat = S @ self.P_.T
        X_res = Xn - X_dyn_hat
        U = X_res @ self.V_
        E = X_res - U @ self.V_.T

        return S, U, E

    def get_monitoring_statistics(self, X):
        S, U, E = self.transform(X)

        inv_cov_d = pinv(
            self.cov_d_ + self.eps * np.eye(self.cov_d_.shape[0])
        )

        inv_cov_s = pinv(
            self.cov_s_ + self.eps * np.eye(self.cov_s_.shape[0])
        )

        Td2 = np.einsum("ij,jk,ik->i", S, inv_cov_d, S)
        Ts2 = np.einsum("ij,jk,ik->i", U, inv_cov_s, U)
        Q = np.sum(E ** 2, axis=1)

        trace_d = np.trace(self.cov_d_) + self.eps
        trace_s = np.trace(self.cov_s_) + self.eps
        trace_e = np.trace(self.cov_e_) + self.eps

        phi = (
            Td2 / trace_d
            + Ts2 / trace_s
            + Q / trace_e
        )

        return Td2, Ts2, Q, phi

    def _lagged_projection_matrix(self, Xn, w):
        n = Xn.shape[0]
        S_lag = np.zeros((n - self.d, self.d))

        for r in range(1, self.d + 1):
            S_lag[:, r - 1] = Xn[self.d - r:n - r, :] @ w

        return S_lag

    def _lagged_linear_combination(self, Xn, beta):
        n, m = Xn.shape
        X_pred = np.zeros((n - self.d, m))

        for r in range(1, self.d + 1):
            X_pred += beta[r - 1] * Xn[self.d - r:n - r, :]

        return X_pred

    def _estimate_beta(self, Xn, w):
        S_lag = self._lagged_projection_matrix(Xn, w)
        s_cur = Xn[self.d:, :] @ w

        beta = pinv(
            S_lag.T @ S_lag + self.eps * np.eye(self.d)
        ) @ (S_lag.T @ s_cur)

        return beta

    def _orthogonalize(self, w, W_list):
        w = np.asarray(w, dtype=float).ravel()

        for w_old in W_list:
            w = w - np.dot(w_old, w) * w_old

        return w

    def _random_unit_vector(self, m, W_list):
        for _ in range(100):
            w = self.rng_.normal(size=m)
            w = self._orthogonalize(w, W_list)

            norm = np.linalg.norm(w)

            if norm > self.eps:
                return w / norm

    def _cov2d(self, A):
        A = np.asarray(A, dtype=float)

        if A.ndim == 1:
            A = A.reshape(-1, 1)

        if A.shape[0] <= 1:
            return np.eye(A.shape[1]) * self.eps

        C = np.cov(A, rowvar=False)
        C = np.atleast_2d(C)

        return (C + C.T) / 2.0