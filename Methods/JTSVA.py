import numpy as np
from scipy.linalg import eigh, pinv, solve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class JTSVA:
    def __init__(
        self,
        d=3,
        k=13,
        tol=1e-8,
        max_iter=200,
        cpv_threshold=0.8,
        eps=1e-8,
        n_random_starts=0,
        random_state=None,
    ):
        self.d = d
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.cpv_threshold = cpv_threshold
        self.eps = eps
        self.n_random_starts = n_random_starts
        self.random_state = random_state

    def fit(self, X):
        n, m = X.shape
        self.n_features_in_ = m
        self.k_ = min(int(self.k), m)
        self.rng_ = np.random.default_rng(self.random_state)
        self.scaler_ = StandardScaler()
        Xn = self.scaler_.fit_transform(X)
        self.X_train_norm_ = Xn
        self.X_t_ = Xn[self.d:, :]
        self.X_dot_ = np.diff(Xn, axis=0)
        M = self._build_time_neighborhood_matrix(n, self.d)
        self.M_ = M

        XTX = Xn.T @ Xn
        XTMX = Xn.T @ M @ Xn
        XdotTXdot = self.X_dot_.T @ self.X_dot_

        H0 = XTX + XTMX + XdotTXdot
        H0 = self._sym(H0) + self.eps * np.eye(m)
        self.H0_ = H0

        W_list = []
        Beta_list = []
        lambda_list = []
        obj_list = []
        iter_list = []

        for comp in range(self.k_):
            init_vectors = self._initial_vectors(m, W_list)

            best_result = None

            for w0 in init_vectors:
                result = self._fit_one_component(
                    Xn=Xn,
                    H0=H0,
                    W_list=W_list,
                    w_init=w0,
                )

                if best_result is None or result["objective"] > best_result["objective"]:
                    best_result = result

            w = best_result["w"]
            beta = best_result["beta"]
            eigval = best_result["eigval"]

            w = self._orthonormalize_euclidean(w, W_list)
            w_norm = np.linalg.norm(w)
            w = w / w_norm

            W_list.append(w)
            Beta_list.append(beta)
            lambda_list.append(eigval)
            obj_list.append(best_result["objective"])
            iter_list.append(best_result["n_iter"])

        self.W_ = np.column_stack(W_list)
        self.Beta_ = np.column_stack(Beta_list)
        self.lambda_ = np.asarray(lambda_list)
        self.objective_ = np.asarray(obj_list)
        self.n_iter_ = np.asarray(iter_list)

        self.orthogonality_error_ = np.linalg.norm(
            self.W_.T @ self.W_ - np.eye(self.k_)
        )

        self.S_ = Xn @ self.W_

        I_m = np.eye(m)
        self.R_ = I_m - self.W_ @ self.W_.T

        X_res = Xn @ self.R_
        self.X_res_train_ = X_res

        # PCA for static / residual space
        self.pca_ = self._fit_pca(X_res)
        self.V_ = self.pca_.components_.T

        self.U_ = X_res @ self.V_
        self.E_ = X_res - self.U_ @ self.V_.T

        self.cov_d_ = self._second_moment(self.S_)
        self.cov_s_ = self._second_moment(self.U_)

        return self

    def transform(self, X):
        Xn = self.scaler_.transform(X)

        S = Xn @ self.W_
        X_res = Xn @ self.R_

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

        return Td2, Ts2, Q

    @staticmethod
    def compute_phi(Td2, Ts2, Q, Td2_lim, Ts2_lim, Q_lim, eps=1e-12):
        return (
            Td2 / max(float(Td2_lim), eps)
            + Ts2 / max(float(Ts2_lim), eps)
            + Q / max(float(Q_lim), eps)
        )

    def _fit_one_component(self, Xn, H0, W_list, w_init):
        w = self._orthonormalize_euclidean(w_init, W_list)
        norm_w = np.linalg.norm(w)

        if norm_w <= self.eps:
            w = self._random_unit_vector(Xn.shape[1], W_list)
        else:
            w = w / norm_w

        beta = np.zeros(self.d)
        eigval = np.nan

        for it in range(self.max_iter):
            beta_new = self._update_beta(w, Xn)

            X_hat = self._lagged_linear_combination(Xn, beta_new)

            H = H0 + X_hat.T @ X_hat
            H = self._sym(H) + self.eps * np.eye(H.shape[0])

            C = self.X_t_.T @ X_hat + X_hat.T @ self.X_t_
            C = self._sym(C)

            w_new, eigval = self._solve_w(C, H, W_list)

            if np.dot(w_new, w) < 0:
                w_new = -w_new

            w_diff = np.linalg.norm(w_new - w)
            beta_diff = np.linalg.norm(beta_new - beta)

            w = w_new
            beta = beta_new

            if w_diff < self.tol and beta_diff < self.tol:
                break

        beta = self._update_beta(w, Xn)
        X_hat = self._lagged_linear_combination(Xn, beta)

        objective = float(w.T @ self.X_t_.T @ X_hat @ w)

        return {
            "w": w,
            "beta": beta,
            "eigval": eigval,
            "objective": objective,
            "n_iter": it + 1,
        }

    def _solve_w(self, C, H, W_list):
        m = C.shape[0]

        if len(W_list) == 0:
            vals, vecs = self._safe_generalized_eigh(C, H)

            order = np.argsort(np.real(vals))[::-1]

            for idx in order:
                w = np.real(vecs[:, idx])
                if np.all(np.isfinite(w)):
                    norm = np.linalg.norm(w)
                    if norm > self.eps:
                        return w / norm, float(np.real(vals[idx]))

            return self._random_unit_vector(m, W_list), np.nan

        W_prev = np.column_stack(W_list)

        try:
            H_inv_C = solve(H, C, assume_a="pos")
            H_inv_W = solve(H, W_prev, assume_a="pos")
        except Exception:
            H_inv = pinv(H)
            H_inv_C = H_inv @ C
            H_inv_W = H_inv @ W_prev

        B = W_prev.T @ H_inv_W
        B_inv = pinv(B + self.eps * np.eye(B.shape[0]))

        P_orth = np.eye(m) - H_inv_W @ B_inv @ W_prev.T

        A = P_orth @ H_inv_C

        try:
            vals, vecs = np.linalg.eig(A)
        except Exception:
            vals, vecs = np.linalg.eig(pinv(np.eye(m)) @ A)

        vals = np.real(vals)
        vecs = np.real(vecs)

        order = np.argsort(vals)[::-1]

        for idx in order:
            w = vecs[:, idx]
            w = self._orthonormalize_euclidean(w, W_list)

            norm = np.linalg.norm(w)

            if norm > self.eps and np.all(np.isfinite(w)):
                return w / norm, float(vals[idx])

        return self._random_unit_vector(m, W_list), np.nan

    def _update_beta(self, w, Xn):
        n = Xn.shape[0]
        n_t = n - self.d

        St = np.zeros((n_t, self.d))

        for r in range(1, self.d + 1):
            St[:, r - 1] = Xn[self.d - r:n - r, :] @ w

        Xt_w = self.X_t_ @ w

        StSt = St.T @ St
        beta = pinv(StSt + self.eps * np.eye(self.d)) @ (St.T @ Xt_w)

        if not np.all(np.isfinite(beta)) or np.linalg.norm(beta) <= self.eps:
            beta = np.zeros(self.d)
            beta[0] = 1.0

        den = beta.T @ StSt @ beta

        if den <= self.eps or not np.isfinite(den):
            beta = beta / (np.linalg.norm(beta) + self.eps)
        else:
            beta = beta / np.sqrt(den + self.eps)

        return beta

    def _lagged_linear_combination(self, Xn, beta):
        n, m = Xn.shape
        X_hat = np.zeros((n - self.d, m))

        for r in range(1, self.d + 1):
            X_hat += beta[r - 1] * Xn[self.d - r:n - r, :]

        return X_hat

    def _build_time_neighborhood_matrix(self, n, d):
        G = np.zeros((n, n), dtype=float)

        for i in range(n):
            neighbors = (
                list(range(max(0, i - d), i))
                + list(range(i + 1, min(n, i + d + 1)))
            )

            if len(neighbors) == 0:
                continue

            weight = 1.0 / len(neighbors)
            G[i, neighbors] = weight

        I = np.eye(n)
        M = (I - G).T @ (I - G)

        return self._sym(M)

    def _initial_vectors(self, m, W_list):
        vectors = []
        for j in range(m):
            e = np.zeros(m)
            e[j] = 1.0
            e = self._orthonormalize_euclidean(e, W_list)

            if np.linalg.norm(e) > self.eps:
                vectors.append(e / np.linalg.norm(e))

        for _ in range(int(self.n_random_starts)):
            vectors.append(self._random_unit_vector(m, W_list))

        if len(vectors) == 0:
            vectors.append(self._random_unit_vector(m, W_list))

        return vectors

    def _random_unit_vector(self, m, W_list):
        for _ in range(200):
            w = self.rng_.normal(size=m)
            w = self._orthonormalize_euclidean(w, W_list)

            norm = np.linalg.norm(w)

            if norm > self.eps:
                return w / norm

        raise RuntimeError("Cannot initialize a valid projection vector. Try reducing k.")

    def _orthonormalize_euclidean(self, w, W_list):
        w = np.asarray(w, dtype=float).ravel()

        for w_old in W_list:
            w = w - (w_old @ w) * w_old

        return w

    def _fit_pca(self, X):
        n, m = X.shape
        max_comp = min(n, m)

        if isinstance(self.cpv_threshold, float) and 0 < self.cpv_threshold < 1:
            pca = PCA(n_components=self.cpv_threshold, svd_solver="full")
        else:
            c = int(self.cpv_threshold)
            c = max(1, min(c, max_comp))
            pca = PCA(n_components=c, svd_solver="full")

        pca.fit(X)
        return pca

    def _second_moment(self, A):
        A = np.asarray(A, dtype=float)

        if A.ndim == 1:
            A = A.reshape(-1, 1)

        n = A.shape[0]

        if n <= 1:
            return self.eps * np.eye(A.shape[1])

        C = A.T @ A / (n - 1)
        C = np.atleast_2d(C)

        return self._sym(C)

    @staticmethod
    def _sym(A):
        return (A + A.T) / 2.0

    def _safe_generalized_eigh(self, C, H):
        try:
            vals, vecs = eigh(C, H)
        except Exception:
            A = pinv(H) @ C
            vals, vecs = np.linalg.eig(A)
            vals = np.real(vals)
            vecs = np.real(vecs)

        return vals, vecs