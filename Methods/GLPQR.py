import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from scipy.linalg import solve_triangular

class GLPQR:
    def __init__(
        self,
        sigma=2500,
        lam=0.14,
        alpha_glpqr=0.5,
        beta=1e-3,
        n_components=3,
        eps=1e-10,
        max_iter_lasso=10000,
        random_state=None,
        zero_diag_lle=False,
    ):
        self.sigma = sigma
        self.lam = lam
        self.alpha = alpha_glpqr
        self.beta = beta
        self.n_components = n_components
        self.eps = eps
        self.max_iter_lasso = max_iter_lasso
        self.random_state = random_state
        self.zero_diag_lle = zero_diag_lle

    def fit(self, X, Y):
        n, p = X.shape
        q = Y.shape[1]

        self.n_samples_ = n
        self.n_features_in_ = p
        self.n_quality_in_ = q

        self.scaler_x_ = StandardScaler()
        self.scaler_y_ = StandardScaler()

        X_norm = self.scaler_x_.fit_transform(X)
        Y_norm = self.scaler_y_.fit_transform(Y)

        self.training_X_ = X_norm

        K_raw = self._rbf_kernel(X_norm, sigma=self.sigma)

        self.K_train_raw_ = K_raw
        self.K_train_ = self._center_kernel(K_raw)

        # Kernel sparse LLE
        W = self._ks_lle(self.K_train_, lam=self.lam)
        self.W_lle_ = W

        B = self._solve_glpqr(
            K=self.K_train_,
            Y=Y_norm,
            W=W,
            alpha=self.alpha,
            beta=self.beta,
            n_components=self.n_components,
            q=q,
        )

        self.B_ = B
        r = B.shape[1]
        self.n_components_ = r

        V = self.K_train_ @ self.B_
        self.V_train_ = V

        # regression from latent embedding to quality variables
        VTV = V.T @ V + self.eps * np.eye(r)
        Q = np.linalg.pinv(VTV) @ V.T @ Y_norm
        self.Q_ = Q

        QQT = Q @ Q.T

        U_q, _, _ = np.linalg.svd(QQT)

        q_rel = min(q, r)
        self.q_rel_ = q_rel

        self.W_Q_ = U_q[:, :q_rel]

        if r > q_rel:
            self.W_U_ = U_q[:, q_rel:]
            self.has_quality_independent_ = True
        else:
            self.W_U_ = None
            self.has_quality_independent_ = False

        Ty = V @ self.W_Q_
        self.Sigma_y_ = self._cov2d(Ty)

        inv_Sigma_y = np.linalg.pinv(
            self.Sigma_y_ + self.eps * np.eye(self.Sigma_y_.shape[0])
        )

        self.Dy_train_ = self._row_quadratic_form(Ty, inv_Sigma_y)

        if self.has_quality_independent_:
            T0 = V @ self.W_U_
            self.Sigma_0_ = self._cov2d(T0)

            inv_Sigma_0 = np.linalg.pinv(
                self.Sigma_0_ + self.eps * np.eye(self.Sigma_0_.shape[0])
            )

            self.D0_train_ = self._row_quadratic_form(T0, inv_Sigma_0)
        else:
            self.Sigma_0_ = None
            self.D0_train_ = np.zeros(n)

        return self

    def transform(self, X):
        X_norm = self.scaler_x_.transform(X)

        K_new = self._rbf_kernel(
            X_norm,
            self.training_X_,
            sigma=self.sigma,
        )

        K_new_c = self._center_new_kernel(
            K_new,
            self.K_train_raw_,
        )

        V = K_new_c @ self.B_

        return V

    def get_monitoring_statistics(self, X):
        V = self.transform(X)

        Ty = V @ self.W_Q_

        inv_Sigma_y = np.linalg.pinv(
            self.Sigma_y_ + self.eps * np.eye(self.Sigma_y_.shape[0])
        )

        Dy = self._row_quadratic_form(Ty, inv_Sigma_y)

        if self.has_quality_independent_:
            T0 = V @ self.W_U_

            inv_Sigma_0 = np.linalg.pinv(
                self.Sigma_0_ + self.eps * np.eye(self.Sigma_0_.shape[0])
            )

            D0 = self._row_quadratic_form(T0, inv_Sigma_0)
        else:
            D0 = np.zeros(V.shape[0])

        return Dy, D0

    @staticmethod
    def _rbf_kernel(X, Y=None, sigma=2500):
        if Y is None:
            Y = X

        sq_dists = cdist(X, Y, metric="sqeuclidean")
        K = np.exp(-sq_dists / sigma)

        return K

    @staticmethod
    def _center_kernel(K):
        n = K.shape[0]
        one = np.ones((n, n)) / n

        Kc = K - one @ K - K @ one + one @ K @ one

        return (Kc + Kc.T) / 2.0

    @staticmethod
    def _center_new_kernel(K_new, K_train_raw):
        train_col_mean = np.mean(K_train_raw, axis=0, keepdims=True)
        new_row_mean = np.mean(K_new, axis=1, keepdims=True)
        train_total_mean = np.mean(K_train_raw)

        K_new_c = K_new - train_col_mean - new_row_mean + train_total_mean

        return K_new_c

    def _ks_lle(self, K, lam=0.14):
        n = K.shape[0]

        K_reg = K + self.eps * np.eye(n)

        try:
            L = np.linalg.cholesky(K_reg)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(n))

        X_lasso = np.asfortranarray(L.T)

        # Solve L @ Y_lasso = K once.
        Y_lasso = solve_triangular(
            L,
            K,
            lower=True,
            check_finite=False,
        )

        lasso = Lasso(
            alpha=lam / n,
            fit_intercept=False,
            max_iter=self.max_iter_lasso,
            tol=1e-4,
            selection="cyclic",
            precompute=True,
            copy_X=False,
            random_state=self.random_state,
        )

        lasso.fit(X_lasso, Y_lasso)

        W = lasso.coef_.copy()

        if self.zero_diag_lle:
            np.fill_diagonal(W, 0.0)

        return W
        
    def _solve_glpqr(self, K, Y, W, alpha, beta, n_components, q):
        n = K.shape[0]

        I = np.eye(n)
        M = (I - W).T @ (I - W)

        YYT = Y @ Y.T

        A1 = K @ YYT @ K
        A2 = K @ M @ K
        A3 = K @ K

        LHS = -A1 + beta * alpha * A2 - beta * (1.0 - alpha) / n * A3
        LHS = (LHS + LHS.T) / 2.0

        RHS = K + self.eps * np.eye(n)
        RHS = (RHS + RHS.T) / 2.0

        if n_components is None:
            r = q
        else:
            r = int(n_components)

        r = max(1, min(r, n))

        try:
            eig_vals, eig_vecs = eigh(
                LHS,
                RHS,
                subset_by_index=[0, r - 1],
            )
        except Exception:
            RHS = K + 1e-6 * np.eye(n)
            RHS = (RHS + RHS.T) / 2.0

            eig_vals, eig_vecs = eigh(
                LHS,
                RHS,
                subset_by_index=[0, r - 1],
            )

        self.eig_values_ = eig_vals

        B = eig_vecs

        return B

    def _cov2d(self, A):
        A = np.asarray(A, dtype=float)

        if A.ndim == 1:
            A = A.reshape(-1, 1)

        if A.shape[0] <= 1:
            return self.eps * np.eye(A.shape[1])

        C = np.cov(A, rowvar=False)
        C = np.atleast_2d(C)

        return (C + C.T) / 2.0

    @staticmethod
    def _row_quadratic_form(A, M):
        A = np.asarray(A, dtype=float)

        if A.shape[0] == 0:
            return np.asarray([])

        return np.einsum("ij,jk,ik->i", A, M, A)
