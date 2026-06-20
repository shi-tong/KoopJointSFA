try:
    from Methods.KPCA import KPCA
    from Methods.SFA import SFA
    from Methods.KSFA import KSFA
    from Methods.KoopJointSFA import KoopJointSFA
    from Methods.GLPQR import GLPQR
    from Methods.JTSVA import JTSVA
    from Methods.DiCCA import DiCCA
except Exception as e:
    print(e)
import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import cumulative_trapezoid
def kde_limit(stat, conf_level=0.99, grids=5000):
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

def KPCAMonitor(
    train,
    test,
    alpha=0.99,
    n_components=30,
    kernel="rbf",
    gamma=0.1,
    return_state = False,
    **kwargs
):
    kpca = KPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        **kwargs
    )

    kpca.fit(train)

    T2_train, SPE_train = kpca.get_monitoring_statistics(train)
    T2_test, SPE_test = kpca.get_monitoring_statistics(test)

    T2_limit = kde_limit(T2_train, alpha)
    SPE_limit = kde_limit(SPE_train, alpha)

    limit = np.array([
        T2_limit,
        SPE_limit,
    ])

    test_stat = np.vstack([
        T2_test,
        SPE_test,
    ])

    alarm = np.any(test_stat > limit[:, None], axis=0)

    if return_state:
        train_stat = np.vstack([T2_train, SPE_train])
        test_stat = np.vstack([T2_test, SPE_test])
        return limit, test_stat, alarm, (train_stat, test_stat) 
    else:
        return limit, test_stat, alarm, None

def SFAMonitor(train, test, alpha=0.99, n_components=50, return_state = False):
    sfa = SFA(n_components=n_components)
    sfa.fit(train)

    T2_train, S2_train = sfa.get_monitoring_statistics(train)
    T2_test, S2_test = sfa.get_monitoring_statistics(test)

    T2_limit = kde_limit(T2_train, alpha)
    S2_limit = kde_limit(S2_train, alpha)

    limit = np.array([
        T2_limit,
        S2_limit,
    ])

    test_stat = np.vstack([
        T2_test,
        S2_test,
    ])

    alarm = np.any(test_stat > limit[:, None], axis=0)

    if return_state:
        train_stat = np.vstack([T2_train, S2_train])
        test_stat = np.vstack([T2_test, S2_test])
        return limit, test_stat, alarm, (train_stat, test_stat)
    else:
        return limit, test_stat, alarm, None

def KSFAMonitor(
    train,
    test,
    alpha=0.99,
    n_components=10,
    kernel="rbf",
    gamma=None,
    n_landmarks=120,
    random_state=42,
    return_state=False,
    **kwargs,
):
    ksfa = KSFA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        n_landmarks=n_landmarks,
        random_state=random_state,
        **kwargs,
    )

    ksfa.fit(train)

    T2_train, S2_train = ksfa.get_monitoring_statistics(train)
    T2_test, S2_test = ksfa.get_monitoring_statistics(test)

    T2_limit = kde_limit(T2_train, alpha)
    S2_limit = kde_limit(S2_train, alpha)

    limit = np.array([
        T2_limit,
        S2_limit,
    ])

    test_stat = np.vstack([
        T2_test,
        S2_test,
    ])

    alarm = np.any(test_stat > limit[:, None], axis=0)

    if return_state:
        train_stat = np.vstack([
            T2_train,
            S2_train,
        ])
        # test_stat = (np.vstack([T2_test,S2_test]), np.vstack([T2_limit,S2_limit]))
        test_stat = np.vstack([T2_test,S2_test])

        return limit, test_stat, alarm, (train_stat, test_stat)

    return limit, test_stat, alarm, None

def KoopJointSFAMonitor(
    train,
    test,
    alpha=0.99,
    n_components=3,
    dictionary="mixed",
    n_centers=60,
    n_lags=1,
    lag_step=2,
    joint_alpha=0.1,
    center_method="kmeans",
    gamma=0.0001,
    random_state=42,
    use_combined_alarm=True,
    return_state=False
):
    kpsfa = KoopJointSFA(
        n_components=n_components,
        dictionary=dictionary,
        n_lags=n_lags,
        lag_step=lag_step,
        n_centers=n_centers,
        center_method=center_method,
        gamma=gamma,
        joint_alpha=joint_alpha,
        control_quantile=alpha,
        random_state=random_state,
    )

    kpsfa.fit(train)

    train_stats = kpsfa.monitor(train)
    test_stats = kpsfa.monitor(test)

    T2_train = train_stats["T2"]
    S2_train = train_stats["S2"]
    Qkoop_train = train_stats["Qkoop"]
    combined_train = train_stats["combined"]

    T2_test = test_stats["T2"]
    S2_test = test_stats["S2"]
    Qkoop_test = test_stats["Qkoop"]
    combined_test = test_stats["combined"]

    T2_limit = kpsfa.T2_limit_
    S2_limit = kpsfa.S2_limit_
    Qkoop_limit = kpsfa.Qkoop_limit_
    combined_limit = kpsfa.combined_limit_

    if use_combined_alarm:
        limit = np.array([combined_limit])
        test_stat = np.vstack([combined_test])
        train_stat = np.vstack([combined_train])
        alarm = combined_test > combined_limit
    else:
        limit = np.array([T2_limit, S2_limit, Qkoop_limit])
        test_stat = np.vstack([T2_test, S2_test, Qkoop_test])
        train_stat = np.vstack([T2_train, S2_train, Qkoop_train])
        alarm = np.any(test_stat > limit[:, None], axis=0)

    if return_state:
        return limit, test_stat, alarm, (train_stat, test_stat)

    return limit, test_stat, alarm, None

def GLPQRMonitor(train_X, train_Y, test_X, alpha=0.99, return_state = False, **kwargs):
    model = GLPQR(**kwargs)
    # 1. Fit model using all IDV(0) normal samples
    model.fit(train_X, train_Y)

    # 2. Use training statistics to determine control limits
    Dy_train = np.asarray(model.Dy_train_, dtype=float)
    D0_train = np.asarray(model.D0_train_, dtype=float)

    Dy_limit = kde_limit(Dy_train, alpha)
    D0_limit = kde_limit(D0_train, alpha)

    # 3. Compute test statistics
    Dy_test, D0_test = model.get_monitoring_statistics(test_X)

    Dy_test = np.asarray(Dy_test, dtype=float)
    D0_test = np.asarray(D0_test, dtype=float)

    limit = np.array([
        Dy_limit,
        D0_limit,
    ])

    train_stat = np.vstack([
        Dy_train,
        D0_train,
    ])

    test_stat = np.vstack([
        Dy_test,
        D0_test,
    ])
    alarm_y = Dy_test > Dy_limit
    alarm_0 = D0_test > D0_limit
    alarm = alarm_y | alarm_0

    if return_state:
        train_stat = np.vstack([Dy_train, D0_train])
        test_stat = np.vstack([Dy_test, D0_test])
        return limit, test_stat, alarm, (train_stat, test_stat)
    else:
        return limit, test_stat, alarm, None


def JTSVAMonitor(
    train_X,
    test_X,
    val_X=None,
    alpha=0.99,
    d=3,
    k=13,
    return_state=False,
    **kwargs
):
    model = JTSVA(d=d, k=k, **kwargs)
    model.fit(train_X)

    if val_X is None:
        val_X = train_X

    Td2_val, Ts2_val, Q_val = model.get_monitoring_statistics(val_X)
    Td2_lim = kde_limit(Td2_val, alpha)
    Ts2_lim = kde_limit(Ts2_val, alpha)
    Q_lim = kde_limit(Q_val, alpha)
    phi_val = model.compute_phi(
        Td2=Td2_val,
        Ts2=Ts2_val,
        Q=Q_val,
        Td2_lim=Td2_lim,
        Ts2_lim=Ts2_lim,
        Q_lim=Q_lim,
        eps=model.eps,
    )
    phi_lim = kde_limit(phi_val, alpha)
    Td2_test, Ts2_test, Q_test = model.get_monitoring_statistics(test_X)

    phi_test = model.compute_phi(
        Td2=Td2_test,
        Ts2=Ts2_test,
        Q=Q_test,
        Td2_lim=Td2_lim,
        Ts2_lim=Ts2_lim,
        Q_lim=Q_lim,
        eps=model.eps,
    )

    limit = np.array([phi_lim])
    test_stat = np.vstack([phi_test])

    alarm = phi_test > phi_lim

    if return_state:
        train_stat = np.vstack([phi_val])
        test_stat = np.vstack([phi_test])
        return limit, test_stat, alarm, (train_stat, test_stat)
    else:
        return limit, test_stat, alarm, None


def DiCCAMonitor(train_X, test_X, alpha=0.99, d=3, k=13, return_state = True, **kwargs):
    model = DiCCA(d=d, k=k, **kwargs)
    model.fit(train_X)

    Td2_train, Ts2_train, Q_train, phi_train = model.get_monitoring_statistics(train_X)
    Td2_test, Ts2_test, Q_test, phi_test = model.get_monitoring_statistics(test_X)

    # Td2_limit = np.nanpercentile(Td2_train, alpha * 100)
    # Ts2_limit = np.nanpercentile(Ts2_train, alpha * 100)
    # Q_limit = np.nanpercentile(Q_train, alpha * 100)

    phi_limit = kde_limit(phi_train, alpha)

    limit = np.array([phi_limit])
    test = np.vstack([phi_test])

    alarm = np.any(test > limit[:, None], axis=0)

    if return_state:
        train_stat = np.vstack([phi_train])
        test_stat = np.vstack([phi_test])
        return limit, test_stat, alarm, (train_stat, test_stat)
    else:
        return limit, test_stat, alarm, None