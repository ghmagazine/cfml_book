from copy import copy

import numpy as np
from sklearn.linear_model import LogisticRegression


def calc_avg(dataset: dict) -> np.ndarray:
    """AVG推定量を実行する."""
    return dataset["r"].mean()


def calc_dm(dataset: dict, pi: np.ndarray, q_hat: np.ndarray) -> float:
    """DM推定量を実行する."""
    return (q_hat * pi).sum(1).mean()


def calc_ips(dataset: dict, pi: np.ndarray, max_value: float = 100) -> float:
    """IPS推定量を実行する."""
    num_data = dataset["num_data"]
    a, r, pi_0 = dataset["a"], dataset["r"], dataset["pi_0"]

    idx = np.arange(num_data)
    w = pi[idx, a] / pi_0[idx, a]  # importance weights

    return np.minimum((w * r).mean(), max_value)


def calc_dr(
    dataset: dict, pi: np.ndarray, q_hat: np.ndarray, max_value: float = 100
) -> float:
    """DR推定量を実行する."""
    num_data = dataset["num_data"]
    a, r, pi_0 = dataset["a"], dataset["r"], dataset["pi_0"]

    idx = np.arange(num_data)
    w = pi[idx, a] / pi_0[idx, a]  # importance weights

    dr = (q_hat * pi).sum(1)  # direct method
    dr += w * (r - q_hat[idx, a])  # correction term

    return np.minimum(dr.mean(), max_value)


def calc_mips(
    dataset: dict,
    pi: np.ndarray,
    replace_c: int = 0,
    is_estimate_w: bool = False,
) -> float:
    """MIPS推定量を実行する."""
    num_data = dataset["num_data"]
    num_actions, num_clusters = dataset["num_actions"], dataset["num_clusters"]
    x, a, c, r = dataset["x"], dataset["a"], copy(dataset["c"]), dataset["r"]
    pi_0, phi_a = dataset["pi_0"], copy(dataset["phi_a"])
    min_value, max_value = r.min(), r.max()

    if replace_c > 0:
        c[c >= num_clusters - replace_c] = num_clusters - replace_c - 1
        phi_a[phi_a >= num_clusters - replace_c] = num_clusters - replace_c - 1

    if is_estimate_w:
        x_c = np.c_[x, np.eye(num_clusters)[c]]
        pi_a_x_c_model = LogisticRegression(C=5, random_state=12345)
        pi_a_x_c_model.fit(x_c, a)

        w_x_a_full = pi / pi_0
        pi_a_x_c_hat = np.zeros((num_data, num_actions))
        pi_a_x_c_hat[:, np.unique(a)] = pi_a_x_c_model.predict_proba(x_c)
        w_x_c_hat = (pi_a_x_c_hat * w_x_a_full).sum(1)

        return np.clip((w_x_c_hat * r).mean(), min_value, max_value)

    else:
        pi_0_c = np.zeros((num_data, num_clusters - replace_c))
        pi_c = np.zeros((num_data, num_clusters - replace_c))
        for c_ in range(num_clusters - replace_c):
            pi_0_c[:, c_] = pi_0[:, phi_a == c_].sum(1)
            pi_c[:, c_] = pi[:, phi_a == c_].sum(1)

        # 周辺重要度重み
        w_x_c = pi_c[np.arange(num_data), c] / pi_0_c[np.arange(num_data), c]

        return np.clip((w_x_c * r).mean(), min_value, max_value)


def calc_offcem(
    dataset: dict,
    pi: np.ndarray,
    q_hat: np.ndarray,
    replace_c: int = 0,
    max_value: float = 100,
) -> float:
    """OffCEM推定量を実行する."""
    num_data = dataset["num_data"]
    num_clusters = dataset["num_clusters"]
    a, c, r = dataset["a"], copy(dataset["c"]), dataset["r"]
    pi_0, phi_a = dataset["pi_0"], copy(dataset["phi_a"])

    if replace_c > 0:
        c[c >= num_clusters - replace_c] = num_clusters - replace_c - 1
        phi_a[phi_a >= num_clusters - replace_c] = num_clusters - replace_c - 1

    idx = np.arange(num_data)
    pi_0_c = np.zeros((num_data, num_clusters - replace_c))
    pi_c = np.zeros((num_data, num_clusters - replace_c))
    for c_ in range(num_clusters - replace_c):
        pi_0_c[:, c_] = pi_0[:, phi_a == c_].sum(1)
        pi_c[:, c_] = pi[:, phi_a == c_].sum(1)

    # 周辺重要度重み
    w_x_c = pi_c[np.arange(num_data), c] / pi_0_c[np.arange(num_data), c]

    return np.minimum(
        ((q_hat * pi).sum(1) + w_x_c * (r - q_hat[idx, a])).mean(), max_value
    )
