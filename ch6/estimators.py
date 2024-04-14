import numpy as np

from sklearn.linear_model import LogisticRegression


def calc_online(dataset_online: dict) -> float:
    """短期実験におけるAVG推定量を実行する."""
    r_on, w_on = dataset_online["r_t"], dataset_online["w"]

    estimated_value_of_pi = (w_on * r_on.mean(1)).sum() / w_on.sum()
    estimated_value_of_pi_0 = ((1 - w_on) * r_on.mean(1)).sum() / (1 - w_on).sum()
    selection_result = estimated_value_of_pi > estimated_value_of_pi_0

    return estimated_value_of_pi, selection_result


def calc_ips(dataset: dict, pi: np.ndarray) -> float:
    """ログデータにおけるIPS推定量を実行する."""
    num_data, T = dataset["num_data"], dataset["T"]
    a_t, r_t, pi_0 = dataset["a_t"], dataset["r_t"], dataset["pi_0"]

    target_pscore = np.zeros((num_data, T), dtype=float)
    logging_pscore = np.zeros((num_data, T), dtype=float)
    for t in range(T):
        target_pscore[:, t] = pi[np.arange(num_data), a_t[:, t]]
        logging_pscore[:, t] = pi_0[np.arange(num_data), a_t[:, t]]

    w_t = target_pscore / logging_pscore  # importance weights

    ips_estimate = 0.0
    for t in range(T):
        ips_estimate += w_t[:, : t + 1].prod(1) * r_t[:, t] / T

    estimated_value_of_pi = ips_estimate.mean()
    estimated_value_of_pi_0 = (r_t.mean(1)).mean()
    selection_result = estimated_value_of_pi > estimated_value_of_pi_0

    return estimated_value_of_pi, selection_result


def calc_new(dataset: dict, dataset_online: dict, pi: np.ndarray) -> float:
    """短期実験データとログデータを用いた新推定量を実行する.
    なおここではIPS推定量との公平な比較を行うため、回帰モデル\hat{f}の部分は省略している.
    """
    x, r_t = dataset["x"], dataset["r_t"]
    x_on, r_on, w_on = dataset_online["x"], dataset_online["r_t"], dataset_online["w"]
    lr = LogisticRegression(C=100, random_state=12345)
    x_r, x_r_on = np.c_[x, r_t[:, 0]], np.c_[x_on, r_on]
    lr.fit(x_r_on, w_on)
    p_x_r = lr.predict_proba(x_r)
    w_x_r_hat = p_x_r[:, 1] / p_x_r[:, 0]

    estimated_value_of_pi = (w_x_r_hat * r_t.mean(1)).mean()
    estimated_value_of_pi_0 = (r_t.mean(1)).mean()
    selection_result = estimated_value_of_pi > estimated_value_of_pi_0

    return estimated_value_of_pi, selection_result
