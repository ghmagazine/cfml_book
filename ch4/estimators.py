import numpy as np


def calc_avg(dataset: dict) -> float:
    """AVG推定量を実行する."""
    return dataset["r_h"].sum(1).mean()


def calc_tis(dataset: dict) -> float:
    """TIS推定量を実行する."""
    num_data, H = dataset["num_data"], dataset["H"]
    a_h, r_h = dataset["a_h"], dataset["r_h"]
    pi, pi_0 = dataset["pi"], dataset["pi_0"]

    target_pscore = np.zeros((num_data, H), dtype=float)
    logging_pscore = np.zeros((num_data, H), dtype=float)
    for h in range(H):
        target_pscore[:, h] = pi[np.arange(num_data), a_h[:, h], h]
        logging_pscore[:, h] = pi_0[np.arange(num_data), a_h[:, h], h]

    w_h = target_pscore / logging_pscore

    return (w_h.prod(1) * r_h.sum(1)).mean()


def calc_sis(dataset: dict) -> float:
    """SIS推定量を実行する."""
    num_data, H = dataset["num_data"], dataset["H"]
    a_h, r_h = dataset["a_h"], dataset["r_h"]
    pi, pi_0 = dataset["pi"], dataset["pi_0"]

    target_pscore = np.zeros((num_data, H), dtype=float)
    logging_pscore = np.zeros((num_data, H), dtype=float)
    for h in range(H):
        target_pscore[:, h] = pi[np.arange(num_data), a_h[:, h], h]
        logging_pscore[:, h] = pi_0[np.arange(num_data), a_h[:, h], h]

    w_h = target_pscore / logging_pscore

    sis_estimate = 0.0
    for h in range(H):
        sis_estimate += w_h[:, : h + 1].prod(1) * r_h[:, h]

    return sis_estimate.mean()


def calc_dr(dataset: dict, Q_hat: np.ndarray) -> float:
    """DR推定量を実行する."""
    num_data, H = dataset["num_data"], dataset["H"]
    s_h, a_h, r_h = dataset["s_h"], dataset["a_h"], dataset["r_h"]
    pi, pi_0 = dataset["pi"], dataset["pi_0"]

    target_pscore = np.zeros((num_data, H), dtype=float)
    logging_pscore = np.zeros((num_data, H), dtype=float)
    for h in range(H):
        target_pscore[:, h] = pi[np.arange(num_data), a_h[:, h], h]
        logging_pscore[:, h] = pi_0[np.arange(num_data), a_h[:, h], h]

    w_h = target_pscore / logging_pscore

    dr_estimate = 0
    for h in range(H):
        if h == H - 1:
            dr_estimate += w_h.prod(1) * (
                r_h[:, h] - Q_hat[s_h[:, h], a_h[:, h]]
            ) + w_h[:, :h].prod(1) * (pi[:, :, 0] * Q_hat[s_h[:, 0], :]).sum(1)
        elif h == 0:
            dr_estimate += w_h[:, 0] * (r_h[:, h] - Q_hat[s_h[:, h], a_h[:, h]]) + (
                pi[:, :, 0] * Q_hat[s_h[:, 0], :]
            ).sum(1)
        else:
            dr_estimate += w_h[:, : h + 1].prod(1) * (
                r_h[:, h] - Q_hat[s_h[:, h], a_h[:, h]]
            ) + w_h[:, :h].prod(1) * (pi[:, :, h + 1] * Q_hat[s_h[:, h + 1], :]).sum(1)

    return dr_estimate.mean()


def calc_mis(dataset: dict, dataset_test: dict) -> float:
    """MIS推定量を実行する."""
    num_states, num_actions = dataset["num_states"], dataset["num_actions"]
    s_h_0, a_h_0, r_h_0 = dataset["s_h"], dataset["a_h"], dataset["r_h"]
    s_h, a_h = dataset_test["s_h"], dataset_test["a_h"]

    p_s_a = np.zeros((num_states, num_actions))
    p_s_a_0 = np.zeros((num_states, num_actions))
    for s in range(num_states):
        for a in range(num_actions):
            p_s_a[s, a] = ((s_h == s) * (a_h == a)).mean()
            p_s_a_0[s, a] = ((s_h_0 == s) * (a_h_0 == a)).mean()

    # 周辺重要度重みの推定
    mis_estimate = 0.0
    for h in range(dataset["H"]):
        w_h = p_s_a[s_h_0[:, h], a_h_0[:, h]] / p_s_a_0[s_h_0[:, h], a_h_0[:, h]]
        mis_estimate += w_h * r_h_0[:, h]

    return mis_estimate.mean()
