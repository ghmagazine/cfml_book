import numpy as np


def calc_avg(dataset: dict, pi: np.ndarray) -> float:
    """AVG推定量を実行する."""
    return dataset["r_k"].sum(1).mean()


def calc_ips(dataset: dict, pi: np.ndarray) -> float:
    """ランキングにおけるIPS推定量を実行する."""
    num_data, K = dataset["num_data"], dataset["K"]
    a_k, r_k, pi_0 = dataset["a_k"], dataset["r_k"], dataset["pi_0"]

    target_pscore = np.zeros((num_data, K), dtype=float)
    logging_pscore = np.zeros((num_data, K), dtype=float)
    for k in range(K):
        target_pscore[:, k] = pi[np.arange(num_data), a_k[:, k]]
        logging_pscore[:, k] = pi_0[np.arange(num_data), a_k[:, k]]

    # ポジションレベルの重要度重み
    w_k = target_pscore / logging_pscore

    # ランキングレベルの重要度重み
    rank_weight = w_k.prod(1)

    return (rank_weight * r_k.sum(1)).mean()


def calc_iips(dataset: dict, pi: np.ndarray) -> float:
    """ランキングにおけるIIPS推定量を実行する."""
    num_data, K = dataset["num_data"], dataset["K"]
    a_k, r_k, pi_0 = dataset["a_k"], dataset["r_k"], dataset["pi_0"]

    target_pscore = np.zeros((num_data, K), dtype=float)
    logging_pscore = np.zeros((num_data, K), dtype=float)
    for k in range(K):
        target_pscore[:, k] = pi[np.arange(num_data), a_k[:, k]]
        logging_pscore[:, k] = pi_0[np.arange(num_data), a_k[:, k]]

    # ポジションレベルの重要度重み
    w_k = target_pscore / logging_pscore

    return (w_k * r_k).sum(1).mean()


def calc_rips(dataset: dict, pi: np.ndarray) -> float:
    """ランキングにおけるRIPS推定量を実行する."""
    num_data, K = dataset["num_data"], dataset["K"]
    a_k, r_k, pi_0 = dataset["a_k"], dataset["r_k"], dataset["pi_0"]

    target_pscore = np.zeros((num_data, K), dtype=float)
    logging_pscore = np.zeros((num_data, K), dtype=float)
    for k in range(K):
        target_pscore[:, k] = pi[np.arange(num_data), a_k[:, k]]
        logging_pscore[:, k] = pi_0[np.arange(num_data), a_k[:, k]]

    rips_estimate = 0.0
    w_k = target_pscore / logging_pscore
    for k in range(K):
        # トップkにおける重要度重み
        top_k_weight = w_k[:, : k + 1].prod(1)
        rips_estimate += top_k_weight * r_k[:, k]

    return rips_estimate.mean()


def calc_aips(dataset: dict, pi: np.ndarray, max_k: int = 20) -> float:
    """ランキングにおけるAIPS推定量を実行する."""
    num_data, K, C = dataset["num_data"], dataset["K"], dataset["C"]
    a_k, r_k, pi_0 = dataset["a_k"], dataset["r_k"], dataset["pi_0"]

    target_pscore = np.zeros((num_data, K), dtype=float)
    logging_pscore = np.zeros((num_data, K), dtype=float)
    for k in range(K):
        target_pscore[:, k] = pi[np.arange(num_data), a_k[:, k]]
        logging_pscore[:, k] = pi_0[np.arange(num_data), a_k[:, k]]

    aips_estimate = 0.0
    w_k = target_pscore / logging_pscore
    for k in range(K):
        w_k_ = w_k * C[:, k, :]
        adaptive_weight = np.where(w_k_ == 0, 1.0, w_k_)
        adaptive_weight[:, max_k:] = 1.0
        aips_estimate += adaptive_weight.prod(1) * r_k[:, k]

    return aips_estimate.mean()


def calc_weights(dataset: dict, pi: np.ndarray) -> np.ndarray:
    """ランキングにおける各種重要度重みを計算する."""
    num_data, K = dataset["num_data"], dataset["K"]
    a_k, pi_0 = dataset["a_k"], dataset["pi_0"]

    target_pscore = np.zeros((num_data, K), dtype=float)
    logging_pscore = np.zeros((num_data, K), dtype=float)
    for k in range(K):
        target_pscore[:, k] = pi[np.arange(num_data), a_k[:, k]]
        logging_pscore[:, k] = pi_0[np.arange(num_data), a_k[:, k]]

    # ポジションレベルの重要度重み
    pos_weight = target_pscore / logging_pscore
    pos_weight_max = pos_weight.max(0).mean()

    # ランキングレベルの重要度重み
    rank_weight = pos_weight.prod(1)
    rank_weight_max = rank_weight.max()

    # トップkにおける重要度重み
    topk_weight = np.zeros((num_data, K))
    for k in range(K):
        topk_weight[:, k] = pos_weight[:, : k + 1].prod(1)
    topk_weight_max = topk_weight.max(0).mean()

    return rank_weight_max, pos_weight_max, topk_weight_max
