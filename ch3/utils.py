from copy import deepcopy

import numpy as np
from pandas import DataFrame
from sklearn.utils import check_random_state
from scipy.stats import rankdata


def sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    """与えられた方策に従い、行動を高速に抽出する."""
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs
    sampled_actions = flg.argmax(axis=1)
    return sampled_actions


def sigmoid(x: np.ndarray) -> np.ndarray:
    """シグモイド関数."""
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))


def softmax(x: np.ndarray) -> np.ndarray:
    """ソフトマックス関数."""
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


def calc_weights(dataset: dict, pi: np.ndarray) -> np.ndarray:
    """重要度重みと周辺重要度重みを計算する."""
    num_data, pi_0 = dataset["num_data"], dataset["pi_0"]
    num_clusters, idx = dataset["num_clusters"], np.arange(num_data)
    w_x_a = pi[idx, dataset["a"]] / pi_0[idx, dataset["a"]]

    pi_0_c = np.zeros((num_data, num_clusters))
    pi_c = np.zeros((num_data, num_clusters))
    for c_ in range(num_clusters):
        pi_0_c[:, c_] = pi_0[:, dataset["phi_a"] == c_].sum(1)
        pi_c[:, c_] = pi[:, dataset["phi_a"] == c_].sum(1)
    w_x_c = pi_c[idx, dataset["c"]] / pi_0_c[idx, dataset["c"]]

    return w_x_a, w_x_c


def eps_greedy_policy(
    q_func: np.ndarray,
    k: int = 5,
    eps: float = 0.1,
) -> np.ndarray:
    """epsilon-greedy法により方策を定義する."""
    is_topk = rankdata(-q_func, method="ordinal", axis=1) <= k
    pi = ((1.0 - eps) / k) * is_topk + eps / q_func.shape[1]

    return pi / pi.sum(1)[:, np.newaxis]


def aggregate_simulation_results(
    estimated_policy_value_list: list,
    policy_value: float,
    experiment_config_name: str,
    experiment_config_value: int,
) -> DataFrame:
    """各推定量の推定値から平均二乗誤差や二乗バイアス、バリアンスなどの実験結果を集計する."""
    result_df = (
        DataFrame(DataFrame(estimated_policy_value_list).stack())
        .reset_index(1)
        .rename(columns={"level_1": "est", 0: "value"})
    )
    result_df[experiment_config_name] = experiment_config_value
    result_df["se"] = (result_df.value - policy_value) ** 2
    result_df["bias"] = 0
    result_df["variance"] = 0
    result_df["true_value"] = policy_value
    sample_mean = DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
    for est_ in sample_mean["est"]:
        estimates = result_df.loc[result_df["est"] == est_, "value"].values
        mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
        mean_estimates = np.ones_like(estimates) * mean_estimates
        result_df.loc[result_df["est"] == est_, "bias"] = (
            policy_value - mean_estimates
        ) ** 2
        result_df.loc[result_df["est"] == est_, "variance"] = (
            estimates - mean_estimates
        ) ** 2

    return result_df


def remove_outliers(result_df: DataFrame, estimators: list = None) -> DataFrame:
    """シミュレーション結果における外れ値に対処する."""
    result_df_ = deepcopy(result_df)
    for metric in ["bias", "variance"]:
        if estimators is None:
            threshold = np.percentile(result_df_.loc[:, metric], 99)
            result_df_.loc[result_df_[metric] > threshold, metric] = threshold
        else:
            for est in estimators:
                threshold = np.percentile(
                    result_df_.loc[result_df_.est == est, metric], 99
                )
                result_df_.loc[
                    (result_df_[metric] > threshold) & (result_df_.est == est), metric
                ] = threshold
    result_df_.loc[:, "se"] = result_df_.loc[:, "bias"] + result_df_.loc[:, "variance"]

    return result_df_
