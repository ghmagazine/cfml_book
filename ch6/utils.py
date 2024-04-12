from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.utils import check_random_state
from scipy.stats import rankdata
import torch


def sample_action_fast(pi: np.ndarray, random_state: int = 12345) -> np.ndarray:
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=pi.shape[0])[:, np.newaxis]
    cum_pi = pi.cumsum(axis=1)
    flg = cum_pi > uniform_rvs
    sampled_actions = flg.argmax(axis=1)
    return sampled_actions


def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.exp(np.minimum(x, 0)) / (1.0 + np.exp(-np.abs(x)))


def softmax(x: np.ndarray) -> np.ndarray:
    b = np.max(x, axis=1)[:, np.newaxis]
    numerator = np.exp(x - b)
    denominator = np.sum(numerator, axis=1)[:, np.newaxis]
    return numerator / denominator


def eps_greedy_policy(
    q_func: np.ndarray,
    eps: float = 0.2,
) -> np.ndarray:
    "Generate an evaluation policy via the epsilon-greedy rule."
    is_topk = rankdata(-q_func, method="ordinal", axis=1) == 3
    pi = (1.0 - eps) * is_topk + eps / q_func.shape[1]

    return pi / pi.sum(1)[:, np.newaxis]


def softmax_policy(
    q_func: np.ndarray,
    beta: float = 1.0,
) -> np.ndarray:
    "Generate an evaluation policy via the softmax rule."

    return softmax(beta * q_func)


def aggregate_simulation_results(
    estimated_policy_value_list: list,
    selection_result_list: list,
    policy_value_of_pi: float,
    experiment_config_name: str,
    experiment_config_value: int,
) -> DataFrame:
    """各推定量の推定値から平均二乗誤差や二乗バイアス、バリアンスなどの実験結果を集計する."""
    estimation_result_df = (
        DataFrame(estimated_policy_value_list)
        .stack()
        .reset_index(1)
        .rename(columns={"level_1": "est", 0: "value"})
    )
    selection_result_df = (
        DataFrame(selection_result_list)
        .stack()
        .reset_index(1)
        .rename(columns={"level_1": "est2", 0: "selection"})
    )
    result_df = pd.concat([estimation_result_df, selection_result_df], axis=1)
    result_df[experiment_config_name] = experiment_config_value
    result_df["se"] = (result_df.value - policy_value_of_pi) ** 2
    result_df["bias"] = 0
    result_df["variance"] = 0
    result_df["true_value"] = policy_value_of_pi
    sample_mean = DataFrame(result_df.groupby(["est"]).mean().value).reset_index()
    for est_ in sample_mean["est"]:
        estimates = result_df.loc[result_df["est"] == est_, "value"].values
        mean_estimates = sample_mean.loc[sample_mean["est"] == est_, "value"].values
        mean_estimates = np.ones_like(estimates) * mean_estimates
        result_df.loc[result_df["est"] == est_, "bias"] = (
            policy_value_of_pi - mean_estimates
        ) ** 2
        result_df.loc[result_df["est"] == est_, "variance"] = (
            estimates - mean_estimates
        ) ** 2

    return result_df


@dataclass
class RegBasedPolicyDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert self.context.shape[0] == self.action.shape[0] == self.reward.shape[0]

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
        )

    def __len__(self):
        return self.context.shape[0]


@dataclass
class GradientBasedPolicyDataset(torch.utils.data.Dataset):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    q_hat: np.ndarray
    pi_0: np.ndarray

    def __post_init__(self):
        """initialize class"""
        assert (
            self.context.shape[0]
            == self.action.shape[0]
            == self.reward.shape[0]
            == self.pscore.shape[0]
            == self.q_hat.shape[0]
            == self.pi_0.shape[0]
        )

    def __getitem__(self, index):
        return (
            self.context[index],
            self.action[index],
            self.reward[index],
            self.pscore[index],
            self.q_hat[index],
            self.pi_0[index],
        )

    def __len__(self):
        return self.context.shape[0]
