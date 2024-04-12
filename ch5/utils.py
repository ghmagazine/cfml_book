from dataclasses import dataclass

import numpy as np
from sklearn.utils import check_random_state
import torch


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


def logging_policy(
    q_func: np.ndarray,
    beta: float = 1.0,
    sigma: float = 1.0,
    lam: float = 0.5,
    random_state: int = 12345,
) -> np.ndarray:
    """ソフトマックス関数により方策を定義する."""
    random_ = check_random_state(random_state)
    noise = random_.normal(scale=sigma, size=q_func.shape)
    pi = softmax(beta * (lam * q_func + (1.0 - lam) * noise))

    return pi / pi.sum(1)[:, np.newaxis]


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
