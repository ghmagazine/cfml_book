import numpy as np
from sklearn.utils import check_random_state

from utils import sample_action_fast, sigmoid, logging_policy


def generate_synthetic_data(
    num_data: int,
    theta_g: np.ndarray,
    M_g: np.ndarray,
    b_g: np.ndarray,
    theta_h: np.ndarray,
    M_h: np.ndarray,
    b_h: np.ndarray,
    phi_a: np.ndarray,
    lambda_: float = 0.5,
    dim_context: int = 5,
    num_actions: int = 50,
    num_clusters: int = 3,
    beta: float = 1.0,
    lam: float = 0.5,
    sigma: float = 1.0,
    random_state: int = 12345,
) -> dict:
    """オフ方策学習におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    x = random_.normal(size=(num_data, dim_context))
    one_hot_a, one_hot_c = np.eye(num_actions), np.eye(num_clusters)

    # 期待報酬関数を定義する
    g_x_c = sigmoid(
        (x - x ** 2) @ theta_g + (x ** 3 + x ** 2 - x) @ M_g @ one_hot_c + b_g
    )
    h_x_a = sigmoid(
        (x ** 3 + x ** 2 - x) @ theta_h + (x - x ** 2) @ M_h @ one_hot_a + b_h
    )
    q_x_a = (1 - lambda_) * g_x_c[:, phi_a] + lambda_ * h_x_a

    # データ収集方策を定義する
    pi_0 = logging_policy(q_x_a, beta=beta, sigma=sigma, lam=lam)
    idx = np.arange(num_data)
    pi_0_c = np.zeros((num_data, num_clusters))
    for c_ in range(num_clusters):
        pi_0_c[:, c_] = pi_0[:, phi_a == c_].sum(1)

    # 行動や報酬を抽出する
    a = sample_action_fast(pi_0, random_state=random_state)
    q_x_a_factual = q_x_a[idx, a]
    r = random_.binomial(n=1, p=q_x_a_factual)

    return dict(
        num_data=num_data,
        num_actions=num_actions,
        num_clusters=num_clusters,
        x=x,
        a=a,
        c=phi_a[a],
        r=r,
        phi_a=phi_a,
        pi_0=pi_0,
        pi_0_c=pi_0_c,
        pscore=pi_0[idx, a],
        pscore_c=pi_0_c[idx, phi_a[a]],
        g_x_c=(1 - lambda_) * g_x_c,
        h_x_a=lambda_ * h_x_a,
        q_x_a=q_x_a,
    )
