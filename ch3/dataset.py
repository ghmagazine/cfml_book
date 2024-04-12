import numpy as np
from sklearn.utils import check_random_state

from utils import sample_action_fast, softmax, eps_greedy_policy


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
    num_def_actions: int = 0,
    num_clusters: int = 3,
    beta: float = -3.0,
    random_state: int = 12345,
) -> dict:
    """行動特徴量を活用したオフ方策評価におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    x = random_.normal(size=(num_data, dim_context))
    one_hot_a, one_hot_c = np.eye(num_actions), np.eye(num_clusters)

    # 期待報酬関数を定義する
    g_x_c = (
        (x - x ** 2) @ theta_g + (x ** 3 + x ** 2 - x) @ M_g @ one_hot_c + b_g
    ) / 10
    h_x_a = (
        (x ** 3 + x ** 2 - x) @ theta_h + (x - x ** 2) @ M_h @ one_hot_a + b_h
    ) / 10
    q_x_a = (1 - lambda_) * g_x_c[:, phi_a] + lambda_ * h_x_a

    # データ収集方策を定義する
    pi_0 = softmax(beta * q_x_a)
    pi_0[:, :num_def_actions] = 0
    pi_0 = pi_0 / pi_0.sum(1)[:, np.newaxis]

    # 行動や報酬を抽出する
    a = sample_action_fast(pi_0, random_state=random_state)
    q_x_a_factual = q_x_a[np.arange(num_data), a]
    r = random_.normal(q_x_a_factual)

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
        g_x_c=(1 - lambda_) * g_x_c,
        h_x_a=lambda_ * h_x_a,
        q_x_a=q_x_a,
    )


def calc_true_value(
    dim_context: int,
    num_actions: int,
    num_clusters: int,
    theta_g: np.ndarray,
    M_g: np.ndarray,
    b_g: np.ndarray,
    phi_a: np.ndarray,
    theta_h: np.ndarray,
    M_h: np.ndarray,
    b_h: np.ndarray,
    lambda_: float,
) -> float:
    """評価方策の真の性能を近似する."""
    test_bandit_data = generate_synthetic_data(
        num_data=10000,
        dim_context=dim_context,
        num_actions=num_actions,
        num_clusters=num_clusters,
        theta_g=theta_g,
        M_g=M_g,
        b_g=b_g,
        theta_h=theta_h,
        M_h=M_h,
        b_h=b_h,
        lambda_=lambda_,
        phi_a=phi_a,
    )

    q_x_a = test_bandit_data["q_x_a"]
    pi = eps_greedy_policy(q_x_a)

    return (q_x_a * pi).sum(1).mean()
