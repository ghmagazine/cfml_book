import numpy as np
from sklearn.utils import check_random_state

from utils import sample_action_fast, sigmoid, softmax, eps_greedy_policy


def generate_synthetic_data(
    num_data: int,
    dim_context: int,
    num_actions: int,
    K: int,
    theta: np.ndarray,  # d x |A|
    M: np.ndarray,  # d x |A|
    b: np.ndarray,  # |A| x 1
    W: np.ndarray,  # K x K
    beta: float = -1.0,
    reward_noise: float = 0.5,
    p: list = [1.0, 0.0, 0.0],  # independent, cascade, all
    p_rand: float = 0.0,
    is_test: bool = False,
    random_state: int = 12345,
) -> dict:
    """ランキングにおけるオフ方策評価におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    x, e_a = random_.normal(size=(num_data, dim_context)), np.eye(num_actions)
    base_q_func = sigmoid((x ** 3 + x ** 2 - x) @ theta + (x - x ** 2) @ M @ e_a + b)

    # ユーザ行動モデルを抽出する
    user_behavior_matrix = np.r_[
        np.eye(K),  # independent
        np.tril(np.ones((K, K))),  # cascade
        np.ones((K, K)),  # all
    ].reshape((3, K, K))
    user_behavior_idx = random_.choice(3, p=p, size=num_data)
    C_ = user_behavior_matrix[user_behavior_idx]

    user_behavior_matrix_rand = random_.choice(
        [-1, 0, 1], p=[0.2, 0.4, 0.4], size=7 * K * K
    ).reshape((7, K, K))
    user_behavior_rand_idx = random_.choice(7, size=num_data)
    C_rand = user_behavior_matrix_rand[user_behavior_rand_idx]

    is_rand = random_.binomial(2, p=p_rand, size=num_data).reshape(num_data, 1, 1)
    C = np.clip(C_ + is_rand * C_rand, 0, 1)

    if is_test:
        pi_0 = eps_greedy_policy(base_q_func)
    else:
        pi_0 = softmax(beta * base_q_func)
    # 行動を抽出する
    a_k = np.zeros((num_data, K), dtype=int)
    r_k = np.zeros((num_data, K), dtype=float)
    q_k = np.zeros((num_data, K), dtype=float)
    for k in range(K):
        a_k_ = sample_action_fast(pi_0, random_state=random_state + k)
        a_k[:, k] = a_k_
    # 報酬を抽出する
    idx = np.arange(num_data)
    for k in range(K):
        q_func_factual = base_q_func[idx, a_k[:, k]] / K
        for l in range(K):
            if l != k:
                q_func_factual += (
                    C[:, k, l] * W[k, l] * base_q_func[idx, a_k[:, l]] / np.abs(l - k)
                )
        q_k[:, k] = q_func_factual
        r_k[:, k] = random_.normal(q_func_factual, scale=reward_noise)

    return dict(
        num_data=num_data,
        K=K,
        num_actions=num_actions,
        x=x,
        a_k=a_k,
        r_k=r_k,
        C=C,
        pi_0=pi_0,
        q_k=q_k,
        base_q_func=base_q_func,
    )


def calc_true_value(
    dim_context: int,
    num_actions: int,
    K: int,
    p: list,
    theta: np.ndarray,
    M: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    p_rand: float = 0.0,
    num_data: int = 100000,
) -> float:
    """評価方策の真の性能を近似する."""
    test_bandit_data = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        K=K,
        theta=theta,
        M=M,
        b=b,
        W=W,
        is_test=True,
        p=p,
        p_rand=p_rand,
        random_state=12345,
    )

    return test_bandit_data["q_k"].sum(1).mean()
