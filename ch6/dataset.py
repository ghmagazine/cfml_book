import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import rankdata

from utils import sample_action_fast, sigmoid, softmax, eps_greedy_policy


def generate_synthetic_data(
    num_data: int,
    dim_context: int,
    num_actions: int,
    T: int,
    theta: np.ndarray,  # d x |A|
    M: np.ndarray,  # d x |A|
    b: np.ndarray,  # |A| x 1
    W: np.ndarray,  # T x T
    eps: float = 0.0,
    beta: float = 1.0,
    reward_noise: float = 0.5,
    p: list = [0.0, 1.0, 0.0],
    p_rand: float = 0.2,
    is_online: bool = False,
    random_state: int = 12345,
) -> dict:
    """オフ方策評価におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    x, e_a = random_.normal(size=(num_data, dim_context)), np.eye(num_actions)

    # 期待報酬関数を定義する
    base_q_func = (
        sigmoid((x ** 3 + x ** 2 - x) @ theta + (x - x ** 2) @ M @ e_a + b) / T
    )

    user_behavior_matrix = np.r_[
        np.eye(T),  # independent
        np.tril(np.ones((T, T))),  # cascade
        np.ones((T, T)),  # all
    ].reshape((3, T, T))
    user_behavior_idx = random_.choice(3, p=p, size=num_data)
    C_ = user_behavior_matrix[user_behavior_idx]

    user_behavior_matrix_rand = random_.choice(
        [-1, 0, 1], p=[0.2, 0.4, 0.4], size=7 * T * T
    ).reshape((7, T, T))
    user_behavior_rand_idx = random_.choice(7, size=num_data)
    C_rand = user_behavior_matrix_rand[user_behavior_rand_idx]

    is_rand = random_.binomial(2, p=p_rand, size=num_data).reshape(num_data, 1, 1)
    C = np.clip(C_ + is_rand * C_rand, 0, 1)

    if is_online:
        w = random_.binomial(1, p=0.5, size=num_data)[:, np.newaxis]
    else:
        w = random_.binomial(1, p=0.0, size=num_data)[:, np.newaxis]

    # データ収集方策を定義する
    pi_0 = w * softmax(beta * base_q_func)
    pi_0 += (1 - w) * eps_greedy_policy(base_q_func, eps=eps)

    # 行動や報酬を抽出する
    a_t = np.zeros((num_data, T), dtype=int)
    r_t = np.zeros((num_data, T), dtype=float)
    q_t = np.zeros((num_data, T), dtype=float)
    for t in range(T):
        a_t_ = sample_action_fast(pi_0, random_state=random_state + t)
        a_t[:, t] = a_t_
    idx = np.arange(num_data)
    for t in range(T):
        q_func_factual = base_q_func[idx, a_t[:, t]]
        for t_ in range(T):
            if t_ != t:
                q_func_factual += (
                    C[:, t, t_]
                    * W[t, t_]
                    * base_q_func[idx, a_t[:, t]]
                    / np.abs(t - t_)
                )
        q_t[:, t] = q_func_factual
        r_t[:, t] = random_.normal(q_func_factual, scale=reward_noise)

    return dict(
        num_data=num_data,
        T=T,
        num_actions=num_actions,
        x=x,
        w=w.flatten(),
        a_t=a_t,
        r_t=r_t,
        pi_0=pi_0,
        q_t=q_t,
        base_q_func=base_q_func,
    )


def calc_true_value(
    dim_context: int,
    num_actions: int,
    T: int,
    theta: np.ndarray,
    M: np.ndarray,
    b: np.ndarray,
    W: np.ndarray,
    eps: float = 0.0,
    beta: float = 1.0,
    num_data: int = 100000,
) -> float:
    """評価方策の真の性能を近似する."""
    bandit_data = generate_synthetic_data(
        num_data=num_data,
        dim_context=dim_context,
        num_actions=num_actions,
        T=T,
        theta=theta,
        M=M,
        b=b,
        W=W,
        eps=eps,
        beta=beta,
        is_online=True,
        random_state=12345,
    )
    w, q_t = bandit_data["w"], bandit_data["q_t"]
    value_of_pi = (w * q_t.mean(1)).sum() / w.sum()
    value_of_pi_0 = ((1 - w) * q_t.mean(1)).sum() / (1 - w).sum()

    return value_of_pi_0, value_of_pi


def generate_synthetic_data2(
    num_data: int,
    dim_context: int,
    num_actions: int,
    beta: float,
    theta_1: np.ndarray,
    M_1: np.ndarray,
    b_1: np.ndarray,
    theta_0: np.ndarray,
    M_0: np.ndarray,
    b_0: np.ndarray,
    random_state: int = 12345,
) -> dict:
    """オフ方策学習におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    x = random_.normal(size=(num_data, dim_context))
    one_hot_a = np.eye(num_actions)

    # 期待報酬関数を定義する
    q_x_a_0 = (
        (x ** 3 + x ** 2 - x) @ theta_0 + (x - x ** 2) @ M_0 @ one_hot_a + b_0
    ) / num_actions
    q_x_a_1 = (
        (x - x ** 2) @ theta_1 + (x ** 3 + x ** 2 - x) @ M_1 @ one_hot_a + b_1
    ) / num_actions
    cate_x_a = q_x_a_1 - q_x_a_0
    q_x_a_1 += rankdata(cate_x_a, axis=1, method="dense") <= num_actions * 0.5
    q_x_a_0 += rankdata(cate_x_a, axis=1, method="dense") <= num_actions * 0.5
    q_x_a_1 -= rankdata(cate_x_a, axis=1, method="dense") >= num_actions * 0.8
    q_x_a_0 -= rankdata(cate_x_a, axis=1, method="dense") >= num_actions * 0.8

    # データ収集方策を定義する
    pi_0 = softmax(beta * cate_x_a)

    # 行動や報酬を抽出する
    a = sample_action_fast(pi_0, random_state=random_state)
    a_mat = np.zeros((num_data, num_actions), dtype=int)
    a_mat[np.arange(num_data), a] = 1
    pscore_mat = a_mat * pi_0 + (1 - a_mat) * (1 - pi_0)

    q_x_a_factual = a_mat * q_x_a_1 + (1 - a_mat) * q_x_a_0
    r_mat = random_.normal(q_x_a_factual)

    return dict(
        num_data=num_data,
        num_actions=num_actions,
        x=x,
        a=a,
        r=r_mat[np.arange(num_data), a],
        a_mat=a_mat,
        r_mat=r_mat,
        pi_0=pi_0,
        pscore=pi_0[np.arange(num_data), a],
        pscore_mat=pscore_mat,
        q_x_a_1=q_x_a_1,
        q_x_a_0=q_x_a_0,
        cate_x_a=cate_x_a,
    )
