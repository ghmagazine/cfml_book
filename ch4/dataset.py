import numpy as np
from sklearn.utils import check_random_state

from utils import sample_action_fast, sigmoid, softmax, eps_greedy_policy


def generate_synthetic_data(
    num_data: int,
    dim_state: int,
    num_states: int,
    num_actions: int,
    H: int,
    theta: np.ndarray,  # d x |A|
    M: np.ndarray,  # d x |A|
    b: np.ndarray,  # |A| x 1
    init_dist: np.ndarray,  # |S|
    trans_probs: np.ndarray,  # |S| x |S| x |A|
    beta: float = -1.0,
    reward_noise: float = 0.5,
    is_test: bool = False,
    random_state: int = 12345,
) -> dict:
    """強化学習におけるオフ方策評価におけるログデータを生成する."""
    random_ = check_random_state(random_state)
    S = random_.normal(size=(num_states, dim_state))
    e_a = np.eye(num_actions)
    q_s_a = sigmoid((S ** 3 + S ** 2 - S) @ theta + (S - S ** 2) @ M @ e_a + b) / H

    s_h = np.zeros((num_data, H), dtype=int)
    a_h = np.zeros((num_data, H), dtype=int)
    r_h = np.zeros((num_data, H))
    q_h = np.zeros((num_data, H))
    pi = np.zeros((num_data, num_actions, H))
    pi_0 = np.zeros((num_data, num_actions, H))

    s_h[:, 0] = sample_action_fast(
        np.tile(init_dist, (num_data, 1)), random_state=random_state
    )
    for h in range(H):
        if is_test:
            pi_0[:, :, h] = eps_greedy_policy(q_s_a[s_h[:, h]])
        else:
            pi_0[:, :, h] = softmax(beta * q_s_a[s_h[:, h]])
        pi[:, :, h] = eps_greedy_policy(q_s_a[s_h[:, h]])
        a_h[:, h] = sample_action_fast(pi_0[:, :, h], random_state=random_state + h)
        q_h[:, h] = q_s_a[s_h[:, h], a_h[:, h]]
        r_h[:, h] = random_.normal(q_h[:, h], scale=reward_noise)
        if h < H - 1:
            s_h[:, h + 1] = sample_action_fast(
                trans_probs[s_h[:, h], :, a_h[:, h]], random_state=random_state + h
            )

    return dict(
        num_data=num_data,
        H=H,
        num_states=num_states,
        num_actions=num_actions,
        s_h=s_h,
        a_h=a_h,
        r_h=r_h,
        S=S,
        pi_0=pi_0,
        pi=pi,
        q_h=q_h,
        q_s_a=q_s_a,
    )


def calc_true_value(
    dim_state: int,
    num_states: int,
    num_actions: int,
    H: int,
    theta: np.ndarray,
    M: np.ndarray,
    b: np.ndarray,
    init_dist: np.ndarray,
    trans_probs: np.ndarray,
    num_data: int = 100000,
) -> float:
    """評価方策の真の性能を近似する."""
    test_bandit_data = generate_synthetic_data(
        num_data=num_data,
        dim_state=dim_state,
        num_states=num_states,
        num_actions=num_actions,
        H=H,
        theta=theta,
        M=M,
        b=b,
        init_dist=init_dist,
        trans_probs=trans_probs,
        is_test=True,
    )

    return test_bandit_data, test_bandit_data["q_h"].sum(1).mean()
