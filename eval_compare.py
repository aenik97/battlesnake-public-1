"""Compare CNN model vs heuristic baseline over N games."""

import numpy as np
from snake_env import BattlesnakeEnv, MOVE_NAMES, MOVE_IDX

N_GAMES = 200
N_ENEMIES = 3


def run_games(use_model: bool, n_games: int = N_GAMES):
    """Run n_games episodes. Returns (wins, avg_len, avg_reward)."""
    import onnxruntime as rt
    import os

    session = None
    input_name = None
    if use_model and os.path.exists("model.onnx"):
        session = rt.InferenceSession("model.onnx")
        input_name = session.get_inputs()[0].name

    env = BattlesnakeEnv(n_enemies=N_ENEMIES)

    wins, total_len, total_reward = 0, 0, 0.0

    for _ in range(n_games):
        obs, _ = env.reset()
        done = False
        ep_len, ep_rew = 0, 0.0

        while not done:
            if session is not None:
                tensor = obs[np.newaxis]  # (1, 6, 11, 11)
                logits = session.run(None, {input_name: tensor})[0][0]
                action = int(np.argmax(logits))
            else:
                from logic import choose_move_heuristic
                state = env._make_state(env.my_body, env.my_health,
                                        [(e["body"], e["health"]) for e in env.enemies])
                action = MOVE_IDX[choose_move_heuristic(state)]

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_len += 1
            ep_rew += reward

        if ep_rew > 1.0:
            wins += 1
        total_len += ep_len
        total_reward += ep_rew

    return wins / n_games, total_len / n_games, total_reward / n_games


def main():
    print(f"Running {N_GAMES} games each, {N_ENEMIES} enemies per game...\n")

    print("=== Baseline: Heuristic ===")
    wr, al, ar = run_games(use_model=False)
    print(f"  Win rate:    {wr:.1%}")
    print(f"  Avg length:  {al:.1f} turns")
    print(f"  Avg reward:  {ar:.2f}\n")

    print("=== CNN Model (model.onnx) ===")
    wr2, al2, ar2 = run_games(use_model=True)
    print(f"  Win rate:    {wr2:.1%}")
    print(f"  Avg length:  {al2:.1f} turns")
    print(f"  Avg reward:  {ar2:.2f}\n")

    print("=== Delta (Model vs Heuristic) ===")
    print(f"  Win rate:   {wr2 - wr:+.1%}")
    print(f"  Avg length: {al2 - al:+.1f} turns")
    print(f"  Avg reward: {ar2 - ar:+.2f}")


if __name__ == "__main__":
    main()
