"""PPO fine-tuning with warm start from IL weights."""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym

from snake_env import BattlesnakeEnv


class SnakeFeaturesExtractor(BaseFeaturesExtractor):
    """CNN feature extractor — same architecture as IL model minus the final head."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, features_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)


def load_il_weights(model: PPO, il_path: str = "model_il.pt") -> None:
    """Copy IL feature-extractor weights into the PPO policy."""
    il_state = torch.load(il_path, map_location="cpu")
    fe = model.policy.features_extractor.cnn

    # IL Sequential indices 0-12 map to FE indices 0-10 (net.0..net.12 → 0..10)
    # net.13 (final Linear 128→4) is the policy head — not loaded here.
    new_state = {}
    for k, v in il_state.items():
        if k.startswith("net.13"):
            continue
        new_k = k.replace("net.", "", 1)
        new_state[new_k] = v

    missing, unexpected = fe.load_state_dict(new_state, strict=False)
    print(f"Warm start: loaded IL weights  missing={missing}  unexpected={unexpected}")


class PPOInference(nn.Module):
    """Thin wrapper to export only the action-logit path to ONNX."""

    def __init__(self, policy):
        super().__init__()
        self.fe = policy.features_extractor
        self.mlp_pi = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_net(self.mlp_pi(self.fe(x)))


def export_onnx(model: PPO, path: str = "model.onnx") -> None:
    policy = model.policy
    policy.eval()
    wrapper = PPOInference(policy)
    wrapper.eval()
    dummy = torch.zeros(1, 6, 11, 11)
    torch.onnx.export(wrapper, (dummy,), path, dynamo=True)
    print(f"Saved {path}")


def main():
    env = make_vec_env(lambda: BattlesnakeEnv(n_enemies=3), n_envs=4)
    eval_env = make_vec_env(lambda: BattlesnakeEnv(n_enemies=3), n_envs=2)

    policy_kwargs = {
        "features_extractor_class": SnakeFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": [],
    }

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    import os
    if os.path.exists("model_ppo.zip"):
        # Round 2+: warm start from previous PPO checkpoint
        prev = PPO.load("model_ppo", env=env)
        model.policy.load_state_dict(prev.policy.state_dict())
        print("Warm start: loaded model_ppo.zip weights")
    elif os.path.exists("model_il.pt"):
        load_il_weights(model)
        print("Warm start: loaded IL weights")
    else:
        print("Starting PPO from random weights")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=".",
        eval_freq=10_000,
        n_eval_episodes=20,
        verbose=1,
    )

    round_num = 2 if os.path.exists("model_ppo.zip") else 1
    print(f"Starting PPO training round {round_num} for 500k steps...")
    model.learn(total_timesteps=500_000, callback=eval_callback)
    model.save("model_ppo")
    print("Saved model_ppo.zip")

    export_onnx(model, "model.onnx")

    # Keep a versioned copy with weights embedded (no external .data file)
    # so onnxruntime can load it as a standalone file in the next round.
    import onnx as _onnx
    _m = _onnx.load("model.onnx")
    versioned_path = f"model_v{round_num}.onnx"
    _onnx.save(_m, versioned_path, save_as_external_data=False)
    print(f"Saved {versioned_path} (for use as opponent in next round)")


if __name__ == "__main__":
    main()
