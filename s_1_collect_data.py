import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image
from tqdm import trange


def preprocess_obs(obs: np.ndarray, size: int = 64) -> np.ndarray:
    img = Image.fromarray(obs)
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def make_env(render_mode=None, seed=None):
    env = gym.make(
        "CarRacing-v3",
        continuous=True,
        render_mode=render_mode,
        domain_randomize=False,
    )
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="data/random_rollouts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(
        render_mode="human" if args.render else None,
        seed=args.seed,
    )

    for ep in trange(args.episodes, desc="Collecting rollouts"):
        obs, info = env.reset(seed=args.seed + ep)

        observations = []
        actions = []
        rewards = []
        dones = []

        done = False

        while not done:
            action = env.action_space.sample()

            obs_small = preprocess_obs(obs)
            observations.append(obs_small)
            actions.append(action.astype(np.float32))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rewards.append(np.float32(reward))
            dones.append(done)

        np.savez_compressed(
            out_dir / f"rollout_{ep:05d}.npz",
            observations=np.asarray(observations, dtype=np.uint8),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.bool_),
        )

    env.close()
    print(f"Saved {args.episodes} rollouts in {out_dir}")


if __name__ == "__main__":
    main()
