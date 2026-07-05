import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cma
import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from s_2_train_vae import ConvVAE
from s_4b_train_mdn_rnn import MDNRNN


VAE = None
MDN = None
Z_DIM = None
HIDDEN_DIM = None
DEVICE = "cpu"


class LinearController:
    def __init__(self, z_dim, hidden_dim, action_dim=3):
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.input_dim = z_dim + hidden_dim
        self.num_params = action_dim * self.input_dim + action_dim

    def set_params(self, params):
        params = np.asarray(params, dtype=np.float32)
        split = self.action_dim * self.input_dim
        self.W = params[:split].reshape(self.action_dim, self.input_dim)
        self.b = params[split:]

    def act(self, z, h):
        x = np.concatenate([z, h], axis=0)
        raw = self.W @ x + self.b
        return np.array(
            [
                np.tanh(raw[0]),
                (np.tanh(raw[1]) + 1.0) / 2.0,
                (np.tanh(raw[2]) + 1.0) / 2.0,
            ],
            dtype=np.float32,
        )


def preprocess_obs(obs):
    img = Image.fromarray(obs).resize((64, 64), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def init_worker(vae_path, mdn_path):
    global VAE, MDN, Z_DIM, HIDDEN_DIM, DEVICE

    DEVICE = "cpu"

    vae_ckpt = torch.load(vae_path, map_location=DEVICE)
    VAE = ConvVAE(z_dim=vae_ckpt["z_dim"]).to(DEVICE)
    VAE.load_state_dict(vae_ckpt["model_state_dict"])
    VAE.eval()
    Z_DIM = vae_ckpt["z_dim"]

    mdn_ckpt = torch.load(mdn_path, map_location=DEVICE)
    MDN = MDNRNN(
        z_dim=mdn_ckpt["z_dim"],
        action_dim=mdn_ckpt["action_dim"],
        hidden_dim=mdn_ckpt["hidden_dim"],
        num_mixtures=mdn_ckpt["num_mixtures"],
    ).to(DEVICE)
    MDN.load_state_dict(mdn_ckpt["model_state_dict"])
    MDN.eval()
    HIDDEN_DIM = mdn_ckpt["hidden_dim"]

    torch.set_num_threads(1)


def evaluate_candidate(args_tuple):
    params, episodes, seed, max_steps = args_tuple

    controller = LinearController(Z_DIM, HIDDEN_DIM)
    controller.set_params(params)

    rewards = []

    for ep in range(episodes):
        env = gym.make(
            "CarRacing-v3",
            continuous=True,
            domain_randomize=False,
        )

        obs, _ = env.reset(seed=seed + ep)

        h = torch.zeros(1, 1, HIDDEN_DIM, device=DEVICE)
        c = torch.zeros(1, 1, HIDDEN_DIM, device=DEVICE)

        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            obs_tensor = preprocess_obs(obs).to(DEVICE)

            with torch.no_grad():
                mu, _ = VAE.encode(obs_tensor)
                z_tensor = mu
                z = z_tensor.squeeze(0).cpu().numpy()

            h_np = h.squeeze(0).squeeze(0).cpu().numpy()
            action = controller.act(z, h_np)

            next_obs, reward, terminated, truncated, _ = env.step(action)

            za = np.concatenate([z, action], axis=0).astype(np.float32)
            za_tensor = torch.from_numpy(za).view(1, 1, -1).to(DEVICE)

            with torch.no_grad():
                _, (h, c) = MDN.lstm(za_tensor, (h, c))

            total_reward += float(reward)
            done = terminated or truncated
            obs = next_obs
            steps += 1

        env.close()
        rewards.append(total_reward)

    return float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", default="models/vae.pt")
    parser.add_argument("--mdn-rnn-path", default="models/mdn_rnn.pt")
    parser.add_argument("--save-path", default="models/controller_cmaes.npz")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    ckpt = torch.load(args.vae_path, map_location="cpu")
    z_dim = ckpt["z_dim"]

    mdn_ckpt = torch.load(args.mdn_rnn_path, map_location="cpu")
    hidden_dim = mdn_ckpt["hidden_dim"]

    controller = LinearController(z_dim, hidden_dim)

    print(f"Controller parameters: {controller.num_params}")
    print(f"Workers: {args.workers}")

    x0 = np.zeros(controller.num_params, dtype=np.float32)

    # Bias inicial: recto, acelerar, no frenar.
    # Con nuestra función de acción:
    # gas   = (tanh(raw[1]) + 1) / 2
    # brake = (tanh(raw[2]) + 1) / 2
    x0[-3:] = np.array([0.0, 1.5, -3.0], dtype=np.float32)

    es = cma.CMAEvolutionStrategy(
        x0,
        args.sigma,
        {
            "popsize": args.population_size,
            "seed": args.seed,
        },
    )

    best_reward = -np.inf
    best_params = None

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(args.vae_path, args.mdn_rnn_path),
    ) as executor:

        for generation in range(args.generations):
            start = time.time()
            solutions = es.ask()

            jobs = [
                (
                    np.asarray(params, dtype=np.float32),
                    args.eval_episodes,
                    args.seed + generation * 100_000 + i * 1_000,
                    args.max_steps,
                )
                for i, params in enumerate(solutions)
            ]

            rewards = list(executor.map(evaluate_candidate, jobs))
            fitnesses = [-r for r in rewards]

            es.tell(solutions, fitnesses)

            gen_best = float(np.max(rewards))
            gen_mean = float(np.mean(rewards))

            if gen_best > best_reward:
                idx = int(np.argmax(rewards))
                best_reward = gen_best
                best_params = np.asarray(solutions[idx], dtype=np.float32)

                np.savez_compressed(
                    args.save_path,
                    params=best_params,
                    best_reward=best_reward,
                    z_dim=z_dim,
                    hidden_dim=hidden_dim,
                )

            print(
                f"Generation {generation + 1}/{args.generations} | "
                f"gen_best={gen_best:.2f} | "
                f"gen_mean={gen_mean:.2f} | "
                f"best={best_reward:.2f} | "
                f"duration={time.time() - start:.1f}s"
            )

    print(f"Saved best controller to {args.save_path}")


if __name__ == "__main__":
    main()