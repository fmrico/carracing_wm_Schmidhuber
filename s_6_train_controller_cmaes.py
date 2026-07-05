import argparse
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import cma
import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from s_2_train_vae import ConvVAE
from s_4b_train_mdn_rnn import MDNRNN


_GLOBALS = {}


def preprocess_obs(obs):
    img = Image.fromarray(obs)
    img = img.resize((64, 64), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).unsqueeze(0)


def make_env(render_mode=None):
    return gym.make(
        "CarRacing-v3",
        continuous=True,
        render_mode=render_mode,
        domain_randomize=False,
    )


class LinearController:
    def __init__(self, z_dim, hidden_dim, action_dim=3):
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.input_dim = z_dim + hidden_dim
        self.num_params = action_dim * self.input_dim + action_dim
        self.W = np.zeros((action_dim, self.input_dim), dtype=np.float32)
        self.b = np.zeros(action_dim, dtype=np.float32)

    def set_params(self, params):
        params = np.asarray(params, dtype=np.float32)
        split = self.action_dim * self.input_dim
        self.W = params[:split].reshape(self.action_dim, self.input_dim)
        self.b = params[split:]

    def act(self, z, h):
        x = np.concatenate([z, h], axis=0)
        raw = self.W @ x + self.b

        steering = np.tanh(raw[0])
        gas = (np.tanh(raw[1]) + 1.0) / 2.0
        brake = (np.tanh(raw[2]) + 1.0) / 2.0

        return np.array([steering, gas, brake], dtype=np.float32)


def load_vae(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = ConvVAE(z_dim=checkpoint["z_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["z_dim"]


def load_mdn_rnn(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = MDNRNN(
        z_dim=checkpoint["z_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_mixtures=checkpoint["num_mixtures"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def evaluate_controller(params, vae, mdn_rnn, controller, device, episodes, seed, max_steps, render=False):
    controller.set_params(params)
    rewards = []

    for ep in range(episodes):
        env = make_env(render_mode="human" if render else None)
        obs, _ = env.reset(seed=seed + ep)

        hidden = None
        total_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            obs_tensor = preprocess_obs(obs).to(device)

            with torch.no_grad():
                mu, _ = vae.encode(obs_tensor)
                z_tensor = mu
                z = z_tensor.squeeze(0).cpu().numpy()

            if hidden is None:
                h_np = np.zeros(mdn_rnn.hidden_dim, dtype=np.float32)
            else:
                h_np = hidden[0].squeeze(0).squeeze(0).detach().cpu().numpy()

            action = controller.act(z, h_np)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            za = np.concatenate([z, action], axis=0).astype(np.float32)
            za_tensor = torch.from_numpy(za).view(1, 1, -1).to(device)

            with torch.no_grad():
                _, _, _, hidden = mdn_rnn(za_tensor, hidden)

            total_reward += float(reward)
            done = terminated or truncated
            obs = next_obs
            steps += 1

        env.close()
        rewards.append(total_reward)

    return float(np.mean(rewards))


def init_worker(vae_path, mdn_rnn_path, device):
    global _GLOBALS
    torch.set_num_threads(1)

    vae, z_dim = load_vae(vae_path, device)
    mdn_rnn, mdn_info = load_mdn_rnn(mdn_rnn_path, device)
    controller = LinearController(z_dim=z_dim, hidden_dim=mdn_info["hidden_dim"])

    _GLOBALS = {
        "vae": vae,
        "mdn_rnn": mdn_rnn,
        "controller": controller,
        "device": device,
    }


def worker_eval(args_tuple):
    params, episodes, seed, max_steps = args_tuple
    return evaluate_controller(
        params=params,
        vae=_GLOBALS["vae"],
        mdn_rnn=_GLOBALS["mdn_rnn"],
        controller=_GLOBALS["controller"],
        device=_GLOBALS["device"],
        episodes=episodes,
        seed=seed,
        max_steps=max_steps,
        render=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", type=str, default="models/vae.pt")
    parser.add_argument("--mdn-rnn-path", type=str, default="models/mdn_rnn.pt")
    parser.add_argument("--save-path", type=str, default="models/controller_cmaes.npz")
    parser.add_argument("--resume-cma", type=str)
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render-best", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() and args.workers <= 1 else "cpu"
    else:
        device = args.device

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cma_path = save_path.with_suffix(".cma.pkl")

    print(f"Using device: {device}")

    vae, z_dim = load_vae(args.vae_path, device)
    mdn_rnn, mdn_info = load_mdn_rnn(args.mdn_rnn_path, device)
    controller = LinearController(z_dim=z_dim, hidden_dim=mdn_info["hidden_dim"])

    print(f"Controller parameters: {controller.num_params}")
    print(f"Workers: {args.workers}")

    x0 = np.zeros(controller.num_params, dtype=np.float32)
    x0[-3:] = np.array([0.0, 1.5, -3.0], dtype=np.float32)

    if args.resume_cma:
        with open(args.resume_cma, "rb") as f:
            es = pickle.load(f)
        print(f"Resumed CMA-ES from {args.resume_cma}")
    else:
        es = cma.CMAEvolutionStrategy(
            x0,
            args.sigma,
            {
                "popsize": args.population_size,
                "seed": args.seed,
            },
        )

    best_reward = -np.inf
    best_params = x0.copy()

    if save_path.exists():
        previous = np.load(save_path)
        best_reward = float(previous["best_reward"])
        best_params = previous["params"].astype(np.float32)
        print(f"Loaded previous best_reward={best_reward:.2f}")

    for generation in range(args.generations):
        start_time = time.time()
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

        if args.workers > 1:
            with Pool(
                processes=args.workers,
                initializer=init_worker,
                initargs=(args.vae_path, args.mdn_rnn_path, "cpu"),
            ) as pool:
                generation_rewards = pool.map(worker_eval, jobs)
        else:
            generation_rewards = [
                evaluate_controller(
                    params=job[0],
                    vae=vae,
                    mdn_rnn=mdn_rnn,
                    controller=controller,
                    device=device,
                    episodes=job[1],
                    seed=job[2],
                    max_steps=job[3],
                    render=False,
                )
                for job in jobs
            ]

        fitnesses = [-r for r in generation_rewards]
        es.tell(solutions, fitnesses)

        gen_best = float(np.max(generation_rewards))
        gen_mean = float(np.mean(generation_rewards))

        if gen_best > best_reward:
            best_reward = gen_best
            best_idx = int(np.argmax(generation_rewards))
            best_params = np.asarray(solutions[best_idx], dtype=np.float32)

        duration = time.time() - start_time

        print(
            f"Generation {generation + 1}/{args.generations} | "
            f"gen_best={gen_best:.2f} | "
            f"gen_mean={gen_mean:.2f} | "
            f"best={best_reward:.2f} | "
            f"duration={duration:.1f}s"
        )

        np.savez_compressed(
            save_path,
            params=best_params,
            best_reward=best_reward,
            z_dim=z_dim,
            hidden_dim=mdn_info["hidden_dim"],
            action_dim=3,
        )

        with open(cma_path, "wb") as f:
            pickle.dump(es, f)

    print(f"Saved controller to {save_path}")
    print(f"Saved CMA-ES state to {cma_path}")

    if args.render_best:
        evaluate_controller(
            params=best_params,
            vae=vae,
            mdn_rnn=mdn_rnn,
            controller=controller,
            device=device,
            episodes=1,
            seed=args.seed + 999_999,
            max_steps=args.max_steps,
            render=True,
        )


if __name__ == "__main__":
    main()