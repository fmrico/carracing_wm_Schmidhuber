import argparse
import time
from pathlib import Path

import cma
import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F

from s_2_train_vae import ConvVAE
from s_4b_train_mdn_rnn import MDNRNN


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def preprocess_obs(obs, device):
    tensor = torch.from_numpy(obs).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    return F.interpolate(tensor, size=(64, 64), mode="bilinear", align_corners=False)


def make_env(render_mode=None):
    return gym.make(
        "CarRacing-v3",
        continuous=True,
        render_mode=render_mode,
        domain_randomize=False,
    )


class LinearController:
    def __init__(self, z_dim, hidden_dim, action_dim=3, device="cpu"):
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.device = device
        self.input_dim = z_dim + hidden_dim
        self.num_params = action_dim * self.input_dim + action_dim

        self.W = torch.zeros((action_dim, self.input_dim), dtype=torch.float32, device=device)
        self.b = torch.zeros(action_dim, dtype=torch.float32, device=device)

    def set_params(self, params):
        params = np.asarray(params, dtype=np.float32)
        split = self.action_dim * self.input_dim

        self.W = torch.from_numpy(params[:split].reshape(self.action_dim, self.input_dim)).to(self.device)
        self.b = torch.from_numpy(params[split:]).to(self.device)

    def act(self, z, h):
        x = torch.cat([z, h], dim=0)
        raw = self.W @ x + self.b

        steering = torch.tanh(raw[0])
        gas = (torch.tanh(raw[1]) + 1.0) / 2.0
        brake = (torch.tanh(raw[2]) + 1.0) / 2.0

        return torch.stack([steering, gas, brake])


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


def evaluate_controller(
    params,
    vae,
    mdn_rnn,
    controller,
    device,
    episodes,
    seed,
    max_steps,
    render=False,
    candidate_index=None,
    total_candidates=None,
    log_episode_rewards=False,
):
    controller.set_params(params)

    rewards = []
    env = make_env(render_mode="human" if render else None)
    eval_start = time.time()

    try:
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)

            hidden = (
                torch.zeros(1, 1, mdn_rnn.hidden_dim, device=device),
                torch.zeros(1, 1, mdn_rnn.hidden_dim, device=device),
            )
            total_reward = 0.0
            done = False
            steps = 0

            with torch.no_grad():
                while not done and steps < max_steps:
                    obs_tensor = preprocess_obs(obs, device)

                    mu, _ = vae.encode(obs_tensor)
                    z_tensor = mu.squeeze(0)
                    h_tensor = hidden[0].squeeze(0).squeeze(0)

                    action_tensor = controller.act(z_tensor, h_tensor)
                    action = action_tensor.cpu().numpy()

                    next_obs, reward, terminated, truncated, info = env.step(action)

                    za_tensor = torch.cat([z_tensor, action_tensor], dim=0).view(1, 1, -1)
                    _, _, _, hidden = mdn_rnn(za_tensor, hidden)

                    total_reward += float(reward)
                    done = terminated or truncated
                    obs = next_obs
                    steps += 1

            rewards.append(total_reward)

            if log_episode_rewards:
                prefix = ""
                if candidate_index is not None and total_candidates is not None:
                    prefix = f"Candidate {candidate_index}/{total_candidates} | "
                log(
                    f"{prefix}episode {ep + 1}/{episodes} | reward={total_reward:.2f} | "
                    f"steps={steps}"
                )
    finally:
        env.close()

    mean_reward = float(np.mean(rewards))
    elapsed = time.time() - eval_start

    if candidate_index is not None and total_candidates is not None:
        log(
            f"Candidate {candidate_index}/{total_candidates} finished | "
            f"mean_reward={mean_reward:.2f} | elapsed={elapsed:.1f}s"
        )

    return mean_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", type=str, default="models/vae.pt")
    parser.add_argument("--mdn-rnn-path", type=str, default="models/mdn_rnn.pt")
    parser.add_argument("--save-path", type=str, default="models/controller_cmaes.npz")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render-best", action="store_true")
    parser.add_argument("--log-every-candidates", type=int, default=8)
    parser.add_argument("--log-episode-rewards", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    log(f"Using device: {device}")
    log(
        "Configuration: "
        f"generations={args.generations}, population_size={args.population_size}, "
        f"eval_episodes={args.eval_episodes}, sigma={args.sigma}, "
        f"max_steps={args.max_steps}, seed={args.seed}"
    )

    log(f"Loading VAE from {args.vae_path}")
    vae, z_dim = load_vae(args.vae_path, device)
    log(f"Loading MDN-RNN from {args.mdn_rnn_path}")
    mdn_rnn, mdn_info = load_mdn_rnn(args.mdn_rnn_path, device)

    controller = LinearController(
        z_dim=z_dim,
        hidden_dim=mdn_info["hidden_dim"],
        action_dim=3,
        device=device,
    )

    log(f"z_dim={z_dim}")
    log(f"hidden_dim={mdn_info['hidden_dim']}")
    log(f"controller parameters={controller.num_params}")

    x0 = np.zeros(controller.num_params, dtype=np.float32)

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

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    log(f"Starting CMA-ES optimization, saving best controller to {save_path}")

    for generation in range(args.generations):
        solutions = es.ask()
        fitnesses = []
        generation_rewards = []
        generation_start = time.time()

        log(f"Starting generation {generation + 1}/{args.generations}")

        for i, params in enumerate(solutions):
            candidate_index = i + 1
            should_log_candidate = (
                candidate_index == 1
                or candidate_index % max(1, args.log_every_candidates) == 0
                or candidate_index == len(solutions)
            )

            reward = evaluate_controller(
                params=params,
                vae=vae,
                mdn_rnn=mdn_rnn,
                controller=controller,
                device=device,
                episodes=args.eval_episodes,
                seed=args.seed + generation * 100_000 + i * 1_000,
                max_steps=args.max_steps,
                render=False,
                candidate_index=candidate_index if should_log_candidate else None,
                total_candidates=len(solutions) if should_log_candidate else None,
                log_episode_rewards=args.log_episode_rewards and should_log_candidate,
            )

            fitnesses.append(-reward)
            generation_rewards.append(reward)

            if reward > best_reward:
                best_reward = reward
                best_params = np.asarray(params, dtype=np.float32)
                log(
                    f"New best controller found | generation={generation + 1} | "
                    f"candidate={candidate_index} | reward={best_reward:.2f}"
                )

        es.tell(solutions, fitnesses)

        elapsed = time.time() - start_time
        generation_elapsed = time.time() - generation_start

        log(
            f"Generation {generation + 1}/{args.generations} | "
            f"gen_best={np.max(generation_rewards):.2f} | "
            f"gen_mean={np.mean(generation_rewards):.2f} | "
            f"best={best_reward:.2f} | "
            f"gen_duration={generation_elapsed:.1f}s | "
            f"elapsed={elapsed:.1f}s"
        )

        log(f"Saving best controller checkpoint to {save_path}")
        np.savez_compressed(
            save_path,
            params=best_params,
            best_reward=best_reward,
            z_dim=z_dim,
            hidden_dim=mdn_info["hidden_dim"],
            action_dim=3,
        )

    log(f"Saved best controller to {save_path}")

    if args.render_best:
        log("Rendering best controller")
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
            candidate_index=1,
            total_candidates=1,
            log_episode_rewards=True,
        )


if __name__ == "__main__":
    main()