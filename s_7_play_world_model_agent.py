import argparse
from pathlib import Path

import cma
import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from s_2_train_vae import ConvVAE
from s_4b_train_mdn_rnn import MDNRNN


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


def evaluate_controller(params, vae, mdn_rnn, controller, device, episodes, seed, render=False):
    controller.set_params(params)

    rewards = []

    for ep in range(episodes):
        env = make_env(render_mode="human" if render else None)
        obs, info = env.reset(seed=seed + ep)

        h = torch.zeros(1, 1, mdn_rnn.hidden_dim, device=device)
        c = torch.zeros(1, 1, mdn_rnn.hidden_dim, device=device)
        previous_action = np.zeros(3, dtype=np.float32)

        total_reward = 0.0
        done = False

        while not done:
            obs_tensor = preprocess_obs(obs).to(device)

            with torch.no_grad():
                mu, logvar = vae.encode(obs_tensor)
                z_tensor = mu
                z = z_tensor.squeeze(0).cpu().numpy()

            h_np = h.squeeze(0).squeeze(0).detach().cpu().numpy()
            action = controller.act(z, h_np)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

            # Actualizamos la memoria del MDN-RNN con z_t y a_t
            za = np.concatenate([z, action], axis=0).astype(np.float32)
            za_tensor = torch.from_numpy(za).view(1, 1, -1).to(device)

            with torch.no_grad():
                _, (h, c) = mdn_rnn.lstm(za_tensor, (h, c))

            obs = next_obs
            previous_action = action

        env.close()
        rewards.append(total_reward)

    return float(np.mean(rewards))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", type=str, default="models/vae.pt")
    parser.add_argument("--mdn-rnn-path", type=str, default="models/mdn_rnn.pt")
    parser.add_argument("--save-path", type=str, default="models/controller_cmaes.npz")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--render-best", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    vae, z_dim = load_vae(args.vae_path, device)
    mdn_rnn, mdn_info = load_mdn_rnn(args.mdn_rnn_path, device)

    controller = LinearController(
        z_dim=z_dim,
        hidden_dim=mdn_info["hidden_dim"],
        action_dim=3,
    )

    print(f"Controller parameters: {controller.num_params}")

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
    best_params = None

    for generation in range(args.generations):
        solutions = es.ask()
        fitnesses = []

        for i, params in enumerate(solutions):
            reward = evaluate_controller(
                params=params,
                vae=vae,
                mdn_rnn=mdn_rnn,
                controller=controller,
                device=device,
                episodes=args.eval_episodes,
                seed=args.seed + generation * 10_000 + i * 100,
                render=False,
            )

            fitnesses.append(-reward)

            if reward > best_reward:
                best_reward = reward
                best_params = np.asarray(params, dtype=np.float32)

        es.tell(solutions, fitnesses)

        print(
            f"Generation {generation + 1}/{args.generations} | "
            f"best_reward={best_reward:.2f} | "
            f"gen_mean_reward={-np.mean(fitnesses):.2f}"
        )

        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.save_path,
            params=best_params,
            best_reward=best_reward,
            z_dim=z_dim,
            hidden_dim=mdn_info["hidden_dim"],
        )

    print(f"Saved best controller to {args.save_path}")

    if args.render_best and best_params is not None:
        evaluate_controller(
            params=best_params,
            vae=vae,
            mdn_rnn=mdn_rnn,
            controller=controller,
            device=device,
            episodes=1,
            seed=args.seed + 999_999,
            render=True,
        )


if __name__ == "__main__":
    main()