import argparse

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

from s_2_train_vae import ConvVAE
from s_4b_train_mdn_rnn import MDNRNN
from s_6_train_controller_cmaes import LinearController


def preprocess_obs(obs):
    img = Image.fromarray(obs).resize((64, 64), Image.BILINEAR)
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


def load_vae(path, device):
    ckpt = torch.load(path, map_location=device)
    model = ConvVAE(z_dim=ckpt["z_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["z_dim"]


def load_mdn_rnn(path, device):
    ckpt = torch.load(path, map_location=device)

    model = MDNRNN(
        z_dim=ckpt["z_dim"],
        action_dim=ckpt["action_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_mixtures=ckpt["num_mixtures"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def run_episode(env, vae, mdn_rnn, controller, device, seed):
    obs, _ = env.reset(seed=seed)

    hidden = None
    total_reward = 0.0
    episode_length = 0
    done = False

    while not done:
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
        episode_length += 1
        done = terminated or truncated
        obs = next_obs

    return total_reward, episode_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", type=str, default="models/vae.pt")
    parser.add_argument("--mdn-rnn-path", type=str, default="models/mdn_rnn.pt")
    parser.add_argument("--controller-path", type=str, default="models/controller_cmaes.npz")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    vae, z_dim = load_vae(args.vae_path, device)
    mdn_rnn, mdn_info = load_mdn_rnn(args.mdn_rnn_path, device)

    data = np.load(args.controller_path)
    params = data["params"]

    controller = LinearController(
        z_dim=z_dim,
        hidden_dim=mdn_info["hidden_dim"],
        action_dim=3,
    )
    controller.set_params(params)

    env = make_env(render_mode="human" if args.render else None)

    rewards = []
    lengths = []

    for ep in range(args.episodes):
        reward, length = run_episode(
            env=env,
            vae=vae,
            mdn_rnn=mdn_rnn,
            controller=controller,
            device=device,
            seed=args.seed + ep,
        )

        rewards.append(reward)
        lengths.append(length)

        print(
            f"Episode {ep + 1}/{args.episodes} | "
            f"reward={reward:.2f} | length={length}"
        )

    env.close()

    rewards = np.asarray(rewards)
    lengths = np.asarray(lengths)

    print()
    print("Evaluation summary")
    print("==================")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {rewards.mean():.2f}")
    print(f"Std reward: {rewards.std():.2f}")
    print(f"Min reward: {rewards.min():.2f}")
    print(f"Max reward: {rewards.max():.2f}")
    print(f"Mean length: {lengths.mean():.2f}")
    print(f"Completed laps (>900): {(rewards > 900).mean():.1%}")


if __name__ == "__main__":
    main()
