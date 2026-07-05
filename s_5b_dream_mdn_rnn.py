import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from s_2_train_vae import ConvVAE
from s_4b_train_mdn_rnn import MDNRNN


def load_vae(path, device):
    checkpoint = torch.load(path, map_location=device)
    model = ConvVAE(z_dim=checkpoint["z_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


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
    return model


def sample_next_z(log_pi, mu, log_sigma, temperature):
    pi = torch.softmax(log_pi / temperature, dim=-1)
    k = torch.multinomial(pi.view(-1), num_samples=1).item()

    sigma = torch.exp(log_sigma[0, 0, k]) * temperature
    eps = torch.randn_like(sigma)

    return mu[0, 0, k] + eps * sigma


def decode_z(vae, z):
    img = vae.decode(z.view(1, -1))
    img = img.squeeze(0).cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", default="models/vae.pt")
    parser.add_argument("--mdn-rnn-path", default="models/mdn_rnn.pt")
    parser.add_argument("--encoded-rollout", default="data/encoded_rollouts/rollout_00000.npz")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device == "auto" and device != "cuda":
        device = "cpu"

    vae = load_vae(args.vae_path, device)
    mdn_rnn = load_mdn_rnn(args.mdn_rnn_path, device)

    data = np.load(args.encoded_rollout)
    z_real = data["z"].astype(np.float32)
    actions = data["actions"].astype(np.float32)

    z = torch.from_numpy(z_real[args.start]).view(1, 1, -1).to(device)
    hidden = None

    images = []

    with torch.no_grad():
        images.append(decode_z(vae, z.view(-1)))

        for i in range(args.steps):
            t = args.start + i
            action = torch.from_numpy(actions[t]).view(1, 1, -1).to(device)

            x = torch.cat([z, action], dim=-1)

            log_pi, mu, log_sigma, hidden = mdn_rnn(x, hidden)

            z_next = sample_next_z(
                log_pi=log_pi,
                mu=mu,
                log_sigma=log_sigma,
                temperature=args.temperature,
            )

            z = z_next.view(1, 1, -1)
            images.append(decode_z(vae, z_next))

    rows = int(np.ceil(len(images) / args.cols))

    fig, axes = plt.subplots(rows, args.cols, figsize=(2 * args.cols, 2 * rows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(f"t+{i}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()