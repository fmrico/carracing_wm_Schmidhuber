import argparse
from pathlib import Path

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

    return model, checkpoint


def predict_next_z(model, z_t, action_t, hidden):
    x = torch.cat([z_t, action_t], dim=-1)

    log_pi, mu, log_sigma, hidden = model(x, hidden)

    # Elegimos la gaussiana más probable.
    k = torch.argmax(log_pi, dim=-1).item()

    # Usamos la media de esa gaussiana como predicción determinista.
    z_pred = mu[0, 0, k]

    return z_pred, hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", type=str, default="models/vae.pt")
    parser.add_argument("--mdn-rnn-path", type=str, default="models/mdn_rnn.pt")
    parser.add_argument("--raw-rollout", type=str, default="data/random_rollouts/rollout_00000.npz")
    parser.add_argument("--encoded-rollout", type=str, default="data/encoded_rollouts/rollout_00000.npz")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    vae = load_vae(args.vae_path, device)
    mdn_rnn, checkpoint = load_mdn_rnn(args.mdn_rnn_path, device)

    raw_data = np.load(args.raw_rollout)
    encoded_data = np.load(args.encoded_rollout)

    observations = raw_data["observations"]
    z = encoded_data["z"].astype(np.float32)
    actions = encoded_data["actions"].astype(np.float32)

    hidden = None

    real_images = []
    predicted_images = []
    mse_values = []

    with torch.no_grad():
        for t in range(args.start, args.start + args.steps):
            z_t = torch.from_numpy(z[t]).view(1, 1, -1).to(device)
            action_t = torch.from_numpy(actions[t]).view(1, 1, -1).to(device)

            z_pred, hidden = predict_next_z(
                model=mdn_rnn,
                z_t=z_t,
                action_t=action_t,
                hidden=hidden,
            )

            z_true = torch.from_numpy(z[t + 1]).to(device)
            mse = torch.mean((z_pred - z_true) ** 2).item()
            mse_values.append(mse)

            recon = vae.decode(z_pred.view(1, -1))
            recon = recon.squeeze(0).cpu().numpy()
            recon = np.transpose(recon, (1, 2, 0))
            recon = np.clip(recon, 0.0, 1.0)

            real_next = observations[t + 1]

            real_images.append(real_next)
            predicted_images.append(recon)

    print(f"Mean latent MSE over {args.steps} steps: {np.mean(mse_values):.6f}")

    fig, axes = plt.subplots(2, args.steps, figsize=(2 * args.steps, 4))

    for i in range(args.steps):
        axes[0, i].imshow(real_images[i])
        axes[0, i].set_title(f"real {i + 1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(predicted_images[i])
        axes[1, i].set_title(f"pred {i + 1}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
