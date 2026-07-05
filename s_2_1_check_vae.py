import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from s_2_train_vae import ConvVAE


def load_vae(path, device):
    checkpoint = torch.load(path, map_location=device)

    model = ConvVAE(z_dim=checkpoint["z_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vae-path",
        default="models/vae.pt",
    )

    parser.add_argument(
        "--rollout",
        default="data/random_rollouts/rollout_00000.npz",
    )

    parser.add_argument(
        "--frame",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    vae = load_vae(args.vae_path, device)

    data = np.load(args.rollout)

    image = data["observations"][args.frame]

    x = image.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        recon, mu, logvar = vae(x)

    recon = recon.squeeze(0).cpu().numpy()
    recon = np.transpose(recon, (1, 2, 0))

    print("Latent vector z:")
    print(mu.squeeze(0).cpu().numpy())

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(recon)
    plt.title("Reconstruction")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
