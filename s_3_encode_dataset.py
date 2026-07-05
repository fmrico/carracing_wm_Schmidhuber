import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from s_2_train_vae import ConvVAE


def load_vae(path: str, device: str):
    checkpoint = torch.load(path, map_location=device)
    z_dim = checkpoint["z_dim"]

    vae = ConvVAE(z_dim=z_dim).to(device)
    vae.load_state_dict(checkpoint["model_state_dict"])
    vae.eval()

    return vae, z_dim


def encode_observations(vae, observations: np.ndarray, batch_size: int, device: str):
    obs = observations.astype(np.float32) / 255.0
    obs = np.transpose(obs, (0, 3, 1, 2))

    tensor = torch.from_numpy(obs)
    loader = DataLoader(
        TensorDataset(tensor),
        batch_size=batch_size,
        shuffle=False,
    )

    all_mu = []

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            mu, logvar = vae.encode(batch)
            all_mu.append(mu.cpu().numpy())

    return np.concatenate(all_mu, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=str, default="data/random_rollouts")
    parser.add_argument("--vae-path", type=str, default="models/vae.pt")
    parser.add_argument("--out-dir", type=str, default="data/encoded_rollouts")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    rollout_dir = Path(args.rollout_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae, z_dim = load_vae(args.vae_path, device)
    print(f"Loaded VAE with z_dim={z_dim}")

    files = sorted(rollout_dir.glob("rollout_*.npz"))
    if not files:
        raise FileNotFoundError(f"No rollout files found in {rollout_dir}")

    for file_path in files:
        data = np.load(file_path)

        observations = data["observations"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]

        z = encode_observations(
            vae=vae,
            observations=observations,
            batch_size=args.batch_size,
            device=device,
        )

        out_path = out_dir / file_path.name

        np.savez_compressed(
            out_path,
            z=z.astype(np.float32),
            actions=actions.astype(np.float32),
            rewards=rewards.astype(np.float32),
            dones=dones.astype(np.bool_),
        )

        print(f"Encoded {file_path.name}: {observations.shape} -> {z.shape}")

    print(f"Encoded rollouts saved in {out_dir}")


if __name__ == "__main__":
    main()