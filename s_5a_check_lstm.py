import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from s_4a_train_lstm import LatentLSTM


def load_lstm(path: str, device: str):
    checkpoint = torch.load(path, map_location=device)

    model = LatentLSTM(
        z_dim=checkpoint["z_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/lstm.pt")
    parser.add_argument("--data-dir", type=str, default="data/encoded_rollouts")
    parser.add_argument("--rollout", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model, checkpoint = load_lstm(args.model_path, device)

    files = sorted(Path(args.data_dir).glob("rollout_*.npz"))
    if not files:
        raise FileNotFoundError(f"No encoded rollouts found in {args.data_dir}")

    data = np.load(files[args.rollout])
    z = data["z"].astype(np.float32)
    actions = data["actions"].astype(np.float32)

    steps = min(args.steps, len(z) - 1)

    x = np.concatenate([z[:steps], actions[:steps]], axis=-1)
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        z_pred = model(x).squeeze(0).cpu().numpy()

    z_true = z[1 : steps + 1]

    mse_per_step = ((z_pred - z_true) ** 2).mean(axis=1)

    print(f"Rollout: {files[args.rollout].name}")
    print(f"Mean MSE over {steps} steps: {mse_per_step.mean():.6f}")

    plt.figure(figsize=(10, 4))
    plt.plot(mse_per_step)
    plt.title("LSTM prediction error per timestep")
    plt.xlabel("Timestep")
    plt.ylabel("MSE(z_pred, z_true)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    dim = 0
    plt.plot(z_true[:, dim], label="true z[0]")
    plt.plot(z_pred[:, dim], label="pred z[0]")
    plt.title("Example latent dimension prediction")
    plt.xlabel("Timestep")
    plt.ylabel("z[0]")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()