import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class EncodedRolloutDataset(Dataset):
    def __init__(self, data_dir: str, sequence_length: int):
        self.files = sorted(Path(data_dir).glob("rollout_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No encoded rollouts found in {data_dir}")

        self.sequence_length = sequence_length
        self.index = []

        for file_idx, file_path in enumerate(self.files):
            data = np.load(file_path)
            n = len(data["z"])

            # Necesitamos z[t:t+L] y z[t+1:t+L+1]
            for start in range(0, n - sequence_length - 1):
                self.index.append((file_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, start = self.index[idx]
        data = np.load(self.files[file_idx])

        z = data["z"]
        actions = data["actions"]

        z_seq = z[start : start + self.sequence_length]
        a_seq = actions[start : start + self.sequence_length]
        z_next = z[start + 1 : start + self.sequence_length + 1]

        x = np.concatenate([z_seq, a_seq], axis=-1)

        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(z_next.astype(np.float32)),
        )


class LatentLSTM(nn.Module):
    def __init__(self, z_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=z_dim + action_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.output = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        h_seq, _ = self.lstm(x)
        z_pred = self.output(h_seq)
        return z_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/encoded_rollouts")
    parser.add_argument("--save-path", type=str, default="models/lstm.pt")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    dataset = EncodedRolloutDataset(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
    )

    sample_x, sample_y = dataset[0]
    input_dim = sample_x.shape[-1]
    z_dim = sample_y.shape[-1]
    action_dim = input_dim - z_dim

    print(f"z_dim={z_dim}")
    print(f"action_dim={action_dim}")
    print(f"sequence_length={args.sequence_length}")
    print(f"training sequences={len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    model = LatentLSTM(
        z_dim=z_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for x, z_next in dataloader:
            x = x.to(device)
            z_next = z_next.to(device)

            z_pred = model(x)
            loss = loss_fn(z_pred, z_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={total_loss / len(dataloader):.6f}"
        )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "z_dim": z_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "sequence_length": args.sequence_length,
        },
        save_path,
    )

    print(f"Saved LSTM world model to {save_path}")


if __name__ == "__main__":
    main()