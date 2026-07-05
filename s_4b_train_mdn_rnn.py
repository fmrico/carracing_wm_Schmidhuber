import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class EncodedRolloutDataset(IterableDataset):
    def __init__(self, data_dir: str, sequence_length: int, log_every_files: int = 10):
        self.files = sorted(Path(data_dir).glob("rollout_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No encoded rollouts found in {data_dir}")

        self.sequence_length = sequence_length
        self.sequences_per_file = []
        self.total_sequences = 0
        self.z_dim = None
        self.action_dim = None

        log(f"Scanning encoded dataset in {data_dir} ({len(self.files)} rollout files)")

        for file_idx, file_path in enumerate(self.files):
            with np.load(file_path) as data:
                z = data["z"]
                actions = data["actions"]

                if self.z_dim is None:
                    self.z_dim = z.shape[-1]
                    self.action_dim = actions.shape[-1]

                n = len(z)
                num_sequences = max(0, n - sequence_length - 1)

            self.sequences_per_file.append(num_sequences)
            self.total_sequences += num_sequences

            if (
                file_idx == 0
                or (file_idx + 1) % log_every_files == 0
                or file_idx + 1 == len(self.files)
            ):
                log(
                    f"Indexed {file_idx + 1}/{len(self.files)} encoded rollouts "
                    f"({self.total_sequences} sequences so far)"
                )

        log(f"Encoded dataset ready with {self.total_sequences} sequences")

    def __len__(self):
        return self.total_sequences

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        rng = np.random.default_rng(torch.initial_seed() + worker_id)
        file_indices = np.arange(len(self.files))
        rng.shuffle(file_indices)

        for file_idx in file_indices[worker_id::num_workers]:
            num_sequences = self.sequences_per_file[file_idx]
            if num_sequences <= 0:
                continue

            with np.load(self.files[file_idx]) as data:
                z = data["z"].astype(np.float32)
                actions = data["actions"].astype(np.float32)
                start_indices = rng.permutation(num_sequences)

                for start in start_indices:
                    z_seq = z[start : start + self.sequence_length]
                    a_seq = actions[start : start + self.sequence_length]
                    z_next = z[start + 1 : start + self.sequence_length + 1]

                    x = np.concatenate([z_seq, a_seq], axis=-1)

                    yield torch.from_numpy(x), torch.from_numpy(z_next)


class MDNRNN(nn.Module):
    def __init__(self, z_dim: int, action_dim: int, hidden_dim: int = 256, num_mixtures: int = 5):
        super().__init__()

        self.z_dim = z_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_mixtures = num_mixtures

        self.lstm = nn.LSTM(
            input_size=z_dim + action_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, num_mixtures * (1 + 2 * z_dim))

    def forward(self, x, hidden=None):
        h_seq, hidden = self.lstm(x, hidden)
        y = self.fc(h_seq)

        b, t, _ = y.shape
        k = self.num_mixtures
        z = self.z_dim

        y = y.view(b, t, k, 1 + 2 * z)

        log_pi = y[:, :, :, 0]
        mu = y[:, :, :, 1 : 1 + z]
        log_sigma = y[:, :, :, 1 + z :]

        log_pi = torch.log_softmax(log_pi, dim=-1)

        # Estabilización importante: evita sigmas casi cero.
        log_sigma = torch.clamp(log_sigma, min=-2.5, max=2.0)

        return log_pi, mu, log_sigma, hidden


def mdn_loss(log_pi, mu, log_sigma, target):
    """
    log_pi:    [B, T, K]
    mu:        [B, T, K, Z]
    log_sigma: [B, T, K, Z]
    target:    [B, T, Z]
    """

    target = target.unsqueeze(2)

    inv_sigma = torch.exp(-log_sigma)

    log_prob = -0.5 * ((target - mu) * inv_sigma).pow(2)
    log_prob = log_prob - log_sigma
    log_prob = log_prob - 0.5 * math.log(2.0 * math.pi)

    log_prob = log_prob.sum(dim=-1)

    log_prob_mixture = torch.logsumexp(log_pi + log_prob, dim=-1)

    return -log_prob_mixture.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/encoded_rollouts")
    parser.add_argument("--save-path", type=str, default="models/mdn_rnn.pt")
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-mixtures", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--dataset-log-every", type=int, default=10)
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
        f"sequence_length={args.sequence_length}, hidden_dim={args.hidden_dim}, "
        f"num_mixtures={args.num_mixtures}, batch_size={args.batch_size}, "
        f"epochs={args.epochs}, lr={args.learning_rate}, grad_clip={args.grad_clip}, "
        f"num_workers={args.num_workers}"
    )

    dataset = EncodedRolloutDataset(
        args.data_dir,
        args.sequence_length,
        log_every_files=max(1, args.dataset_log_every),
    )

    z_dim = dataset.z_dim
    action_dim = dataset.action_dim

    log(f"z_dim={z_dim}")
    log(f"action_dim={action_dim}")
    log(f"hidden_dim={args.hidden_dim}")
    log(f"num_mixtures={args.num_mixtures}")
    log(f"training sequences={len(dataset)}")

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (device == "cuda"),
        "drop_last": True,
    }

    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(**loader_kwargs)
    log(
        f"DataLoader ready with {len(dataset)} sequences across {len(loader)} batches "
        f"(num_workers={args.num_workers}, pin_memory={device == 'cuda'})"
    )
    log("Data pipeline mode: shuffled rollouts with shuffled sequences inside each rollout")

    model = MDNRNN(
        z_dim=z_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_mixtures=args.num_mixtures,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    log("Model and optimizer initialized")

    for epoch in range(args.epochs):
        model.train()
        start_time = time.time()
        total_loss = 0.0

        log(f"Starting epoch {epoch + 1}/{args.epochs}")

        for batch_idx, (x, z_next) in enumerate(loader, start=1):
            x = x.to(device, non_blocking=True)
            z_next = z_next.to(device, non_blocking=True)

            log_pi, mu, log_sigma, _ = model(x)
            loss = mdn_loss(log_pi, mu, log_sigma, z_next)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            total_loss += loss.item()

            if batch_idx == 1 or batch_idx % max(1, args.log_every) == 0 or batch_idx == len(loader):
                elapsed = time.time() - start_time
                log(
                    f"Epoch {epoch + 1}/{args.epochs} | "
                    f"batch {batch_idx}/{len(loader)} | "
                    f"nll={loss.item():.6f} | "
                    f"elapsed={elapsed:.1f}s"
                )

        epoch_loss = total_loss / len(loader)
        duration = time.time() - start_time

        log(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"nll={epoch_loss:.6f} | "
            f"duration={duration:.1f}s"
        )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"Saving MDN-RNN checkpoint to {save_path}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "z_dim": z_dim,
            "action_dim": action_dim,
            "hidden_dim": args.hidden_dim,
            "num_mixtures": args.num_mixtures,
            "sequence_length": args.sequence_length,
        },
        save_path,
    )

    log(f"Saved MDN-RNN world model to {save_path}")


if __name__ == "__main__":
    main()