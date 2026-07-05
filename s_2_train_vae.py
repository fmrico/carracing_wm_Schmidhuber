import argparse
import os
from pathlib import Path
from time import strftime, time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def log(message: str):
    print(f"[{strftime('%H:%M:%S')}] {message}", flush=True)


class RolloutImageDataset(IterableDataset):
    def __init__(self, data_dir: str, log_every_files: int = 10):
        self.files = sorted(Path(data_dir).glob("rollout_*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No rollout files found in {data_dir}")

        log(f"Scanning dataset in {data_dir} ({len(self.files)} rollout files)")

        self.frames_per_file = []
        total_frames = 0
        for file_idx, file_path in enumerate(self.files):
            with np.load(file_path) as data:
                n = len(data["observations"])

            self.frames_per_file.append(n)
            total_frames += n

            if (
                file_idx == 0
                or (file_idx + 1) % log_every_files == 0
                or file_idx + 1 == len(self.files)
            ):
                log(
                    f"Indexed {file_idx + 1}/{len(self.files)} rollout files "
                    f"({total_frames} frames so far)"
                )

        self.total_frames = total_frames
        log(f"Dataset ready with {total_frames} frames")

    def __len__(self):
        return self.total_frames

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
            with np.load(self.files[file_idx]) as data:
                observations = data["observations"]
                frame_indices = rng.permutation(len(observations))

                for frame_idx in frame_indices:
                    obs = observations[frame_idx].astype(np.float32) / 255.0
                    obs = np.transpose(obs, (2, 0, 1))
                    yield torch.from_numpy(obs)


class ConvVAE(nn.Module):
    def __init__(self, z_dim: int = 32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 4x4
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc_mu = nn.Linear(256 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, z_dim)

        self.decoder_input = nn.Linear(z_dim, 256 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 64x64
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, beta: float):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/random_rollouts")
    parser.add_argument("--save-path", type=str, default="models/vae.pt")
    parser.add_argument("--z-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.0001)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--dataset-log-every", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--prefetch-factor", type=int, default=4)
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
        f"z_dim={args.z_dim}, batch_size={args.batch_size}, epochs={args.epochs}, "
        f"lr={args.learning_rate}, beta={args.beta}, num_workers={args.num_workers}"
    )

    dataset = RolloutImageDataset(
        args.data_dir,
        log_every_files=max(1, args.dataset_log_every),
    )
    dataloader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": (device == "cuda"),
    }

    if args.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = args.prefetch_factor

    dataloader = DataLoader(**dataloader_kwargs)

    log(
        f"DataLoader ready with {len(dataset)} samples across {len(dataloader)} batches "
        f"(num_workers={args.num_workers}, pin_memory={device == 'cuda'})"
    )
    log("Data pipeline mode: shuffled rollouts with shuffled frames inside each rollout")

    model = ConvVAE(z_dim=args.z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    log("Model and optimizer initialized")

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time()

        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        log(f"Starting epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(dataloader, start=1):
            batch = batch.to(device, non_blocking=(device == "cuda"))

            recon, mu, logvar = model(batch)
            loss, recon_loss, kl_loss = vae_loss(
                recon, batch, mu, logvar, beta=args.beta
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()

            if batch_idx == 1 or batch_idx % max(1, args.log_every) == 0 or batch_idx == len(dataloader):
                elapsed = time() - epoch_start
                log(
                    f"Epoch {epoch + 1}/{args.epochs} | batch {batch_idx}/{len(dataloader)} | "
                    f"loss={loss.item():.6f} | recon={recon_loss.item():.6f} | "
                    f"kl={kl_loss.item():.6f} | elapsed={elapsed:.1f}s"
                )

        n = len(dataloader)

        log(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={total_loss / n:.6f} | "
            f"recon={total_recon / n:.6f} | "
            f"kl={total_kl / n:.6f} | "
            f"duration={time() - epoch_start:.1f}s"
        )

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"Saving VAE checkpoint to {save_path}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "z_dim": args.z_dim,
        },
        save_path,
    )

    log(f"Saved VAE to {save_path}")


if __name__ == "__main__":
    main()