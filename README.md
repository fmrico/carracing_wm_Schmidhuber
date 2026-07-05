# CarRacing World Model Experiment

This directory contains a reproducible CarRacing world-model pipeline inspired by the classic VAE + recurrent dynamics + controller setup.

It covers:

- random rollout collection from `CarRacing-v3`
- VAE training on resized observations
- latent dataset encoding
- latent dynamics training with either a deterministic LSTM baseline or an MDN-RNN world model
- qualitative inspection scripts for the learned models
- controller optimization with CMA-ES
- playback of the final agent

## Directory Layout

- `data/random_rollouts/`: raw random trajectories collected from the environment
- `data/encoded_rollouts/`: VAE-encoded latent trajectories
- `models/`: saved checkpoints for the VAE, LSTM, MDN-RNN, and controller
- `s_1_collect_data.py` to `s_7_play_world_model_agent.py`: the main experiment pipeline

Generated datasets are ignored by Git through `.gitignore`, while the folder structure is preserved with `.gitkeep` files.

## Installation

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install the local package plus the runtime dependencies used by this experiment:

```bash
pip install -e .
pip install torch numpy matplotlib pillow tqdm cma "gymnasium[box2d]"
```

Notes:

- `pip install -e .` installs the local `gymnasium_env` package defined in `pyproject.toml`.
- `torch` installation may need a CUDA-specific command depending on your system. If so, install PyTorch first from the official instructions and then run the remaining packages.
- All major training scripts default to `--device auto`, which uses CUDA when available.

## Reproducing the Experiment

All commands below assume you are running them from this directory:

```bash
cd carracing_wm_Schmidhuber
```

### 1. Collect Random Rollouts

This creates compressed rollout files in `data/random_rollouts/`.

```bash
python s_1_collect_data.py --episodes 1000 --seed 0
```

Useful options:

- `--episodes`: number of random episodes to collect
- `--render`: show the environment while collecting
- `--out-dir`: choose a different dataset directory

Expected output:

- `data/random_rollouts/rollout_00000.npz`, `rollout_00001.npz`, ...

### 2. Train the VAE

The VAE learns a compact latent representation from the resized `64x64` RGB observations.

```bash
python s_2_train_vae.py \
	--data-dir data/random_rollouts \
	--save-path models/vae.pt \
	--epochs 10 \
	--batch-size 256
```

Useful options:

- `--z-dim`: latent size, default `32`
- `--num-workers`: data loader workers
- `--log-every`: progress frequency during training
- `--dataset-log-every`: progress frequency while scanning rollout files

Expected output:

- `models/vae.pt`

### 2.1. Inspect the VAE Reconstruction

This is an optional sanity check before moving on.

```bash
python s_2_1_check_vae.py \
	--vae-path models/vae.pt \
	--rollout data/random_rollouts/rollout_00000.npz \
	--frame 0
```

This opens a Matplotlib window with the original frame, the reconstruction, and prints the latent vector.

### 3. Encode the Rollout Dataset into Latents

This runs the trained VAE over every collected rollout and stores latent sequences in `data/encoded_rollouts/`.

```bash
python s_3_encode_dataset.py \
	--rollout-dir data/random_rollouts \
	--vae-path models/vae.pt \
	--out-dir data/encoded_rollouts \
	--batch-size 512
```

Expected output:

- `data/encoded_rollouts/rollout_00000.npz`, `rollout_00001.npz`, ...

Each encoded rollout stores at least:

- `z`: latent sequence from the VAE encoder mean
- `actions`: action sequence
- `rewards`
- `dones`

### 4A. Train the Deterministic LSTM Baseline

This step is optional. It is useful as a simpler baseline and for debugging latent dynamics, but it is not the model used by the final controller.

```bash
python s_4a_train_lstm.py \
	--data-dir data/encoded_rollouts \
	--save-path models/lstm.pt \
	--sequence-length 64 \
	--epochs 10
```

Expected output:

- `models/lstm.pt`

### 4B. Train the MDN-RNN World Model

This is the recurrent latent dynamics model used by the controller phase.

```bash
python s_4b_train_mdn_rnn.py \
	--data-dir data/encoded_rollouts \
	--save-path models/mdn_rnn.pt \
	--sequence-length 64 \
	--hidden-dim 256 \
	--num-mixtures 5 \
	--epochs 10 \
	--batch-size 128
```

Useful options:

- `--grad-clip`: gradient clipping threshold
- `--num-workers`: data loader workers
- `--prefetch-factor`: worker prefetch depth
- `--log-every`: progress frequency during training
- `--dataset-log-every`: progress frequency while indexing encoded rollouts

Expected output:

- `models/mdn_rnn.pt`

### 5A. Inspect the Deterministic LSTM Baseline

Optional diagnostic for the baseline model.

```bash
python s_5a_check_lstm.py \
	--model-path models/lstm.pt \
	--data-dir data/encoded_rollouts \
	--rollout 0 \
	--steps 200
```

This plots latent prediction error over time.

### 5B. Inspect One-Step MDN-RNN Predictions

Optional diagnostic for the main world model.

```bash
python s_5b_check_mdn_rnn.py \
	--vae-path models/vae.pt \
	--mdn-rnn-path models/mdn_rnn.pt \
	--raw-rollout data/random_rollouts/rollout_00000.npz \
	--encoded-rollout data/encoded_rollouts/rollout_00000.npz \
	--steps 8
```

This decodes predicted next latents and compares them visually to the real next observations.

### 5C. Roll Out a Dream Trajectory with the MDN-RNN

Optional qualitative check of long-horizon latent imagination.

```bash
python s_5b_dream_mdn_rnn.py \
	--vae-path models/vae.pt \
	--mdn-rnn-path models/mdn_rnn.pt \
	--encoded-rollout data/encoded_rollouts/rollout_00000.npz \
	--start 0 \
	--steps 32 \
	--temperature 1.0
```

This samples successive latent states from the MDN-RNN and decodes them back to images.

### 6. Train the Controller with CMA-ES

This phase keeps the VAE and MDN-RNN fixed and optimizes a linear controller in the latent-plus-hidden-state space.

```bash
python s_6_train_controller_cmaes.py \
	--vae-path models/vae.pt \
	--mdn-rnn-path models/mdn_rnn.pt \
	--save-path models/controller_cmaes.npz \
	--generations 200 \
	--population-size 32 \
	--eval-episodes 4 \
	--max-steps 1000s
```s

Useful options:

- `--sigma`: initial CMA-ES sampling scale
- `--workers`: number of parallel candidate evaluations
- `--resume-cma`: resume the CMA-ES optimizer state from a previous `.cma.pkl` file
- `--render-best`: render the final best controller after optimization

Expected output:

- `models/controller_cmaes.npz`
- `models/controller_cmaes.cma.pkl`

You can stop and continue the controller search later.

First run:

```bash
python s_6_train_controller_cmaes.py \
	--generations 300 \s
	--population-size 32 \
	--eval-episodes 4 \
	--max-steps 1000 \
	--sigma 0.1 \
	--workers 16
```

Continue later from the saved CMA-ES state:

```bash
python s_6_train_controller_cmaes.py \
	--generations 1500 \
	--resume-cma models/controller_cmaes.cma.pkl \
	--workers 16
```

When resuming:

- `--resume-cma` restores the CMA-ES internal optimizer state
- `models/controller_cmaes.npz` is still used to recover the best controller found so far
- `--generations` means additional generations to run from the resumed state

### 7. Run the Trained World-Model Agent

Use the saved controller together with the trained VAE and MDN-RNN to play CarRacing.

```bash
python s_7_play_world_model_agent.py \
	--vae-path models/vae.pt \
	--mdn-rnn-path models/mdn_rnn.pt \
	--controller-path models/controller_cmaes.npz \
	--episodes 3 \
	--render
```

This should load all trained components and run the final policy in the environment.

## Minimal Reproduction Sequence

If you only want the main experiment without optional checks, run these commands in order:

```bash
python s_1_collect_data.py --episodes 1000 --seed 0
python s_2_train_vae.py --data-dir data/random_rollouts --save-path models/vae.pt
python s_3_encode_dataset.py --rollout-dir data/random_rollouts --vae-path models/vae.pt --out-dir data/encoded_rollouts
python s_4b_train_mdn_rnn.py --data-dir data/encoded_rollouts --save-path models/mdn_rnn.pt
python s_6_train_controller_cmaes.py --vae-path models/vae.pt --mdn-rnn-path models/mdn_rnn.pt --save-path models/controller_cmaes.npz
python s_7_play_world_model_agent.py --vae-path models/vae.pt --mdn-rnn-path models/mdn_rnn.pt --controller-path models/controller_cmaes.npz --render
```
