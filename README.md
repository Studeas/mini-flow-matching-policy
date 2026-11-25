# Mini Flow Matching · PushT

A small example of conditional flow matching for PushT. An MLP learns to turn
noise into a sequence of actions, given the current state of the pusher and the
T-shaped block. The model uses state arrays, with no images or observation history.

## Code

```text
mini-flow-matching/
├── mini-flow-matching.py   # Model, flow matching loss, sampling, and training
├── data_prep.py            # Data download, episode split, and DataLoader
├── visualize.py            # Loss curves, coverage evaluation, and videos
├── pyproject.toml          # Dependencies
└── uv.lock                 # Fixed dependency versions
```

The core training step interpolates between noise and demonstrated actions:

```python
x = (1 - t) * noise + t * actions
loss = mse(model(x, tau, obs), actions - noise)
```

Here, `tau` is the time along the flow from noise to actions, not the robot's
physical time.

## Running

The project uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run mini-flow-matching.py
```

The default run trains on CPU for 600 epochs, then saves a model, plots, and two
videos in `results/`. A short run uses `--epochs 2`; `--eval-only` evaluates the
saved model without training. Each training run starts from scratch and replaces
previous outputs.

## Data

The first run downloads the PushT demonstrations and keeps a small array cache
in `data/pusht.npz`. The 206 episodes are split into 185 for training and 21 for
validation. Normalization uses only the training data.

```text
obs      [B, 6]       agent xy, block xy, sin(angle), cos(angle)
actions  [B, 16, 2]   the next 16 target xy positions for the pusher
```

Actions are absolute positions in environment pixels. Sequences stay within one
episode; short sequences at the end repeat the final action.

## Results

`loss.png` shows training and validation loss. `coverage.png` and `coverage.json`
compare the original demonstrations, the policy starting from training poses,
and the policy starting from 50 random test poses. `train.mp4` and `test.mp4`
show the first episode from each policy group, without selecting by outcome.

Coverage is the area shared by the block and the target, divided by the target
area. The reported value is the **mean of each episode's maximum coverage**.
The policy runs for 300 steps, observing again after every 4 actions. Recorded
demonstrations have different lengths, so this is not an equal-duration
comparison. Performance from training poses does not measure generalization.

## Sources

- [PushT environment](https://github.com/huggingface/gym-pusht)
- [Diffusion Policy demonstrations](https://diffusion-policy.cs.columbia.edu/)

Generated data, results, and the local environment are excluded from Git.
