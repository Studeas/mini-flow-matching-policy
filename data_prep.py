from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import zarr

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "pusht.npz"
HORIZON = 16


def load_data():
    if not DATA_FILE.exists():
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        url = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
        print("Downloading PushT demonstrations...", flush=True)

        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "pusht.zip"
            urllib.request.urlretrieve(url, archive)

            with zarr.ZipStore(str(archive), mode="r") as store:
                data = zarr.open_group(
                    store,
                    path="pusht/pusht_cchi_v7_replay.zarr",
                    mode="r",
                )
                np.savez_compressed(
                    DATA_FILE,
                    states=data["data/state"][:],
                    actions=data["data/action"][:],
                    ends=data["meta/episode_ends"][:],
                )

    with np.load(DATA_FILE) as data:
        states = data["states"]
        actions = data["actions"]
        ends = data["ends"]

    train_ids = np.random.default_rng(0).permutation(len(ends))[:185]
    return states, actions, ends, train_ids


def encode_obs(state):
    state = np.asarray(state, dtype=np.float32)

    return np.concatenate(
        [
            state[..., :4],
            np.sin(state[..., 4:5]),
            np.cos(state[..., 4:5]),
        ],
        axis=-1,
    )


def make_dataloaders(batch_size=256):
    states, actions, ends, train_ids = load_data()
    starts = np.concatenate(([0], ends[:-1]))
    train_obs, train_actions, val_obs, val_actions = [], [], [], []

    for episode in range(len(ends)):
        start, end = int(starts[episode]), int(ends[episode])
        for t in range(start, end):
            indices = np.minimum(np.arange(t, t + HORIZON), end - 1)

            if episode in train_ids:
                train_obs.append(encode_obs(states[t]))
                train_actions.append(actions[indices])
            else:
                val_obs.append(encode_obs(states[t]))
                val_actions.append(actions[indices])

    train_obs = torch.tensor(np.array(train_obs))
    train_actions = torch.tensor(np.array(train_actions))
    val_obs = torch.tensor(np.array(val_obs))
    val_actions = torch.tensor(np.array(val_actions))

    stats = {
        "obs_mean": train_obs.mean(0),
        "obs_std": train_obs.std(0).clamp_min(1e-6),
        "action_mean": train_actions.mean((0, 1)),
        "action_std": train_actions.std((0, 1)).clamp_min(1e-6),
    }

    train = TensorDataset(
        (train_obs - stats["obs_mean"]) / stats["obs_std"],
        (train_actions - stats["action_mean"]) / stats["action_std"],
    )
    val = TensorDataset(
        (val_obs - stats["obs_mean"]) / stats["obs_std"],
        (val_actions - stats["action_mean"]) / stats["action_std"],
    )

    return (
        DataLoader(train, batch_size, shuffle=True),
        DataLoader(val, batch_size),
        stats,
    )
