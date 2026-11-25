import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import gymnasium as gym
import gym_pusht
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_prep import load_data, encode_obs


def plot_loss(history, output):
    values = np.array(history)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(values[:, 0], values[:, 1], label="train", marker=".")
    ax.plot(values[:, 0], values[:, 2], label="validation", marker=".")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Flow matching loss")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output / "loss.png", dpi=150)
    plt.close(fig)


def set_pose(env, state):
    env.agent.position = tuple(state[:2].astype(float))
    env.block.angle = float(state[4])
    env.block.position = tuple(state[2:4].astype(float))
    env.space.reindex_shapes_for_body(env.block)

    return env.get_obs()


def reference_coverage(env, states, ends):
    scores = []
    start = 0
    env.reset(seed=0)

    for end in ends:
        best = 0.0
        for state in states[start:end]:
            set_pose(env, state)
            best = max(best, float(env._get_coverage()))

        scores.append(best)
        start = int(end)

    return scores


@torch.no_grad()
def rollout(
    env,
    model,
    stats,
    sample_actions,
    seed,
    initial_state=None,
    video=None,
):
    torch.manual_seed(seed)
    state, _ = env.reset(seed=seed)
    if initial_state is not None:
        state = set_pose(env, initial_state)

    writer = None
    if video is not None:
        writer = imageio.get_writer(video, fps=10)
        writer.append_data(env.render())

    best = float(env._get_coverage())
    try:
        for step in range(0, 300, 4):
            obs = torch.from_numpy(encode_obs(state)).unsqueeze(0)
            obs = (obs - stats["obs_mean"]) / stats["obs_std"]

            actions = sample_actions(model, obs)[0]
            actions = actions * stats["action_std"] + stats["action_mean"]

            for action in actions[:4]:
                state, _, _, _, info = env.step(action.numpy().clip(0, 512))
                best = max(best, float(info["coverage"]))
                if writer is not None:
                    writer.append_data(env.render())
    finally:
        if writer is not None:
            writer.close()

    return best


def plot_coverage(scores, output):
    names, means = [], []
    for name, values in scores.items():
        names.append(name)
        means.append(float(np.mean(values)))
        print(f"{name}: {means[-1]:.1%} ({len(values)} episodes)", flush=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, means, color=["gray", "steelblue", "seagreen"])
    for i, value in enumerate(means):
        ax.text(i, value + 0.02, f"{value:.1%}", ha="center")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean maximum coverage")

    fig.tight_layout()
    fig.savefig(output / "coverage.png", dpi=150)
    plt.close(fig)

    (output / "coverage.json").write_text(json.dumps(scores, indent=2))


@torch.no_grad()
def evaluate(model, stats, sample_actions, output):
    model.eval()
    states, _, ends, train_ids = load_data()
    starts = np.concatenate(([0], ends[:-1]))

    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="state",
        render_mode="rgb_array",
        visualization_width=256,
        visualization_height=256,
    ).unwrapped
    try:
        scores = {
            "demonstrations": reference_coverage(env, states, ends),
            "train starts": [],
            "test starts": [],
        }

        for i, episode in enumerate(train_ids):
            video = output / "train.mp4" if i == 0 else None
            score = rollout(
                env,
                model,
                stats,
                sample_actions,
                6000 + i,
                states[starts[episode]],
                video,
            )
            scores["train starts"].append(score)

        for i in range(50):
            video = output / "test.mp4" if i == 0 else None
            score = rollout(
                env,
                model,
                stats,
                sample_actions,
                5000 + i,
                video=video,
            )
            scores["test starts"].append(score)
    finally:
        env.close()

    plot_coverage(scores, output)
