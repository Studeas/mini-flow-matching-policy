import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_prep import ROOT, HORIZON, make_dataloaders
from visualize import plot_loss, evaluate

OUTPUT = ROOT / "results"


class FlowPolicy(nn.Module):
    def __init__(
        self,
        obs_dim=6,
        action_dim=2,
        horizon=HORIZON,
        hidden_dim=512,
        time_dim=32,
    ):
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.time_dim = time_dim

        chunk_dim = horizon * action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim + chunk_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, chunk_dim),
        )

    def time_embedding(self, tau):
        frequency = torch.exp(
            torch.linspace(0, math.log(1000), self.time_dim // 2)
        )
        phase = tau[:, None] * frequency[None, :]

        return torch.cat([phase.sin(), phase.cos()], dim=-1)

    def forward(self, x, tau, obs):
        batch = x.shape[0]
        inputs = torch.cat(
            [x.reshape(batch, -1), self.time_embedding(tau), obs],
            dim=-1,
        )

        return self.net(inputs).reshape(batch, self.horizon, self.action_dim)


def flow_matching_loss(model, obs, actions):
    # sample Gaussian noise
    noise = torch.randn_like(actions)

    # sample flow time
    tau = torch.rand(len(actions))
    t = tau[:, None, None]

    # construct interpolated actions
    x = (1 - t) * noise + t * actions

    # compute target velocity
    target = actions - noise

    # predict velocity
    prediction = model(x, tau, obs)

    # compute loss
    return F.mse_loss(prediction, target)


@torch.no_grad()
def sample_actions(model, obs, num_steps=20):
    x = torch.randn(len(obs), model.horizon, model.action_dim)
    dt = 1.0 / num_steps

    for step in range(num_steps):
        tau = torch.full((len(obs),), step * dt)
        x = x + dt * model(x, tau, obs)

    return x


@torch.no_grad()
def validation_loss(model, loader):
    model.eval()
    total = 0.0

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)

        for obs, actions in loader:
            total += flow_matching_loss(model, obs, actions).item() * len(obs)

    return total / len(loader.dataset)


def train(epochs):
    torch.manual_seed(0)
    train_loader, val_loader, stats = make_dataloaders()

    model = FlowPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0

        for obs, actions in train_loader:
            loss = flow_matching_loss(model, obs, actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total += loss.item() * len(obs)

        if epoch % 10 == 0 or epoch == epochs:
            train_loss = total / len(train_loader.dataset)
            val_loss = validation_loss(model, val_loader)

            history.append([epoch, train_loss, val_loss])
            print(
                f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}",
                flush=True,
            )
            plot_loss(history, OUTPUT)

    torch.save(
        {"model": model.state_dict(), "stats": stats},
        OUTPUT / "model.pt",
    )
    return model, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be positive")

    torch.set_num_threads(4)
    OUTPUT.mkdir(exist_ok=True)

    if args.eval_only:
        checkpoint = torch.load(OUTPUT / "model.pt", weights_only=True)
        model = FlowPolicy()
        model.load_state_dict(checkpoint["model"])
        stats = checkpoint["stats"]
    else:
        model, stats = train(args.epochs)

    evaluate(model, stats, sample_actions, OUTPUT)
