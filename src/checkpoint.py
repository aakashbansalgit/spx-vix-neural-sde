import os
import datetime
import glob
import torch
import json
from smile import Smile
from config import DEVICE


class Encoder(json.JSONEncoder):
    """JSON encoder that handles Smile objects."""
    def default(self, obj):
        if isinstance(obj, Smile):
            return obj.to_json()
        return super().default(obj)


def save_checkpoint(epoch, loss, model):
    """Save model weights and optimizer state to a timestamped .ckpt file."""
    name = model["name"]
    state = {
        "loss": loss.item(),
        "epoch": epoch,
        **{k: m.state_dict() for k, m in model["nets"].items()},
        "optimizer": model["optimizer"].state_dict(),
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")
    path = f"checkpoints/{name}/epoch{epoch}-{timestamp}-{state['loss']:.5f}.ckpt"
    torch.save(state, path)
    print(f"Saved {path}  (epoch {epoch}, loss {state['loss']:.4f})")


def load(path, model):
    """Load a checkpoint into model, initialising the optimizer if needed."""
    print(f"Loading {path}")
    state = torch.load(path, map_location=DEVICE)
    for k, m in state.items():
        if k not in ("loss", "epoch", "optimizer"):
            model["nets"][k].load_state_dict(m)
    if model["optimizer"] is None:
        model["optimizer"] = torch.optim.Adam(params_of_model(model), lr=0.001)
    model["optimizer"].load_state_dict(state["optimizer"])
    print(f"Loaded {path}  (epoch {state['epoch']}, loss {state['loss']:.4f})")
    return model, state["epoch"] + 1


def params_of_model(model):
    """Return a flat list of all trainable parameters across all networks in model."""
    params = []
    for net in model["nets"].values():
        params += list(net.parameters())
    return params


def load_last_checkpoint(model):
    """Load the most recently modified checkpoint, or start from epoch 1 if none exist."""
    name = model["name"]
    checkpoints = sorted(glob.glob(f"checkpoints/{name}/*.ckpt"), key=os.path.getmtime)
    model["optimizer"] = torch.optim.Adam(params_of_model(model), lr=0.001)
    if not checkpoints:
        return model, 1
    return load(checkpoints[-1], model)


def warm_start(path, model):
    """Initialise the networks from another run's checkpoint and start a fresh
    optimizer at epoch 1.

    This is how a new calibration reuses an old one. With an expanding window
    the previous run covered fewer maturities; with a rolling window it covered
    an earlier date. Either way the starting surface is already close, so the
    fit needs far fewer epochs than a cold start."""
    state = torch.load(path, map_location=DEVICE)
    for k, m in state.items():
        if k not in ("loss", "epoch", "optimizer"):
            model["nets"][k].load_state_dict(m)
    model["optimizer"] = torch.optim.Adam(params_of_model(model), lr=0.001)
    print(f"Warm start from {path}  (its epoch {state['epoch']}, loss {state['loss']:.4f})")
    return model, 1
