"""
Entry point for training the Neural SDE calibration model.

    python train.py --date 2021-10-01 --maturities maturities_monthly.json --epochs 10000
    python train.py --weighting dynamic --sigma2 0.5
    python train.py --warm-start checkpoints/<run>/<file>.ckpt --name rolling_2021-10-08

Checkpoints, smiles and a metrics.csv of loss, weights, coverage and
deviation are written to checkpoints/<name>/ every --save-step epochs. A run
resumes from its own latest checkpoint unless --warm-start is given.
"""
import argparse
import json
import time
import torch
import checkpoint
import my_model
from data import load_dataset
from generator import GeneratorIto
from train_utils import train, wrap_loss_func, prepare_data, checkpoint_func


def parse_args():
    p = argparse.ArgumentParser(description="SPX-VIX joint Neural SDE calibration")
    p.add_argument("--date", default="2021-10-01", help="calibration date, YYYY-MM-DD")
    p.add_argument("--maturities", default="maturities_monthly.json")
    p.add_argument("--params", default="params.json", help="training hyperparameters")
    p.add_argument("--epochs", type=int, default=None, help="stop after this many epochs")
    p.add_argument("--save-step", type=int, default=100)
    p.add_argument("--name", default=None, help="run name, used for the checkpoint folder")
    p.add_argument("--weighting", choices=["fixed", "dynamic"], default=None)
    p.add_argument("--sigma2", type=float, default=None)
    p.add_argument("--warm-start", default=None, help="checkpoint to initialise from")
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.maturities) as f:
        maturities = json.load(f)
    with open(args.params) as f:
        params = json.load(f)
    for key in ("weighting", "sigma2", "batch_size"):
        if getattr(args, key) is not None:
            params[key] = getattr(args, key)
    if args.name:
        my_model.MODEL["name"] = args.name

    start = time.perf_counter()
    data = load_dataset(args.date, maturities)
    print(f"Data loaded ({time.perf_counter() - start:.1f}s)")

    smiles, maturities, fSPX = prepare_data(data, args.date, maturities)
    dt = torch.tensor(params["dt"] / 365)
    gen = GeneratorIto(dt, my_model.V_AND_MUY(my_model.MODEL), fSPX)
    if args.warm_start:
        model, epoch = checkpoint.warm_start(args.warm_start, my_model.MODEL)
    else:
        model, epoch = checkpoint.load_last_checkpoint(my_model.MODEL)

    _, T0, loss_func = wrap_loss_func(args.date, params, smiles, maturities, gen, dt, fSPX)
    print(f"Setup complete ({time.perf_counter() - start:.1f}s). "
          f"Weighting: {params.get('weighting', 'fixed')}. Starting at epoch {epoch}.")

    train(model, epoch, loss_func, checkpoint_func, save_step=args.save_step,
          max_epochs=args.epochs)


if __name__ == "__main__":
    main()
