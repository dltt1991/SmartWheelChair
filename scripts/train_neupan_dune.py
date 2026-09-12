#!/usr/bin/env python3
"""Explicitly train a NeuPAN DUNE model for the wheelchair footprint.

This script never trains implicitly from a navigation launch.  Run it in the
Python 3.8 ROS Noetic container after installing the upstream ``py38`` branch.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="NeuPAN DUNE training YAML")
    parser.add_argument("--output", type=Path, required=True, help="new checkpoint directory")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--data-size", type=int, default=100000)
    args = parser.parse_args()
    try:
        from neupan import neupan
    except ImportError as exc:
        raise SystemExit("Install NeuPAN py38 dependencies first: %s" % exc)
    if args.epochs < 1 or args.data_size < 10:
        parser.error("epochs must be positive and data-size >= 10")
    args.output.mkdir(parents=True, exist_ok=False)
    # direct_train disables upstream interactive missing-checkpoint prompts.
    # Use DUNETrain directly to select an explicit output directory instead of
    # upstream's default sys.path[0]/model, which would dirty the repository.
    planner = neupan.init_from_yaml(args.config, device="cpu", train={"direct_train": True})
    from neupan.blocks.dune_train import DUNETrain
    dune = planner.pan.dune_layer
    trainer = DUNETrain(dune.model, dune.G, dune.h, str(args.output.resolve()))
    checkpoint = trainer.start(data_size=args.data_size, data_range=[-5., -5., 5., 5.],
                               epoch=args.epochs, batch_size=256, lr=5e-5,
                               save_freq=max(1, args.epochs), valid_freq=max(1, args.epochs))
    print("Checkpoint:", checkpoint)


if __name__ == "__main__":
    main()
