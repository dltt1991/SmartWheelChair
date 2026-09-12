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
    parser.add_argument("--output", type=Path, help="document expected output directory")
    args = parser.parse_args()
    try:
        from neupan import neupan
    except ImportError as exc:
        raise SystemExit("Install NeuPAN py38 dependencies first: %s" % exc)
    planner = neupan.init_from_yaml(args.config)
    planner.train_dune()
    if args.output:
        print("DUNE checkpoints are written by NeuPAN below", args.output)


if __name__ == "__main__":
    main()
