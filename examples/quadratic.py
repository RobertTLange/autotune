#!/usr/bin/env python3
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, default=0.0)
    args = parser.parse_args()
    score = 1.0 - ((args.x - 0.7) ** 2)
    print(f"autotune_metric={score}")


if __name__ == "__main__":
    main()
