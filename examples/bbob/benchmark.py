#!/usr/bin/env python3
"""Fast, dependency-free, two-dimensional BBOB-style objectives."""

import argparse
import math
from collections.abc import Callable

Point = tuple[float, float]
Objective = Callable[[Point], float]
SEARCH_BOUND = 5.0


def sphere(point: Point) -> float:
    x1, x2 = point
    return x1**2 + x2**2


def ellipsoid(point: Point) -> float:
    x1, x2 = point
    return x1**2 + 1_000_000.0 * x2**2


def rosenbrock(point: Point) -> float:
    x1, x2 = point
    return (1.0 - x1) ** 2 + 100.0 * (x2 - x1**2) ** 2


def rastrigin(point: Point) -> float:
    return 20.0 + sum(value**2 - 10.0 * math.cos(2.0 * math.pi * value) for value in point)


OBJECTIVES: dict[str, Objective] = {
    "ellipsoid": ellipsoid,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "sphere": sphere,
}


def bounded_coordinate(raw_value: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(value) or abs(value) > SEARCH_BOUND:
        raise argparse.ArgumentTypeError(f"must be finite and between {-SEARCH_BOUND:g} and {SEARCH_BOUND:g}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", choices=sorted(OBJECTIVES), default="rosenbrock")
    parser.add_argument("--x1", type=bounded_coordinate, default=-1.2)
    parser.add_argument("--x2", type=bounded_coordinate, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric = OBJECTIVES[args.function]((args.x1, args.x2))
    print(f"autotune_metric={metric:.17g}")


if __name__ == "__main__":
    main()
