#!/usr/bin/env python3
"""Fast, dependency-free, five-dimensional BBOB-style objectives."""

import argparse
import math
from collections.abc import Callable

Point = tuple[float, ...]
Objective = Callable[[Point], float]
DIMENSION = 5
SEARCH_BOUND = 5.0
DEFAULT_POINT = (-1.2, 1.0, 1.0, 1.0, 1.0)


def sphere(point: Point) -> float:
    return sum(value**2 for value in point)


def ellipsoid(point: Point) -> float:
    return sum(
        1_000_000.0 ** (index / (len(point) - 1)) * value**2
        for index, value in enumerate(point)
    )


def rosenbrock(point: Point) -> float:
    return sum(
        (1.0 - current) ** 2 + 100.0 * (following - current**2) ** 2
        for current, following in zip(point, point[1:])
    )


def rastrigin(point: Point) -> float:
    return 10.0 * len(point) + sum(
        value**2 - 10.0 * math.cos(2.0 * math.pi * value) for value in point
    )


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
    for index, default in enumerate(DEFAULT_POINT, start=1):
        parser.add_argument(f"--x{index}", type=bounded_coordinate, default=default)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    point = tuple(getattr(args, f"x{index}") for index in range(1, DIMENSION + 1))
    metric = OBJECTIVES[args.function](point)
    print(f"autotune_metric={metric:.17g}")


if __name__ == "__main__":
    main()
