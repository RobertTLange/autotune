#!/usr/bin/env python3

x = 0.0
penalty = 0.05

score = 1.0 - ((x - 0.7) ** 2) - penalty
print(f"autotune_metric={score}")
