# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [TBD] - TBD

### Added

- Zero-install npx execution for Autotune and integrity-locked Headless fallback execution.
- Hash-locked, cached Optuna and `cmaes` controller provisioning isolated from training-script dependencies.

### Fixed

- Bounded runner cancellation, output capture, and cleanup for detached or reparented subprocesses.
- Runtime cache recovery, macOS Python detection, Centaur timeout handling, and cross-platform process cleanup.

### Security

- Explicit executable trust, absolute tool resolution, isolated fallback installation, and hash-verified runtime provisioning.
- PID generation and ancestry fencing before descendant or process-group signals.

## [0.1.0] - 2026-07-23

### Added

- Agent-assisted hyperparameter analysis, trial execution, results inspection, progress plotting, and study resumption.
- Optuna samplers, pruning, persistent studies, and iterative agentic search-space refinement.
- Centaur sampler support and runnable BBOB, MNIST, CIFAR-10 speedrun, nanochat, and PID-controller examples.
- GitHub-hosted Codex skill installation through `npx skills`.

### Security

- Private run artifacts, allowlisted agent environments, immutable nanochat data snapshots, and supervised BBOB process cleanup.
- Cross-platform release gates for Node.js 22 and 24 on Linux and macOS.

[TBD]: https://github.com/RobertTLange/autotune/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/RobertTLange/autotune/releases/tag/v0.1.0
