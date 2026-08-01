# Autotune Python SDK

Typed synchronous and asynchronous interfaces for the `autotune` command-line tool. The CLI remains the execution engine and the authority for option validation.

## Requirements

- Python 3.10 or newer
- Node.js 22 or newer
- An `autotune` executable on `PATH`

Select another executable with `binary=...`, `AUTOTUNE_CLI_BIN`, or `AUTOTUNE_BIN`.

## Usage

```python
from autotune_cli import Autotune

autotune = Autotune()
space = autotune.analyze("train.py", agent="codex")
result = autotune.run("train.py", trials=20, yes=True, agent="codex")
print(result.best_trial)
```

`run()` is intentionally noninteractive: pass `yes=True`, or provide `config="search-space.yaml"`. A configured run is automatically passed to the CLI with `--yes`.

Use `invoke()` for raw CLI features:

```python
print(autotune.invoke(["--help"]).stdout)
```

The async client has the same API:

```python
from autotune_cli import AsyncAutotune

result = await AsyncAutotune().run("train.py", trials=20, yes=True)
```
