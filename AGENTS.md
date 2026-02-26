# Liger Kernel — Agent Instructions

## Cursor Cloud specific instructions

### Overview

Liger Kernel is a Python library of high-performance Triton GPU kernels for LLM training. It is a single Python package (not a multi-service app). Source code lives in `src/liger_kernel/`, tests in `test/`.

### Key commands

| Task | Command | Notes |
|------|---------|-------|
| Install dev deps | `pip install -e ".[dev]"` | Installs torch, triton, transformers, ruff, pytest, etc. |
| Lint / format | `make checkstyle` | Runs `ruff check` + `ruff format` |
| Pre-commit hooks | `prek install` then `prek run -a` | Uses [prek](https://prek.j178.dev/) (Rust-based pre-commit alternative) |
| Unit tests | `make test` | **Requires GPU** — runs pytest on `test/` (ignoring convergence) |
| Convergence tests | `make test-convergence` | **Requires GPU** — runs mini-model convergence checks |
| Single test | `python -m pytest test/path/test_file.py::test_name` | Standard pytest invocation |
| Benchmarks | `make run-benchmarks` | **Requires GPU** |
| Docs server | `make serve` | MkDocs dev server |

### GPU requirement

All Triton kernel forward/backward passes require a GPU (CUDA, ROCm, XPU, or NPU). In a CPU-only Cloud Agent VM:

- **Works:** `make checkstyle`, `prek run -a`, module imports, module instantiation, monkey-patching API calls.
- **Does not work:** `make test`, `make test-convergence`, `make run-benchmarks`, or any code that actually invokes Triton kernels (forward pass).

### Gotchas

- If `prek install` fails with a `core.hooksPath` error, run `git config --unset-all --local core.hooksPath` and `git config --unset-all --global core.hooksPath` first.
- Scripts installed by pip (`ruff`, `pytest`, `prek`, etc.) go to `~/.local/bin`. Ensure it is on `PATH`.
- `setup.py` auto-detects the GPU platform. On a CPU-only machine it falls back to CUDA deps (torch + triton for CPU).
