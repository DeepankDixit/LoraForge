# The LoraForge Makefile — What It Is and How to Use It

---

## What a Makefile Is

A Makefile is a **shortcut system for the terminal**. Nothing more.

Instead of typing a long, complex command every time you want to do something — like running tests, formatting code, or launching an activity — you define a short name for that command in the Makefile. Then you just type `make <name>` and Make runs the full command for you.

**Example:**
```
Without Makefile:
  pytest tests/ -v -x --ignore=tests/slow -m "not gpu and not slow"

With Makefile:
  make test
```

Both do the exact same thing. The Makefile is just saving you from memorising and typing the long version.

Make is not a programming language like Python. It is a **build automation tool** — originally designed in 1976 for compiling C code, but now used in almost every kind of software project because the "shortcut" pattern is universally useful. The syntax in a Makefile is minimal: you define a target name, then the command(s) that run when you invoke that target.

---

## Makefile Syntax — The Three Things You Need to Know

```makefile
# 1. A comment starts with #

# 2. A target looks like this:
target-name:
	command to run       ← IMPORTANT: this indent is a TAB, not spaces

# 3. A target can depend on another target:
full-pipeline: activity1 activity2 activity3
	@echo "All done"
```

The indentation before each command **must be a real TAB character**, not spaces. This is the one quirk of Makefile syntax that trips people up. VS Code's Makefile extension enforces this automatically.

The `@echo` prefix means "print this message to the terminal but don't print the echo command itself." Without `@`, Make prints the command AND its output. With `@`, it only prints the output.

---

## The `.PHONY` Declaration

At the top of the LoraForge Makefile you see:
```makefile
.PHONY: install test test-all lint format typecheck clean \
        activity1 activity2 ...
```

Originally, Make was designed for building files. If you had a target called `test`, Make would look for a file named `test` on disk. If that file existed, Make would think "nothing to do" and skip running the command.

`.PHONY` tells Make: "these targets are not file names — always run them." Since none of our targets actually build a file called `test` or `activity1`, they must all be declared `.PHONY`.

---

## Every Target in the LoraForge Makefile

### Setup

**`make install`**
```makefile
pip install -e ".[dev]"
pip install -e ".[modelopt]"
pre-commit install
```
Installs the LoraForge Python package in **editable mode** (`-e`), meaning changes you make to the source code immediately take effect without reinstalling. The `[dev]` part installs development dependencies (pytest, black, ruff, mypy). The `[modelopt]` part installs NVIDIA ModelOpt for quantization (Activity 2). `pre-commit install` sets up git hooks that automatically run linting before every commit.

Run this once when you first set up the project on a new machine.

---

### Code Quality

These targets have nothing to do with ML or GPU work. They are about keeping the Python code clean and correct.

**`make test`**
```makefile
pytest tests/ -v -x --ignore=tests/slow -m "not gpu and not slow"
```
Runs the fast unit tests. The flags mean:
- `-v` = verbose output (show each test name)
- `-x` = stop immediately on first failure (don't run 50 more tests after one breaks)
- `--ignore=tests/slow` = skip the slow test directory
- `-m "not gpu and not slow"` = skip any test marked `@pytest.mark.gpu` or `@pytest.mark.slow`

This is the test command you run constantly during development — it finishes in seconds.

**`make test-all`**
```makefile
pytest tests/ -v --cov=. --cov-report=term-missing
```
Runs ALL tests including slow ones. The `--cov` flags add code coverage reporting — it shows you exactly which lines of code were executed during the tests and which were not. You run this before submitting a pull request or publishing a release, not constantly during development.

**`make lint`**
```makefile
ruff check .
```
Ruff is a very fast Python linter. A **linter** reads your code without running it and flags problems: unused imports, variables that shadow built-ins, calling a function with the wrong number of arguments, using deprecated APIs, and dozens of other categories. It does not change any files — it only reports problems.

**`make format`**
```makefile
black . && isort .
```
Black is a code formatter — it automatically rewrites your Python files to follow a consistent style (consistent spacing, line length, quote style, etc.). isort specifically sorts your import statements alphabetically and groups them (standard library → third party → local). Unlike a linter, these commands **actually modify your files**. Run this before committing code.

**`make typecheck`**
```makefile
mypy . --ignore-missing-imports
```
mypy is a static type checker. It reads your Python type annotations (the `def foo(x: int) -> str:` syntax) and verifies that you are not passing the wrong types anywhere — without running the code. For example, if a function expects a `str` and you pass an `int`, mypy catches it. The `--ignore-missing-imports` flag tells mypy not to complain about third-party libraries that don't have type stubs.

---

### Activities

These are the ML pipeline commands. Each one runs a specific activity's main entry point.

**`make activity1`**
```makefile
python activity1_baseline/main.py --config activity1_baseline/config.yaml
```
Runs the complete Activity 1 pipeline: merge the LoRA adapter → launch vLLM → benchmark TTFT and throughput → evaluate perplexity → generate baseline report. Requires a GPU with ≥24GB VRAM. Takes 60–120 minutes end-to-end.

**`make activity2`**
```makefile
python activity2_quantization/main.py --config activity2_quantization/config.yaml
```
Runs the quantization pipeline: applies FP8, INT4 AWQ, and INT8 SmoothQuant to the merged model → benchmarks each → selects the best format for the configured accuracy budget. Requires NVIDIA ModelOpt.

**`make activity3`**
```makefile
python activity3_kv_cache/main.py --config activity3_kv_cache/config.yaml
```
Runs KV cache optimizations: enables prefix caching, tests quantized KV cache, benchmarks TTFT reduction across varying shared prefix lengths.

**`make activity4-train`**
```makefile
python activity4_speculative_decoding/main.py --config ... --mode train
```
Trains the EAGLE draft model. This is the long step — training a small auxiliary model for 3 epochs on ~70K examples. Takes several hours. You only do this once.

**`make activity4-bench`**
```makefile
python activity4_speculative_decoding/main.py --config ... --mode benchmark
```
Runs speculative decoding benchmarks using the already-trained EAGLE draft model. Measures acceptance rate and generation speedup across temperatures and batch sizes.

**`make activity5-dashboard`**
```makefile
streamlit run activity5_benchmark_dashboard/reporting/dashboard/app.py --server.port 8501
```
Launches the Streamlit dashboard. After running Activities 1–4, open your browser at `http://localhost:8501` to see the interactive comparison charts, Pareto curves, and cost estimator. This runs a web server locally — Ctrl+C to stop it.

**`make activity6`**
```makefile
python activity6_domain_cpt/main.py --config activity6_domain_cpt/config.yaml
```
Runs the optional domain-adaptive Continued Pre-Training pipeline. Run this before Activity 0 if you want to try the full CPT → SFT → inference optimization pipeline.

---

### The Full Pipeline

**`make full-pipeline`**
```makefile
$(MAKE) activity1
$(MAKE) activity2
$(MAKE) activity3
$(MAKE) activity4-bench
@echo "Pipeline complete. Run 'make activity5-dashboard' to view results."
```
Runs Activities 1 → 2 → 3 → 4 (benchmark only, skips training) in sequence. `$(MAKE)` is how a Makefile calls another target inside itself. This is the "run everything" command for a demo or final benchmark. Activity 5 is excluded because it is an interactive dashboard — you launch it separately.

---

### Cleanup

**`make clean`**
```makefile
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type d -name "*.egg-info" -exec rm -rf {} +
find . -name "*.pyc" -delete
```
Deletes all auto-generated Python cache files and build artifacts. These are created automatically when Python runs and are completely safe to delete — Python regenerates them the next time the code runs. Clean when your IDE is showing stale errors, or before packaging a release.

---

## What Makefile Is NOT

It is **not** the list of ML activities in the LoraForge sense. The ML activities (0 through 6) are the experiments and implementations. The Makefile is just the keyboard shortcut panel for running them. The two things happen to use the same word "activity" because the Makefile targets are named after the activities they run.

It is **not** a package manager like pip. It does not install anything by itself — the `make install` target happens to call `pip install`, but that is just a shell command that Make is executing. Make itself knows nothing about Python packages.

It is **not** required to run the project. Every `make <target>` call is equivalent to typing the underlying command directly. The Makefile just saves you from having to remember and type those commands.

---

## Quick Reference Card

| Command | What it does | When to run it |
|---|---|---|
| `make install` | Install all Python dependencies | Once, on project setup |
| `make test` | Fast unit tests (no GPU, no slow tests) | Constantly during development |
| `make test-all` | All tests with coverage report | Before committing / releasing |
| `make lint` | Check code for problems (read-only) | Before committing |
| `make format` | Auto-format code (modifies files) | Before committing |
| `make typecheck` | Check type annotations | Before committing |
| `make activity1` | Run Activity 1 end-to-end | On GPU machine, after Activity 0 |
| `make activity2` | Run Activity 2 (quantization) | On GPU machine, after Activity 1 |
| `make activity3` | Run Activity 3 (KV cache) | On GPU machine, after Activity 2 |
| `make activity4-train` | Train EAGLE draft model | On GPU machine, once |
| `make activity4-bench` | Benchmark speculative decoding | On GPU machine, after training |
| `make activity5-dashboard` | Launch results dashboard | On any machine with results JSON |
| `make activity6` | Run optional CPT pipeline | On GPU machine, before Activity 0 |
| `make full-pipeline` | Run Activities 1–4 in sequence | Full demo run |
| `make clean` | Delete Python cache files | When IDE shows stale errors |
