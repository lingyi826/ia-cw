# Intelligent Agents Coursework

This repository contains the rule-based agent, tabular Q-learning agent and
reproducible experiments for the Eregion Works environment. This README covers
execution and reproduction. The report explains the normative specification,
agent design, experimental findings and limitations.

## Contents

1. [Requirements and setup](#1-requirements-and-setup)
2. [Repository files](#2-repository-files)
3. [Running the agents](#3-running-the-agents)
4. [Experiment configurations](#4-experiment-configurations)
5. [Reproducing the evaluation](#5-reproducing-the-evaluation)
6. [Reproducing training and saving Q-values](#6-reproducing-training-and-saving-q-values)
7. [Generating figures and tables](#7-generating-figures-and-tables)
8. [Output definitions and verification](#8-output-definitions-and-verification)
9. [Troubleshooting](#9-troubleshooting)

## 1. Requirements and setup

Run all commands from the **repository root**, containing `README.md`,
`pyproject.toml`, `examples/` and `src/`.

Requirements are Python **3.11 or newer** and **PyYAML**. The project selects
Python 3.11 in `.python-version` and locks PyYAML to 6.0.3. The commands have
been checked with Python 3.11.18. All experiments run on a CPU; the tabular
learner and SVG visualiser require no PyTorch, NumPy, pandas or Matplotlib.

If the existing environment already runs the project, use it directly. Otherwise,
choose one setup route.

**With uv:**

```bash
uv sync --locked
```

Prefix subsequent Python commands with `uv run`, for example:

```bash
uv run python -m examples.evaluate --episodes 5 --seed 7
```

**With Python and pip, on Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install PyYAML==6.0.3
```

Use `.\.venv\Scripts\python.exe` in place of `python` in subsequent commands.
On macOS/Linux, create the environment with `python3 -m venv .venv`, then use
`.venv/bin/python` to install the same dependency and run the modules.

The commands below use `python` for readability. Use the same environment
interpreter throughout. Check it with:

```bash
python -c "import sys, yaml; print(sys.executable); print(yaml.__version__)"
```

Installation may require internet access. Training, evaluation and plotting use
local files after setup. No `evaluation` extra is defined or required.

## 2. Repository files

| File or directory | Purpose |
|---|---|
| `examples/rulebased.py` | Task 2 decision rules in `RuleBasedAgent.decide()` |
| `examples/qlearning.py` | Task 3 action selection, Q-value updates and JSON save/load |
| `examples/random.py` | Seeded random-action baseline |
| `examples/learning.py` | Observation feature encoder |
| `examples/demo.py` | Agent runs with step-by-step JSON output |
| `examples/train_qlearning.py` | Training and saving per-mode Q-tables |
| `examples/evaluate.py` | Multi-seed evaluation and metric calculation |
| `examples/visualise.py` | Summary CSV to SVG figure and Markdown table |
| `src/environment.yaml` | Base environment configuration |
| `src/` | Simulator, task lifecycle, actions, rewards and base metrics |
| `artifacts/qvalues/` | Nine supplied Q-table JSON files |
| `results/` | Recorded episode CSV, summary CSV, SVG figure and Markdown table |
| `pyproject.toml`, `uv.lock` | Project dependencies and lockfile |
| `COURSEWORK-SPEC.md` | Coursework requirements |

## 3. Running the agents

Run a rule-based or random episode using the current YAML configuration:

```bash
python -m examples.demo --agent rulebased --episodes 1 --seed 7
python -m examples.demo --agent random --episodes 1 --seed 7
```

Run the coursework's short Q-learning training demonstration:

```bash
python -m examples.demo --agent qlearning --train-episodes 10 --episodes 1 --seed 7
```

Each command assigns the selected policy type to all three workers. The demo
prints a JSON `step` record per joint step and an `episode_summary` after each
evaluation episode. Q-learning trains first and then evaluates without learning.
Demo training remains in memory unless `--save-qvalues-dir` is supplied.

For a quiet run of a **saved policy without retraining**, use:

```bash
python -m examples.evaluate --modes individual --agents qlearning --episodes 1 --seed 7 --output-dir results/qlearning_check
```

Replace `individual` with `simultaneous` or `sequential` to select another mode.
The evaluator loads the corresponding Q-tables automatically.

`examples.demo` reads the YAML task settings directly; it does not select a pure
experiment mode. The current YAML uses cooperation probability `0.5` and mode
`simultaneous`. Use the evaluator for the controlled modes described below.

For a saved-policy demo, provide `--config` pointing to a YAML with matching
mode settings, `--train-episodes 0`, and a mode-specific Q-value directory.
The path conventions are:

| Runner | `--qvalues-dir` | Expected file layout |
|---|---|---|
| `examples.evaluate` | `artifacts/qvalues` | `<root>/<mode>/<worker>.json` |
| `examples.demo` | e.g. `artifacts/qvalues/individual` | `<directory>/<worker>.json` |

## 4. Experiment configurations

Training and evaluation derive three configurations from `src/environment.yaml`
using `scenario_for_mode()` in `examples/train_qlearning.py`:

| Mode | `cooperation_probability` | `cooperative_mode` |
|---|---:|---|
| `individual` | `0.0` | ignored |
| `simultaneous` | `1.0` | `simultaneous` |
| `sequential` | `1.0` | `sequential` |

These two fields are overridden in memory. No YAML edits or three separate YAML
files are needed. All other settings come from the base configuration.

The recorded experiments used:

| Parameter | Value |
|---|---|
| Base seed | `7` |
| Workers / rooms | `3` / `3` |
| Horizon | `240` joint steps |
| Initial cleanliness / accessibility threshold | `20` / `1` per room |
| Work damage / cleaning increase | `2` / `2` |
| Restoration interval | `48` steps |
| Work required | `4` units per participant |
| Deadline | `16` steps per task or sequential phase |
| Spawn / initial task probability | `0.7` / `1.0` |
| Completion reward | `2.0` per participant |
| Work / Clean cost | `0.02` / `0.02` |
| Inaccessible-room cost | `0.1` |
| Stop when all rooms are locked | `true` |

Training, evaluation and demo accept `--config path/to/config.yaml` for an
alternative base configuration. Keep the common settings unchanged when
reproducing the recorded results.

## 5. Reproducing the evaluation

### Full comparison using the supplied Q-values

```bash
python -m examples.evaluate --episodes 5 --seed 7
```

This runs all three policies in all three task modes, with five episodes per
mode–policy pair: **45 episodes and 9 summary groups**. It loads the supplied
Q-tables, disables Q-learning exploration and updates, and performs no training.
Seeded tie-breaking between equally best Q-actions remains enabled.

The evaluation seeds are `1000007` through `1000011` for every group. Training
and evaluation use separate episodes and seeds. For Task 2, use the six random
and rule-based groups. For Task 3, compare all nine groups.

Outputs are:

| File | Contents |
|---|---|
| `results/evaluation_episodes.csv` | One row per episode, including seeds, outcomes and action counts |
| `results/evaluation_summary.csv` | One row per mode–policy pair, with metric means and sample standard deviations |

The terminal prints nine `evaluation_summary` records and a final
`evaluation_files` record containing paths and elapsed time. The full evaluation
was previously measured at **9.141 seconds** on the author's CPU-only Windows
machine, excluding installation and training. Runtime varies by machine; the
coursework limit is ten minutes for evaluation.

The default command replaces existing CSV files in `results/`. To preserve the
recorded outputs, run:

```bash
python -m examples.evaluate --episodes 5 --seed 7 --output-dir results/reproduced
```

### Task 2 only or a selected task mode

Evaluate random and rule-based policies only; this requires no Q-value files:

```bash
python -m examples.evaluate --agents random rulebased --episodes 5 --seed 7 --output-dir results/task2
```

This produces 30 episode rows and six summary rows. To run all three policies in
one task mode:

```bash
python -m examples.evaluate --modes simultaneous --episodes 5 --seed 7 --output-dir results/simultaneous_check
```

`--episodes` sets the episode count **per group**. `--modes` and `--agents` accept
one or more names. `--qvalues-dir` selects the saved-policy root and defaults to
`artifacts/qvalues`. Use separate output directories for different experiments.

## 6. Reproducing training and saving Q-values

The supplied policies are stored at:

```text
artifacts/qvalues/individual/{alice,bob,carol}.json
artifacts/qvalues/simultaneous/{alice,bob,carol}.json
artifacts/qvalues/sequential/{alice,bob,carol}.json
```

There are nine files. Each stores the feature names, transition count, learning
parameters and Q-table entries. Evaluation can load them without retraining.

To reproduce training while retaining the supplied files:

```bash
python -m examples.train_qlearning --episodes 200 --seed 7 --output-dir artifacts/qvalues_retrained
```

The script trains fresh workers for each mode, retaining their tables across
episodes within that mode. It uses training seeds `7` through `206`. Each mode
has 200 episodes and 48,000 joint training steps in the recorded run. Learning
rate is `0.15`, discount factor is `0.95`, and exploration follows the coursework
epsilon schedule in `examples/qlearning.py`.

After each mode, a `training_summary` record gives the elapsed time, step count,
number of Q-entries and saved paths. Recorded CPU-only times were:

| Mode | Episodes | Training steps | Time |
|---|---:|---:|---:|
| Individual | 200 | 48,000 | 33.772 s |
| Simultaneous | 200 | 48,000 | 40.602 s |
| Sequential | 200 | 48,000 | 37.315 s |
| Total | 600 | 144,000 | 111.689 s |

Evaluate the newly trained policies with:

```bash
python -m examples.evaluate --episodes 5 --seed 7 --qvalues-dir artifacts/qvalues_retrained --output-dir results/retrained
```

Omitting the training command's `--output-dir` writes to `artifacts/qvalues` and
replaces the supplied policies. `--modes` can restrict training to selected modes;
subsequent evaluation must select modes for which matching saved files exist.

## 7. Generating figures and tables

After the full evaluation in the default `results/` directory:

```bash
python -m examples.visualise
```

The script reads `results/evaluation_summary.csv` and writes:

- `results/evaluation_comparison.svg`: four grouped bar charts.
- `results/evaluation_summary.md`: the corresponding summary table.

The charts show task completion, normalised cleanliness, N3 Work compliance and
social welfare. Bars represent means and error bars represent sample standard
deviations. The Markdown table rounds values to three decimal places; the CSV
retains the calculation precision.

For a run in a separate directory, specify all output paths:

```bash
python -m examples.visualise --input results/reproduced/evaluation_summary.csv --output results/reproduced/evaluation_comparison.svg --table results/reproduced/evaluation_summary.md
```

For newly trained policies, replace `results/reproduced` with `results/retrained`.
Open the SVG in a browser or an SVG-capable document application, and the table
in a Markdown preview.

The supplied visualiser requires **all nine mode–policy groups**. Task 2-only or
single-mode CSV files can be inspected as tables, but cannot be plotted directly
with this script. Use the full evaluation for the supplied comparison figure.

## 8. Output definitions and verification

Summary columns ending in `_mean` and `_std` contain the mean and **sample
standard deviation of episode-level values** within each group.

| Metric | Definition used by the code |
|---|---|
| `task_completion_rate` | completed / (completed + expired + pending); each cooperative task counted once |
| `tasks_completed_per_step` | completed tasks / actual joint steps |
| `office_cleanliness` | Mean post-step room cleanliness divided by each room's initial cleanliness |
| `room_availability` | Accessible room snapshots / all post-step room snapshots |
| `mean_return` | Average accumulated reward across workers |
| `social_welfare` | Sum of workers' accumulated rewards |
| `norm_compliance_rate` | 1 − violating Work attempts / all Work attempts |

A violating Work attempt is useless (zero progress), unsafe (accepted but its
target room is inaccessible after the joint step), or rejected. An attempt is
counted at most once in `norm_violations`. The compliance rate is `1.0` when no
Work is attempted; it measures the N3 Work prohibition, not all proposed norms.
Detailed counts are retained in the episode CSV and calculated in
`examples.evaluate.evaluate_episode()`.

For the full five-seed evaluation, check these counts on PowerShell:

```powershell
(Get-ChildItem artifacts/qvalues -Recurse -Filter *.json).Count
(Import-Csv results/evaluation_episodes.csv).Count
(Import-Csv results/evaluation_summary.csv).Count
```

Expected counts are **9, 45 and 9**, respectively. Adjust the result paths if
using a different output directory. Each summary group should have five episodes
with seeds `1000007`–`1000011`, and the figure should contain four panels.

Recorded reference outputs are available in `results/`, including
[`evaluation_summary.md`](results/evaluation_summary.md). Compare a reproduced
run with these files rather than copying numeric results into this README.

For reproducibility, retain the same source version, common environment settings,
Q-value files and seeds. Worker RNG seeds use `base_seed * 100 + worker_index`
during training and `evaluation_seed * 100 + worker_index` during evaluation,
with workers in scenario order. Evaluation recreates policies for each episode.
Matched seeds do not guarantee identical task arrivals under different policies,
because their actions alter worker availability and later sampling.

## 9. Troubleshooting

| Issue | Action |
|---|---|
| `python` is not recognised | Use `uv run python` or `.\.venv\Scripts\python.exe` on Windows |
| `No module named yaml` | Install PyYAML into the interpreter being used, or run `uv sync --locked` |
| `No module named examples` or `src` | Run from the repository root using `python -m examples.<module>` |
| uv reports access denied | Close processes holding environment files open; if the existing interpreter works, run it directly without syncing |
| Missing Q-value JSON | Check the nine files and the runner's directory convention in Section 3 |
| Too much demo output | Use `examples.evaluate` for short summaries and CSV output |
| Visualiser reports missing groups | Supply a complete nine-group summary CSV |
| Cannot write CSV | Close the file in Excel or choose a different `--output-dir` |

All supported options are available through:

```bash
python -m examples.demo --help
python -m examples.train_qlearning --help
python -m examples.evaluate --help
python -m examples.visualise --help
```
