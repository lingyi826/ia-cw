# Intelligent Agents Coursework

## 1. Setup

Use Python 3.11 or newer and run all commands from the repository root, which
contains `examples/`, `src/` and `pyproject.toml`. 

With uv, install the project dependencies:

```bash
uv sync
```

The commands below use `python`. When using the uv environment, prefix them with
`uv run`, for example `uv run python -m examples.demo --agent rulebased`.
If a working environment is already active, use `python` directly.

## 2. Running the agents

Run one episode with the random baseline:

```bash
python -m examples.demo --agent random --episodes 1 --seed 7
```

Run one episode with the rule-based agent:

```bash
python -m examples.demo --agent rulebased --episodes 1 --seed 7
```

Train Q-learning for ten episodes and then run one evaluation episode:

```bash
python -m examples.demo --agent qlearning --train-episodes 10 --episodes 1 --seed 7
```

Each command uses the selected policy for all three workers. The demo reads
`src/environment.yaml` directly and prints step records followed by an
`episode_summary`. The ten-episode command is a short demonstration; the saved
policies used in the experiments were trained for 200 episodes per task mode.

## 3. Reproducing the evaluation

The repository includes trained Q-values for Alice, Bob and Carol in each mode:

```text
artifacts/qvalues/individual/{alice,bob,carol}.json
artifacts/qvalues/simultaneous/{alice,bob,carol}.json
artifacts/qvalues/sequential/{alice,bob,carol}.json
```

Evaluate the saved learning policies alongside the random and rule-based agents:

```bash
python -m examples.evaluate --episodes 5 --seed 7 --output-dir results/reproduced
```

This loads the saved Q-values without retraining or updating them. New results
are written to `results/reproduced`, keeping the recorded files in `results/`.

The evaluator runs all three configurations automatically:

| Configuration | `cooperation_probability` | `cooperative_mode` |
|---|---:|---|
| Individual | `0.0` | ignored |
| Simultaneous cooperative | `1.0` | `simultaneous` |
| Sequential cooperative | `1.0` | `sequential` |

These settings are applied in memory; no YAML edits are needed. All other
environment settings are read from `src/environment.yaml` and should remain
unchanged to reproduce the recorded experiment.

`--episodes 5` means five evaluation episodes for each agent in each configuration.
`--seed 7` is the base seed, producing evaluation seeds `1000007` to `1000011`.
The full comparison therefore contains **45 episodes and 9 summary groups**.
For Task 2, compare random with rule-based results. For Task 3, compare Q-learning
with both baselines.

The command produces:

- `results/reproduced/evaluation_episodes.csv`: results for each episode.
- `results/reproduced/evaluation_summary.csv`: means and sample standard
  deviations for each configuration and agent.

The recorded full evaluation took approximately **9 seconds** on my
CPU-only Windows machine, excluding installation and training.

## 4. Reproducing the figures and tables

After running the evaluation above, run:

```bash
python -m examples.visualise --input results/reproduced/evaluation_summary.csv --output results/reproduced/evaluation_comparison.svg --table results/reproduced/evaluation_summary.md
```

This generates:

- `evaluation_comparison.svg`: charts for task completion, office cleanliness,
  N3 Work compliance and social welfare, with sample standard deviation error bars.
- `evaluation_summary.md`: the corresponding table of means and standard deviations.

Both files are saved in `results/reproduced`. Open the SVG in a browser and the
Markdown table in a Markdown preview. The visualiser uses all nine groups from
the full evaluation.

## 5. Reproducing training and saving Q-values

To repeat training, run:

```bash
python -m examples.train_qlearning --episodes 200 --seed 7 --output-dir artifacts/qvalues_retrained
```

This trains a fresh set of agents for each of the three task configurations.
Each configuration uses 200 episodes, training seeds `7` to `206`, and 48,000
joint training steps in the recorded run. Training and evaluation episodes are
separate.

The command saves nine JSON files under `artifacts/qvalues_retrained`, using the
same mode and worker directory layout as the supplied Q-values. This keeps the
original policies in `artifacts/qvalues` available for evaluation without training.

Recorded training times on my CPU-only Windows machine were:

| Configuration | Training time |
|---|---:|
| Individual | 33.772 s |
| Simultaneous | 40.602 s |
| Sequential | 37.315 s |
| Total | 111.689 s |

To evaluate these newly trained policies and generate their figures, run:

```bash
python -m examples.evaluate --episodes 5 --seed 7 --qvalues-dir artifacts/qvalues_retrained --output-dir results/retrained
python -m examples.visualise --input results/retrained/evaluation_summary.csv --output results/retrained/evaluation_comparison.svg --table results/retrained/evaluation_summary.md
```

Use the same code version, base environment settings and seeds when reproducing
the experiment. The recorded data and visualisations are included in `results/`
for comparison; their interpretation is presented in the report.
