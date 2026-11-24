# Intelligent Agents Coursework

Private repository: <https://github.com/lingyi826/ia-cw>

This repository contains the normative rule-based agent, tabular Q-learning
agent, reproducible evaluation code, saved Q-values and visualisation code for
the Eregion Works coursework environment.

## Requirements

- Python 3.11 or newer
- PyYAML

The visualisation code uses only the Python standard library and does not need
Matplotlib, pandas or another plotting package.

With `uv`, install the project from the repository root:

```bash
uv sync
```

Commands below may then be prefixed with `uv run`, for example:

```bash
uv run python -m examples.demo --agent rulebased
```

If the existing virtual environment is already active, use `python` directly.

## Main coursework files

- `examples/rulebased.py`: norm-derived Work, Clean and Idle decision rules.
- `examples/qlearning.py`: epsilon-greedy action selection, Q-learning update,
  and JSON Q-value save/load methods.
- `examples/train_qlearning.py`: reproducible training in all three task modes.
- `examples/evaluate.py`: multi-seed comparison of random, rule-based and
  Q-learning agents.
- `examples/visualise.py`: SVG graph and Markdown table generation.
- `artifacts/qvalues/`: trained per-mode, per-agent Q-tables.
- `results/`: episode data, summary statistics, graph and report table.

## Run an individual agent

Run the rule-based agent:

```bash
python -m examples.demo --agent rulebased --episodes 1
```

Run the random-action baseline:

```bash
python -m examples.demo --agent random --episodes 1
```

Run Q-learning with the saved individual-task Q-values and no retraining:

```bash
python -m examples.demo --agent qlearning --train-episodes 0 --episodes 1 --qvalues-dir artifacts/qvalues/individual
```

`examples.demo` prints detailed step records followed by an episode summary.
Use `examples.evaluate` for the quieter full experiment.

## Task configurations

The evaluation code derives all configurations from `src/environment.yaml`
without editing it or maintaining duplicate YAML files:

| Configuration | `cooperation_probability` | `cooperative_mode` |
|---|---:|---|
| Individual | `0.0` | ignored |
| Simultaneous cooperative | `1.0` | `simultaneous` |
| Sequential cooperative | `1.0` | `sequential` |

The conversion is implemented by `scenario_for_mode()` in
`examples/train_qlearning.py` and is shared by training and evaluation.

## Train and save Q-values

Train separate Q-tables for all three task configurations:

```bash
python -m examples.train_qlearning --episodes 200
```

The command writes nine JSON files:

```text
artifacts/qvalues/
├── individual/{alice,bob,carol}.json
├── simultaneous/{alice,bob,carol}.json
└── sequential/{alice,bob,carol}.json
```

Every JSON file contains the encoder feature names, observed-transition count,
learning parameters and Q-table entries. Running the training command again
replaces the files with newly trained values.

The submitted Q-values were trained for 200 episodes in each configuration.
Each configuration used 48,000 joint training steps. The measured times were:

| Configuration | Training time |
|---|---:|
| Individual | 33.772 s |
| Simultaneous | 40.602 s |
| Sequential | 37.315 s |
| Total | 111.689 s |

Training time is not included in the evaluation runtime below.

## Reproduce the evaluation

Run the complete experiment using the saved Q-values:

```bash
python -m examples.evaluate --episodes 5
```

This evaluates random, rule-based and Q-learning agents in individual,
simultaneous and sequential configurations. All agents use the same five
evaluation seeds: `1000007` to `1000011`. Training and evaluation are separate;
the evaluation command calls policies with learning disabled.

The command produces:

```text
results/evaluation_episodes.csv
results/evaluation_summary.csv
```

`evaluation_episodes.csv` contains 45 rows: three configurations multiplied by
three agent types and five seeds. `evaluation_summary.csv` contains the mean and
sample standard deviation for every configuration-agent pair.

The measured runtime for all 45 evaluation episodes was 9.141 seconds on the
author's CPU-only Windows machine, well below the ten-minute limit.

## Metrics

- **Task completion rate:** completed tasks divided by completed, expired and
  pending tasks.
- **Tasks completed per step:** completed tasks divided by joint environment
  steps.
- **Office cleanliness:** mean post-step room cleanliness divided by each
  room's initial cleanliness.
- **Room availability:** accessible room-steps divided by all room-steps.
- **N3 work compliance:** proportion of Work attempts that were useful, safe
  and accepted.
- **Social welfare:** sum of all agent returns.
- Additional counts include useful, useless and unsafe Work, rejected actions,
  Clean actions and Idle actions.

Task completion provides evidence about the N1/N2 commitments. Work compliance
provides evidence about the N3 prohibition. A high prohibition-compliance score
does not by itself demonstrate that a learning agent represents or has learned
the norm. For example, an agent that never attempts Work creates no unsafe Work
but also fails its task commitments.

## Generate the graph and report table

After evaluation, run:

```bash
python -m examples.visualise
```

This uses `results/evaluation_summary.csv` and creates:

```text
results/evaluation_comparison.svg
results/evaluation_summary.md
```

The SVG contains grouped bars with standard-deviation error bars for task
completion, office cleanliness, N3 compliance and social welfare. SVG is a
vector format that can be inserted directly into Word. The Markdown file
contains the same mean and standard-deviation values in table form.

## Reproducibility notes

- The random baseline uses its seeded random-number generator.
- Evaluation policies are recreated for each episode using the evaluation seed.
- Q-learning policies load saved Q-values and run greedily during evaluation.
- Training seeds and evaluation seeds are separate.
- Raw episode results are retained instead of reporting only calculated means.
