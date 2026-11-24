"""Train and save one tabular Q-learning policy for each task mode."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from examples.qlearning import QLearningAgent
from src.configuration import load_config
from src.environment import Environment
from src.metrics import summarise_episode
from src.runner import rollout
from src.scenario import Scenario


TASK_MODES = ("individual", "simultaneous", "sequential")


def scenario_for_mode(scenario: Scenario, mode: str) -> Scenario:
    """Return the same scenario with only the requested task type enabled."""
    if mode == "individual":
        tasks = replace(
            scenario.tasks,
            cooperation_probability=0.0,
            cooperative_mode="simultaneous",
        )
    elif mode in ("simultaneous", "sequential"):
        tasks = replace(
            scenario.tasks,
            cooperation_probability=1.0,
            cooperative_mode=mode,
        )
    else:
        raise ValueError(f"Unknown task mode: {mode}")
    return replace(scenario, tasks=tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=TASK_MODES,
        default=list(TASK_MODES),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qvalues"))
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    loaded = load_config(args.config)
    base_seed = loaded.seed if args.seed is None else args.seed

    for mode in args.modes:
        scenario = scenario_for_mode(loaded.scenario, mode)
        environment = Environment(scenario, seed=base_seed)
        policies = {
            actor: QLearningAgent(scenario, seed=base_seed * 100 + index)
            for index, actor in enumerate(environment.state.agents)
        }

        started = perf_counter()
        training_steps = 0
        for episode in range(args.episodes):
            environment.reset(seed=base_seed + episode)
            metrics = summarise_episode(rollout(environment, policies, learning=True))
            training_steps += metrics.steps

        mode_dir = args.output_dir / mode
        files: dict[str, str] = {}
        q_entries: dict[str, int] = {}
        for actor, policy in policies.items():
            path = mode_dir / f"{actor}.json"
            policy.save_values(path)
            files[actor] = str(path)
            q_entries[actor] = len(policy.values)

        print(
            json.dumps(
                {
                    "type": "training_summary",
                    "mode": mode,
                    "episodes": args.episodes,
                    "training_steps": training_steps,
                    "elapsed_seconds": round(perf_counter() - started, 3),
                    "q_entries": q_entries,
                    "files": files,
                },
                allow_nan=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
