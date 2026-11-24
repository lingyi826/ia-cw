"""Evaluate random, rule-based and learned policies in all three task modes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

from examples.qlearning import QLearningAgent
from examples.random import RandomAgent
from examples.rulebased import RuleBasedAgent
from examples.train_qlearning import TASK_MODES, scenario_for_mode
from src.actions import Clean, Idle, Work
from src.agents import Policy
from src.configuration import load_config
from src.environment import Environment
from src.metrics import EpisodeMetrics, iter_episode_metrics
from src.runner import rollout
from src.scenario import Scenario


AGENT_TYPES = ("random", "rulebased", "qlearning")
SUMMARY_METRICS = (
    "mean_return",
    "social_welfare",
    "tasks_completed",
    "tasks_expired",
    "task_completion_rate",
    "tasks_completed_per_step",
    "room_availability",
    "office_cleanliness",
    "norm_compliance_rate",
    "work_attempts",
    "useful_work",
    "useless_work",
    "unsafe_work",
    "rejected_actions",
    "clean_actions",
    "idle_actions",
)


def make_policies(
    agent_type: str,
    scenario: Scenario,
    seed: int,
    qvalues_dir: Path,
    mode: str,
) -> dict[str, Policy]:
    """Create fresh deterministic policies for one evaluation episode."""
    policies: dict[str, Policy] = {}
    for index, actor in enumerate(spec.id for spec in scenario.agents):
        policy_seed = seed * 100 + index
        if agent_type == "random":
            policies[actor] = RandomAgent(scenario, seed=policy_seed)
        elif agent_type == "rulebased":
            policies[actor] = RuleBasedAgent(scenario, seed=policy_seed)
        elif agent_type == "qlearning":
            policy = QLearningAgent(scenario, seed=policy_seed)
            path = qvalues_dir / mode / f"{actor}.json"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path}. Run python -m examples.train_qlearning first."
                )
            policy.load_values(path)
            policy.set_training(False)
            policies[actor] = policy
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    return policies


def evaluate_episode(
    scenario: Scenario,
    agent_type: str,
    mode: str,
    seed: int,
    qvalues_dir: Path,
) -> dict[str, object]:
    """Run one episode and calculate performance, cleanliness and compliance."""
    environment = Environment(scenario, seed=seed)
    policies = make_policies(agent_type, scenario, seed, qvalues_dir, mode)

    metrics = EpisodeMetrics()
    cleanliness_total = 0.0
    cleanliness_samples = 0
    total_actions = 0
    work_attempts = useful_work = useless_work = unsafe_work = 0
    violating_work = rejected_actions = clean_actions = idle_actions = 0

    results = rollout(environment, policies, learning=False)
    for result, metrics in iter_episode_metrics(results):
        for room in result.after.rooms.values():
            cleanliness_total += room.cleanliness / room.initial_cleanliness
            cleanliness_samples += 1

        for outcome in result.outcomes.values():
            total_actions += 1
            if not outcome.accepted:
                rejected_actions += 1

            if isinstance(outcome.executed, Clean):
                clean_actions += 1
            elif isinstance(outcome.executed, Idle):
                idle_actions += 1

            if isinstance(outcome.requested, Work):
                work_attempts += 1
                useless = outcome.progress == 0
                unsafe = outcome.accepted and not result.after.rooms[
                    outcome.requested.room_id
                ].accessible
                useful_work += not useless
                useless_work += useless
                unsafe_work += unsafe
                violating_work += useless or unsafe or not outcome.accepted

    office_cleanliness = cleanliness_total / cleanliness_samples
    norm_compliance_rate = (
        1.0 - violating_work / work_attempts if work_attempts else 1.0
    )
    return {
        "mode": mode,
        "agent": agent_type,
        "seed": seed,
        "steps": metrics.steps,
        "agent_returns": json.dumps(dict(metrics.agent_returns), sort_keys=True),
        "social_welfare": metrics.social_welfare,
        "institutional_cost": metrics.institutional_cost,
        "sanction_penalties": metrics.sanction_penalties,
        "tasks_completed": metrics.tasks_completed,
        "tasks_expired": metrics.tasks_expired,
        "tasks_pending": metrics.tasks_pending,
        "mean_return": metrics.mean_return,
        "task_completion_rate": metrics.task_completion_rate,
        "tasks_completed_per_step": metrics.tasks_completed_per_step,
        "room_availability": metrics.room_availability,
        "office_cleanliness": office_cleanliness,
        "total_actions": total_actions,
        "work_attempts": work_attempts,
        "useful_work": useful_work,
        "useless_work": useless_work,
        "unsafe_work": unsafe_work,
        "norm_violations": violating_work,
        "norm_compliance_rate": norm_compliance_rate,
        "rejected_actions": rejected_actions,
        "clean_actions": clean_actions,
        "idle_actions": idle_actions,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for mode in TASK_MODES:
        for agent_type in AGENT_TYPES:
            group = [
                row for row in rows if row["mode"] == mode and row["agent"] == agent_type
            ]
            if not group:
                continue
            summary: dict[str, object] = {
                "mode": mode,
                "agent": agent_type,
                "episodes": len(group),
            }
            for metric in SUMMARY_METRICS:
                values = [float(row[metric]) for row in group]
                summary[f"{metric}_mean"] = mean(values)
                summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--modes", nargs="+", choices=TASK_MODES, default=list(TASK_MODES)
    )
    parser.add_argument(
        "--agents", nargs="+", choices=AGENT_TYPES, default=list(AGENT_TYPES)
    )
    parser.add_argument("--qvalues-dir", type=Path, default=Path("artifacts/qvalues"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be greater than zero")

    loaded = load_config(args.config)
    base_seed = loaded.seed if args.seed is None else args.seed
    started = perf_counter()
    rows: list[dict[str, object]] = []
    for mode in args.modes:
        scenario = scenario_for_mode(loaded.scenario, mode)
        for agent_type in args.agents:
            for episode in range(args.episodes):
                evaluation_seed = base_seed + 1_000_000 + episode
                row = evaluate_episode(
                    scenario,
                    agent_type,
                    mode,
                    evaluation_seed,
                    args.qvalues_dir,
                )
                row["episode"] = episode + 1
                rows.append(row)

    episode_path = args.output_dir / "evaluation_episodes.csv"
    summary_path = args.output_dir / "evaluation_summary.csv"
    write_csv(episode_path, rows)
    summaries = summarise(rows)
    write_csv(summary_path, summaries)

    for summary in summaries:
        print(
            json.dumps(
                {
                    "type": "evaluation_summary",
                    "mode": summary["mode"],
                    "agent": summary["agent"],
                    "episodes": summary["episodes"],
                    "task_completion_rate_mean": summary[
                        "task_completion_rate_mean"
                    ],
                    "office_cleanliness_mean": summary["office_cleanliness_mean"],
                    "norm_compliance_rate_mean": summary[
                        "norm_compliance_rate_mean"
                    ],
                    "social_welfare_mean": summary["social_welfare_mean"],
                },
                allow_nan=False,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "type": "evaluation_files",
                "episodes": str(episode_path),
                "summary": str(summary_path),
                "elapsed_seconds": round(perf_counter() - started, 3),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
