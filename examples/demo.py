"""Run one of the four example agents without sanctions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import asdict, fields
from pathlib import Path

from src.agents import Policy
from src.configuration import load_config
from src.environment import Environment
from examples.qlearning import QLearningAgent
from src.metrics import EpisodeMetrics, iter_episode_metrics
from src.runner import rollout
from examples.rulebased import RuleBasedAgent
from examples.random import RandomAgent

def metric_values(metrics: EpisodeMetrics) -> dict[str, object]:
    values = {item.name: getattr(metrics, item.name) for item in fields(metrics)}
    values["agent_returns"] = dict(metrics.agent_returns)
    values.update(
        mean_return=metrics.mean_return,
        task_completion_rate=metrics.task_completion_rate,
        tasks_completed_per_step=metrics.tasks_completed_per_step,
        room_availability=metrics.room_availability,
    )
    return values


def run_episode(
    environment: Environment,
    policies: Mapping[str, Policy],
    *,
    agent: str,
    phase: str,
    episode: int,
) -> EpisodeMetrics:
    metrics = EpisodeMetrics()
    results = rollout(environment, policies, learning=phase == "training")
    for result, metrics in iter_episode_metrics(results):
        actions = {
            actor: {
                "requested": {
                    "type": type(outcome.requested).__name__,
                    **asdict(outcome.requested),
                },
                "executed": {
                    "type": type(outcome.executed).__name__,
                    **asdict(outcome.executed),
                },
                "accepted": outcome.accepted,
                "reason": outcome.reason,
                "progress": outcome.progress,
            }
            for actor, outcome in result.outcomes.items()
        }
        print(
            json.dumps(
                {
                    "type": "step",
                    "agent": agent,
                    "phase": phase,
                    "episode": episode,
                    "seed": result.after.episode_seed,
                    "step": result.after.step,
                    "actions": actions,
                    "state": asdict(result.before),
                    "next_state": asdict(result.after),
                    "rewards": {
                        actor: {**asdict(reward), "total": reward.total}
                        for actor, reward in result.rewards.items()
                    },
                    "metrics": metric_values(metrics),
                },
                allow_nan=False,
            ),
            flush=True,
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent", choices=("rulebased", "ppo", "qlearning", "coordinator", "random"), default="rulebased"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-episodes", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--qvalues-dir",
        type=Path,
        help="Load per-agent Q-values from DIR/<agent>.json before running.",
    )
    parser.add_argument(
        "--save-qvalues-dir",
        type=Path,
        help="Save trained per-agent Q-values to DIR/<agent>.json.",
    )
    args = parser.parse_args()

    if args.agent != "qlearning" and (args.qvalues_dir or args.save_qvalues_dir):
        parser.error("Q-value file options can only be used with --agent qlearning")

    # If no args.config, default to `environment.yaml`
    loaded = load_config(args.config)
    seed = loaded.seed if args.seed is None else args.seed
    environment = Environment(
        loaded.scenario,
        seed=seed,
    )

    policies: dict[str, Policy] = {}
    for index, actor in enumerate(environment.state.agents):
        if args.agent == "rulebased":
            policies[actor] = RuleBasedAgent(loaded.scenario, seed=seed * 100 + index)
        if args.agent == "random":
            policies[actor] = RandomAgent(loaded.scenario, seed=seed * 100 + index)
        elif args.agent == "qlearning":
            policy = QLearningAgent(loaded.scenario, seed=seed * 100 + index)
            if args.qvalues_dir is not None:
                policy.load_values(args.qvalues_dir / f"{actor}.json")
            policies[actor] = policy

    ## RL only
    training_steps = 0
    if args.agent in ("ppo", "qlearning"):
        for episode in range(args.train_episodes):
            environment.reset(seed=seed + episode)
            metrics = run_episode(
                environment,
                policies,
                agent=args.agent,
                phase="training",
                episode=episode + 1,
            )
            training_steps += metrics.steps

    if args.agent == "qlearning" and args.save_qvalues_dir is not None:
        for actor, policy in policies.items():
            assert isinstance(policy, QLearningAgent)
            policy.save_values(args.save_qvalues_dir / f"{actor}.json")

    for episode in range(args.episodes):
        evaluation_seed = seed + 1_000_000 + episode
        environment.reset(seed=evaluation_seed)
        metrics = run_episode(
            environment,
            policies,
            agent=args.agent,
            phase="evaluation",
            episode=episode + 1,
        )
        print(
            json.dumps(
                {
                    "type": "episode_summary",
                    "phase": "evaluation",
                    "episode": episode + 1,
                    "agent": args.agent,
                    "seed": evaluation_seed,
                    "training_steps": training_steps,
                    **metric_values(metrics),
                },
                allow_nan=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
