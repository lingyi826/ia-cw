"""Episode returns, task outcomes, compliance and room availability."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from src.events import TaskEvent
from src.transitions import StepResult


@dataclass(frozen=True)
class EpisodeMetrics:
    """Undiscounted totals for an episode so far; undefined rates are None."""

    steps: int = 0
    agent_returns: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    social_welfare: float = 0.0
    institutional_cost: float = 0.0
    sanction_penalties: float = 0.0
    tasks_completed: int = 0
    tasks_expired: int = 0
    tasks_pending: int = 0
    authorisations_unexercised: int = 0
    accessible_room_steps: int = 0
    room_steps: int = 0

    @property
    def mean_return(self) -> float:
        return sum(self.agent_returns.values()) / len(self.agent_returns)

    @property
    def task_completion_rate(self) -> float | None:
        """Completed / (completed + expired + pending); count each shared task once."""
        total = self.tasks_completed + self.tasks_expired + self.tasks_pending
        return self.tasks_completed / total if total else None

    @property
    def tasks_completed_per_step(self) -> float:
        """Completed tasks / joint steps; count each cooperative task once."""
        return self.tasks_completed / self.steps

    @property
    def room_availability(self) -> float:
        """Accessible rooms / all rooms, pooled over post-transition snapshots."""
        return self.accessible_room_steps / self.room_steps


def iter_episode_metrics(
    results: Iterable[StepResult],
) -> Iterator[tuple[StepResult, EpisodeMetrics]]:
    """Yield each live transition with a cumulative metrics snapshot.

    Consume a contiguous rollout starting at step zero without storing its
    history. Pending tasks are those remaining at the current step.
    """
    returns: dict[str, float] = {}
    steps = completed = expired = pending = satisfied = violated = unexercised = 0
    accessible = room_steps = 0
    welfare = institutional_cost = penalties = 0.0
    for result in results:
        if steps == 0:
            returns = dict.fromkeys(result.before.agents, 0.0)
        steps += 1
        for actor, reward in result.rewards.items():
            returns[actor] += reward.total
            penalties += reward.sanction_total
        welfare += result.social_welfare
        for event in result.events:
            if isinstance(event, TaskEvent):
                completed += event.kind == "completed"
                expired += event.kind == "expired"

        accessible += sum(room.accessible for room in result.after.rooms.values())
        room_steps += len(result.after.rooms)
        pending = len(result.after.tasks)
        yield result, EpisodeMetrics(
            steps=steps,
            agent_returns=MappingProxyType(dict(returns)),
            social_welfare=welfare,
            institutional_cost=institutional_cost,
            sanction_penalties=penalties,
            tasks_completed=completed,
            tasks_expired=expired,
            tasks_pending=pending,
            authorisations_unexercised=unexercised,
            accessible_room_steps=accessible,
            room_steps=room_steps,
        )


def summarise_episode(results: Iterable[StepResult]) -> EpisodeMetrics:
    """Consume one complete rollout and return its final cumulative metrics.

    Cooperative task events count once per task, not once per rewarded
    participant. Rewards come directly from StepResult.
    """
    metrics = EpisodeMetrics()
    for _, metrics in iter_episode_metrics(results):
        pass
    return metrics
