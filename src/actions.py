"""Physical actions, joint requests and resolved outcomes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True, slots=True)
class Work:
    """Select a room directly, without pathfinding, and attempt work there.

    Inaccessible rooms reject entry. A sequential follower cannot work before
    handoff, including the step when the leader finishes. Accepted work moves
    the actor and consumes task_damage cleanliness even without useful progress.
    Eligible work in a matching task type adds one unit of progress.
    Concurrent workers can exhaust a room in one joint step.
    """

    room_id: str


@dataclass(frozen=True, slots=True)
class Clean:
    """Move directly to an accessible room and restore cleanliness.

    Accepted cleaning adds clean_increase cleanliness. Locked rooms
    reject cleaning without moving the actor. Work and cleaning effects are
    combined before cleanliness is clamped to zero and initial_cleanliness.
    """

    room_id: str


@dataclass(frozen=True, slots=True)
class Idle:
    pass

Action: TypeAlias = Work | Clean | Idle 


@dataclass(frozen=True, slots=True)
class Outcome:
    """Separate an action's intention, resolved execution, and useful progress.

    requested preserves the controller's intention; executed is the resolved
    action. A rejected attempt has accepted=False, executes Idle, gives a reason,
    and makes zero progress. Acceptance alone does not establish task progress:
    accepted Work can consume cleanliness while progress remains zero.
    progress counts useful work units credited before task settlement.
    """

    requested: Action
    executed: Action
    accepted: bool
    reason: str | None = None
    progress: int = 0


@dataclass(frozen=True, slots=True)
class JointAction:
    step: int
    actions: Mapping[str, Action]

    def __post_init__(self) -> None:
        actions = dict(sorted(self.actions.items()))
        object.__setattr__(self, "actions", actions)
