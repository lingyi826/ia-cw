"""Build physical observations and merge data supplied by extension models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from src.actions import Action, Clean, Idle, Work
from src.state import AgentState, RoomState, TaskState, WorldState


@dataclass(frozen=True, slots=True)
class Observation:
    """
    An observation class can be used to extrate data about the state of the world, defined by WorldState.
    """
    state: WorldState
    subject: str
    available_actions: tuple[Action, ...] = ()

    def __post_init__(self) -> None:
        actions = tuple(self.available_actions)
        object.__setattr__(self, "available_actions", actions)

    @property
    def agent(self) -> AgentState:
        return self.state.agents[self.subject]

    @property
    def room(self) -> RoomState | None:
        return self.state.rooms[self.agent.room_id] if self.agent.room_id is not None else None

    @property
    def task(self) -> TaskState | None:
        return self.state.tasks[self.agent.task_id] if self.agent.task_id is not None else None

    @property
    def task_actions(self) -> tuple[Work | None, Clean | None, Idle]:
        """Work, clean and idle in the task room, or current room when unassigned.

        Room binding comes from the physical task and location records. No
        alternative room is selected. Unavailable physical actions are masked.
        """
        room_id = self.task.room_id if self.task is not None else self.agent.room_id
        work = Work(room_id) if room_id is not None else None
        clean = Clean(room_id) if room_id is not None else None
        return (
            work if work in self.available_actions else None,
            clean if clean in self.available_actions else None,
            Idle(),
        )

def observe(
    state: WorldState,
) -> Mapping[str, Observation]:
    observations: dict[str, Observation] = {}
    accessible = tuple(room.id for room in state.rooms.values() if room.accessible)
    for actor in state.agents.values():
        task = state.tasks.get(actor.task_id) if actor.task_id is not None else None
        gated = (
            task is not None
            and task.mode == "sequential"
            and actor.id != task.leader
            and task.leader is not None
            and not task.portion_complete(task.leader)
        )
        actions: list[Action] = [] if gated else [Work(room) for room in accessible]
        actions.extend(Clean(room) for room in accessible)
        actions.append(Idle())
        bare = Observation(state, actor.id, available_actions=tuple(actions))

        observations[actor.id] = replace(bare, available_actions=tuple(actions))

    return MappingProxyType(
        {
            actor: replace(
                observation,
            )
            for actor, observation in observations.items()
        }
    )