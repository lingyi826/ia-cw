"""Room, agent, task and world snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

TaskMode: TypeAlias = Literal["individual", "simultaneous", "sequential"]


@dataclass(frozen=True, slots=True)
class RoomState:
    id: str
    task_type: str
    initial_cleanliness: int
    cleanliness: int
    accessible_threshold: int
    restore_remaining: int = 0
    features: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", dict(sorted(self.features.items())))

    @property
    def accessible(self) -> bool:
        return self.cleanliness >= self.accessible_threshold


@dataclass(frozen=True, slots=True)
class AgentState:
    id: str
    room_id: str | None
    task_id: str | None = None
    reward_total: float = 0.0
    features: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", dict(sorted(self.features.items())))


@dataclass(frozen=True, slots=True)
class TaskState:
    id: str
    room_id: str
    task_type: str
    participants: tuple[str, ...]
    mode: TaskMode
    progress: Mapping[str, int]
    work_required: int
    deadline: int
    remaining: int
    leader: str | None = None


    def __post_init__(self) -> None:
        participants = tuple(self.participants)
        progress = dict(sorted(self.progress.items()))
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "progress", progress)

    def portion_complete(self, agent_id: str) -> bool:
        return self.progress[agent_id] >= self.work_required

    def can_progress(self, agent_id: str) -> bool:
        return not self.portion_complete(agent_id) and (
            self.mode != "sequential"
            or agent_id == self.leader
            or (self.leader is not None and self.portion_complete(self.leader))
        )


@dataclass(frozen=True, slots=True)
class WorldState:
    episode_seed: int
    step: int
    rooms: Mapping[str, RoomState]
    agents: Mapping[str, AgentState]
    tasks: Mapping[str, TaskState]
    next_task_id: int = 0
    terminated: bool = False
    truncated: bool = False
    features: dict[str, str | int | float | bool | None] = field(
        default_factory=dict,
        metadata={"doc": "Named scalar environment features, visible in observations."},
    )
    """
    Used to store the current state of the world/enivronment (synonymous). 
    
    Agents are a mapping of agent IDs to AgentState objects, 
    Rooms are a mapping of room IDs to RoomState objects, and tasks are a mapping of task IDs to TaskState objects. 
    
    The next_task_id is used to generate unique task IDs for new tasks. 
    
    The terminated and truncated flags indicate whether the episode has ended. 
    """

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", dict(sorted(self.features.items())))
        for name, kind in (("rooms", RoomState), ("agents", AgentState), ("tasks", TaskState)):
            mapping = dict(sorted(getattr(self, name).items()))
            object.__setattr__(self, name, mapping)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated
