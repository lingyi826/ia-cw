"""Physical scenario settings for rooms, agents, tasks and the world."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class RoomSpec:
    """Physical room settings with cleanliness capacity and no occupancy limit.

    initial_cleanliness is the initial and maximum cleanliness. Rooms below
    accessible_threshold reject both Work and Clean. Eligible work can progress
    a task in any accessible room with a matching task_type.
    """

    id: str = field(metadata={"doc": "Unique room ID."})
    task_type: str = field(
        default="general", metadata={"doc": "Task type accepted by work in this room."}
    )
    initial_cleanliness: int = field(
        default=6, metadata={"doc": "Initial and maximum cleanliness units."}
    )
    accessible_threshold: int = field(
        default=1, metadata={"doc": "Minimum cleanliness units permitting entry."}
    )
    features: dict[str, str | int | float | bool | None] = field(
        default_factory=dict, metadata={"doc": "Named scalar room features."}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", dict(sorted(self.features.items())))


@dataclass(frozen=True, slots=True)
class AgentSpec:
    id: str = field(
        metadata={"doc": "Unique physical participant ID, independent of its policy."}
    )
    initial_room: str | None = field(
        default=None, metadata={"doc": "Initial room ID, or None for outside."}
    )
    features: dict[str, str | int | float | bool | None] = field(
        default_factory=dict, metadata={"doc": "Named scalar agent features."}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", dict(sorted(self.features.items())))


@dataclass(frozen=True, slots=True)
class WorldConfig:
    horizon: int = field(default=200, metadata={"doc": "Maximum joint steps in one episode."})
    task_damage: int = field(
        default=1, metadata={"doc": "Cleanliness units consumed per accepted Work action."}
    )
    clean_increase: int = field(
        default=1,
        metadata={
            "doc": "Cleanliness units added per accepted Clean action, up to room capacity."
        },
    )
    restore_steps: int = field(
        default=20,
        metadata={"doc": "Locked-room restoration ticks; zero disables restoration."},
    )
    inaccessible_cost: float = field(
        default=0.0, metadata={"doc": "Reward cost per rejected entry or forced exit."}
    )
    work_cost: float = field(
        default=0.0, metadata={"doc": "Reward cost per accepted Work action."}
    )
    clean_cost: float = field(
        default=0.0, metadata={"doc": "Reward cost per accepted Clean action."}
    )
    stop_when_all_locked: bool = field(
        default=False,
        metadata={"doc": "Terminate if every room is locked and no restoration is pending."},
    )
    features: dict[str, str | int | float | bool | None] = field(
        default_factory=dict, metadata={"doc": "Named scalar environment features."}
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", dict(sorted(self.features.items())))


@dataclass(frozen=True, slots=True)
class TaskConfig:
    work_required: int = field(
        default=3, metadata={"doc": "Work units each participant must complete per task."}
    )
    deadline: int = field(
        default=15, metadata={"doc": "Joint-step budget per task, or per sequential phase."}
    )
    spawn_probability: float = field(
        default=0.25, metadata={"doc": "Per-free-agent task-spawn probability after each step."}
    )
    initial_probability: float = field(
        default=1.0, metadata={"doc": "Per-agent task-spawn probability on reset."}
    )
    cooperation_probability: float = field(
        default=0.0,
        metadata={
            "doc": "Probability a spawn requires two agents; one forbids individual fallback."
        },
    )
    cooperative_mode: Literal["simultaneous", "sequential"] = field(
        default="simultaneous",
        metadata={
            "doc": "Whether paired portions can progress concurrently or require a leader handoff."
        },
    )
    reward: float = field(
        default=1.0,
        metadata={"doc": "External reward paid to each participant on joint completion."},
    )


@dataclass(frozen=True, slots=True)
class Scenario:
    rooms: tuple[RoomSpec, ...] = field(
        metadata={"doc": "Nonempty room specifications with unique IDs."}
    )
    agents: tuple[AgentSpec, ...] = field(
        metadata={"doc": "Nonempty physical participant specifications with unique IDs."}
    )
    world: WorldConfig = field(
        default_factory=WorldConfig,
        metadata={"doc": "Physical dynamics and episode termination parameters."},
    )
    tasks: TaskConfig = field(
        default_factory=TaskConfig,
        metadata={"doc": "Task generation, progress, timing, and payoff parameters."},
    )

    def __post_init__(self) -> None:
        rooms, agents = tuple(self.rooms), tuple(self.agents)
        object.__setattr__(self, "rooms", tuple(sorted(rooms, key=lambda item: item.id)))
        object.__setattr__(self, "agents", tuple(sorted(agents, key=lambda item: item.id)))
