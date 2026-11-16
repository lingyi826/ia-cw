"""Task and room lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TaskEvent:
    kind: Literal["spawned", "completed", "expired", "defected", "handoff"]
    task_id: str
    participants: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", tuple(self.participants))


@dataclass(frozen=True, slots=True)
class RoomEvent:
    kind: Literal["locked", "restored", "ejected"]
    room_id: str
    agents: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "agents", tuple(self.agents))
