"""External reward components and totals."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Reward:
    task: float = 0.0
    work_cost: float = 0.0
    clean_cost: float = 0.0
    inaccessible_cost: float = 0.0
    central_sanction: float = 0.0
    peer_sanction: float = 0.0
    peer_action_cost: float = 0.0
    peer_omission_cost: float = 0.0
    peer_success_bonus: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.task
            + self.peer_success_bonus
            - self.work_cost
            - self.clean_cost
            - self.inaccessible_cost
            - self.central_sanction
            - self.peer_sanction
            - self.peer_action_cost
            - self.peer_omission_cost
        )

    @property
    def sanction_total(self) -> float:
        return self.central_sanction + self.peer_sanction
