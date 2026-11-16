"""Physical task and room features shared by PPO and Q-learning."""

from __future__ import annotations

from src.observations import Observation
from src.scenario import Scenario


class TaskEncoder:
    """Encode physical observations without norm advice or a coordinator plan."""

    feature_names = (
        "time.fraction",
        "task.active",
        "task.remaining_work_fraction",
        "task.deadline_fraction",
        "task.ready",
        "task.cooperative",
        "task.partner_remaining_fraction",
        "room.cleanliness",
        "work.available",
        "clean.available",
    )
    size = len(feature_names)

    def __init__(self, scenario: Scenario) -> None:
        self._scenario = scenario

    def __call__(self, observation: Observation) -> tuple[float, ...]:
        task = observation.task
        work, clean, _ = observation.task_actions
        partner = (
            next((actor for actor in task.participants if actor != observation.subject), None)
            if task is not None
            else None
        )
        room_id = task.room_id if task is not None else observation.agent.room_id
        room = observation.state.rooms[room_id] if room_id is not None else None
        return (
            observation.state.step / self._scenario.world.horizon,
            float(task is not None),
            0.0
            if task is None
            else (task.work_required - task.progress[observation.subject]) / task.work_required,
            0.0 if task is None else task.remaining / task.deadline,
            float(task is not None and task.can_progress(observation.subject)),
            float(task is not None and len(task.participants) > 1),
            0.0
            if task is None or partner is None
            else (task.work_required - task.progress[partner]) / task.work_required,
            0.0 if room is None else room.cleanliness / room.initial_cleanliness,
            float(work is not None),
            float(clean is not None),
        )
