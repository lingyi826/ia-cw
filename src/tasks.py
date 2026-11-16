"""Task assignment and shared task lifecycles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from random import Random
from typing import Literal

from src.actions import Outcome
from src.events import TaskEvent
from src.rewards import Reward
from src.scenario import Scenario
from src.state import AgentState, TaskState


def spawn_tasks(
    scenario: Scenario,
    agents: Mapping[str, AgentState],
    tasks: Mapping[str, TaskState],
    next_id: int,
    rng: Random,
    probability: float,
) -> tuple[dict[str, AgentState], dict[str, TaskState], int, tuple[TaskEvent, ...]]:
    """Assign tasks only to free, sampled participants; strict pairs never fall back.
    Every agent is assigned to at most one task, and every task is assigned to at most two agents.
    An agent is assigend a task with a given probability each time this function is called.

    
    When a task is assigned to an agent, the task is generated as a TaskState
    the task state has an ID, a room ID, a task type, a list of participants, 
    a mode (individual, simultaneous, sequential), progress for each participant, 
    work required, deadline, remaining time, and an optional leader. 

    If a task has a leader, the leader must complete their portion before the other participants 
    can continue.

    Returns the new agent and task states, the next task ID, and a list of TaskEvents."""
    new_agents, new_tasks = dict(agents), dict(tasks)
    candidates = [
        aid
        for aid in sorted(agents)
        if agents[aid].task_id is None and rng.random() < probability
    ]
    rng.shuffle(candidates)
    events: list[TaskEvent] = []
    while candidates:
        aid = candidates.pop()
        cooperate = rng.random() < scenario.tasks.cooperation_probability
        if cooperate and not candidates:
            continue
        participants = (aid, candidates.pop()) if cooperate else (aid,)
        room = rng.choice(scenario.rooms)
        task_id = f"task-{next_id}"
        next_id += 1
        mode: Literal["individual", "simultaneous", "sequential"] = (
            scenario.tasks.cooperative_mode if cooperate else "individual"
        )
        task = TaskState(
            id=task_id,
            room_id=room.id,
            task_type=room.task_type,
            participants=participants,
            mode=mode,
            progress={actor: 0 for actor in participants},
            work_required=scenario.tasks.work_required,
            deadline=scenario.tasks.deadline,
            remaining=scenario.tasks.deadline,
            leader=aid if mode == "sequential" else None,
        )
        new_tasks[task_id] = task
        for actor in participants:
            new_agents[actor] = replace(new_agents[actor], task_id=task_id)
        events.append(TaskEvent("spawned", task_id, participants))
    return new_agents, new_tasks, next_id, tuple(events)


def settle_tasks(
    scenario: Scenario,
    agents: Mapping[str, AgentState],
    tasks: Mapping[str, TaskState],
    outcomes: Mapping[str, Outcome],
    rewards: Mapping[str, Reward],
) -> tuple[
    dict[str, AgentState], dict[str, TaskState], dict[str, Reward], tuple[TaskEvent, ...]
]:
    """Settle (end) each task once, including both partners and their shared clock."""
    new_agents = dict(agents)
    new_tasks: dict[str, TaskState] = {}
    new_rewards = dict(rewards)
    events: list[TaskEvent] = []
    for task_id in sorted(tasks):
        before = tasks[task_id]
        progress = {
            aid: min(before.work_required, before.progress[aid] + outcomes[aid].progress)
            for aid in before.participants
        }
        updated = replace(before, progress=progress)
        complete = [updated.portion_complete(aid) for aid in before.participants]
        if all(complete):
            for aid in before.participants:
                new_rewards[aid] = replace(
                    new_rewards[aid], task=new_rewards[aid].task + scenario.tasks.reward
                )
                new_agents[aid] = replace(new_agents[aid], task_id=None)
            events.append(TaskEvent("completed", task_id, before.participants))
            continue
        handoff = (
            before.mode == "sequential"
            and before.leader is not None
            and not before.portion_complete(before.leader)
            and updated.portion_complete(before.leader)
        )
        remaining = before.deadline if handoff else before.remaining - 1
        if remaining == 0:
            for aid in before.participants:
                new_agents[aid] = replace(new_agents[aid], task_id=None)
            events.append(TaskEvent("expired", task_id, before.participants))
            if len(complete) == 2 and sum(complete) == 1:
                events.append(TaskEvent("defected", task_id, before.participants))
            continue
        new_tasks[task_id] = replace(updated, remaining=remaining)
        if handoff:
            events.append(TaskEvent("handoff", task_id, before.participants))
    return new_agents, new_tasks, new_rewards, tuple(events)
