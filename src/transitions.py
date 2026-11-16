"""One simultaneous physical transition with explicit normative consequences."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from random import Random
from types import MappingProxyType

from src.actions import Clean, Idle, JointAction, Outcome, Work
from src.events import RoomEvent, TaskEvent
from src.observations import Observation, observe
from src.rewards import Reward
from src.scenario import Scenario
from src.state import AgentState, RoomState, WorldState
from src.tasks import settle_tasks, spawn_tasks

@dataclass(frozen=True, slots=True)
class StepResult:
    before: WorldState
    after: WorldState
    outcomes: Mapping[str, Outcome]
    rewards: Mapping[str, Reward]
    events: tuple[TaskEvent | RoomEvent, ...]
    observations: Mapping[str, Observation]
    institutional_cost: float

    def __post_init__(self) -> None:
        for name in ("outcomes", "rewards", "observations"):
            mapping = dict(sorted(getattr(self, name).items()))
            object.__setattr__(self, name, mapping)
        for name in (
            "events",
        ):
            records = tuple(getattr(self, name))
            object.__setattr__(self, name, records)

    @property
    def terminated(self) -> bool:
        return self.after.terminated

    @property
    def truncated(self) -> bool:
        return self.after.truncated

    @property
    def done(self) -> bool:
        return self.after.done

    @property
    def social_welfare(self) -> float:
        return sum(reward.total for reward in self.rewards.values())


def initial_state(scenario: Scenario, seed: int, rng: Random) -> WorldState:
    rooms = {
        spec.id: RoomState(
            id=spec.id,
            task_type=spec.task_type,
            initial_cleanliness=spec.initial_cleanliness,
            cleanliness=spec.initial_cleanliness,
            accessible_threshold=spec.accessible_threshold,
            features=spec.features,
        )
        for spec in scenario.rooms
    }
    agents = {
        spec.id: AgentState(id=spec.id, room_id=spec.initial_room, features=spec.features)
        for spec in scenario.agents
    }
    agents, tasks, next_id, _ = spawn_tasks(
        scenario,
        agents,
        {}, # Current tasks.
        0,
        rng,
        scenario.tasks.initial_probability,
    )
    return WorldState(
        seed, 0, rooms, agents, tasks, next_task_id=next_id, features=scenario.world.features
    )


def _add_reward(rewards: dict[str, Reward], aid: str, component: str, amount: float) -> None:
    rewards[aid] = replace(
        rewards[aid], **{component: getattr(rewards[aid], component) + amount}
    )


def _resolve_actions(
    before: WorldState, joint: JointAction, scenario: Scenario
) -> tuple[dict[str, AgentState], dict[str, RoomState], dict[str, Outcome], dict[str, Reward]]:
    """
    Takes as input the current world state (as before), any joint actions, and the scenario.

    Processes the single and joint actions for each agent, and updates the world.

    Returns the new agent and room states, as well as the outcomes and rewards for each agent.

    The WorldState (before) contains all agents and rooms.
    """
    agents, rooms = dict(before.agents), dict(before.rooms)
    outcomes, rewards = {}, {aid: Reward() for aid in before.agents}
    deltas = {rid: 0 for rid in before.rooms}

    ## For every agent, check their action. If the action modifies the room (Work or Clean),
    # update the room's cleanliness and the agent's state. If the action is invalid, record a rejection reason.
    for aid in before.agents:
        action = joint.actions[aid]
        actor = before.agents[aid]
        task = before.tasks.get(actor.task_id) if actor.task_id is not None else None
        rejection = None
        progress = 0
        if isinstance(action, (Work, Clean)):
            target = before.rooms[action.room_id]
            if not target.accessible:
                rejection = "room_inaccessible"
                _add_reward(rewards, aid, "inaccessible_cost", scenario.world.inaccessible_cost)
            elif (
                isinstance(action, Work)
                and task is not None
                and task.mode == "sequential"
                and task.leader is not None
                and aid != task.leader
                and not task.portion_complete(task.leader)
            ):
                rejection = "waiting_for_leader"
            else:
                agents[aid] = replace(actor, room_id=action.room_id)
                if isinstance(action, Clean):
                    deltas[action.room_id] += scenario.world.clean_increase
                    _add_reward(rewards, aid, "clean_cost", scenario.world.clean_cost)
                else:
                    deltas[action.room_id] -= scenario.world.task_damage
                    _add_reward(rewards, aid, "work_cost", scenario.world.work_cost)
                    if (
                        task is not None
                        and task.task_type == target.task_type
                        and task.can_progress(aid)
                    ):
                        progress = min(1, task.work_required - task.progress[aid])
        outcomes[aid] = Outcome(
            requested=action,
            executed=Idle() if rejection else action,
            accepted=rejection is None,
            reason=rejection,
            progress=progress,
        )
    for rid, room in before.rooms.items():
        rooms[rid] = replace(
            room,
            cleanliness=max(0, min(room.initial_cleanliness, room.cleanliness + deltas[rid])),
        )
    return agents, rooms, outcomes, rewards


def _settle_rooms(
    before: WorldState,
    scenario: Scenario,
    agents: Mapping[str, AgentState],
    rooms: Mapping[str, RoomState],
    rewards: dict[str, Reward],
) -> tuple[dict[str, AgentState], dict[str, RoomState], tuple[RoomEvent, ...]]:
    """
    Update room states and eject agents from inaccessible rooms, returning new states and events.
    Called after actions have been resolved and before tasks are settled in the main transition function advance().
    """
    new_agents, new_rooms = dict(agents), dict(rooms)
    events: list[RoomEvent] = []
    for rid, room in rooms.items():
        if before.rooms[rid].accessible and not room.accessible:
            events.append(RoomEvent("locked", rid))
        expelled = tuple(
            aid for aid, actor in agents.items() if actor.room_id == rid and not room.accessible
        )
        if expelled:
            events.append(RoomEvent("ejected", rid, expelled))
            for aid in expelled:
                new_agents[aid] = replace(new_agents[aid], room_id=None)
                _add_reward(rewards, aid, "inaccessible_cost", scenario.world.inaccessible_cost)
        remaining = room.restore_remaining
        if not room.accessible and remaining == 0 and scenario.world.restore_steps:
            remaining = scenario.world.restore_steps
        if remaining:
            remaining -= 1
            if remaining == 0:
                new_rooms[rid] = replace(
                    room, cleanliness=room.initial_cleanliness, restore_remaining=0
                )
                events.append(RoomEvent("restored", rid))
                continue
        new_rooms[rid] = replace(room, restore_remaining=remaining)
    return new_agents, new_rooms, tuple(events)


def advance(
    before: WorldState,
    joint: JointAction,
    scenario: Scenario,
    observations: Mapping[str, Observation],
    rng: Random,
) -> StepResult:
    """
    Called by Environment.step() to advance the world state by one step, given a joint action and the current state.

    Environment.step() is called by rollout() in runner.py. rollout() is called by the main loop, and is responsible for running an entire episode (many timesteps).
    """
    if before.done:
        raise RuntimeError("The episode has ended; reset before taking another step")

    agents, rooms, outcomes, rewards = _resolve_actions(before, joint, scenario)

    agents, tasks, rewards, task_events = settle_tasks(
        scenario, agents, before.tasks, outcomes, rewards
    )


    agents, rooms, room_events = _settle_rooms(before, scenario, agents, rooms, rewards)
    for aid, agent in agents.items():
        agents[aid] = replace(
            agent,
            reward_total=agent.reward_total + rewards[aid].total,
        )
    step = before.step + 1
    truncated = step >= scenario.world.horizon
    # Find bool to mark if we terminate or not!
    terminated = (
        scenario.world.stop_when_all_locked
        and all(not room.accessible for room in rooms.values())
        and not any(room.restore_remaining for room in rooms.values())
    )
    next_id = before.next_task_id
    spawn_events: tuple[TaskEvent, ...] = ()
    if not (truncated or terminated):
        # Spawn some more new tasks to do.
        agents, tasks, next_id, spawn_events = spawn_tasks(
            scenario,
            agents,
            tasks,
            next_id,
            rng,
            scenario.tasks.spawn_probability,
        )
    after = WorldState(
        before.episode_seed,
        step,
        rooms,
        agents,
        tasks,
        next_id,
        terminated,
        truncated,
        features=scenario.world.features,
    )
    next_observations = observe(after)
    return StepResult(
        before=before,
        after=after,
        outcomes=outcomes,
        rewards=rewards,
        events=task_events + room_events + spawn_events,
        observations=next_observations,
        institutional_cost=None,
    )
