"""Directed work, cleaning and idle commitments, followed by a rule-based agent."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from src.actions import Clean, Idle, Outcome, Work
from src.agents import Agent, Decision

from src.observations import Observation
from src.scenario import Scenario
from src.state import WorldState

import random
from typing import cast

from src.agents import Decision, Experience, LearningAgent
from examples.learning import TaskEncoder
from src.actions import Idle
from src.agents import Agent, Decision
from src.observations import Observation
from src.scenario import Scenario


class RuleBasedAgent(Agent):
    """Choose actions using one normative specification for every task mode.

    N1 Commitment: complete assigned work before the deadline.
    N2 Commitment: contribute whenever the current task can progress.
    N3 Prohibition: do not perform useless or room-closing Work.
    N4 Authorization: Clean or Idle when its condition is satisfied.
    N5 Power: room locking, restoration and action rejection belong to the
        environment, so the agent respects the actions made available to it.
    N6 Sanction: costs and sanctions are applied by the environment; the agent
        avoids the actions that trigger them.
    """

    def __init__(self, scenario: Scenario, *, seed: int = 0) -> None:
        self.encoder = TaskEncoder(scenario)
        self.random = random.Random(seed)
        self.values: dict[tuple[tuple[int, ...], int], float] = {} #Stores the Q-Values of states

    def key(self, observation: Observation) -> tuple[int, ...]:
        return tuple(round(4 * value) for value in self.encoder(observation))

    def decide(self, observation: Observation) -> Decision:
        """
        Given the observation, make a decision on what action to take
        The observation will contain all information about agents, rooms, tasks, and the current state.
        """
        state = self.key(observation)
        # actions = observation.task_actions
        # available = [index for index, action in enumerate(actions) if action is not None]

        # # The observation contains information about agents, rooms, tasks, and the current state.       
        # print("***************************")
        # print("OBSERVATION INFORMATION: ")
        # print("***************************")
        # # To get all agents: 
        # print(observation.state.agents)
        # print("***************************")
        # # To get all rooms: 
        # print(observation.state.rooms)
        # print("***************************")
        # # To get all tasks: 
        # print(observation.state.tasks)
        # print("***************************")

        # # The observation also contains the subject, which is the agent that is making the decision.
        # # To get the subject: 
        # print(observation.subject)
        # print("***************************")
        # # The observation also contains the available actions, which are the actions that the agent can take. 
        # # To get the available actions:
        # print(observation.available_actions)
        # print("***************************")

        ## Your solution should implement a decision rule here that follows the norm specification from Task 1.
        # The Observation should give you all the information that you need. Specific information about a task 
        # can be accessed by using the task id to index into the observation.state.tasks dictionary.

        ## TODO ADD YOUR IMPLEMENTATION HERE
        task = observation.task
        work, clean, idle = observation.task_actions
        scenario = self.encoder._scenario

        # N3 PROHIBITION: without a task, Work would damage a room but produce
        # no progress.
        # N4 AUTHORIZATION: a free agent may Clean a dirty room or Idle when the
        # room is already clean.
        if task is None:
            room = observation.room
            if room is not None and clean is not None and room.cleanliness < room.initial_cleanliness:
                return Decision(clean, (state, "N4: authorised maintenance"))
            return Decision(idle, (state, "N4: authorised idle"))

        room = observation.state.rooms[task.room_id]

        # N2 GENERAL COOPERATION COMMITMENT: can_progress() is the one common
        # test for all task configurations. For an individual task it means the
        # agent still has useful work. For cooperative tasks it also respects
        # any dependency on another participant, without separate policies for
        # simultaneous and sequential tasks.
        eligible_to_contribute = task.can_progress(observation.subject)
        if not eligible_to_contribute:
            if clean is not None and room.cleanliness < room.initial_cleanliness:
                return Decision(clean, (state, "N2/N4: clean while contribution is not due"))
            return Decision(idle, (state, "N2/N4: wait until contribution is due"))

        # N3 needs the total predicted damage, so count every agent who could
        # Work in this office during the same step. This also covers agents on
        # different individual tasks that happen to share an office.
        expected_workers = 0
        for actor, agent in observation.state.agents.items():
            if agent.task_id is None:
                continue
            other_task = observation.state.tasks[agent.task_id]
            if other_task.room_id == room.id and other_task.can_progress(actor):
                expected_workers += 1

        cleanliness_after_work = (room.cleanliness - scenario.world.task_damage * max(1, expected_workers))

        # N3 PROHIBITION takes priority over N1: do not Work if the predicted
        # joint damage would close the office.
        # N4 AUTHORIZATION allows Clean as the safe alternative.
        if cleanliness_after_work < room.accessible_threshold:
            if clean is not None:
                return Decision(clean, (state, "N3/N4: clean before unsafe work"))
            return Decision(idle, (state, "N3: unsafe work is prohibited"))

        # N1 TASK COMMITMENT: after the higher-priority safety and cooperation
        # checks have passed, available Work is useful and should be performed.
        if work is not None:
            return Decision(work, (state, "N1/N2: make a valid task contribution"))

        # N4 AUTHORIZATION: if Work is temporarily unavailable for another
        # reason, Clean a dirty accessible room or otherwise Idle.
        if clean is not None and room.cleanliness < room.initial_cleanliness:
            return Decision(clean, (state, "N4: authorised maintenance"))

        return Decision(Idle(), (state, "N4: authorised idle"))
