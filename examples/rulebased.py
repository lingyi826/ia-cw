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
from src.observations import Observation
from src.scenario import Scenario


class RuleBasedAgent(Agent):
    """Learn when to work, clean or idle.

    Physical task and room features are grouped into quarter-unit bins.
    The environment binds actions to the task room or current room.
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
        actions = observation.task_actions
        available = [index for index, action in enumerate(actions) if action is not None]

        # The observation contains information about agents, rooms, tasks, and the current state.       
        print("***************************")
        print("OBSERVATION INFORMATION: ")
        print("***************************")
        # To get all agents: 
        print(observation.state.agents)
        print("***************************")
        # To get all rooms: 
        print(observation.state.rooms)
        print("***************************")
        # To get all tasks: 
        print(observation.state.tasks)
        print("***************************")

        # The observation also contains the subject, which is the agent that is making the decision.
        # To get the subject: 
        print(observation.subject)
        print("***************************")
        # The observation also contains the available actions, which are the actions that the agent can take. 
        # To get the available actions:
        print(observation.available_actions)
        print("***************************")

        ## Your solution should implement a decision rule here that follows the norm specification from Task 1.
        # The Observation should give you all the information that you need. Specific information about a task 
        # can be accessed by using the task id to index into the observation.state.tasks dictionary.
        # For example, to get the details of task with id 1:
        task_id = 1
        task_details = observation.state.tasks[task_id]
        print(f"Details of task with id {task_id}: {task_details}")

        ## NOTE: If you can't access the information you need to write your implementation, reach out to the TAs for help.

        ## TODO ADD YOUR IMPLEMENTATION HERE
        
        # Starting template agent, return a random action to take.
        index = self.random.choice(available)
        action = actions[index]
        assert action is not None
        return Decision(action, (state, index))