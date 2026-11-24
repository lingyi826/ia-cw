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

from src.agents import Decision
from examples.learning import TaskEncoder
from src.observations import Observation
from src.scenario import Scenario


class RandomAgent(Agent):
    """Selects random actions at any given timestep, according to a specific probability

    Each action that an agent can do is automatically filtered for the decide function. If an agent has no assigned task, then it cannot work. Likewise for clean.
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

        """Here are some basic actions that could be returned:
        available:  (Work(room_id='office2'), Clean(room_id='office2'), Idle())
        available:  (Work(room_id='office2'), Clean(room_id='office2'), Idle())
        available:  (None, None, Idle())

        The action can be: 
        - Work(room_id='some_id')
        - Idle()
        - Clean(room_id='some_id')
        """

        ## The random agent will randomly choose each task:
        # Work, Clean, Idle
        probs = [0.5, 0.2, 0.1]
        potential_actions = [Work, Clean, Idle]
        choice = self.random.choices(potential_actions, weights=probs, k=1)[0]

        # Choose the first action whose type matches the one chosen
        for index, action in enumerate(actions):
            if isinstance(action, choice):
                return Decision(action, (state, index))

        ## Our choice wasn't available, so just return a random action to take instead.
        index = self.random.choice(available)
        action = actions[index]
        assert action is not None
        return Decision(action, (state, index))
"""
Looking for observe? Observe is only used in learning. Because rule based agents do not learn it is not used here.
"""
