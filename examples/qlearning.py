"""Tabular Q learning over work, clean and idle using external rewards."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import cast
from src.agents import Decision, Experience, LearningAgent
from examples.learning import TaskEncoder
from src.observations import Observation
from src.scenario import Scenario


class QLearningAgent(LearningAgent):
    """
    Learn when to work, clean or idle.

    Physical task and room features are grouped into quarter-unit bins.
    The environment binds actions to the task room or current room.
    """

    def __init__(self, scenario: Scenario, *, seed: int = 0) -> None:
        self.encoder = TaskEncoder(scenario)
        self.random = random.Random(seed)
        self.training = True

######  KEY VARIABLES USED FOR ANSWERING QUESITONS. #####
        self.values: dict[tuple[tuple[int, ...], int], float] = {} #Stores the Q-Values of states
        self.observed = 0 #Track number of observed transitions
        self.discount_factor = 0.95
        self.learning_rate = 0.15

    def set_training(self, training: bool) -> None:
        self.training = training

    def key(self, observation: Observation) -> tuple[int, ...]:
        return tuple(round(4 * value) for value in self.encoder(observation))

    def save_values(self, path: str | Path) -> None:
        """Save the learned Q-table in a readable, deterministic JSON file."""
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        entries = [
            {
                "state": list(state),
                "action": action,
                "value": value,
            }
            for (state, action), value in sorted(self.values.items())
        ]
        payload = {
            "format": "tabular-q-values-v1",
            "feature_names": list(self.encoder.feature_names),
            "observed_transitions": self.observed,
            "discount_factor": self.discount_factor,
            "learning_rate": self.learning_rate,
            "entries": entries,
        }
        file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_values(self, path: str | Path) -> None:
        """Load a Q-table previously written by :meth:`save_values`."""
        file = Path(path)
        payload = json.loads(file.read_text(encoding="utf-8"))
        if payload.get("format") != "tabular-q-values-v1":
            raise ValueError(f"Unsupported Q-value format in {file}")
        if tuple(payload.get("feature_names", ())) != self.encoder.feature_names:
            raise ValueError(f"Q-value features in {file} do not match TaskEncoder")

        self.values = {
            (tuple(int(value) for value in entry["state"]), int(entry["action"])): float(
                entry["value"]
            )
            for entry in payload["entries"]
        }
        self.observed = int(payload["observed_transitions"])

    '''
    Given a set of observations choses an action to take. When not training greedily slectes the action with
    the greatest comrresponding Q-Value. When training also has a chance of instead selecting an action at 
    random.
    IN:
        Observation: values representing the state of the environment, contains
                     state: WorldState
                     subject: str
                     norms: tuple[NormInstance, ...]
                     available_actions: tuple[Action, ...]
    OUT:
        Decision:  The action taken by the agent, and the context of the action, contains
                   action: Action
                   metadata: object
    SIDE EFFECTS:
        None
    '''
    def decide(self, observation: Observation) -> Decision:
        state = self.key(observation)
        actions = observation.task_actions
        available = [index for index, action in enumerate(actions) if action is not None]

        #TODO REPLACE `index = self.random.choice(available)` WITH GREEDY ACTION SELECTION
        # During training, sometimes explore with a random action.  Epsilon
        # gradually falls from 0.4, but never goes below 0.05.
        epsilon = max(0.05, 0.4 * (1 - self.observed / 100000))
        if self.training and self.random.random() < epsilon:
            index = self.random.choice(available)
        else:
            # Otherwise use the best learned action.  Choosing randomly between
            # equal Q-values avoids always favouring the first action slot.
            best_value = max(self.values.get((state, slot), 0.0) for slot in available)
            best_actions = [
                slot
                for slot in available
                if self.values.get((state, slot), 0.0) == best_value
            ]
            index = self.random.choice(best_actions)
        action = actions[index]
        assert action is not None
        return Decision(action, (state, index))
    
    '''
    Updates the Q-Value of a state from a single observed transition.
    IN:
        Experience: an observed transition from the environment contains
                    observation: Observation
                    decision: Decision
                    result: StepResult
                    next_observation: Observation
    OUT:
        None
    SIDE EFFECTS:
        self.values
    '''
    def observe(self, experience: Experience) -> None:
        state, index = cast(tuple[tuple[int, ...], int], experience.decision.metadata)
        next_state = self.key(experience.next_observation)
        next_actions = experience.next_observation.task_actions
        future = max(
            self.values.get((next_state, slot), 0.0)
            for slot, action in enumerate(next_actions)
            if action is not None
        )
        reward = experience.reward.total
        old_q = self.values.get((state, index), 0.0) #current Q-Value

        # A terminal transition has no next-state value.  For any other
        # transition, include the discounted estimate of the best next action.
        if experience.result.done:
            target = reward
        else:
            target = reward + self.discount_factor * future

        # Standard Q-learning update:
        # Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
        self.values[state, index] = old_q + self.learning_rate * (target - old_q)

        self.observed += 1
