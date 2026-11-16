"""Agent decisions, learning experiences and policy contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.actions import Action
from src.observations import Observation
from src.rewards import Reward
from src.transitions import StepResult


@dataclass(frozen=True)
class Decision:
    """An action plus metadata owned exclusively by its policy."""

    action: Action
    metadata: object | None = None


@dataclass(frozen=True)
class Experience:
    """The exact decision and external transition delivered back to a learner."""

    observation: Observation
    decision: Decision
    result: StepResult
    next_observation: Observation

    @property
    def reward(self) -> Reward:
        return self.result.rewards[self.observation.subject]


class Policy(Protocol):
    def decide(self, observation: Observation) -> Decision:
        """Choose an action from a named observation."""
        ...


@runtime_checkable
class LearningPolicy(Policy, Protocol):
    def observe(self, experience: Experience) -> None:
        """Learn from the environment's external reward and causal events."""
        ...


@runtime_checkable
class ModeAwarePolicy(Policy, Protocol):
    def set_training(self, training: bool) -> None:
        """Select learning or evaluation behaviour when coordinated by a runner."""
        ...


class Agent(ABC):
    @abstractmethod
    def decide(self, observation: Observation) -> Decision: ...


class LearningAgent(Agent):
    @abstractmethod
    def observe(self, experience: Experience) -> None: ...
