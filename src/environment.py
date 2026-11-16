"""State ownership"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from random import Random

from src.actions import JointAction
from src.configuration import load_config
from src.observations import Observation, observe
from src.scenario import Scenario
from src.state import WorldState
from src.transitions import StepResult, advance, initial_state


class Environment:
    """Adopt candidate states only after all model callbacks have succeeded."""

    def __init__(
        self,
        scenario: Scenario | None = None,
        *,
        seed: int | None = None,
        config: str | PathLike[str] | None = None,
    ) -> None:
        if scenario is None:
            loaded = load_config(config)
            scenario = loaded.scenario
            seed = loaded.seed if seed is None else seed
        elif seed is None:
            seed = 0
        self._scenario = scenario
        self._seed = seed
        self._random = Random(self._seed)
        self._state = initial_state(scenario, self._seed, self._random)
        self._observations = observe(self._state)

    @property
    def state(self) -> WorldState:
        return self._state

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    def observe(self) -> Mapping[str, Observation]:
        return self._observations

    def reset(self, *, seed: int | None = None) -> Mapping[str, Observation]:
        episode_seed = self._seed if seed is None else seed
        rng = Random(episode_seed)
        candidate = initial_state(self._scenario, episode_seed, rng)
        observations = observe(candidate)
        self._state, self._observations, self._random = candidate, observations, rng
        return observations

    def step(self, joint: JointAction) -> StepResult:
        rng = Random(0)
        rng.setstate(self._random.getstate())
        result = advance(
            self._state,
            joint,
            self._scenario,
            self._observations,
            rng,
        )
        self._state, self._observations, self._random = result.after, result.observations, rng
        return result