"""Coordinate policies and yield joint transitions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from src.actions import JointAction
from src.agents import Experience, LearningPolicy, ModeAwarePolicy, Policy
from src.environment import Environment
from src.transitions import StepResult


def rollout(
    environment: Environment, policies: Mapping[str, Policy], *, learning: bool = True
) -> Iterator[StepResult]:
    """Yield transitions in memory, passing each decision back to its learner.

    This function does not reset the environment or save outputs. Policy failures
    before step() leave the world unchanged. A learner failure after step() is
    propagated after the valid physical transition has already been committed.
    """
    ordered = sorted(policies)
    for actor in ordered:
        policy = policies[actor]
        if isinstance(policy, ModeAwarePolicy):
            policy.set_training(learning)
    while not environment.state.done:
        observations = environment.observe()
        ## For every agent, get their decision based on their observation and policy. 
        # The policy is stored in a separate list
        
        # An agent can decide to do any action that is available to them in a room.
        # They can clean, work, or do nothing in any work.
        decisions = {actor: policies[actor].decide(observations[actor]) for actor in ordered}
        result = environment.step(
            JointAction(
                environment.state.step, {actor: decisions[actor].action for actor in ordered}
            )
        )
        if learning:
            for actor in ordered:
                policy = policies[actor]
                if isinstance(policy, LearningPolicy):
                    # This policy.observe() is the agents policy observation. it is only used in learning
                    policy.observe(
                        Experience(
                            observations[actor],
                            decisions[actor],
                            result,
                            result.observations[actor],
                        )
                    )
        yield result
