"""Read physical environment settings from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from os import PathLike
from pathlib import Path

from yaml import safe_load

from src.scenario import AgentSpec, RoomSpec, Scenario, TaskConfig, WorldConfig


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    scenario: Scenario
    seed: int


def default_config_path() -> Path:
    """Locate the editable source file, or the installed distribution's copy."""
    source = Path(__file__).with_name("environment.yaml")
    if source.is_file():
        return source
    try:
        installed = distribution("internalisation-sim")
    except PackageNotFoundError:
        return source
    for entry in installed.files or ():
        if entry.as_posix().endswith("share/internalisation-sim/environment.yaml"):
            return Path(str(installed.locate_file(entry)))
    return source


def load_config(path: str | PathLike[str] | None = None) -> EnvironmentConfig:
    """Load YAML into scenario records, using Python field defaults."""
    file = default_config_path() if path is None else Path(path)
    values = safe_load(file.read_text(encoding="utf-8"))
    scenario = Scenario(
        rooms=tuple(RoomSpec(**room) for room in values["rooms"]),
        agents=tuple(AgentSpec(**agent) for agent in values["agents"]),
        world=WorldConfig(**values["world"]),
        tasks=TaskConfig(**values["tasks"]),
    )
    return EnvironmentConfig(scenario, values["seed"])
