"""Configuration management for MuZero training.

Loads config from TOML files with environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli


@dataclass
class NetworkConfig:
    """Neural network configuration."""

    hidden_dim: int = 256
    num_res_blocks: int = 16
    num_observation_planes: int = 21
    action_space_size: int = 65536
    support_size: int = 0  # For scalar value/reward heads


@dataclass
class MctsConfig:
    """MCTS configuration."""

    num_simulations: int = 800
    dirichlet_alpha: float = 0.3
    exploration_fraction: float = 0.25
    pb_c_base: float = 19652.0
    pb_c_init: float = 1.25


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 32  # Reduced from 1024 for memory efficiency on M3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    unroll_steps: int = 5
    td_steps: int = 10
    discount: float = 1.0
    gradient_scale: float = 0.5  # Scale for dynamics gradients


@dataclass
class SelfPlayConfig:
    """Self-play configuration."""

    games_per_iteration: int = 100
    num_workers: int = 4
    temperature: float = 1.0
    temperature_drop_move: int = 30


@dataclass
class ReplayConfig:
    """Replay buffer configuration."""

    buffer_size: int = 100000
    priority_alpha: float = 1.0


@dataclass
class PathsConfig:
    """File paths configuration."""

    data_dir: str = "data/games"
    checkpoint_dir: str = "checkpoints"
    config_file: str = "config.toml"


@dataclass
class Config:
    """Complete MuZero configuration."""

    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MctsConfig = field(default_factory=MctsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> Config:
        """Load configuration from a TOML file."""
        path = Path(path)
        with path.open("rb") as f:
            data = tomli.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        """Create configuration from a dictionary."""
        return cls(
            network=_parse_section(data.get("network", {}), NetworkConfig),
            mcts=_parse_section(data.get("mcts", {}), MctsConfig),
            training=_parse_section(data.get("training", {}), TrainingConfig),
            selfplay=_parse_section(data.get("selfplay", {}), SelfPlayConfig),
            replay=_parse_section(data.get("replay", {}), ReplayConfig),
            paths=_parse_section(data.get("paths", {}), PathsConfig),
        )

    def apply_env_overrides(self) -> Config:
        """Apply environment variable overrides.

        Environment variables follow the pattern:
        MUZERO_{SECTION}_{KEY} = value

        Example: MUZERO_MCTS_NUM_SIMULATIONS=100
        """
        sections = {
            "network": self.network,
            "mcts": self.mcts,
            "training": self.training,
            "selfplay": self.selfplay,
            "replay": self.replay,
            "paths": self.paths,
        }

        for section_name, section_obj in sections.items():
            prefix = f"MUZERO_{section_name.upper()}_"
            for env_key, env_value in os.environ.items():
                if env_key.startswith(prefix):
                    field_name = env_key[len(prefix) :].lower()
                    if hasattr(section_obj, field_name):
                        field_type = type(getattr(section_obj, field_name))
                        try:
                            setattr(section_obj, field_name, field_type(env_value))
                        except (ValueError, TypeError):
                            pass  # Skip invalid conversions

        return self


def _parse_section(data: dict[str, Any], cls: type) -> Any:
    """Parse a config section into a dataclass instance."""
    # Filter to only fields that exist in the dataclass
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}
    return cls(**filtered_data)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration with environment overrides.

    Args:
        path: Path to config TOML file. If None, uses default config.

    Returns:
        Complete configuration with environment overrides applied.
    """
    if path is not None:
        config = Config.from_toml(path)
    else:
        config = Config()

    return config.apply_env_overrides()
