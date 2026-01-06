"""Replay buffer for MuZero training.

Loads game trajectories from MessagePack files and samples batches
with unrolled trajectories for training.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch

from muzero.utils import NUM_OBSERVATION_PLANES


@dataclass
class GameStep:
    """A single step in a game trajectory."""

    observation: np.ndarray  # (21, 8, 8)
    action: int  # Action index
    mcts_policy: dict[int, float]  # Sparse policy {action_idx: probability}
    reward: float  # Step reward (usually 0 for chess)


@dataclass
class GameTrajectory:
    """A complete game trajectory."""

    steps: list[GameStep]
    outcome: float  # +1 white wins, -1 black wins, 0 draw
    metadata: dict[str, Any]

    def __len__(self) -> int:
        return len(self.steps)


@dataclass
class TrainingBatch:
    """A batch of training samples."""

    observations: torch.Tensor  # (B, 21, 8, 8)
    actions: torch.Tensor  # (B, K) actions for K unroll steps
    target_policies: torch.Tensor  # (B, K+1, action_space_size)
    target_values: torch.Tensor  # (B, K+1)
    target_rewards: torch.Tensor  # (B, K)
    sample_indices: list[int] | None = None  # For priority updates

    def to(self, device: torch.device) -> TrainingBatch:
        """Move batch to device."""
        return TrainingBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            target_policies=self.target_policies.to(device),
            target_values=self.target_values.to(device),
            target_rewards=self.target_rewards.to(device),
            sample_indices=self.sample_indices,
        )


class ReplayBuffer:
    """Replay buffer that loads games from disk and samples training batches."""

    def __init__(
        self,
        data_dir: str | Path,
        buffer_size: int = 100000,
        action_space_size: int = 65536,
        priority_alpha: float = 1.0,
    ) -> None:
        """Initialize the replay buffer.

        Args:
            data_dir: Directory containing game MessagePack files
            buffer_size: Maximum number of games to keep in memory
            action_space_size: Size of the action space for dense policy
            priority_alpha: Exponent for priority-based sampling (0 = uniform)
        """
        self.data_dir = Path(data_dir)
        self.buffer_size = buffer_size
        self.action_space_size = action_space_size
        self.priority_alpha = priority_alpha

        # Cached games (most recent)
        self.games: list[GameTrajectory] = []

        # Index mapping: (game_idx, step_idx) for sampling
        self._position_index: list[tuple[int, int]] = []

        # Priority tracking for prioritized experience replay
        self._priorities: np.ndarray = np.array([], dtype=np.float32)
        self._max_priority: float = 1.0

    def load_games(self, max_games: int | None = None) -> int:
        """Load games from the data directory.

        Args:
            max_games: Maximum number of games to load (None for buffer_size)

        Returns:
            Number of games loaded
        """
        if not self.data_dir.exists():
            return 0

        max_games = max_games or self.buffer_size
        # Search recursively for game files in subdirectories
        game_files = sorted(self.data_dir.glob("**/*.msgpack"))[-max_games:]

        loaded = 0
        for game_file in game_files:
            try:
                game = self._load_game(game_file)
                if game is not None and len(game) > 0:
                    self.games.append(game)
                    loaded += 1
            except Exception:
                continue  # Skip corrupted files

        self._rebuild_index()
        return loaded

    def add_game(self, trajectory: GameTrajectory) -> None:
        """Add a game to the buffer.

        Args:
            trajectory: Game trajectory to add
        """
        self.games.append(trajectory)

        # Trim buffer if needed
        if len(self.games) > self.buffer_size:
            self.games = self.games[-self.buffer_size :]

        self._rebuild_index()

    def sample_batch(
        self,
        batch_size: int,
        unroll_steps: int,
        td_steps: int,
        discount: float = 1.0,
    ) -> TrainingBatch:
        """Sample a batch of training data with optional priority sampling.

        Args:
            batch_size: Number of samples in the batch
            unroll_steps: Number of steps to unroll for dynamics
            td_steps: Number of steps for TD target computation
            discount: Discount factor for value targets

        Returns:
            TrainingBatch ready for training
        """
        if not self._position_index:
            raise RuntimeError("No games in buffer. Call load_games() first.")

        # Sample positions with priority-based sampling
        if self.priority_alpha > 0 and len(self._priorities) > 0:
            # P(i) = p_i^alpha / sum(p^alpha)
            probs = self._priorities**self.priority_alpha
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
                indices = np.random.choice(
                    len(self._position_index), size=batch_size, p=probs, replace=True
                ).tolist()
            else:
                # Fallback to uniform if all priorities are zero
                indices = random.choices(range(len(self._position_index)), k=batch_size)
        else:
            # Uniform sampling (priority_alpha = 0)
            indices = random.choices(range(len(self._position_index)), k=batch_size)

        positions = [self._position_index[i] for i in indices]

        observations = []
        actions_list = []
        target_policies = []
        target_values = []
        target_rewards = []

        for game_idx, step_idx in positions:
            game = self.games[game_idx]
            obs, acts, pols, vals, rews = self._make_sample(
                game, step_idx, unroll_steps, td_steps, discount
            )
            observations.append(obs)
            actions_list.append(acts)
            target_policies.append(pols)
            target_values.append(vals)
            target_rewards.append(rews)

        return TrainingBatch(
            observations=torch.tensor(np.array(observations), dtype=torch.float32),
            actions=torch.tensor(np.array(actions_list), dtype=torch.long),
            target_policies=torch.tensor(np.array(target_policies), dtype=torch.float32),
            target_values=torch.tensor(np.array(target_values), dtype=torch.float32),
            target_rewards=torch.tensor(np.array(target_rewards), dtype=torch.float32),
            sample_indices=indices,
        )

    def update_priorities(
        self,
        indices: list[int],
        td_errors: np.ndarray,
        epsilon: float = 1e-6,
    ) -> None:
        """Update priorities based on TD errors.

        Args:
            indices: Batch indices that were sampled (from sample_indices)
            td_errors: Absolute TD errors for each sample
            epsilon: Small constant for numerical stability
        """
        new_priorities = np.abs(td_errors) + epsilon
        for idx, priority in zip(indices, new_priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = priority
        if len(new_priorities) > 0:
            self._max_priority = max(self._max_priority, float(np.max(new_priorities)))

    def _make_sample(
        self,
        game: GameTrajectory,
        start_idx: int,
        unroll_steps: int,
        td_steps: int,
        discount: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create a single training sample from a game position.

        Returns:
            Tuple of (observation, actions, policies, values, rewards)
        """
        # Initial observation
        observation = game.steps[start_idx].observation

        # Collect K+1 policies, K+1 values, K actions, K rewards
        actions = np.zeros(unroll_steps, dtype=np.int64)
        policies = np.zeros((unroll_steps + 1, self.action_space_size), dtype=np.float32)
        values = np.zeros(unroll_steps + 1, dtype=np.float32)
        rewards = np.zeros(unroll_steps, dtype=np.float32)

        for k in range(unroll_steps + 1):
            step_idx = start_idx + k

            if step_idx < len(game):
                step = game.steps[step_idx]

                # Policy (convert sparse to dense)
                for action_idx, prob in step.mcts_policy.items():
                    policies[k, action_idx] = prob

                # Value target (n-step return)
                values[k] = self._compute_n_step_return(game, step_idx, td_steps, discount)

                # Action and reward for unroll (not for last policy)
                if k < unroll_steps:
                    actions[k] = step.action
                    rewards[k] = step.reward
            else:
                # Past end of game: use uniform policy and game outcome
                policies[k] = 1.0 / self.action_space_size
                values[k] = game.outcome
                if k < unroll_steps:
                    actions[k] = 0  # Dummy action
                    rewards[k] = 0.0

        return observation, actions, policies, values, rewards

    def _compute_n_step_return(
        self,
        game: GameTrajectory,
        start_idx: int,
        n: int,
        discount: float,
    ) -> float:
        """Compute n-step bootstrapped return.

        z_t = u_{t+1} + gamma*u_{t+2} + ... + gamma^{n-1}*u_{t+n} + gamma^n * v_{t+n}

        For chess with discount=1 and rewards=0, this simplifies to
        the game outcome at the final position.
        """
        if discount == 1.0:
            # Simplified: just use game outcome
            return game.outcome

        # General case with discounting
        total = 0.0
        for i in range(n):
            step_idx = start_idx + i
            if step_idx >= len(game) - 1:
                break
            total += (discount**i) * game.steps[step_idx].reward

        # Bootstrap from final value
        bootstrap_idx = min(start_idx + n, len(game) - 1)
        if bootstrap_idx < len(game):
            # Use game outcome as terminal value
            total += (discount**n) * game.outcome

        return total

    def _rebuild_index(self) -> None:
        """Rebuild the position index and priority array."""
        self._position_index = []
        for game_idx, game in enumerate(self.games):
            for step_idx in range(len(game)):
                self._position_index.append((game_idx, step_idx))

        # Initialize priorities with max priority for new positions
        num_positions = len(self._position_index)
        self._priorities = np.full(num_positions, self._max_priority, dtype=np.float32)

    def _load_game(self, path: Path) -> GameTrajectory | None:
        """Load a game from MessagePack file."""
        with path.open("rb") as f:
            data = msgpack.unpack(f, raw=False, strict_map_key=False)

        steps = []
        for step_data in data.get("steps", []):
            # Handle observation - could be flattened or pre-encoded
            obs_data = step_data.get("observation", [])
            if isinstance(obs_data, list) and len(obs_data) == NUM_OBSERVATION_PLANES * 64:
                obs = np.array(obs_data, dtype=np.float32).reshape(NUM_OBSERVATION_PLANES, 8, 8)
            else:
                obs = np.zeros((NUM_OBSERVATION_PLANES, 8, 8), dtype=np.float32)

            # Handle sparse policy
            policy_data = step_data.get("mcts_policy", {})
            if isinstance(policy_data, dict):
                policy = {int(k): float(v) for k, v in policy_data.items()}
            else:
                policy = {}

            steps.append(
                GameStep(
                    observation=obs,
                    action=int(step_data.get("action", 0)),
                    mcts_policy=policy,
                    reward=float(step_data.get("reward", 0.0)),
                )
            )

        return GameTrajectory(
            steps=steps,
            outcome=float(data.get("outcome", 0.0)),
            metadata=data.get("metadata", {}),
        )

    def __len__(self) -> int:
        """Return number of positions in buffer."""
        return len(self._position_index)

    @property
    def num_games(self) -> int:
        """Return number of games in buffer."""
        return len(self.games)


def create_synthetic_game(
    num_steps: int = 50,
    action_space_size: int = 65536,
) -> GameTrajectory:
    """Create a synthetic game for testing.

    Args:
        num_steps: Number of steps in the game
        action_space_size: Size of action space

    Returns:
        Synthetic GameTrajectory
    """
    steps = []
    for _ in range(num_steps):
        # Random observation
        obs = np.random.randn(NUM_OBSERVATION_PLANES, 8, 8).astype(np.float32)

        # Random sparse policy with ~30 legal moves
        num_legal = random.randint(20, 40)
        legal_actions = random.sample(range(action_space_size), num_legal)
        probs = np.random.dirichlet([1.0] * num_legal)
        policy = {a: float(p) for a, p in zip(legal_actions, probs)}

        # Random action from policy
        action = random.choice(legal_actions)

        steps.append(
            GameStep(
                observation=obs,
                action=action,
                mcts_policy=policy,
                reward=0.0,
            )
        )

    # Random outcome
    outcome = random.choice([-1.0, 0.0, 1.0])

    return GameTrajectory(
        steps=steps,
        outcome=outcome,
        metadata={"synthetic": True},
    )
