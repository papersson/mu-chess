"""MuZero training loop.

Implements the training algorithm from the MuZero paper:
- Unrolled training with K imaginary steps
- Combined loss: policy, value, and reward
- Gradient scaling for dynamics network
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from muzero.networks import MuZeroNetwork, scale_gradient
from muzero.replay import ReplayBuffer, TrainingBatch, create_synthetic_game
from muzero.utils import get_device

if TYPE_CHECKING:
    from muzero.config import Config


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""

    total_loss: float
    policy_loss: float
    value_loss: float
    reward_loss: float


class MuZeroTrainer:
    """Trainer for MuZero networks."""

    def __init__(
        self,
        config: Config,
        device: str = "auto",
    ) -> None:
        """Initialize the trainer.

        Args:
            config: Complete MuZero configuration
            device: Device to train on ("auto", "cuda", "mps", "cpu")
        """
        self.config = config
        self.device = get_device(device)

        # Create network
        self.network = MuZeroNetwork(config.network).to(self.device)

        # Create optimizer with weight decay
        self.optimizer = Adam(
            self.network.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            data_dir=config.paths.data_dir,
            buffer_size=config.replay.buffer_size,
            action_space_size=config.network.action_space_size,
        )

        # Training state
        self.training_step = 0

    def train_step(self, batch: TrainingBatch) -> TrainingMetrics:
        """Run a single training step.

        Args:
            batch: Training batch

        Returns:
            Training metrics
        """
        self.network.train()
        batch = batch.to(self.device)

        # Compute loss
        loss, metrics = self._compute_loss(batch)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_step += 1

        return metrics

    def _compute_loss(
        self, batch: TrainingBatch
    ) -> tuple[torch.Tensor, TrainingMetrics]:
        """Compute the MuZero loss.

        L = Σₜ [ L_policy(πₜ, pₜ) + L_value(zₜ, vₜ) + L_reward(uₜ, rₜ) ]

        Args:
            batch: Training batch

        Returns:
            Tuple of (loss tensor, metrics)
        """
        batch_size = batch.observations.shape[0]
        unroll_steps = batch.actions.shape[1]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_reward_loss = 0.0

        # Initial inference on real observation
        hidden, policy_logits, value = self.network.initial_inference(
            batch.observations
        )

        # Loss at t=0
        policy_loss = self._policy_loss(policy_logits, batch.target_policies[:, 0])
        value_loss = self._value_loss(value, batch.target_values[:, 0])

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()

        total_loss = policy_loss + value_loss

        # Unroll K steps
        for k in range(unroll_steps):
            # Get actions for this step
            actions = batch.actions[:, k]

            # Recurrent inference
            hidden, reward, policy_logits, value = self.network.recurrent_inference(
                hidden, actions
            )

            # Scale gradient for dynamics (per MuZero paper)
            hidden = scale_gradient(hidden, self.config.training.gradient_scale)

            # Losses at step k+1
            policy_loss = self._policy_loss(
                policy_logits, batch.target_policies[:, k + 1]
            )
            value_loss = self._value_loss(value, batch.target_values[:, k + 1])
            reward_loss = self._reward_loss(reward, batch.target_rewards[:, k])

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_reward_loss += reward_loss.item()

            total_loss = total_loss + policy_loss + value_loss + reward_loss

        # Average over unroll steps
        num_predictions = unroll_steps + 1
        metrics = TrainingMetrics(
            total_loss=total_loss.item() / num_predictions,
            policy_loss=total_policy_loss / num_predictions,
            value_loss=total_value_loss / num_predictions,
            reward_loss=total_reward_loss / unroll_steps if unroll_steps > 0 else 0.0,
        )

        return total_loss / num_predictions, metrics

    def _policy_loss(
        self, policy_logits: torch.Tensor, target_policy: torch.Tensor
    ) -> torch.Tensor:
        """Compute policy loss (cross-entropy).

        Args:
            policy_logits: Predicted policy logits (B, action_space_size)
            target_policy: Target policy distribution (B, action_space_size)

        Returns:
            Policy loss
        """
        # Cross-entropy: -Σ target * log(softmax(logits))
        log_probs = F.log_softmax(policy_logits, dim=-1)
        return -torch.sum(target_policy * log_probs, dim=-1).mean()

    def _value_loss(
        self, value: torch.Tensor, target_value: torch.Tensor
    ) -> torch.Tensor:
        """Compute value loss (MSE).

        Args:
            value: Predicted value (B,)
            target_value: Target value (B,)

        Returns:
            Value loss
        """
        return F.mse_loss(value, target_value)

    def _reward_loss(
        self, reward: torch.Tensor, target_reward: torch.Tensor
    ) -> torch.Tensor:
        """Compute reward loss (MSE).

        Args:
            reward: Predicted reward (B,)
            target_reward: Target reward (B,)

        Returns:
            Reward loss
        """
        return F.mse_loss(reward, target_reward)

    def train(
        self,
        num_steps: int,
        checkpoint_interval: int = 1000,
        log_interval: int = 100,
    ) -> None:
        """Main training loop.

        Args:
            num_steps: Number of training steps
            checkpoint_interval: Steps between checkpoints
            log_interval: Steps between log messages
        """
        # Load games if available
        num_games = self.replay_buffer.load_games()
        if num_games == 0:
            print("No games found, using synthetic data for testing")
            for _ in range(100):
                self.replay_buffer.add_game(create_synthetic_game())

        print(f"Training with {len(self.replay_buffer)} positions from {self.replay_buffer.num_games} games")
        print(f"Device: {self.device}")

        progress = tqdm(range(num_steps), desc="Training")
        running_loss = 0.0
        running_policy = 0.0
        running_value = 0.0
        running_reward = 0.0

        for step in progress:
            # Sample batch
            batch = self.replay_buffer.sample_batch(
                batch_size=self.config.training.batch_size,
                unroll_steps=self.config.training.unroll_steps,
                td_steps=self.config.training.td_steps,
                discount=self.config.training.discount,
            )

            # Train step
            metrics = self.train_step(batch)

            # Update running averages
            running_loss += metrics.total_loss
            running_policy += metrics.policy_loss
            running_value += metrics.value_loss
            running_reward += metrics.reward_loss

            # Log progress
            if (step + 1) % log_interval == 0:
                avg_loss = running_loss / log_interval
                avg_policy = running_policy / log_interval
                avg_value = running_value / log_interval
                avg_reward = running_reward / log_interval

                progress.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    policy=f"{avg_policy:.4f}",
                    value=f"{avg_value:.4f}",
                    reward=f"{avg_reward:.4f}",
                )

                running_loss = 0.0
                running_policy = 0.0
                running_value = 0.0
                running_reward = 0.0

            # Checkpoint
            if (step + 1) % checkpoint_interval == 0:
                self.save_checkpoint(f"checkpoint_{step + 1}.pt")

        # Final checkpoint
        self.save_checkpoint("checkpoint_final.pt")

    def save_checkpoint(self, filename: str) -> Path:
        """Save a training checkpoint.

        Args:
            filename: Checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = Path(self.config.paths.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / filename

        checkpoint = {
            "training_step": self.training_step,
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "network": self.config.network.__dict__,
                "training": self.config.training.__dict__,
            },
        }

        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load a training checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.training_step = checkpoint["training_step"]
        self.network.load_state_dict(checkpoint["network_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])

    def evaluate(self, num_batches: int = 10) -> TrainingMetrics:
        """Evaluate the network on held-out data.

        Args:
            num_batches: Number of batches to evaluate

        Returns:
            Average metrics
        """
        self.network.eval()

        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_reward = 0.0

        with torch.no_grad():
            for _ in range(num_batches):
                batch = self.replay_buffer.sample_batch(
                    batch_size=self.config.training.batch_size,
                    unroll_steps=self.config.training.unroll_steps,
                    td_steps=self.config.training.td_steps,
                    discount=self.config.training.discount,
                )
                batch = batch.to(self.device)

                _, metrics = self._compute_loss(batch)

                total_loss += metrics.total_loss
                total_policy += metrics.policy_loss
                total_value += metrics.value_loss
                total_reward += metrics.reward_loss

        return TrainingMetrics(
            total_loss=total_loss / num_batches,
            policy_loss=total_policy / num_batches,
            value_loss=total_value / num_batches,
            reward_loss=total_reward / num_batches,
        )
