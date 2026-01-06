"""MuZero neural network architectures.

Implements the three core networks:
- Representation h(observation) -> hidden_state
- Dynamics g(hidden_state, action) -> (next_hidden_state, reward)
- Prediction f(hidden_state) -> (policy, value)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from muzero.config import NetworkConfig


class ResidualBlock(nn.Module):
    """Pre-activation residual block.

    Uses the pre-activation variant: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        return out + residual


class RepresentationNetwork(nn.Module):
    """Representation network h(observation) -> hidden_state.

    Encodes the raw observation (board state) into a latent hidden state
    that the dynamics network can operate on.

    Input: (batch, num_planes, 8, 8) - e.g., (B, 21, 8, 8) for chess
    Output: (batch, hidden_dim, 8, 8)
    """

    def __init__(
        self,
        num_observation_planes: int,
        hidden_dim: int,
        num_res_blocks: int,
    ) -> None:
        super().__init__()

        # Initial convolution to project observation to hidden dimension
        self.conv_in = nn.Conv2d(
            num_observation_planes, hidden_dim, kernel_size=3, padding=1, bias=False
        )
        self.bn_in = nn.BatchNorm2d(hidden_dim)

        # Stack of residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Encode observation to hidden state.

        Args:
            observation: Board state tensor (B, num_planes, 8, 8)

        Returns:
            Hidden state tensor (B, hidden_dim, 8, 8)
        """
        x = F.relu(self.bn_in(self.conv_in(observation)))

        for block in self.res_blocks:
            x = block(x)

        return x


class DynamicsNetwork(nn.Module):
    """Dynamics network g(hidden_state, action) -> (next_hidden_state, reward).

    Models the game dynamics in latent space. Given a hidden state and an action,
    predicts the next hidden state and the reward for that transition.

    Input:
        - hidden_state: (batch, hidden_dim, 8, 8)
        - action: (batch,) action indices

    Output:
        - next_hidden_state: (batch, hidden_dim, 8, 8)
        - reward: (batch,) scalar rewards
    """

    def __init__(
        self,
        hidden_dim: int,
        num_res_blocks: int,
        action_space_size: int,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_space_size = action_space_size

        # Action encoding: We encode the action as a spatial plane
        # For chess moves encoded as (from_sq, to_sq), we create a 2-plane encoding
        # where one plane has 1 at the from-square and another at the to-square
        self.action_planes = 2

        # Convolution to combine hidden state and action encoding
        self.conv_in = nn.Conv2d(
            hidden_dim + self.action_planes, hidden_dim, kernel_size=3, padding=1, bias=False
        )
        self.bn_in = nn.BatchNorm2d(hidden_dim)

        # Residual blocks for state transition
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])

        # Reward prediction head
        self.reward_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.reward_bn = nn.BatchNorm2d(1)
        self.reward_fc = nn.Linear(64, 1)  # 8x8 = 64

    def _encode_action(
        self, action: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Encode action indices as spatial planes.

        For chess moves encoded in 16-bit format: from(6) | to(6) | flags(4)

        Args:
            action: Action indices (B,)
            batch_size: Batch size
            device: Device for tensor creation

        Returns:
            Action planes (B, 2, 8, 8) with from and to squares marked
        """
        planes = torch.zeros(batch_size, 2, 8, 8, device=device)

        # Extract from and to squares from the action encoding
        # Assuming format: bits 0-5 = from, bits 6-11 = to
        from_sq = action & 0x3F  # Lower 6 bits
        to_sq = (action >> 6) & 0x3F  # Next 6 bits

        # Convert square indices to (row, col)
        from_row = from_sq // 8
        from_col = from_sq % 8
        to_row = to_sq // 8
        to_col = to_sq % 8

        # Mark squares in the planes
        batch_indices = torch.arange(batch_size, device=device)
        planes[batch_indices, 0, from_row, from_col] = 1.0
        planes[batch_indices, 1, to_row, to_col] = 1.0

        return planes

    def forward(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next state and reward from state-action pair.

        Args:
            hidden_state: Current hidden state (B, hidden_dim, 8, 8)
            action: Action indices (B,)

        Returns:
            Tuple of (next_hidden_state, reward)
        """
        batch_size = hidden_state.shape[0]
        device = hidden_state.device

        # Encode action as spatial planes
        action_planes = self._encode_action(action, batch_size, device)

        # Concatenate hidden state and action encoding
        x = torch.cat([hidden_state, action_planes], dim=1)

        # Process through network
        x = F.relu(self.bn_in(self.conv_in(x)))

        for block in self.res_blocks:
            x = block(x)

        next_hidden = x

        # Predict reward
        reward = self.reward_conv(x)
        reward = F.relu(self.reward_bn(reward))
        reward = reward.view(batch_size, -1)
        reward = self.reward_fc(reward).squeeze(-1)

        return next_hidden, reward


class PredictionNetwork(nn.Module):
    """Prediction network f(hidden_state) -> (policy, value).

    Evaluates a hidden state to produce a policy distribution and value estimate.

    Input: hidden_state (batch, hidden_dim, 8, 8)
    Output:
        - policy: (batch, action_space_size) logits
        - value: (batch,) scalar in [-1, 1]
    """

    def __init__(
        self,
        hidden_dim: int,
        action_space_size: int,
    ) -> None:
        super().__init__()

        # Policy head
        self.policy_conv = nn.Conv2d(hidden_dim, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 64, action_space_size)  # 32 channels * 8x8

        # Value head
        self.value_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)  # 8x8 = 64
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate hidden state to get policy and value.

        Args:
            hidden_state: Hidden state tensor (B, hidden_dim, 8, 8)

        Returns:
            Tuple of (policy_logits, value)
        """
        batch_size = hidden_state.shape[0]

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(hidden_state)))
        policy = policy.view(batch_size, -1)
        policy = self.policy_fc(policy)  # Logits, not softmax

        # Value head
        value = F.relu(self.value_bn(self.value_conv(hidden_state)))
        value = value.view(batch_size, -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)).squeeze(-1)

        return policy, value


class MuZeroNetwork(nn.Module):
    """Complete MuZero network combining h, g, and f.

    Provides two main inference methods:
    - initial_inference: For the root node (real observation)
    - recurrent_inference: For tree expansion (imagined states)
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config

        self.representation = RepresentationNetwork(
            num_observation_planes=config.num_observation_planes,
            hidden_dim=config.hidden_dim,
            num_res_blocks=config.num_res_blocks,
        )

        self.dynamics = DynamicsNetwork(
            hidden_dim=config.hidden_dim,
            num_res_blocks=config.num_res_blocks,
            action_space_size=config.action_space_size,
        )

        self.prediction = PredictionNetwork(
            hidden_dim=config.hidden_dim,
            action_space_size=config.action_space_size,
        )

    def initial_inference(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run initial inference from a real observation.

        This is used at the root of the MCTS tree to encode the actual
        game state into the latent representation.

        Args:
            observation: Game observation (B, num_planes, 8, 8)

        Returns:
            Tuple of (hidden_state, policy_logits, value)
        """
        hidden_state = self.representation(observation)
        policy, value = self.prediction(hidden_state)
        return hidden_state, policy, value

    def recurrent_inference(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run recurrent inference for tree expansion.

        This is used during MCTS to expand the tree by imagining
        what would happen after taking an action.

        Args:
            hidden_state: Current hidden state (B, hidden_dim, 8, 8)
            action: Action indices (B,)

        Returns:
            Tuple of (next_hidden_state, reward, policy_logits, value)
        """
        next_hidden, reward = self.dynamics(hidden_state, action)
        policy, value = self.prediction(next_hidden)
        return next_hidden, reward, policy, value

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Get all network weights as a dictionary."""
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load weights from a dictionary."""
        self.load_state_dict(weights)


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Scale gradients during backward pass.

    This is used to scale the gradient of the dynamics network
    relative to other networks, as recommended in the MuZero paper.

    Args:
        tensor: Input tensor
        scale: Gradient scale factor (e.g., 0.5)

    Returns:
        Tensor with scaled gradients
    """
    return tensor * scale + tensor.detach() * (1 - scale)
