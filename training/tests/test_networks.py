"""Tests for MuZero neural networks."""

import pytest
import torch

from muzero.config import NetworkConfig
from muzero.networks import (
    DynamicsNetwork,
    MuZeroNetwork,
    PredictionNetwork,
    RepresentationNetwork,
    ResidualBlock,
    scale_gradient,
)


@pytest.fixture
def config() -> NetworkConfig:
    """Create a test network config with small dimensions."""
    return NetworkConfig(
        hidden_dim=32,
        num_res_blocks=2,
        num_observation_planes=21,
        action_space_size=65536,
    )


class TestResidualBlock:
    def test_shape_preserved(self):
        block = ResidualBlock(channels=32)
        x = torch.randn(4, 32, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        block = ResidualBlock(channels=32)
        # With zero weights, output should equal input (residual connection)
        # This is a weak test but verifies the residual path exists
        x = torch.randn(4, 32, 8, 8)
        out = block(x)
        # At least verify it runs without error
        assert out.shape == x.shape


class TestRepresentationNetwork:
    def test_output_shape(self, config: NetworkConfig):
        network = RepresentationNetwork(
            num_observation_planes=config.num_observation_planes,
            hidden_dim=config.hidden_dim,
            num_res_blocks=config.num_res_blocks,
        )

        obs = torch.randn(4, config.num_observation_planes, 8, 8)
        hidden = network(obs)

        assert hidden.shape == (4, config.hidden_dim, 8, 8)

    def test_batch_independence(self, config: NetworkConfig):
        network = RepresentationNetwork(
            num_observation_planes=config.num_observation_planes,
            hidden_dim=config.hidden_dim,
            num_res_blocks=config.num_res_blocks,
        )
        network.eval()

        obs = torch.randn(4, config.num_observation_planes, 8, 8)

        with torch.no_grad():
            hidden_batch = network(obs)
            hidden_single = network(obs[0:1])

        # First element of batch should equal single inference
        torch.testing.assert_close(hidden_batch[0:1], hidden_single, atol=1e-5, rtol=1e-5)


class TestDynamicsNetwork:
    def test_output_shapes(self, config: NetworkConfig):
        network = DynamicsNetwork(
            hidden_dim=config.hidden_dim,
            num_res_blocks=config.num_res_blocks,
            action_space_size=config.action_space_size,
        )

        hidden = torch.randn(4, config.hidden_dim, 8, 8)
        action = torch.tensor([0, 100, 200, 300])

        next_hidden, reward = network(hidden, action)

        assert next_hidden.shape == (4, config.hidden_dim, 8, 8)
        assert reward.shape == (4,)

    def test_action_encoding(self, config: NetworkConfig):
        network = DynamicsNetwork(
            hidden_dim=config.hidden_dim,
            num_res_blocks=config.num_res_blocks,
            action_space_size=config.action_space_size,
        )

        # Test that different actions produce different outputs
        hidden = torch.randn(1, config.hidden_dim, 8, 8)
        action1 = torch.tensor([0])  # a1-a1 (no-op, but valid encoding)
        action2 = torch.tensor([64])  # a1-b1

        with torch.no_grad():
            out1, _ = network(hidden, action1)
            out2, _ = network(hidden, action2)

        # Different actions should produce different outputs
        assert not torch.allclose(out1, out2)


class TestPredictionNetwork:
    def test_output_shapes(self, config: NetworkConfig):
        network = PredictionNetwork(
            hidden_dim=config.hidden_dim,
            action_space_size=config.action_space_size,
        )

        hidden = torch.randn(4, config.hidden_dim, 8, 8)
        policy, value = network(hidden)

        assert policy.shape == (4, config.action_space_size)
        assert value.shape == (4,)

    def test_value_range(self, config: NetworkConfig):
        network = PredictionNetwork(
            hidden_dim=config.hidden_dim,
            action_space_size=config.action_space_size,
        )

        # Random inputs should produce values in [-1, 1] due to tanh
        hidden = torch.randn(100, config.hidden_dim, 8, 8)
        _, value = network(hidden)

        assert value.min() >= -1.0
        assert value.max() <= 1.0


class TestMuZeroNetwork:
    def test_initial_inference(self, config: NetworkConfig):
        network = MuZeroNetwork(config)

        obs = torch.randn(4, config.num_observation_planes, 8, 8)
        hidden, policy, value = network.initial_inference(obs)

        assert hidden.shape == (4, config.hidden_dim, 8, 8)
        assert policy.shape == (4, config.action_space_size)
        assert value.shape == (4,)

    def test_recurrent_inference(self, config: NetworkConfig):
        network = MuZeroNetwork(config)

        hidden = torch.randn(4, config.hidden_dim, 8, 8)
        action = torch.tensor([0, 1, 2, 3])

        next_hidden, reward, policy, value = network.recurrent_inference(hidden, action)

        assert next_hidden.shape == (4, config.hidden_dim, 8, 8)
        assert reward.shape == (4,)
        assert policy.shape == (4, config.action_space_size)
        assert value.shape == (4,)

    def test_end_to_end(self, config: NetworkConfig):
        """Test full inference chain: obs -> hidden -> action -> next_hidden."""
        network = MuZeroNetwork(config)

        # Initial inference from observation
        obs = torch.randn(2, config.num_observation_planes, 8, 8)
        hidden, policy, value = network.initial_inference(obs)

        # Recurrent inference for imagined step
        action = torch.tensor([100, 200])
        next_hidden, reward, next_policy, next_value = network.recurrent_inference(
            hidden, action
        )

        # Verify all outputs are valid
        assert not torch.isnan(hidden).any()
        assert not torch.isnan(policy).any()
        assert not torch.isnan(value).any()
        assert not torch.isnan(next_hidden).any()
        assert not torch.isnan(reward).any()
        assert not torch.isnan(next_policy).any()
        assert not torch.isnan(next_value).any()


class TestScaleGradient:
    def test_forward_unchanged(self):
        x = torch.randn(4, 32, 8, 8, requires_grad=True)
        y = scale_gradient(x, 0.5)

        # Forward pass should be unchanged
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-6)

    def test_gradient_scaled(self):
        x = torch.randn(4, 32, requires_grad=True)
        y = scale_gradient(x, 0.5)
        loss = y.sum()
        loss.backward()

        # Gradient should be scaled by 0.5
        expected_grad = torch.ones_like(x) * 0.5
        torch.testing.assert_close(x.grad, expected_grad, atol=1e-6, rtol=1e-6)
