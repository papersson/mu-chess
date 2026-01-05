"""ONNX export utilities for MuZero networks.

Exports the MuZero network to two ONNX files:
- initial_inference.onnx: For root node inference (real observation)
- recurrent_inference.onnx: For tree expansion (imagined states)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from muzero.config import NetworkConfig
    from muzero.networks import MuZeroNetwork


class InitialInferenceWrapper(nn.Module):
    """Wrapper for ONNX export of initial inference.

    Takes a raw observation and outputs:
    - hidden_state
    - policy (softmax probabilities)
    - value
    """

    def __init__(self, network: MuZeroNetwork) -> None:
        super().__init__()
        self.representation = network.representation
        self.prediction = network.prediction

    def forward(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run initial inference.

        Args:
            observation: Game observation (B, num_planes, 8, 8)

        Returns:
            Tuple of (hidden_state, policy, value)
        """
        hidden_state = self.representation(observation)
        policy_logits, value = self.prediction(hidden_state)

        # Apply softmax to get probabilities
        policy = torch.softmax(policy_logits, dim=-1)

        return hidden_state, policy, value


class RecurrentInferenceWrapper(nn.Module):
    """Wrapper for ONNX export of recurrent inference.

    Takes a hidden state and action, outputs:
    - next_hidden_state
    - reward
    - policy (softmax probabilities)
    - value
    """

    def __init__(self, network: MuZeroNetwork) -> None:
        super().__init__()
        self.dynamics = network.dynamics
        self.prediction = network.prediction

    def forward(
        self, hidden_state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run recurrent inference.

        Args:
            hidden_state: Current hidden state (B, hidden_dim, 8, 8)
            action: Action indices (B,)

        Returns:
            Tuple of (next_hidden_state, reward, policy, value)
        """
        next_hidden, reward = self.dynamics(hidden_state, action)
        policy_logits, value = self.prediction(next_hidden)

        # Apply softmax to get probabilities
        policy = torch.softmax(policy_logits, dim=-1)

        return next_hidden, reward, policy, value


def export_to_onnx(
    network: MuZeroNetwork,
    output_dir: str | Path,
    config: NetworkConfig,
    opset_version: int = 17,
) -> dict[str, Path]:
    """Export MuZero networks to ONNX format.

    Creates two files:
    - initial_inference.onnx: observation -> (hidden, policy, value)
    - recurrent_inference.onnx: (hidden, action) -> (next_hidden, reward, policy, value)

    Args:
        network: Trained MuZero network
        output_dir: Directory to save ONNX files
        config: Network configuration
        opset_version: ONNX opset version

    Returns:
        Dict mapping model names to output paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    network.eval()

    # Export initial inference
    initial_wrapper = InitialInferenceWrapper(network)
    initial_path = output_dir / "initial_inference.onnx"

    dummy_obs = torch.randn(1, config.num_observation_planes, 8, 8)

    torch.onnx.export(
        initial_wrapper,
        (dummy_obs,),
        initial_path,
        input_names=["observation"],
        output_names=["hidden_state", "policy", "value"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "hidden_state": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Export recurrent inference
    recurrent_wrapper = RecurrentInferenceWrapper(network)
    recurrent_path = output_dir / "recurrent_inference.onnx"

    dummy_hidden = torch.randn(1, config.hidden_dim, 8, 8)
    dummy_action = torch.tensor([0], dtype=torch.long)

    torch.onnx.export(
        recurrent_wrapper,
        (dummy_hidden, dummy_action),
        recurrent_path,
        input_names=["hidden_state", "action"],
        output_names=["next_hidden_state", "reward", "policy", "value"],
        dynamic_axes={
            "hidden_state": {0: "batch_size"},
            "action": {0: "batch_size"},
            "next_hidden_state": {0: "batch_size"},
            "reward": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    print(f"Exported initial_inference to {initial_path}")
    print(f"Exported recurrent_inference to {recurrent_path}")

    return {
        "initial_inference": initial_path,
        "recurrent_inference": recurrent_path,
    }


def verify_onnx_export(
    network: MuZeroNetwork,
    onnx_paths: dict[str, Path],
    config: NetworkConfig,
    atol: float = 1e-5,
) -> bool:
    """Verify ONNX export matches PyTorch outputs.

    Args:
        network: Original PyTorch network
        onnx_paths: Dict of ONNX model paths
        config: Network configuration
        atol: Absolute tolerance for comparison

    Returns:
        True if verification passes
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping verification")
        return True

    network.eval()

    # Test initial inference
    print("Verifying initial_inference...")
    initial_session = ort.InferenceSession(str(onnx_paths["initial_inference"]))

    dummy_obs = torch.randn(1, config.num_observation_planes, 8, 8)

    with torch.no_grad():
        pt_hidden, pt_policy_logits, pt_value = network.initial_inference(dummy_obs)
        pt_policy = torch.softmax(pt_policy_logits, dim=-1)

    onnx_outputs = initial_session.run(
        None, {"observation": dummy_obs.numpy()}
    )
    onnx_hidden, onnx_policy, onnx_value = onnx_outputs

    hidden_match = torch.allclose(pt_hidden, torch.tensor(onnx_hidden), atol=atol)
    policy_match = torch.allclose(pt_policy, torch.tensor(onnx_policy), atol=atol)
    value_match = torch.allclose(pt_value, torch.tensor(onnx_value).squeeze(), atol=atol)

    if not (hidden_match and policy_match and value_match):
        print(f"  Hidden match: {hidden_match}")
        print(f"  Policy match: {policy_match}")
        print(f"  Value match: {value_match}")
        return False

    print("  Initial inference: OK")

    # Test recurrent inference
    print("Verifying recurrent_inference...")
    recurrent_session = ort.InferenceSession(str(onnx_paths["recurrent_inference"]))

    dummy_hidden = torch.randn(1, config.hidden_dim, 8, 8)
    dummy_action = torch.tensor([100], dtype=torch.long)

    with torch.no_grad():
        pt_next_hidden, pt_reward, pt_policy_logits, pt_value = network.recurrent_inference(
            dummy_hidden, dummy_action
        )
        pt_policy = torch.softmax(pt_policy_logits, dim=-1)

    onnx_outputs = recurrent_session.run(
        None,
        {
            "hidden_state": dummy_hidden.numpy(),
            "action": dummy_action.numpy(),
        },
    )
    onnx_next_hidden, onnx_reward, onnx_policy, onnx_value = onnx_outputs

    hidden_match = torch.allclose(pt_next_hidden, torch.tensor(onnx_next_hidden), atol=atol)
    reward_match = torch.allclose(pt_reward, torch.tensor(onnx_reward).squeeze(), atol=atol)
    policy_match = torch.allclose(pt_policy, torch.tensor(onnx_policy), atol=atol)
    value_match = torch.allclose(pt_value, torch.tensor(onnx_value).squeeze(), atol=atol)

    if not (hidden_match and reward_match and policy_match and value_match):
        print(f"  Hidden match: {hidden_match}")
        print(f"  Reward match: {reward_match}")
        print(f"  Policy match: {policy_match}")
        print(f"  Value match: {value_match}")
        return False

    print("  Recurrent inference: OK")

    return True
