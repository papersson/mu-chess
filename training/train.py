#!/usr/bin/env python3
"""MuZero training CLI.

Usage:
    python train.py train [--config CONFIG] [--steps STEPS]
    python train.py export [--config CONFIG] [--checkpoint CHECKPOINT]
    python train.py test [--config CONFIG]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from muzero.config import Config, load_config
from muzero.export import export_to_onnx, verify_onnx_export
from muzero.networks import MuZeroNetwork
from muzero.trainer import MuZeroTrainer
from muzero.utils import get_device


def cmd_train(args: argparse.Namespace) -> int:
    """Run training."""
    config = load_config(args.config)

    # Override batch size if specified
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    print("MuZero Training")
    print("=" * 40)
    print(f"Config: {args.config or 'default'}")
    print(f"Device: {get_device(args.device)}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Steps: {args.steps}")
    print()

    trainer = MuZeroTrainer(config, device=args.device)

    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    trainer.train(
        num_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
    )

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export model to ONNX."""
    config = load_config(args.config)

    print("MuZero ONNX Export")
    print("=" * 40)

    # Create network and load checkpoint
    network = MuZeroNetwork(config.network)

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        import torch

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        network.load_state_dict(checkpoint["network_state"])

    # Export
    output_dir = Path(args.output) if args.output else Path(config.paths.checkpoint_dir)
    onnx_paths = export_to_onnx(network, output_dir, config.network)

    # Verify
    if args.verify:
        print()
        print("Verifying export...")
        if verify_onnx_export(network, onnx_paths, config.network):
            print("Verification passed!")
        else:
            print("Verification FAILED!")
            return 1

    return 0


def cmd_test(args: argparse.Namespace) -> int:
    """Test network forward pass with synthetic data."""
    config = load_config(args.config)

    print("MuZero Network Test")
    print("=" * 40)

    import torch

    device = get_device(args.device)
    network = MuZeroNetwork(config.network).to(device)

    # Test initial inference
    print("\nTesting initial_inference...")
    dummy_obs = torch.randn(
        4, config.network.num_observation_planes, 8, 8, device=device
    )

    with torch.no_grad():
        hidden, policy, value = network.initial_inference(dummy_obs)

    print(f"  Input observation: {dummy_obs.shape}")
    print(f"  Hidden state: {hidden.shape}")
    print(f"  Policy: {policy.shape}")
    print(f"  Value: {value.shape}")

    # Verify policy sums to ~1 after softmax
    policy_probs = torch.softmax(policy, dim=-1)
    policy_sum = policy_probs.sum(dim=-1)
    print(f"  Policy sum (should be ~1): {policy_sum}")

    # Verify value in [-1, 1]
    print(f"  Value range: [{value.min():.3f}, {value.max():.3f}]")

    # Test recurrent inference
    print("\nTesting recurrent_inference...")
    dummy_action = torch.tensor([100, 200, 300, 400], device=device, dtype=torch.long)

    with torch.no_grad():
        next_hidden, reward, next_policy, next_value = network.recurrent_inference(
            hidden, dummy_action
        )

    print(f"  Input hidden: {hidden.shape}")
    print(f"  Input action: {dummy_action.shape}")
    print(f"  Next hidden: {next_hidden.shape}")
    print(f"  Reward: {reward.shape}")
    print(f"  Next policy: {next_policy.shape}")
    print(f"  Next value: {next_value.shape}")

    print("\nAll tests passed!")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MuZero training CLI")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to config.toml file",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the network")
    train_parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=10000,
        help="Number of training steps",
    )
    train_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=None,
        help="Batch size (default: from config, 32)",
    )
    train_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to resume from",
    )
    train_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=1000,
        help="Steps between checkpoints",
    )
    train_parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Steps between log messages",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export to ONNX")
    export_parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to export",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory",
    )
    export_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX export matches PyTorch",
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test network forward pass")

    args = parser.parse_args()

    if args.command == "train":
        return cmd_train(args)
    elif args.command == "export":
        return cmd_export(args)
    elif args.command == "test":
        return cmd_test(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
