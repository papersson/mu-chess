"""Utility functions for MuZero training.

Includes:
- Device detection (MPS/CUDA/CPU)
- Observation encoding for chess
- Action encoding/decoding
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# Board dimensions
BOARD_SIZE = 8
NUM_SQUARES = 64

# Observation plane indices
PLANE_WHITE_PAWN = 0
PLANE_WHITE_KNIGHT = 1
PLANE_WHITE_BISHOP = 2
PLANE_WHITE_ROOK = 3
PLANE_WHITE_QUEEN = 4
PLANE_WHITE_KING = 5
PLANE_BLACK_PAWN = 6
PLANE_BLACK_KNIGHT = 7
PLANE_BLACK_BISHOP = 8
PLANE_BLACK_ROOK = 9
PLANE_BLACK_QUEEN = 10
PLANE_BLACK_KING = 11
PLANE_SIDE_TO_MOVE = 12
PLANE_CASTLING_WK = 13
PLANE_CASTLING_WQ = 14
PLANE_CASTLING_BK = 15
PLANE_CASTLING_BQ = 16
PLANE_EN_PASSANT = 17
PLANE_HALFMOVE = 18
PLANE_FULLMOVE = 19
PLANE_BIAS = 20

NUM_OBSERVATION_PLANES = 21


def get_device(device: str = "auto") -> torch.device:
    """Get the best available device.

    Args:
        device: Device specification. "auto" detects automatically,
                or use "cuda", "mps", "cpu" explicitly.

    Returns:
        torch.device for the selected device.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device)


def encode_observation(position_data: dict[str, Any]) -> np.ndarray:
    """Encode a chess position into observation planes.

    The position_data dictionary should contain:
    - pieces: dict mapping "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"
              to lists of square indices (0-63)
    - side_to_move: "white" or "black"
    - castling: dict with "K", "Q", "k", "q" booleans
    - en_passant: optional square index (0-63) or None
    - halfmove: halfmove clock
    - fullmove: fullmove number

    Alternatively, accepts:
    - bitboards: dict mapping piece chars to 64-bit integers

    Args:
        position_data: Dictionary containing position information

    Returns:
        Observation array of shape (21, 8, 8)
    """
    planes = np.zeros((NUM_OBSERVATION_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Handle bitboard format
    if "bitboards" in position_data:
        _encode_from_bitboards(planes, position_data)
    elif "pieces" in position_data:
        _encode_from_pieces(planes, position_data)
    else:
        # Assume it's a flattened array that just needs reshaping
        if "observation" in position_data:
            return np.array(position_data["observation"], dtype=np.float32).reshape(
                NUM_OBSERVATION_PLANES, BOARD_SIZE, BOARD_SIZE
            )
        raise ValueError("position_data must contain 'pieces', 'bitboards', or 'observation'")

    # Side to move
    if position_data.get("side_to_move") == "white":
        planes[PLANE_SIDE_TO_MOVE] = 1.0

    # Castling rights
    castling = position_data.get("castling", {})
    if castling.get("K", False):
        planes[PLANE_CASTLING_WK] = 1.0
    if castling.get("Q", False):
        planes[PLANE_CASTLING_WQ] = 1.0
    if castling.get("k", False):
        planes[PLANE_CASTLING_BK] = 1.0
    if castling.get("q", False):
        planes[PLANE_CASTLING_BQ] = 1.0

    # En passant
    ep_square = position_data.get("en_passant")
    if ep_square is not None:
        ep_file = ep_square % 8
        planes[PLANE_EN_PASSANT, :, ep_file] = 1.0

    # Move counters (normalized)
    halfmove = position_data.get("halfmove", 0)
    fullmove = position_data.get("fullmove", 1)
    planes[PLANE_HALFMOVE] = min(halfmove / 100.0, 1.0)
    planes[PLANE_FULLMOVE] = min(fullmove / 200.0, 1.0)

    # Bias plane
    planes[PLANE_BIAS] = 1.0

    return planes


def _encode_from_pieces(planes: np.ndarray, position_data: dict[str, Any]) -> None:
    """Encode piece positions from piece lists."""
    pieces = position_data["pieces"]

    piece_to_plane = {
        "P": PLANE_WHITE_PAWN,
        "N": PLANE_WHITE_KNIGHT,
        "B": PLANE_WHITE_BISHOP,
        "R": PLANE_WHITE_ROOK,
        "Q": PLANE_WHITE_QUEEN,
        "K": PLANE_WHITE_KING,
        "p": PLANE_BLACK_PAWN,
        "n": PLANE_BLACK_KNIGHT,
        "b": PLANE_BLACK_BISHOP,
        "r": PLANE_BLACK_ROOK,
        "q": PLANE_BLACK_QUEEN,
        "k": PLANE_BLACK_KING,
    }

    for piece, squares in pieces.items():
        plane_idx = piece_to_plane.get(piece)
        if plane_idx is not None:
            for sq in squares:
                row = sq // 8
                col = sq % 8
                planes[plane_idx, row, col] = 1.0


def _encode_from_bitboards(planes: np.ndarray, position_data: dict[str, Any]) -> None:
    """Encode piece positions from bitboards."""
    bitboards = position_data["bitboards"]

    piece_to_plane = {
        "P": PLANE_WHITE_PAWN,
        "N": PLANE_WHITE_KNIGHT,
        "B": PLANE_WHITE_BISHOP,
        "R": PLANE_WHITE_ROOK,
        "Q": PLANE_WHITE_QUEEN,
        "K": PLANE_WHITE_KING,
        "p": PLANE_BLACK_PAWN,
        "n": PLANE_BLACK_KNIGHT,
        "b": PLANE_BLACK_BISHOP,
        "r": PLANE_BLACK_ROOK,
        "q": PLANE_BLACK_QUEEN,
        "k": PLANE_BLACK_KING,
    }

    for piece, bb in bitboards.items():
        plane_idx = piece_to_plane.get(piece)
        if plane_idx is not None:
            for sq in range(64):
                if bb & (1 << sq):
                    row = sq // 8
                    col = sq % 8
                    planes[plane_idx, row, col] = 1.0


def encode_action_plane(action_idx: int) -> np.ndarray:
    """Create a spatial action plane for the dynamics network.

    Encodes a chess move as two 8x8 planes:
    - Plane 0: from-square marked with 1
    - Plane 1: to-square marked with 1

    Args:
        action_idx: Action index in 16-bit format (from(6) | to(6) | flags(4))

    Returns:
        Action planes of shape (2, 8, 8)
    """
    planes = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # Extract squares from action encoding
    from_sq = action_idx & 0x3F
    to_sq = (action_idx >> 6) & 0x3F

    # Mark squares
    from_row, from_col = from_sq // 8, from_sq % 8
    to_row, to_col = to_sq // 8, to_sq % 8

    planes[0, from_row, from_col] = 1.0
    planes[1, to_row, to_col] = 1.0

    return planes


def decode_action(action_idx: int) -> tuple[int, int, int]:
    """Decode action index to (from_square, to_square, flags).

    Args:
        action_idx: Action index in 16-bit format

    Returns:
        Tuple of (from_square, to_square, flags)
    """
    from_sq = action_idx & 0x3F
    to_sq = (action_idx >> 6) & 0x3F
    flags = (action_idx >> 12) & 0xF
    return from_sq, to_sq, flags


def encode_action(from_sq: int, to_sq: int, flags: int = 0) -> int:
    """Encode a move into action index format.

    Args:
        from_sq: From square (0-63)
        to_sq: To square (0-63)
        flags: Move flags (promotion, castling, etc.)

    Returns:
        Action index
    """
    return from_sq | (to_sq << 6) | (flags << 12)


def create_legal_moves_mask(
    legal_actions: list[int], action_space_size: int = 65536
) -> np.ndarray:
    """Create a mask for legal moves.

    Args:
        legal_actions: List of legal action indices
        action_space_size: Total action space size

    Returns:
        Boolean mask of shape (action_space_size,)
    """
    mask = np.zeros(action_space_size, dtype=np.float32)
    for action in legal_actions:
        mask[action] = 1.0
    return mask


def apply_legal_moves_mask(
    policy_logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Apply legal moves mask to policy logits.

    Sets illegal move logits to -inf so they get zero probability after softmax.

    Args:
        policy_logits: Raw policy logits (B, action_space_size)
        legal_mask: Boolean mask (B, action_space_size) or (action_space_size,)

    Returns:
        Masked logits
    """
    illegal_mask = legal_mask == 0
    masked_logits = policy_logits.clone()
    masked_logits[illegal_mask] = float("-inf")
    return masked_logits
