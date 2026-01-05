//! Property-based tests for chess implementation.
//!
//! These tests use proptest to generate random inputs and verify invariants
//! as specified in SPEC.md Section 7 and Section 10.2.

use muzero_chess::{Bitboard, Color, Move, Piece, PieceType, Position, Square};
use proptest::prelude::*;

// =============================================================================
// Strategies for generating random chess data
// =============================================================================

/// Generate a random square (0-63)
fn arb_square() -> impl Strategy<Value = Square> {
    (0u8..64).prop_map(Square::new_unchecked)
}

/// Generate a random color
fn arb_color() -> impl Strategy<Value = Color> {
    prop_oneof![Just(Color::White), Just(Color::Black)]
}

/// Generate a random piece type
fn arb_piece_type() -> impl Strategy<Value = PieceType> {
    prop_oneof![
        Just(PieceType::Pawn),
        Just(PieceType::Knight),
        Just(PieceType::Bishop),
        Just(PieceType::Rook),
        Just(PieceType::Queen),
        Just(PieceType::King),
    ]
}

/// Generate a valid position by making random moves from starting position.
/// This ensures we only test positions reachable through legal play.
fn arb_valid_position() -> impl Strategy<Value = Position> {
    // Number of random moves to make (0-50)
    (0usize..50).prop_flat_map(|num_moves| {
        // Generate random indices to select moves
        proptest::collection::vec(0usize..256, num_moves).prop_map(move |move_indices| {
            let mut pos = Position::starting();
            for &idx in &move_indices {
                let moves = pos.legal_moves();
                if moves.is_empty() {
                    break; // Game over
                }
                let mv = moves[idx % moves.len()];
                pos = pos.make_move(mv);
            }
            pos
        })
    })
}

/// Generate a random position by placing pieces on the board.
/// This may generate invalid positions, so we filter them.
fn arb_random_position() -> impl Strategy<Value = Position> {
    // Generate piece placements
    let pieces_strategy = proptest::collection::vec(
        (arb_square(), arb_color(), arb_piece_type()),
        0..16,
    );

    (pieces_strategy, arb_color()).prop_filter_map(
        "valid position",
        |(pieces, side_to_move)| {
            // Start with empty position and place pieces
            // Ensure exactly one king per side
            let mut white_king = None;
            let mut black_king = None;
            let mut occupied = Bitboard::EMPTY;

            // First pass: find kings and check for overlaps
            for &(sq, color, piece_type) in &pieces {
                if occupied.contains(sq) {
                    return None; // Overlapping pieces
                }
                occupied.set(sq);

                if piece_type == PieceType::King {
                    match color {
                        Color::White => {
                            if white_king.is_some() {
                                return None; // Multiple white kings
                            }
                            white_king = Some(sq);
                        }
                        Color::Black => {
                            if black_king.is_some() {
                                return None; // Multiple black kings
                            }
                            black_king = Some(sq);
                        }
                    }
                }

                // Pawns can't be on rank 1 or 8
                if piece_type == PieceType::Pawn && (sq.rank() == 0 || sq.rank() == 7) {
                    return None;
                }
            }

            // Must have exactly one king per side
            let _wk = white_king?;
            let _bk = black_king?;

            // Build FEN string
            let mut fen = String::new();
            for rank in (0..8).rev() {
                let mut empty = 0;
                for file in 0..8 {
                    let sq = Square::from_coords(file, rank).unwrap();
                    let piece = pieces.iter().find(|(s, _, _)| *s == sq);
                    if let Some((_, color, pt)) = piece {
                        if empty > 0 {
                            fen.push(char::from_digit(empty, 10).unwrap());
                            empty = 0;
                        }
                        let c = pt.to_char();
                        fen.push(if *color == Color::White {
                            c
                        } else {
                            c.to_ascii_lowercase()
                        });
                    } else {
                        empty += 1;
                    }
                }
                if empty > 0 {
                    fen.push(char::from_digit(empty, 10).unwrap());
                }
                if rank > 0 {
                    fen.push('/');
                }
            }

            // Add side to move, castling (none), en passant (none), clocks
            fen.push(' ');
            fen.push(if side_to_move == Color::White {
                'w'
            } else {
                'b'
            });
            fen.push_str(" - - 0 1");

            // Try to parse - this validates the position
            Position::from_fen(&fen).ok()
        },
    )
}

// =============================================================================
// INV-4: FEN Round-Trip Property Test (SPEC.md §7, §10.2 P1)
// =============================================================================

proptest! {
    /// INV-4: parse(to_fen(pos)) == pos
    /// Any valid position should survive FEN serialization and parsing.
    #[test]
    fn prop_fen_roundtrip(pos in arb_valid_position()) {
        let fen = pos.to_fen();
        let parsed = Position::from_fen(&fen)
            .expect("FEN generated from valid position should parse");
        prop_assert_eq!(parsed.to_fen(), fen, "FEN should round-trip exactly");
    }

    /// Additional FEN round-trip test with randomly constructed positions
    #[test]
    fn prop_fen_roundtrip_random(pos in arb_random_position()) {
        let fen = pos.to_fen();
        let parsed = Position::from_fen(&fen)
            .expect("FEN generated from valid position should parse");
        prop_assert_eq!(parsed.to_fen(), fen, "FEN should round-trip exactly");
    }
}

// =============================================================================
// INV-5: All Generated Moves Are Legal (SPEC.md §7, §10.2 P2)
// =============================================================================

proptest! {
    /// INV-5: All generated moves are legal
    /// No move returned by legal_moves() should leave the king in check.
    #[test]
    fn prop_all_moves_legal(pos in arb_valid_position()) {
        let moves = pos.legal_moves();
        let us = pos.side_to_move();

        for mv in moves {
            let new_pos = pos.make_move(mv);
            // After the move, it's the opponent's turn
            // Check that OUR king (the side that just moved) is not in check
            let our_king = new_pos.king_square(us);
            let in_check = new_pos.is_square_attacked(our_king, us.opposite());
            prop_assert!(
                !in_check,
                "Move {} leaves king in check: {:?}",
                mv,
                pos.to_fen()
            );
        }
    }
}

// =============================================================================
// INV-6: Exactly One King Per Side (SPEC.md §7)
// =============================================================================

proptest! {
    /// INV-6: After any legal move, there is still exactly one king per side.
    #[test]
    fn prop_one_king_per_side(pos in arb_valid_position()) {
        // Initial position has one king per side
        prop_assert_eq!(
            pos.pieces(Color::White, PieceType::King).popcount(),
            1,
            "White should have exactly one king"
        );
        prop_assert_eq!(
            pos.pieces(Color::Black, PieceType::King).popcount(),
            1,
            "Black should have exactly one king"
        );

        // After any move, still one king per side
        for mv in pos.legal_moves() {
            let new_pos = pos.make_move(mv);
            prop_assert_eq!(
                new_pos.pieces(Color::White, PieceType::King).popcount(),
                1,
                "White should have exactly one king after move {}",
                mv
            );
            prop_assert_eq!(
                new_pos.pieces(Color::Black, PieceType::King).popcount(),
                1,
                "Black should have exactly one king after move {}",
                mv
            );
        }
    }
}

// =============================================================================
// Additional Property Tests
// =============================================================================

proptest! {
    /// Square coordinates round-trip
    #[test]
    fn prop_square_coords_roundtrip(file in 0u8..8, rank in 0u8..8) {
        let sq = Square::from_coords(file, rank).unwrap();
        prop_assert_eq!(sq.file(), file);
        prop_assert_eq!(sq.rank(), rank);
    }

    /// Square algebraic notation round-trip
    #[test]
    fn prop_square_algebraic_roundtrip(sq in arb_square()) {
        let notation = sq.to_string();
        let parsed = Square::from_algebraic(&notation).unwrap();
        prop_assert_eq!(parsed, sq);
    }

    /// Move encoding round-trip
    #[test]
    fn prop_move_encoding(from in arb_square(), to in arb_square()) {
        let mv = Move::quiet(from, to);
        prop_assert_eq!(mv.from(), from);
        prop_assert_eq!(mv.to(), to);
    }

    /// Piece char round-trip
    #[test]
    fn prop_piece_char_roundtrip(color in arb_color(), piece_type in arb_piece_type()) {
        let piece = Piece::new(color, piece_type);
        let c = piece.to_char();
        let parsed = Piece::from_char(c).unwrap();
        prop_assert_eq!(parsed.color, piece.color);
        prop_assert_eq!(parsed.piece_type, piece.piece_type);
    }
}
