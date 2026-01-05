//! Observation encoding for neural network input.
//!
//! Converts a chess position to a 21-plane feature representation
//! matching the Python training pipeline format.

use crate::{Bitboard, Color, PieceType, Position};

/// Number of planes in the observation tensor.
pub const NUM_PLANES: usize = 21;

/// Total number of floats in the observation (21 planes × 64 squares).
pub const OBSERVATION_SIZE: usize = NUM_PLANES * 64;

/// Plane indices for the observation tensor.
#[allow(dead_code)]
mod planes {
    // Piece planes (0-11)
    pub const WHITE_PAWN: usize = 0;
    pub const WHITE_KNIGHT: usize = 1;
    pub const WHITE_BISHOP: usize = 2;
    pub const WHITE_ROOK: usize = 3;
    pub const WHITE_QUEEN: usize = 4;
    pub const WHITE_KING: usize = 5;
    pub const BLACK_PAWN: usize = 6;
    pub const BLACK_KNIGHT: usize = 7;
    pub const BLACK_BISHOP: usize = 8;
    pub const BLACK_ROOK: usize = 9;
    pub const BLACK_QUEEN: usize = 10;
    pub const BLACK_KING: usize = 11;

    // Game state planes (12-20)
    pub const SIDE_TO_MOVE: usize = 12;
    pub const CASTLING_WK: usize = 13;
    pub const CASTLING_WQ: usize = 14;
    pub const CASTLING_BK: usize = 15;
    pub const CASTLING_BQ: usize = 16;
    pub const EN_PASSANT: usize = 17;
    pub const HALFMOVE: usize = 18;
    pub const FULLMOVE: usize = 19;
    pub const BIAS: usize = 20;
}

/// Maps (color, piece_type) to plane index.
fn piece_plane(color: Color, piece_type: PieceType) -> usize {
    let base = match color {
        Color::White => 0,
        Color::Black => 6,
    };
    base + piece_type.index()
}

/// Encode a position into a 21-plane observation tensor.
///
/// The output is a flat Vec<f32> with 1344 elements, laid out as:
/// `[plane_0_sq_0, plane_0_sq_1, ..., plane_0_sq_63, plane_1_sq_0, ...]`
///
/// Each square index corresponds to: `rank * 8 + file` (a1=0, h1=7, a2=8, ..., h8=63).
pub fn encode(pos: &Position) -> Vec<f32> {
    let mut obs = vec![0.0f32; OBSERVATION_SIZE];

    // Encode piece planes (0-11)
    for color in [Color::White, Color::Black] {
        for piece_type in PieceType::ALL {
            let plane = piece_plane(color, piece_type);
            let bitboard = pos.pieces(color, piece_type);
            encode_bitboard(&mut obs, plane, bitboard);
        }
    }

    // Encode side to move (plane 12)
    // All 64 squares set to 1.0 if white to move, 0.0 if black
    if pos.side_to_move() == Color::White {
        fill_plane(&mut obs, planes::SIDE_TO_MOVE, 1.0);
    }

    // Encode castling rights (planes 13-16)
    let castling = pos.castling_rights();
    if castling.can_castle_kingside(Color::White) {
        fill_plane(&mut obs, planes::CASTLING_WK, 1.0);
    }
    if castling.can_castle_queenside(Color::White) {
        fill_plane(&mut obs, planes::CASTLING_WQ, 1.0);
    }
    if castling.can_castle_kingside(Color::Black) {
        fill_plane(&mut obs, planes::CASTLING_BK, 1.0);
    }
    if castling.can_castle_queenside(Color::Black) {
        fill_plane(&mut obs, planes::CASTLING_BQ, 1.0);
    }

    // Encode en passant (plane 17)
    // Set 1.0 on all squares of the en passant file
    if let Some(ep_square) = pos.en_passant_square() {
        let file = ep_square.file();
        let file_mask = Bitboard::file(file);
        encode_bitboard(&mut obs, planes::EN_PASSANT, file_mask);
    }

    // Encode halfmove clock (plane 18)
    // Normalized: min(halfmove / 100.0, 1.0)
    let halfmove_norm = (pos.halfmove_clock() as f32 / 100.0).min(1.0);
    fill_plane(&mut obs, planes::HALFMOVE, halfmove_norm);

    // Encode fullmove number (plane 19)
    // Normalized: min(fullmove / 200.0, 1.0)
    let fullmove_norm = (pos.fullmove_number() as f32 / 200.0).min(1.0);
    fill_plane(&mut obs, planes::FULLMOVE, fullmove_norm);

    // Encode bias plane (plane 20)
    // All 1s
    fill_plane(&mut obs, planes::BIAS, 1.0);

    obs
}

/// Encode a bitboard into a plane.
/// Sets 1.0 for each set bit in the bitboard.
#[inline]
fn encode_bitboard(obs: &mut [f32], plane: usize, bitboard: Bitboard) {
    let base = plane * 64;
    for sq in bitboard.iter() {
        obs[base + sq.index()] = 1.0;
    }
}

/// Fill an entire plane with a constant value.
#[inline]
fn fill_plane(obs: &mut [f32], plane: usize, value: f32) {
    let base = plane * 64;
    for i in 0..64 {
        obs[base + i] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_size() {
        let pos = Position::starting();
        let obs = encode(&pos);
        assert_eq!(obs.len(), OBSERVATION_SIZE);
        assert_eq!(obs.len(), 21 * 64);
    }

    #[test]
    fn test_starting_position_pieces() {
        let pos = Position::starting();
        let obs = encode(&pos);

        // White pawns on rank 2 (squares 8-15)
        let white_pawn_base = planes::WHITE_PAWN * 64;
        for sq in 8..16 {
            assert_eq!(obs[white_pawn_base + sq], 1.0, "white pawn at sq {}", sq);
        }
        // No white pawns elsewhere on rank 1
        for sq in 0..8 {
            assert_eq!(obs[white_pawn_base + sq], 0.0, "no white pawn at sq {}", sq);
        }

        // Black pawns on rank 7 (squares 48-55)
        let black_pawn_base = planes::BLACK_PAWN * 64;
        for sq in 48..56 {
            assert_eq!(obs[black_pawn_base + sq], 1.0, "black pawn at sq {}", sq);
        }

        // White king on e1 (square 4)
        let white_king_base = planes::WHITE_KING * 64;
        assert_eq!(obs[white_king_base + 4], 1.0);

        // Black king on e8 (square 60)
        let black_king_base = planes::BLACK_KING * 64;
        assert_eq!(obs[black_king_base + 60], 1.0);
    }

    #[test]
    fn test_side_to_move() {
        // Starting position: white to move
        let pos = Position::starting();
        let obs = encode(&pos);
        let stm_base = planes::SIDE_TO_MOVE * 64;
        for i in 0..64 {
            assert_eq!(obs[stm_base + i], 1.0);
        }

        // After e4: black to move
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
            .unwrap();
        let obs = encode(&pos);
        for i in 0..64 {
            assert_eq!(obs[stm_base + i], 0.0);
        }
    }

    #[test]
    fn test_castling_rights() {
        let pos = Position::starting();
        let obs = encode(&pos);

        // All castling rights present
        let wk_base = planes::CASTLING_WK * 64;
        let wq_base = planes::CASTLING_WQ * 64;
        let bk_base = planes::CASTLING_BK * 64;
        let bq_base = planes::CASTLING_BQ * 64;

        assert_eq!(obs[wk_base], 1.0);
        assert_eq!(obs[wq_base], 1.0);
        assert_eq!(obs[bk_base], 1.0);
        assert_eq!(obs[bq_base], 1.0);

        // No castling rights
        let pos = Position::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w - - 0 1").unwrap();
        let obs = encode(&pos);
        assert_eq!(obs[wk_base], 0.0);
        assert_eq!(obs[wq_base], 0.0);
        assert_eq!(obs[bk_base], 0.0);
        assert_eq!(obs[bq_base], 0.0);
    }

    #[test]
    fn test_en_passant() {
        // Position after e4: en passant on e-file
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
            .unwrap();
        let obs = encode(&pos);

        let ep_base = planes::EN_PASSANT * 64;
        // e-file is file 4, so squares 4, 12, 20, 28, 36, 44, 52, 60 should be 1.0
        for rank in 0..8 {
            let sq = rank * 8 + 4; // e-file
            assert_eq!(obs[ep_base + sq], 1.0, "en passant on e{}", rank + 1);
        }
        // Other files should be 0
        for rank in 0..8 {
            let sq = rank * 8 + 0; // a-file
            assert_eq!(obs[ep_base + sq], 0.0);
        }
    }

    #[test]
    fn test_halfmove_clock() {
        // Halfmove clock = 50 → 0.5
        let pos = Position::from_fen("8/8/8/8/8/8/8/4K2k w - - 50 100").unwrap();
        let obs = encode(&pos);

        let hm_base = planes::HALFMOVE * 64;
        assert!((obs[hm_base] - 0.5).abs() < 1e-5);

        // Halfmove clock = 100 → 1.0 (clamped)
        let pos = Position::from_fen("8/8/8/8/8/8/8/4K2k w - - 100 100").unwrap();
        let obs = encode(&pos);
        assert!((obs[hm_base] - 1.0).abs() < 1e-5);

        // Halfmove clock = 150 → 1.0 (clamped)
        // Note: We can't directly test >100 with a valid FEN since 100+ triggers 50-move rule
        // But we can test that 100 gives 1.0
    }

    #[test]
    fn test_fullmove_number() {
        // Fullmove = 100 → 0.5
        let pos = Position::from_fen("8/8/8/8/8/8/8/4K2k w - - 0 100").unwrap();
        let obs = encode(&pos);

        let fm_base = planes::FULLMOVE * 64;
        assert!((obs[fm_base] - 0.5).abs() < 1e-5);

        // Fullmove = 200 → 1.0
        let pos = Position::from_fen("8/8/8/8/8/8/8/4K2k w - - 0 200").unwrap();
        let obs = encode(&pos);
        assert!((obs[fm_base] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bias_plane() {
        let pos = Position::starting();
        let obs = encode(&pos);

        let bias_base = planes::BIAS * 64;
        for i in 0..64 {
            assert_eq!(obs[bias_base + i], 1.0);
        }
    }
}
