//! Legal move generation for chess.
//!
//! This module generates all legal moves from a position by first generating
//! pseudo-legal moves, then filtering out moves that leave the king in check.

use crate::{attacks, Bitboard, Color, Move, PieceType, Position, Square};

impl Position {
    /// Generate all legal moves from this position
    pub fn legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::with_capacity(256);
        self.generate_pseudo_legal_moves(&mut moves);

        // Filter out illegal moves (those that leave king in check)
        moves.retain(|&mv| !self.leaves_king_in_check(mv));

        moves
    }

    /// Check if a move leaves our king in check
    fn leaves_king_in_check(&self, mv: Move) -> bool {
        let new_pos = self.make_move(mv);
        // After making the move, it's the opponent's turn
        // So we check if our king (the side that just moved) is attacked
        let our_king = new_pos.king_square(self.side_to_move());
        new_pos.is_square_attacked(our_king, self.side_to_move().opposite())
    }

    /// Generate all pseudo-legal moves (not checking for leaving king in check)
    fn generate_pseudo_legal_moves(&self, moves: &mut Vec<Move>) {
        let us = self.side_to_move();

        self.generate_pawn_moves(us, moves);
        self.generate_knight_moves(us, moves);
        self.generate_bishop_moves(us, moves);
        self.generate_rook_moves(us, moves);
        self.generate_queen_moves(us, moves);
        self.generate_king_moves(us, moves);
        self.generate_castling_moves(us, moves);
    }

    /// Generate pawn moves
    fn generate_pawn_moves(&self, us: Color, moves: &mut Vec<Move>) {
        let tables = attacks();
        let pawns = self.pieces(us, PieceType::Pawn);
        let empty = !self.all_pieces();
        let enemies = self.occupancy(us.opposite());

        let (forward_shift, start_rank, promo_rank) = match us {
            Color::White => (8i8, Bitboard::RANK_2, Bitboard::RANK_7),
            Color::Black => (-8i8, Bitboard::RANK_7, Bitboard::RANK_2),
        };

        // Non-promotion pawns
        let non_promo_pawns = pawns & !promo_rank;
        // Promotion pawns
        let promo_pawns = pawns & promo_rank;

        // Single pushes (non-promotion)
        let single_push_targets = if us == Color::White {
            non_promo_pawns.north() & empty
        } else {
            non_promo_pawns.south() & empty
        };

        for to in single_push_targets {
            let from = Square::new_unchecked((to.0 as i8 - forward_shift) as u8);
            moves.push(Move::quiet(from, to));
        }

        // Double pushes
        let double_push_targets = if us == Color::White {
            (((non_promo_pawns & start_rank).north()) & empty).north() & empty
        } else {
            (((non_promo_pawns & start_rank).south()) & empty).south() & empty
        };

        for to in double_push_targets {
            let from = Square::new_unchecked((to.0 as i8 - 2 * forward_shift) as u8);
            moves.push(Move::double_pawn_push(from, to));
        }

        // Captures (non-promotion)
        for from in non_promo_pawns {
            let attacks = tables.pawn_attacks(from, us) & enemies;
            for to in attacks {
                moves.push(Move::capture(from, to));
            }
        }

        // Promotion pushes
        let promo_push_targets = if us == Color::White {
            promo_pawns.north() & empty
        } else {
            promo_pawns.south() & empty
        };

        for to in promo_push_targets {
            let from = Square::new_unchecked((to.0 as i8 - forward_shift) as u8);
            for piece in [
                PieceType::Queen,
                PieceType::Rook,
                PieceType::Bishop,
                PieceType::Knight,
            ] {
                moves.push(Move::promotion(from, to, piece, false));
            }
        }

        // Promotion captures
        for from in promo_pawns {
            let attacks = tables.pawn_attacks(from, us) & enemies;
            for to in attacks {
                for piece in [
                    PieceType::Queen,
                    PieceType::Rook,
                    PieceType::Bishop,
                    PieceType::Knight,
                ] {
                    moves.push(Move::promotion(from, to, piece, true));
                }
            }
        }

        // En passant
        if let Some(ep_square) = self.en_passant_square() {
            // Find pawns that can capture en passant
            // A pawn can capture en passant if it attacks the en passant square
            let ep_attackers = tables.pawn_attacks(ep_square, us.opposite()) & pawns;
            for from in ep_attackers {
                moves.push(Move::en_passant(from, ep_square));
            }
        }
    }

    /// Generate knight moves
    fn generate_knight_moves(&self, us: Color, moves: &mut Vec<Move>) {
        let tables = attacks();
        let knights = self.pieces(us, PieceType::Knight);
        let our_pieces = self.occupancy(us);
        let enemies = self.occupancy(us.opposite());

        for from in knights {
            let attacks = tables.knight_attacks(from) & !our_pieces;
            for to in attacks {
                if enemies.contains(to) {
                    moves.push(Move::capture(from, to));
                } else {
                    moves.push(Move::quiet(from, to));
                }
            }
        }
    }

    /// Generate bishop moves
    fn generate_bishop_moves(&self, us: Color, moves: &mut Vec<Move>) {
        let tables = attacks();
        let bishops = self.pieces(us, PieceType::Bishop);
        let our_pieces = self.occupancy(us);
        let enemies = self.occupancy(us.opposite());
        let occ = self.all_pieces();

        for from in bishops {
            let attacks = tables.bishop_attacks(from, occ) & !our_pieces;
            for to in attacks {
                if enemies.contains(to) {
                    moves.push(Move::capture(from, to));
                } else {
                    moves.push(Move::quiet(from, to));
                }
            }
        }
    }

    /// Generate rook moves
    fn generate_rook_moves(&self, us: Color, moves: &mut Vec<Move>) {
        let tables = attacks();
        let rooks = self.pieces(us, PieceType::Rook);
        let our_pieces = self.occupancy(us);
        let enemies = self.occupancy(us.opposite());
        let occ = self.all_pieces();

        for from in rooks {
            let attacks = tables.rook_attacks(from, occ) & !our_pieces;
            for to in attacks {
                if enemies.contains(to) {
                    moves.push(Move::capture(from, to));
                } else {
                    moves.push(Move::quiet(from, to));
                }
            }
        }
    }

    /// Generate queen moves
    fn generate_queen_moves(&self, us: Color, moves: &mut Vec<Move>) {
        let tables = attacks();
        let queens = self.pieces(us, PieceType::Queen);
        let our_pieces = self.occupancy(us);
        let enemies = self.occupancy(us.opposite());
        let occ = self.all_pieces();

        for from in queens {
            let attacks = tables.queen_attacks(from, occ) & !our_pieces;
            for to in attacks {
                if enemies.contains(to) {
                    moves.push(Move::capture(from, to));
                } else {
                    moves.push(Move::quiet(from, to));
                }
            }
        }
    }

    /// Generate king moves (not including castling)
    fn generate_king_moves(&self, us: Color, moves: &mut Vec<Move>) {
        let tables = attacks();
        let king_sq = self.king_square(us);
        let our_pieces = self.occupancy(us);
        let enemies = self.occupancy(us.opposite());

        let attacks = tables.king_attacks(king_sq) & !our_pieces;
        for to in attacks {
            if enemies.contains(to) {
                moves.push(Move::capture(king_sq, to));
            } else {
                moves.push(Move::quiet(king_sq, to));
            }
        }
    }

    /// Generate castling moves
    fn generate_castling_moves(&self, us: Color, moves: &mut Vec<Move>) {
        let rights = self.castling_rights();
        let them = us.opposite();

        // Can't castle while in check
        if self.is_check() {
            return;
        }

        match us {
            Color::White => {
                // Kingside castling (O-O)
                if rights.can_castle_kingside(us) {
                    // Check that squares between king and rook are empty
                    let between = Square::F1.bitboard() | Square::G1.bitboard();
                    if (self.all_pieces() & between).is_empty() {
                        // Check that king doesn't pass through or end up on attacked square
                        if !self.is_square_attacked(Square::F1, them)
                            && !self.is_square_attacked(Square::G1, them)
                        {
                            moves.push(Move::king_castle(Square::E1, Square::G1));
                        }
                    }
                }
                // Queenside castling (O-O-O)
                if rights.can_castle_queenside(us) {
                    // Check that squares between king and rook are empty
                    let between =
                        Square::B1.bitboard() | Square::C1.bitboard() | Square::D1.bitboard();
                    if (self.all_pieces() & between).is_empty() {
                        // Check that king doesn't pass through or end up on attacked square
                        // Note: b1 doesn't need to be safe, only d1 and c1
                        if !self.is_square_attacked(Square::D1, them)
                            && !self.is_square_attacked(Square::C1, them)
                        {
                            moves.push(Move::queen_castle(Square::E1, Square::C1));
                        }
                    }
                }
            }
            Color::Black => {
                // Kingside castling (O-O)
                if rights.can_castle_kingside(us) {
                    let between = Square::F8.bitboard() | Square::G8.bitboard();
                    if (self.all_pieces() & between).is_empty() && !self.is_square_attacked(Square::F8, them) && !self.is_square_attacked(Square::G8, them) {
                        moves.push(Move::king_castle(Square::E8, Square::G8));
                    }
                }
                // Queenside castling (O-O-O)
                if rights.can_castle_queenside(us) {
                    let between =
                        Square::B8.bitboard() | Square::C8.bitboard() | Square::D8.bitboard();
                    if (self.all_pieces() & between).is_empty() && !self.is_square_attacked(Square::D8, them) && !self.is_square_attacked(Square::C8, them) {
                        moves.push(Move::queen_castle(Square::E8, Square::C8));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starting_position_moves() {
        let pos = Position::starting();
        let moves = pos.legal_moves();

        // Starting position should have exactly 20 legal moves
        // 16 pawn moves (8 single + 8 double) + 4 knight moves
        assert_eq!(moves.len(), 20);
    }

    #[test]
    fn test_position_after_e4() {
        let pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
            .unwrap();
        let moves = pos.legal_moves();

        // Black should have 20 legal moves (same as starting position)
        assert_eq!(moves.len(), 20);
    }

    #[test]
    fn test_en_passant_legal() {
        // Position where white can capture en passant
        let pos =
            Position::from_fen("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3")
                .unwrap();
        let moves = pos.legal_moves();

        // Check that en passant is included
        let ep_moves: Vec<_> = moves.iter().filter(|m| m.is_en_passant()).collect();
        assert_eq!(ep_moves.len(), 1);
        assert_eq!(ep_moves[0].from(), Square::F5);
        assert_eq!(ep_moves[0].to(), Square::E6);
    }

    #[test]
    fn test_promotion_moves() {
        // White pawn about to promote
        let pos = Position::from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1").unwrap();
        let moves = pos.legal_moves();

        // Should have 4 promotion moves + king moves
        let promo_moves: Vec<_> = moves.iter().filter(|m| m.is_promotion()).collect();
        assert_eq!(promo_moves.len(), 4);
    }

    #[test]
    fn test_castling_available() {
        let pos = Position::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let moves = pos.legal_moves();

        // Both castling moves should be available
        let castle_moves: Vec<_> = moves.iter().filter(|m| m.is_castle()).collect();
        assert_eq!(castle_moves.len(), 2);
    }

    #[test]
    fn test_castling_blocked() {
        // Position where castling is blocked by pieces
        let pos = Position::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/RN2K1NR w KQkq - 0 1").unwrap();
        let moves = pos.legal_moves();

        // No castling should be available
        let castle_moves: Vec<_> = moves.iter().filter(|m| m.is_castle()).collect();
        assert_eq!(castle_moves.len(), 0);
    }

    #[test]
    fn test_castling_through_check() {
        // Position where f1 is attacked by black queen on f6
        // Remove e2 pawn to allow clear line, but that's not needed if queen is on f-file
        let pos = Position::from_fen("r3k2r/pppppppp/5q2/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();

        // Verify f1 is attacked by the queen on f6 (through f5-f4-f3-f2... wait, f2 has pawn)
        // Let me use a different approach - queen on f3 with no blocking pawn
        let pos = Position::from_fen("r3k2r/pppppppp/8/8/8/5q2/PPPPP1PP/R3K2R w KQkq - 0 1").unwrap();

        // Verify f1 is attacked
        assert!(pos.is_square_attacked(Square::F1, Color::Black));

        let moves = pos.legal_moves();

        // Only queenside castling should be available (f1 is attacked)
        let castle_moves: Vec<_> = moves.iter().filter(|m| m.is_castle()).collect();
        assert_eq!(castle_moves.len(), 1, "Expected 1 castle move, got {:?}", castle_moves);
        assert!(castle_moves[0].is_queenside_castle());
    }

    #[test]
    fn test_castling_out_of_check() {
        // Position where white king is in check from black rook on e8
        // Black king must not be in check (put it on g7, out of reach of white rooks)
        let pos = Position::from_fen("4r3/6k1/8/8/8/8/8/R3K2R w KQ - 0 1").unwrap();
        assert!(pos.is_check()); // Verify king is in check
        let moves = pos.legal_moves();

        // No castling while in check
        let castle_moves: Vec<_> = moves.iter().filter(|m| m.is_castle()).collect();
        assert_eq!(castle_moves.len(), 0);
    }

    #[test]
    fn test_in_check_must_escape() {
        // Position where white king is in check by queen
        let pos = Position::from_fen("7k/8/8/8/8/8/1q6/K7 w - - 0 1").unwrap();
        let moves = pos.legal_moves();

        // Only king moves that escape check should be legal
        for mv in &moves {
            let new_pos = pos.make_move(*mv);
            let king_sq = new_pos.king_square(Color::White);
            assert!(!new_pos.is_square_attacked(king_sq, Color::Black));
        }
    }

    #[test]
    fn test_checkmate_no_moves() {
        // Fool's mate position - black is checkmated
        let pos = Position::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3")
            .unwrap();

        // Actually this is White to move but let's set up a proper checkmate
        let pos = Position::from_fen("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
            .unwrap();
        // This isn't actually checkmate. Let me use a known checkmate position

        let pos = Position::from_fen("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
            .unwrap();
        // Scholar's mate - black is checkmated
        let moves = pos.legal_moves();
        assert_eq!(moves.len(), 0);
    }

    #[test]
    fn test_stalemate() {
        // Classic stalemate position - black king on a8 cornered by white king on b6 and queen on b8
        // Wait, that would be checkmate. Let me use king on a8, white queen on c7 and king on c6
        // k7/2Q5/2K5/8/8/8/8/8 b - - 0 1
        // Actually that's also not stalemate. Let me use the classic:
        // Black king on h8, white queen on g6, white king on f6
        let pos = Position::from_fen("7k/8/5KQ1/8/8/8/8/8 b - - 0 1").unwrap();
        let moves = pos.legal_moves();

        // Black king has no legal moves but is not in check
        assert_eq!(moves.len(), 0);
        assert!(!pos.is_check());
    }
}
