//! Chess position representation using bitboards.

use crate::{attacks, Bitboard, Color, Move, Piece, PieceType, Square};

/// Castling rights encoded as a 4-bit value
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct CastlingRights(pub(crate) u8);

impl CastlingRights {
    pub const NONE: Self = Self(0);
    pub const WHITE_KINGSIDE: u8 = 1;
    pub const WHITE_QUEENSIDE: u8 = 2;
    pub const BLACK_KINGSIDE: u8 = 4;
    pub const BLACK_QUEENSIDE: u8 = 8;
    pub const WHITE_BOTH: Self = Self(Self::WHITE_KINGSIDE | Self::WHITE_QUEENSIDE);
    pub const BLACK_BOTH: Self = Self(Self::BLACK_KINGSIDE | Self::BLACK_QUEENSIDE);
    pub const ALL: Self = Self(15);

    /// Check if the given color can castle kingside
    #[inline]
    pub const fn can_castle_kingside(self, color: Color) -> bool {
        let flag = match color {
            Color::White => Self::WHITE_KINGSIDE,
            Color::Black => Self::BLACK_KINGSIDE,
        };
        (self.0 & flag) != 0
    }

    /// Check if the given color can castle queenside
    #[inline]
    pub const fn can_castle_queenside(self, color: Color) -> bool {
        let flag = match color {
            Color::White => Self::WHITE_QUEENSIDE,
            Color::Black => Self::BLACK_QUEENSIDE,
        };
        (self.0 & flag) != 0
    }

    /// Remove kingside castling rights for a color
    #[inline]
    pub fn remove_kingside(&mut self, color: Color) {
        let flag = match color {
            Color::White => Self::WHITE_KINGSIDE,
            Color::Black => Self::BLACK_KINGSIDE,
        };
        self.0 &= !flag;
    }

    /// Remove queenside castling rights for a color
    #[inline]
    pub fn remove_queenside(&mut self, color: Color) {
        let flag = match color {
            Color::White => Self::WHITE_QUEENSIDE,
            Color::Black => Self::BLACK_QUEENSIDE,
        };
        self.0 &= !flag;
    }

    /// Remove all castling rights for a color
    #[inline]
    pub fn remove_all(&mut self, color: Color) {
        let flags = match color {
            Color::White => Self::WHITE_KINGSIDE | Self::WHITE_QUEENSIDE,
            Color::Black => Self::BLACK_KINGSIDE | Self::BLACK_QUEENSIDE,
        };
        self.0 &= !flags;
    }

    /// Parse castling rights from FEN string (e.g., "KQkq", "Kq", "-")
    pub fn from_fen(s: &str) -> Option<Self> {
        if s == "-" {
            return Some(Self::NONE);
        }
        let mut rights = Self::NONE;
        for c in s.chars() {
            match c {
                'K' => rights.0 |= Self::WHITE_KINGSIDE,
                'Q' => rights.0 |= Self::WHITE_QUEENSIDE,
                'k' => rights.0 |= Self::BLACK_KINGSIDE,
                'q' => rights.0 |= Self::BLACK_QUEENSIDE,
                _ => return None,
            }
        }
        Some(rights)
    }

    /// Convert to FEN string
    pub fn to_fen(self) -> String {
        if self.0 == 0 {
            return "-".to_string();
        }
        let mut s = String::with_capacity(4);
        if self.0 & Self::WHITE_KINGSIDE != 0 {
            s.push('K');
        }
        if self.0 & Self::WHITE_QUEENSIDE != 0 {
            s.push('Q');
        }
        if self.0 & Self::BLACK_KINGSIDE != 0 {
            s.push('k');
        }
        if self.0 & Self::BLACK_QUEENSIDE != 0 {
            s.push('q');
        }
        s
    }
}

/// The result of a chess game
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GameResult {
    WhiteWins,
    BlackWins,
    Draw(DrawReason),
}

/// Reason for a draw
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DrawReason {
    Stalemate,
    FiftyMoveRule,
    InsufficientMaterial,
    ThreefoldRepetition,
}

/// A complete chess position
#[derive(Clone, PartialEq, Eq)]
pub struct Position {
    /// Piece bitboards indexed by [color][piece_type]
    pieces: [[Bitboard; 6]; 2],
    /// Occupancy bitboards per color
    occupancy: [Bitboard; 2],
    /// All pieces combined
    all_pieces: Bitboard,
    /// Side to move
    side_to_move: Color,
    /// Castling rights
    castling_rights: CastlingRights,
    /// En passant target square (the square where a pawn can be captured en passant)
    en_passant_square: Option<Square>,
    /// Half-move clock for 50-move rule (moves since last pawn move or capture)
    halfmove_clock: u8,
    /// Full-move number (starts at 1, incremented after Black's move)
    fullmove_number: u16,
}

impl Position {
    /// Create an empty position
    pub fn empty() -> Self {
        Position {
            pieces: [[Bitboard::EMPTY; 6]; 2],
            occupancy: [Bitboard::EMPTY; 2],
            all_pieces: Bitboard::EMPTY,
            side_to_move: Color::White,
            castling_rights: CastlingRights::NONE,
            en_passant_square: None,
            halfmove_clock: 0,
            fullmove_number: 1,
        }
    }

    /// Create the starting position
    pub fn starting() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap()
    }

    /// Parse a position from FEN notation
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err("FEN must have at least 4 parts".to_string());
        }

        let mut pos = Position::empty();

        // Parse piece placement
        let mut rank = 7i8;
        let mut file = 0i8;
        for c in parts[0].chars() {
            match c {
                '/' => {
                    rank -= 1;
                    file = 0;
                    if rank < 0 {
                        return Err("Too many ranks in FEN".to_string());
                    }
                }
                '1'..='8' => {
                    file += (c as i8) - ('0' as i8);
                }
                _ => {
                    if file >= 8 {
                        return Err("Too many files in FEN".to_string());
                    }
                    let piece =
                        Piece::from_char(c).ok_or_else(|| format!("Invalid piece: {}", c))?;
                    let sq = Square::from_coords(file as u8, rank as u8).unwrap();
                    pos.set_piece(sq, piece);
                    file += 1;
                }
            }
        }

        // Parse side to move
        pos.side_to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(format!("Invalid side to move: {}", parts[1])),
        };

        // Parse castling rights
        pos.castling_rights = CastlingRights::from_fen(parts[2])
            .ok_or_else(|| format!("Invalid castling rights: {}", parts[2]))?;

        // Parse en passant square
        pos.en_passant_square = if parts[3] == "-" {
            None
        } else {
            Some(
                Square::from_algebraic(parts[3])
                    .ok_or_else(|| format!("Invalid en passant square: {}", parts[3]))?,
            )
        };

        // Parse halfmove clock (optional)
        pos.halfmove_clock = if parts.len() > 4 {
            parts[4]
                .parse()
                .map_err(|_| format!("Invalid halfmove clock: {}", parts[4]))?
        } else {
            0
        };

        // Parse fullmove number (optional)
        pos.fullmove_number = if parts.len() > 5 {
            parts[5]
                .parse()
                .map_err(|_| format!("Invalid fullmove number: {}", parts[5]))?
        } else {
            1
        };

        // Validate position
        pos.validate()?;

        Ok(pos)
    }

    /// Convert position to FEN notation
    pub fn to_fen(&self) -> String {
        let mut fen = String::with_capacity(100);

        // Piece placement
        for rank in (0..8).rev() {
            let mut empty_count = 0;
            for file in 0..8 {
                let sq = Square::from_coords(file, rank).unwrap();
                if let Some(piece) = self.piece_at(sq) {
                    if empty_count > 0 {
                        fen.push(char::from_digit(empty_count, 10).unwrap());
                        empty_count = 0;
                    }
                    fen.push(piece.to_char());
                } else {
                    empty_count += 1;
                }
            }
            if empty_count > 0 {
                fen.push(char::from_digit(empty_count, 10).unwrap());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        // Side to move
        fen.push(' ');
        fen.push(match self.side_to_move {
            Color::White => 'w',
            Color::Black => 'b',
        });

        // Castling rights
        fen.push(' ');
        fen.push_str(&self.castling_rights.to_fen());

        // En passant square
        fen.push(' ');
        if let Some(sq) = self.en_passant_square {
            fen.push_str(&sq.to_string());
        } else {
            fen.push('-');
        }

        // Halfmove clock and fullmove number
        fen.push(' ');
        fen.push_str(&self.halfmove_clock.to_string());
        fen.push(' ');
        fen.push_str(&self.fullmove_number.to_string());

        fen
    }

    /// Validate the position
    fn validate(&self) -> Result<(), String> {
        // Check for exactly one king per side
        for color in [Color::White, Color::Black] {
            let king_count = self.pieces(color, PieceType::King).popcount();
            if king_count != 1 {
                return Err(format!(
                    "{:?} has {} kings, expected 1",
                    color, king_count
                ));
            }
        }

        // Check that pawns are not on promotion ranks
        let white_pawns = self.pieces(Color::White, PieceType::Pawn);
        let black_pawns = self.pieces(Color::Black, PieceType::Pawn);
        if (white_pawns & (Bitboard::RANK_1 | Bitboard::RANK_8)).is_not_empty() {
            return Err("White pawns on invalid rank".to_string());
        }
        if (black_pawns & (Bitboard::RANK_1 | Bitboard::RANK_8)).is_not_empty() {
            return Err("Black pawns on invalid rank".to_string());
        }

        // Check that the side not to move is not in check
        // (This would mean the position came from an illegal move)
        let opponent = self.side_to_move.opposite();
        let opponent_king = self.king_square(opponent);
        if self.is_square_attacked(opponent_king, self.side_to_move) {
            return Err("Opponent king is in check (invalid position)".to_string());
        }

        Ok(())
    }

    /// Set a piece on a square
    fn set_piece(&mut self, sq: Square, piece: Piece) {
        let color_idx = piece.color.index();
        let piece_idx = piece.piece_type.index();
        self.pieces[color_idx][piece_idx].set(sq);
        self.occupancy[color_idx].set(sq);
        self.all_pieces.set(sq);
    }

    /// Remove a piece from a square
    fn remove_piece(&mut self, sq: Square, piece: Piece) {
        let color_idx = piece.color.index();
        let piece_idx = piece.piece_type.index();
        self.pieces[color_idx][piece_idx].clear(sq);
        self.occupancy[color_idx].clear(sq);
        self.all_pieces.clear(sq);
    }

    /// Get the piece at a square, if any
    pub fn piece_at(&self, sq: Square) -> Option<Piece> {
        if !self.all_pieces.contains(sq) {
            return None;
        }

        for color in [Color::White, Color::Black] {
            if !self.occupancy[color.index()].contains(sq) {
                continue;
            }
            for piece_type in PieceType::ALL {
                if self.pieces[color.index()][piece_type.index()].contains(sq) {
                    return Some(Piece::new(color, piece_type));
                }
            }
        }

        None
    }

    /// Get the bitboard for pieces of a given color and type
    #[inline]
    pub fn pieces(&self, color: Color, piece_type: PieceType) -> Bitboard {
        self.pieces[color.index()][piece_type.index()]
    }

    /// Get the occupancy bitboard for a color
    #[inline]
    pub fn occupancy(&self, color: Color) -> Bitboard {
        self.occupancy[color.index()]
    }

    /// Get the occupancy bitboard for all pieces
    #[inline]
    pub fn all_pieces(&self) -> Bitboard {
        self.all_pieces
    }

    /// Get the side to move
    #[inline]
    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }

    /// Get the castling rights
    #[inline]
    pub fn castling_rights(&self) -> CastlingRights {
        self.castling_rights
    }

    /// Get the en passant square
    #[inline]
    pub fn en_passant_square(&self) -> Option<Square> {
        self.en_passant_square
    }

    /// Get the halfmove clock
    #[inline]
    pub fn halfmove_clock(&self) -> u8 {
        self.halfmove_clock
    }

    /// Get the fullmove number
    #[inline]
    pub fn fullmove_number(&self) -> u16 {
        self.fullmove_number
    }

    /// Get the king square for a color
    pub fn king_square(&self, color: Color) -> Square {
        self.pieces(color, PieceType::King).lsb().unwrap()
    }

    /// Check if a square is attacked by a given color
    pub fn is_square_attacked(&self, sq: Square, by_color: Color) -> bool {
        let tables = attacks();
        let occ = self.all_pieces;

        // Pawn attacks (check if sq could be attacked by enemy pawns)
        // We look for pawns that attack sq - which means pawns whose attack pattern includes sq
        let pawn_attackers =
            tables.pawn_attacks(sq, by_color.opposite()) & self.pieces(by_color, PieceType::Pawn);
        if pawn_attackers.is_not_empty() {
            return true;
        }

        // Knight attacks
        let knight_attackers =
            tables.knight_attacks(sq) & self.pieces(by_color, PieceType::Knight);
        if knight_attackers.is_not_empty() {
            return true;
        }

        // King attacks
        let king_attackers = tables.king_attacks(sq) & self.pieces(by_color, PieceType::King);
        if king_attackers.is_not_empty() {
            return true;
        }

        // Bishop/Queen attacks (diagonals)
        let bishop_queen =
            self.pieces(by_color, PieceType::Bishop) | self.pieces(by_color, PieceType::Queen);
        let bishop_attackers = tables.bishop_attacks(sq, occ) & bishop_queen;
        if bishop_attackers.is_not_empty() {
            return true;
        }

        // Rook/Queen attacks (orthogonals)
        let rook_queen =
            self.pieces(by_color, PieceType::Rook) | self.pieces(by_color, PieceType::Queen);
        let rook_attackers = tables.rook_attacks(sq, occ) & rook_queen;
        if rook_attackers.is_not_empty() {
            return true;
        }

        false
    }

    /// Check if the current side to move is in check
    #[inline]
    pub fn is_check(&self) -> bool {
        let king_sq = self.king_square(self.side_to_move);
        self.is_square_attacked(king_sq, self.side_to_move.opposite())
    }

    /// Apply a move and return the new position
    pub fn make_move(&self, mv: Move) -> Position {
        let mut new_pos = self.clone();
        let us = self.side_to_move;
        let them = us.opposite();
        let from = mv.from();
        let to = mv.to();

        // Get the piece being moved
        let piece = self.piece_at(from).expect("No piece at from square");
        debug_assert_eq!(piece.color, us);

        // Handle capture (remove captured piece)
        if mv.is_capture() {
            let capture_sq = if mv.is_en_passant() {
                // En passant: captured pawn is not on the target square
                Square::from_coords(to.file(), from.rank()).unwrap()
            } else {
                to
            };
            let captured = self
                .piece_at(capture_sq)
                .expect("No piece at capture square");
            debug_assert_eq!(captured.color, them);
            new_pos.remove_piece(capture_sq, captured);
        }

        // Move the piece
        new_pos.remove_piece(from, piece);

        // Handle promotion
        let final_piece = if let Some(promo_type) = mv.promotion_piece() {
            Piece::new(us, promo_type)
        } else {
            piece
        };
        new_pos.set_piece(to, final_piece);

        // Handle castling (move the rook)
        if mv.is_castle() {
            let (rook_from, rook_to) = if mv.is_kingside_castle() {
                match us {
                    Color::White => (Square::H1, Square::F1),
                    Color::Black => (Square::H8, Square::F8),
                }
            } else {
                match us {
                    Color::White => (Square::A1, Square::D1),
                    Color::Black => (Square::A8, Square::D8),
                }
            };
            let rook = Piece::new(us, PieceType::Rook);
            new_pos.remove_piece(rook_from, rook);
            new_pos.set_piece(rook_to, rook);
        }

        // Update castling rights
        // If king moves, lose all castling rights
        if piece.piece_type == PieceType::King {
            new_pos.castling_rights.remove_all(us);
        }
        // If rook moves from corner, lose that side's castling rights
        if piece.piece_type == PieceType::Rook {
            match from {
                sq if sq == Square::A1 => new_pos.castling_rights.remove_queenside(Color::White),
                sq if sq == Square::H1 => new_pos.castling_rights.remove_kingside(Color::White),
                sq if sq == Square::A8 => new_pos.castling_rights.remove_queenside(Color::Black),
                sq if sq == Square::H8 => new_pos.castling_rights.remove_kingside(Color::Black),
                _ => {}
            }
        }
        // If a rook is captured, lose that side's castling rights
        if mv.is_capture() {
            match to {
                sq if sq == Square::A1 => new_pos.castling_rights.remove_queenside(Color::White),
                sq if sq == Square::H1 => new_pos.castling_rights.remove_kingside(Color::White),
                sq if sq == Square::A8 => new_pos.castling_rights.remove_queenside(Color::Black),
                sq if sq == Square::H8 => new_pos.castling_rights.remove_kingside(Color::Black),
                _ => {}
            }
        }

        // Update en passant square
        new_pos.en_passant_square = if mv.is_double_pawn_push() {
            // Set en passant square to the square behind the pawn
            Some(Square::from_coords(to.file(), (from.rank() + to.rank()) / 2).unwrap())
        } else {
            None
        };

        // Update clocks
        if piece.piece_type == PieceType::Pawn || mv.is_capture() {
            new_pos.halfmove_clock = 0;
        } else {
            new_pos.halfmove_clock = self.halfmove_clock + 1;
        }

        if us == Color::Black {
            new_pos.fullmove_number += 1;
        }

        // Switch side to move
        new_pos.side_to_move = them;

        new_pos
    }

    /// Check if the game has ended (checkmate, stalemate, or draw by 50-move rule)
    /// Note: Threefold repetition requires external tracking
    pub fn is_terminal(&self) -> bool {
        // 50-move rule
        if self.halfmove_clock >= 100 {
            return true;
        }

        // Checkmate or stalemate (no legal moves)
        self.legal_moves().is_empty()
    }

    /// Get the game result if the position is terminal
    pub fn outcome(&self) -> Option<GameResult> {
        // 50-move rule
        if self.halfmove_clock >= 100 {
            return Some(GameResult::Draw(DrawReason::FiftyMoveRule));
        }

        // Check for insufficient material
        if self.is_insufficient_material() {
            return Some(GameResult::Draw(DrawReason::InsufficientMaterial));
        }

        let moves = self.legal_moves();
        if moves.is_empty() {
            if self.is_check() {
                // Checkmate - the side to move loses
                return Some(match self.side_to_move {
                    Color::White => GameResult::BlackWins,
                    Color::Black => GameResult::WhiteWins,
                });
            } else {
                // Stalemate
                return Some(GameResult::Draw(DrawReason::Stalemate));
            }
        }

        None
    }

    /// Check for insufficient material to checkmate
    fn is_insufficient_material(&self) -> bool {
        // If there are any pawns, rooks, or queens, there's sufficient material
        for color in [Color::White, Color::Black] {
            if self.pieces(color, PieceType::Pawn).is_not_empty()
                || self.pieces(color, PieceType::Rook).is_not_empty()
                || self.pieces(color, PieceType::Queen).is_not_empty()
            {
                return false;
            }
        }

        let white_knights = self.pieces(Color::White, PieceType::Knight).popcount();
        let white_bishops = self.pieces(Color::White, PieceType::Bishop).popcount();
        let black_knights = self.pieces(Color::Black, PieceType::Knight).popcount();
        let black_bishops = self.pieces(Color::Black, PieceType::Bishop).popcount();

        let white_minors = white_knights + white_bishops;
        let black_minors = black_knights + black_bishops;

        // K vs K
        if white_minors == 0 && black_minors == 0 {
            return true;
        }

        // K+N vs K or K+B vs K
        if (white_minors == 0 && black_minors == 1) || (white_minors == 1 && black_minors == 0) {
            return true;
        }

        // K+B vs K+B (same colored bishops)
        if white_knights == 0 && black_knights == 0 && white_bishops == 1 && black_bishops == 1 {
            let white_bishop_sq = self.pieces(Color::White, PieceType::Bishop).lsb().unwrap();
            let black_bishop_sq = self.pieces(Color::Black, PieceType::Bishop).lsb().unwrap();
            // Same color if file+rank parity matches
            let white_color = (white_bishop_sq.file() + white_bishop_sq.rank()) % 2;
            let black_color = (black_bishop_sq.file() + black_bishop_sq.rank()) % 2;
            if white_color == black_color {
                return true;
            }
        }

        false
    }
}

impl std::fmt::Debug for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Position {{")?;
        writeln!(f, "  FEN: {}", self.to_fen())?;
        writeln!(f)?;

        // Print the board
        for rank in (0..8).rev() {
            write!(f, "  {} ", rank + 1)?;
            for file in 0..8 {
                let sq = Square::from_coords(file, rank).unwrap();
                match self.piece_at(sq) {
                    Some(piece) => write!(f, "{} ", piece.to_char())?,
                    None => write!(f, ". ")?,
                }
            }
            writeln!(f)?;
        }
        writeln!(f, "    a b c d e f g h")?;
        writeln!(f)?;
        writeln!(f, "  Side to move: {:?}", self.side_to_move)?;
        writeln!(f, "  Castling: {}", self.castling_rights.to_fen())?;
        writeln!(
            f,
            "  En passant: {}",
            self.en_passant_square
                .map_or("-".to_string(), |sq| sq.to_string())
        )?;
        writeln!(
            f,
            "  Halfmove clock: {}, Fullmove: {}",
            self.halfmove_clock, self.fullmove_number
        )?;
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starting_position() {
        let pos = Position::starting();
        assert_eq!(pos.side_to_move(), Color::White);
        assert_eq!(
            pos.pieces(Color::White, PieceType::Pawn).popcount(),
            8
        );
        assert_eq!(
            pos.pieces(Color::Black, PieceType::Pawn).popcount(),
            8
        );
        assert_eq!(
            pos.pieces(Color::White, PieceType::King).popcount(),
            1
        );
        assert_eq!(
            pos.pieces(Color::Black, PieceType::King).popcount(),
            1
        );
        assert_eq!(pos.king_square(Color::White), Square::E1);
        assert_eq!(pos.king_square(Color::Black), Square::E8);
    }

    #[test]
    fn test_fen_roundtrip() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1",
        ];

        for fen in fens {
            let pos = Position::from_fen(fen).unwrap();
            assert_eq!(pos.to_fen(), fen);
        }
    }

    #[test]
    fn test_is_check() {
        // Position where white king is in check
        let pos = Position::from_fen("7k/8/8/8/8/8/1q6/K7 w - - 0 1").unwrap();
        assert!(pos.is_check());

        // Starting position - no check
        let pos = Position::starting();
        assert!(!pos.is_check());
    }

    #[test]
    fn test_make_move_simple() {
        let pos = Position::starting();
        let mv = Move::double_pawn_push(Square::E2, Square::E4);
        let new_pos = pos.make_move(mv);

        assert_eq!(new_pos.side_to_move(), Color::Black);
        assert!(new_pos.piece_at(Square::E4).is_some());
        assert!(new_pos.piece_at(Square::E2).is_none());
        assert_eq!(new_pos.en_passant_square(), Some(Square::E3));
    }

    #[test]
    fn test_make_move_capture() {
        let pos = Position::from_fen("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2").unwrap();
        let mv = Move::capture(Square::E4, Square::D5);
        let new_pos = pos.make_move(mv);

        assert!(new_pos.piece_at(Square::D5).is_some());
        assert!(new_pos.piece_at(Square::E4).is_none());
        assert_eq!(new_pos.pieces(Color::Black, PieceType::Pawn).popcount(), 7);
    }

    #[test]
    fn test_make_move_castle_kingside() {
        let pos = Position::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1").unwrap();
        let mv = Move::king_castle(Square::E1, Square::G1);
        let new_pos = pos.make_move(mv);

        assert_eq!(new_pos.piece_at(Square::G1).unwrap().piece_type, PieceType::King);
        assert_eq!(new_pos.piece_at(Square::F1).unwrap().piece_type, PieceType::Rook);
        assert!(new_pos.piece_at(Square::E1).is_none());
        assert!(new_pos.piece_at(Square::H1).is_none());
        assert!(!new_pos.castling_rights().can_castle_kingside(Color::White));
        assert!(!new_pos.castling_rights().can_castle_queenside(Color::White));
    }

    #[test]
    fn test_make_move_en_passant() {
        let pos = Position::from_fen("rnbqkbnr/pppp1ppp/8/4pP2/8/8/PPPPP1PP/RNBQKBNR w KQkq e6 0 3").unwrap();
        let mv = Move::en_passant(Square::F5, Square::E6);
        let new_pos = pos.make_move(mv);

        assert_eq!(new_pos.piece_at(Square::E6).unwrap().piece_type, PieceType::Pawn);
        assert!(new_pos.piece_at(Square::E5).is_none()); // Captured pawn removed
        assert!(new_pos.piece_at(Square::F5).is_none());
        assert_eq!(new_pos.pieces(Color::Black, PieceType::Pawn).popcount(), 7);
    }

    #[test]
    fn test_make_move_promotion() {
        let pos = Position::from_fen("8/P7/8/8/8/8/8/4K2k w - - 0 1").unwrap();
        let mv = Move::promotion(Square::A7, Square::A8, PieceType::Queen, false);
        let new_pos = pos.make_move(mv);

        assert_eq!(new_pos.piece_at(Square::A8).unwrap().piece_type, PieceType::Queen);
        assert!(new_pos.piece_at(Square::A7).is_none());
    }

    #[test]
    fn test_is_square_attacked() {
        let pos = Position::starting();

        // e3 is attacked by white pawns
        assert!(pos.is_square_attacked(Square::E3, Color::White));

        // e6 is attacked by black pawns
        assert!(pos.is_square_attacked(Square::E6, Color::Black));

        // a3 is attacked by white knight
        assert!(pos.is_square_attacked(Square::A3, Color::White));

        // e4 is not attacked by anyone in starting position
        assert!(!pos.is_square_attacked(Square::E4, Color::White));
        assert!(!pos.is_square_attacked(Square::E4, Color::Black));
    }
}
