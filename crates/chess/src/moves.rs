//! Chess move encoding using a compact 16-bit representation.
//!
//! Format: `from(6) | to(6) | flags(4)`
//! - Bits 0-5: destination square
//! - Bits 6-11: origin square
//! - Bits 12-15: move flags

use std::fmt;

use crate::{PieceType, Square};

/// Move flags (4 bits) indicating the type of move
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum MoveFlags {
    Quiet = 0,
    DoublePawnPush = 1,
    KingCastle = 2,
    QueenCastle = 3,
    Capture = 4,
    EnPassant = 5,
    // 6, 7 unused
    KnightPromotion = 8,
    BishopPromotion = 9,
    RookPromotion = 10,
    QueenPromotion = 11,
    KnightPromotionCapture = 12,
    BishopPromotionCapture = 13,
    RookPromotionCapture = 14,
    QueenPromotionCapture = 15,
}

impl MoveFlags {
    /// Create flags from raw u8
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(MoveFlags::Quiet),
            1 => Some(MoveFlags::DoublePawnPush),
            2 => Some(MoveFlags::KingCastle),
            3 => Some(MoveFlags::QueenCastle),
            4 => Some(MoveFlags::Capture),
            5 => Some(MoveFlags::EnPassant),
            8 => Some(MoveFlags::KnightPromotion),
            9 => Some(MoveFlags::BishopPromotion),
            10 => Some(MoveFlags::RookPromotion),
            11 => Some(MoveFlags::QueenPromotion),
            12 => Some(MoveFlags::KnightPromotionCapture),
            13 => Some(MoveFlags::BishopPromotionCapture),
            14 => Some(MoveFlags::RookPromotionCapture),
            15 => Some(MoveFlags::QueenPromotionCapture),
            _ => None,
        }
    }

    /// Create promotion flags for a given piece type
    pub fn promotion(piece: PieceType, is_capture: bool) -> Self {
        let base = match piece {
            PieceType::Knight => 8,
            PieceType::Bishop => 9,
            PieceType::Rook => 10,
            PieceType::Queen => 11,
            _ => panic!("Invalid promotion piece"),
        };
        MoveFlags::from_u8(base + if is_capture { 4 } else { 0 }).unwrap()
    }
}

/// A chess move encoded in 16 bits
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Move(pub(crate) u16);

impl Move {
    /// Null move constant (used as a sentinel value)
    pub const NULL: Move = Move(0);

    /// Create a new move
    #[inline]
    pub const fn new(from: Square, to: Square, flags: MoveFlags) -> Self {
        Move((to.0 as u16) | ((from.0 as u16) << 6) | ((flags as u16) << 12))
    }

    /// Create a quiet move (no capture, no special flags)
    #[inline]
    pub const fn quiet(from: Square, to: Square) -> Self {
        Self::new(from, to, MoveFlags::Quiet)
    }

    /// Create a capture move
    #[inline]
    pub const fn capture(from: Square, to: Square) -> Self {
        Self::new(from, to, MoveFlags::Capture)
    }

    /// Create a double pawn push
    #[inline]
    pub const fn double_pawn_push(from: Square, to: Square) -> Self {
        Self::new(from, to, MoveFlags::DoublePawnPush)
    }

    /// Create an en passant capture
    #[inline]
    pub const fn en_passant(from: Square, to: Square) -> Self {
        Self::new(from, to, MoveFlags::EnPassant)
    }

    /// Create a kingside castle
    #[inline]
    pub const fn king_castle(from: Square, to: Square) -> Self {
        Self::new(from, to, MoveFlags::KingCastle)
    }

    /// Create a queenside castle
    #[inline]
    pub const fn queen_castle(from: Square, to: Square) -> Self {
        Self::new(from, to, MoveFlags::QueenCastle)
    }

    /// Create a promotion move
    #[inline]
    pub fn promotion(from: Square, to: Square, piece: PieceType, is_capture: bool) -> Self {
        Self::new(from, to, MoveFlags::promotion(piece, is_capture))
    }

    /// Get the origin square
    #[inline]
    pub const fn from(self) -> Square {
        Square(((self.0 >> 6) & 0x3F) as u8)
    }

    /// Get the destination square
    #[inline]
    pub const fn to(self) -> Square {
        Square((self.0 & 0x3F) as u8)
    }

    /// Get the move flags
    #[inline]
    pub const fn flags(self) -> u8 {
        ((self.0 >> 12) & 0x0F) as u8
    }

    /// Check if this is a capture move
    #[inline]
    pub const fn is_capture(self) -> bool {
        (self.flags() & 4) != 0 || self.flags() == MoveFlags::EnPassant as u8
    }

    /// Check if this is a promotion move
    #[inline]
    pub const fn is_promotion(self) -> bool {
        self.flags() >= 8
    }

    /// Check if this is a castle move
    #[inline]
    pub const fn is_castle(self) -> bool {
        self.flags() == MoveFlags::KingCastle as u8
            || self.flags() == MoveFlags::QueenCastle as u8
    }

    /// Check if this is a kingside castle
    #[inline]
    pub const fn is_kingside_castle(self) -> bool {
        self.flags() == MoveFlags::KingCastle as u8
    }

    /// Check if this is a queenside castle
    #[inline]
    pub const fn is_queenside_castle(self) -> bool {
        self.flags() == MoveFlags::QueenCastle as u8
    }

    /// Check if this is an en passant capture
    #[inline]
    pub const fn is_en_passant(self) -> bool {
        self.flags() == MoveFlags::EnPassant as u8
    }

    /// Check if this is a double pawn push
    #[inline]
    pub const fn is_double_pawn_push(self) -> bool {
        self.flags() == MoveFlags::DoublePawnPush as u8
    }

    /// Get the promotion piece type, if this is a promotion
    pub fn promotion_piece(self) -> Option<PieceType> {
        if !self.is_promotion() {
            return None;
        }
        Some(match self.flags() & 3 {
            0 => PieceType::Knight,
            1 => PieceType::Bishop,
            2 => PieceType::Rook,
            3 => PieceType::Queen,
            _ => unreachable!(),
        })
    }

    /// Get the raw 16-bit encoding
    #[inline]
    pub const fn raw(self) -> u16 {
        self.0
    }

    /// Check if this is a null move
    #[inline]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Move({} -> {}", self.from(), self.to())?;
        if self.is_promotion() {
            write!(f, "={}", self.promotion_piece().unwrap().to_char())?;
        }
        if self.is_castle() {
            write!(
                f,
                " castle:{}",
                if self.is_kingside_castle() { "K" } else { "Q" }
            )?;
        }
        if self.is_en_passant() {
            write!(f, " e.p.")?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.from(), self.to())?;
        if let Some(promo) = self.promotion_piece() {
            write!(f, "{}", promo.to_char().to_ascii_lowercase())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_encoding() {
        let mv = Move::new(Square::E2, Square::E4, MoveFlags::DoublePawnPush);
        assert_eq!(mv.from(), Square::E2);
        assert_eq!(mv.to(), Square::E4);
        assert_eq!(mv.flags(), MoveFlags::DoublePawnPush as u8);
        assert!(mv.is_double_pawn_push());
        assert!(!mv.is_capture());
        assert!(!mv.is_promotion());
    }

    #[test]
    fn test_move_capture() {
        let mv = Move::capture(Square::E4, Square::D5);
        assert!(mv.is_capture());
        assert!(!mv.is_promotion());
        assert!(!mv.is_castle());
    }

    #[test]
    fn test_move_promotion() {
        let mv = Move::promotion(Square::E7, Square::E8, PieceType::Queen, false);
        assert!(mv.is_promotion());
        assert!(!mv.is_capture());
        assert_eq!(mv.promotion_piece(), Some(PieceType::Queen));

        let mv = Move::promotion(Square::E7, Square::D8, PieceType::Knight, true);
        assert!(mv.is_promotion());
        assert!(mv.is_capture());
        assert_eq!(mv.promotion_piece(), Some(PieceType::Knight));
    }

    #[test]
    fn test_move_castle() {
        let mv = Move::king_castle(Square::E1, Square::G1);
        assert!(mv.is_castle());
        assert!(mv.is_kingside_castle());
        assert!(!mv.is_queenside_castle());

        let mv = Move::queen_castle(Square::E1, Square::C1);
        assert!(mv.is_castle());
        assert!(!mv.is_kingside_castle());
        assert!(mv.is_queenside_castle());
    }

    #[test]
    fn test_move_en_passant() {
        let mv = Move::en_passant(Square::E5, Square::D6);
        assert!(mv.is_en_passant());
        assert!(mv.is_capture());
        assert!(!mv.is_promotion());
    }

    #[test]
    fn test_move_display() {
        let mv = Move::quiet(Square::E2, Square::E4);
        assert_eq!(format!("{}", mv), "e2e4");

        let mv = Move::promotion(Square::E7, Square::E8, PieceType::Queen, false);
        assert_eq!(format!("{}", mv), "e7e8q");
    }

    #[test]
    fn test_null_move() {
        assert!(Move::NULL.is_null());
        let mv = Move::quiet(Square::E2, Square::E4);
        assert!(!mv.is_null());
    }
}
