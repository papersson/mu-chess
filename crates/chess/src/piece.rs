use std::fmt;

/// A player color
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    /// Returns the opposite color
    #[inline]
    pub const fn opposite(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    /// Returns the index (0 for White, 1 for Black)
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Direction of pawn movement (+1 for White, -1 for Black in terms of rank)
    #[inline]
    pub const fn pawn_direction(self) -> i8 {
        match self {
            Color::White => 1,
            Color::Black => -1,
        }
    }

    /// Starting rank for pawns (rank index 1 for White, 6 for Black)
    #[inline]
    pub const fn pawn_start_rank(self) -> u8 {
        match self {
            Color::White => 1,
            Color::Black => 6,
        }
    }

    /// Promotion rank for pawns (rank index 7 for White, 0 for Black)
    #[inline]
    pub const fn promotion_rank(self) -> u8 {
        match self {
            Color::White => 7,
            Color::Black => 0,
        }
    }

    /// Back rank (rank index 0 for White, 7 for Black)
    #[inline]
    pub const fn back_rank(self) -> u8 {
        match self {
            Color::White => 0,
            Color::Black => 7,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Color::White => write!(f, "White"),
            Color::Black => write!(f, "Black"),
        }
    }
}

/// A piece type (without color)
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    /// All piece types in order
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    /// Returns the index (0-5)
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    /// Returns the FEN character for this piece type (uppercase)
    #[inline]
    pub const fn to_char(self) -> char {
        match self {
            PieceType::Pawn => 'P',
            PieceType::Knight => 'N',
            PieceType::Bishop => 'B',
            PieceType::Rook => 'R',
            PieceType::Queen => 'Q',
            PieceType::King => 'K',
        }
    }

    /// Parse piece type from FEN character (case-insensitive)
    pub fn from_char(c: char) -> Option<Self> {
        match c.to_ascii_uppercase() {
            'P' => Some(PieceType::Pawn),
            'N' => Some(PieceType::Knight),
            'B' => Some(PieceType::Bishop),
            'R' => Some(PieceType::Rook),
            'Q' => Some(PieceType::Queen),
            'K' => Some(PieceType::King),
            _ => None,
        }
    }

    /// Returns true if this is a sliding piece (bishop, rook, or queen)
    #[inline]
    pub const fn is_slider(self) -> bool {
        matches!(self, PieceType::Bishop | PieceType::Rook | PieceType::Queen)
    }
}

impl fmt::Display for PieceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

/// A piece with color and type
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Piece {
    pub color: Color,
    pub piece_type: PieceType,
}

impl Piece {
    /// Creates a new piece
    #[inline]
    pub const fn new(color: Color, piece_type: PieceType) -> Self {
        Piece { color, piece_type }
    }

    /// White pieces
    pub const WHITE_PAWN: Piece = Piece::new(Color::White, PieceType::Pawn);
    pub const WHITE_KNIGHT: Piece = Piece::new(Color::White, PieceType::Knight);
    pub const WHITE_BISHOP: Piece = Piece::new(Color::White, PieceType::Bishop);
    pub const WHITE_ROOK: Piece = Piece::new(Color::White, PieceType::Rook);
    pub const WHITE_QUEEN: Piece = Piece::new(Color::White, PieceType::Queen);
    pub const WHITE_KING: Piece = Piece::new(Color::White, PieceType::King);

    /// Black pieces
    pub const BLACK_PAWN: Piece = Piece::new(Color::Black, PieceType::Pawn);
    pub const BLACK_KNIGHT: Piece = Piece::new(Color::Black, PieceType::Knight);
    pub const BLACK_BISHOP: Piece = Piece::new(Color::Black, PieceType::Bishop);
    pub const BLACK_ROOK: Piece = Piece::new(Color::Black, PieceType::Rook);
    pub const BLACK_QUEEN: Piece = Piece::new(Color::Black, PieceType::Queen);
    pub const BLACK_KING: Piece = Piece::new(Color::Black, PieceType::King);

    /// Returns the FEN character for this piece
    #[inline]
    pub const fn to_char(self) -> char {
        let c = self.piece_type.to_char();
        match self.color {
            Color::White => c,
            Color::Black => c.to_ascii_lowercase(),
        }
    }

    /// Parse piece from FEN character
    pub fn from_char(c: char) -> Option<Self> {
        let piece_type = PieceType::from_char(c)?;
        let color = if c.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        Some(Piece::new(color, piece_type))
    }
}

impl fmt::Display for Piece {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_char())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_opposite() {
        assert_eq!(Color::White.opposite(), Color::Black);
        assert_eq!(Color::Black.opposite(), Color::White);
    }

    #[test]
    fn test_piece_type_from_char() {
        assert_eq!(PieceType::from_char('P'), Some(PieceType::Pawn));
        assert_eq!(PieceType::from_char('p'), Some(PieceType::Pawn));
        assert_eq!(PieceType::from_char('N'), Some(PieceType::Knight));
        assert_eq!(PieceType::from_char('K'), Some(PieceType::King));
        assert_eq!(PieceType::from_char('x'), None);
    }

    #[test]
    fn test_piece_from_char() {
        let wp = Piece::from_char('P').unwrap();
        assert_eq!(wp.color, Color::White);
        assert_eq!(wp.piece_type, PieceType::Pawn);

        let bk = Piece::from_char('k').unwrap();
        assert_eq!(bk.color, Color::Black);
        assert_eq!(bk.piece_type, PieceType::King);
    }

    #[test]
    fn test_piece_to_char() {
        assert_eq!(Piece::WHITE_PAWN.to_char(), 'P');
        assert_eq!(Piece::BLACK_PAWN.to_char(), 'p');
        assert_eq!(Piece::WHITE_KING.to_char(), 'K');
        assert_eq!(Piece::BLACK_KING.to_char(), 'k');
    }

    #[test]
    fn test_pawn_ranks() {
        assert_eq!(Color::White.pawn_start_rank(), 1);
        assert_eq!(Color::Black.pawn_start_rank(), 6);
        assert_eq!(Color::White.promotion_rank(), 7);
        assert_eq!(Color::Black.promotion_rank(), 0);
    }
}
