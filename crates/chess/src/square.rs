use std::fmt;

use crate::Bitboard;

/// A chess square (0-63) using rank-major ordering.
/// a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Square(pub(crate) u8);

impl Square {
    // Named squares for convenience
    pub const A1: Square = Square(0);
    pub const B1: Square = Square(1);
    pub const C1: Square = Square(2);
    pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);
    pub const F1: Square = Square(5);
    pub const G1: Square = Square(6);
    pub const H1: Square = Square(7);

    pub const A2: Square = Square(8);
    pub const B2: Square = Square(9);
    pub const C2: Square = Square(10);
    pub const D2: Square = Square(11);
    pub const E2: Square = Square(12);
    pub const F2: Square = Square(13);
    pub const G2: Square = Square(14);
    pub const H2: Square = Square(15);

    pub const A3: Square = Square(16);
    pub const B3: Square = Square(17);
    pub const C3: Square = Square(18);
    pub const D3: Square = Square(19);
    pub const E3: Square = Square(20);
    pub const F3: Square = Square(21);
    pub const G3: Square = Square(22);
    pub const H3: Square = Square(23);

    pub const A4: Square = Square(24);
    pub const B4: Square = Square(25);
    pub const C4: Square = Square(26);
    pub const D4: Square = Square(27);
    pub const E4: Square = Square(28);
    pub const F4: Square = Square(29);
    pub const G4: Square = Square(30);
    pub const H4: Square = Square(31);

    pub const A5: Square = Square(32);
    pub const B5: Square = Square(33);
    pub const C5: Square = Square(34);
    pub const D5: Square = Square(35);
    pub const E5: Square = Square(36);
    pub const F5: Square = Square(37);
    pub const G5: Square = Square(38);
    pub const H5: Square = Square(39);

    pub const A6: Square = Square(40);
    pub const B6: Square = Square(41);
    pub const C6: Square = Square(42);
    pub const D6: Square = Square(43);
    pub const E6: Square = Square(44);
    pub const F6: Square = Square(45);
    pub const G6: Square = Square(46);
    pub const H6: Square = Square(47);

    pub const A7: Square = Square(48);
    pub const B7: Square = Square(49);
    pub const C7: Square = Square(50);
    pub const D7: Square = Square(51);
    pub const E7: Square = Square(52);
    pub const F7: Square = Square(53);
    pub const G7: Square = Square(54);
    pub const H7: Square = Square(55);

    pub const A8: Square = Square(56);
    pub const B8: Square = Square(57);
    pub const C8: Square = Square(58);
    pub const D8: Square = Square(59);
    pub const E8: Square = Square(60);
    pub const F8: Square = Square(61);
    pub const G8: Square = Square(62);
    pub const H8: Square = Square(63);

    /// Creates a square from index, returning None if out of range
    #[inline]
    pub const fn new(index: u8) -> Option<Self> {
        if index < 64 {
            Some(Square(index))
        } else {
            None
        }
    }

    /// Creates a square from index, panicking if out of range
    /// Use only when index is known to be valid
    #[inline]
    pub const fn new_unchecked(index: u8) -> Self {
        debug_assert!(index < 64);
        Square(index)
    }

    /// Creates a square from file (0-7) and rank (0-7)
    #[inline]
    pub const fn from_coords(file: u8, rank: u8) -> Option<Self> {
        if file < 8 && rank < 8 {
            Some(Square(rank * 8 + file))
        } else {
            None
        }
    }

    /// Returns the file (0-7, where 0 = a-file)
    #[inline]
    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    /// Returns the rank (0-7, where 0 = rank 1)
    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    /// Returns the raw index (0-63)
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Returns a bitboard with just this square set
    #[inline]
    pub const fn bitboard(self) -> Bitboard {
        Bitboard(1u64 << self.0)
    }

    /// Returns the file character ('a'-'h')
    #[inline]
    pub const fn file_char(self) -> char {
        (b'a' + self.file()) as char
    }

    /// Returns the rank character ('1'-'8')
    #[inline]
    pub const fn rank_char(self) -> char {
        (b'1' + self.rank()) as char
    }

    /// Parse square from algebraic notation (e.g., "e4")
    pub fn from_algebraic(s: &str) -> Option<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return None;
        }
        let file = bytes[0].wrapping_sub(b'a');
        let rank = bytes[1].wrapping_sub(b'1');
        Self::from_coords(file, rank)
    }

    /// Offset the square by (file_delta, rank_delta), returning None if out of bounds
    #[inline]
    pub fn offset(self, file_delta: i8, rank_delta: i8) -> Option<Self> {
        let new_file = self.file() as i8 + file_delta;
        let new_rank = self.rank() as i8 + rank_delta;
        if (0..8).contains(&new_file) && (0..8).contains(&new_rank) {
            Some(Square((new_rank * 8 + new_file) as u8))
        } else {
            None
        }
    }

    /// Iterator over all 64 squares
    pub fn all() -> impl Iterator<Item = Square> {
        (0..64).map(Square)
    }
}

impl fmt::Debug for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.file_char(), self.rank_char())
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.file_char(), self.rank_char())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_coords() {
        assert_eq!(Square::A1.file(), 0);
        assert_eq!(Square::A1.rank(), 0);
        assert_eq!(Square::H1.file(), 7);
        assert_eq!(Square::H1.rank(), 0);
        assert_eq!(Square::A8.file(), 0);
        assert_eq!(Square::A8.rank(), 7);
        assert_eq!(Square::H8.file(), 7);
        assert_eq!(Square::H8.rank(), 7);
        assert_eq!(Square::E4.file(), 4);
        assert_eq!(Square::E4.rank(), 3);
    }

    #[test]
    fn test_square_from_coords() {
        assert_eq!(Square::from_coords(0, 0), Some(Square::A1));
        assert_eq!(Square::from_coords(7, 7), Some(Square::H8));
        assert_eq!(Square::from_coords(4, 3), Some(Square::E4));
        assert_eq!(Square::from_coords(8, 0), None);
        assert_eq!(Square::from_coords(0, 8), None);
    }

    #[test]
    fn test_square_algebraic() {
        assert_eq!(Square::from_algebraic("a1"), Some(Square::A1));
        assert_eq!(Square::from_algebraic("h8"), Some(Square::H8));
        assert_eq!(Square::from_algebraic("e4"), Some(Square::E4));
        assert_eq!(Square::from_algebraic("i1"), None);
        assert_eq!(Square::from_algebraic("a9"), None);
        assert_eq!(Square::from_algebraic(""), None);
        assert_eq!(Square::from_algebraic("a"), None);
    }

    #[test]
    fn test_square_display() {
        assert_eq!(format!("{}", Square::A1), "a1");
        assert_eq!(format!("{}", Square::E4), "e4");
        assert_eq!(format!("{}", Square::H8), "h8");
    }

    #[test]
    fn test_square_offset() {
        assert_eq!(Square::E4.offset(1, 1), Some(Square::F5));
        assert_eq!(Square::E4.offset(-1, -1), Some(Square::D3));
        assert_eq!(Square::A1.offset(-1, 0), None);
        assert_eq!(Square::H8.offset(1, 0), None);
        assert_eq!(Square::A1.offset(0, -1), None);
        assert_eq!(Square::H8.offset(0, 1), None);
    }

    #[test]
    fn test_all_squares() {
        let squares: Vec<_> = Square::all().collect();
        assert_eq!(squares.len(), 64);
        assert_eq!(squares[0], Square::A1);
        assert_eq!(squares[63], Square::H8);
    }
}
