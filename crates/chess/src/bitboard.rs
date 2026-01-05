use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr};

use crate::Square;

/// A 64-bit integer representing a set of squares on a chess board.
/// Each bit corresponds to a square (bit 0 = a1, bit 63 = h8).
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Bitboard(pub u64);

impl Bitboard {
    /// Empty bitboard (no squares set)
    pub const EMPTY: Bitboard = Bitboard(0);

    /// Full bitboard (all squares set)
    pub const ALL: Bitboard = Bitboard(!0u64);

    // File masks (vertical columns)
    pub const FILE_A: Bitboard = Bitboard(0x0101_0101_0101_0101);
    pub const FILE_B: Bitboard = Bitboard(0x0202_0202_0202_0202);
    pub const FILE_C: Bitboard = Bitboard(0x0404_0404_0404_0404);
    pub const FILE_D: Bitboard = Bitboard(0x0808_0808_0808_0808);
    pub const FILE_E: Bitboard = Bitboard(0x1010_1010_1010_1010);
    pub const FILE_F: Bitboard = Bitboard(0x2020_2020_2020_2020);
    pub const FILE_G: Bitboard = Bitboard(0x4040_4040_4040_4040);
    pub const FILE_H: Bitboard = Bitboard(0x8080_8080_8080_8080);

    // Rank masks (horizontal rows)
    pub const RANK_1: Bitboard = Bitboard(0x0000_0000_0000_00FF);
    pub const RANK_2: Bitboard = Bitboard(0x0000_0000_0000_FF00);
    pub const RANK_3: Bitboard = Bitboard(0x0000_0000_00FF_0000);
    pub const RANK_4: Bitboard = Bitboard(0x0000_0000_FF00_0000);
    pub const RANK_5: Bitboard = Bitboard(0x0000_00FF_0000_0000);
    pub const RANK_6: Bitboard = Bitboard(0x0000_FF00_0000_0000);
    pub const RANK_7: Bitboard = Bitboard(0x00FF_0000_0000_0000);
    pub const RANK_8: Bitboard = Bitboard(0xFF00_0000_0000_0000);

    // Diagonal masks
    pub const LIGHT_SQUARES: Bitboard = Bitboard(0x55AA_55AA_55AA_55AA);
    pub const DARK_SQUARES: Bitboard = Bitboard(0xAA55_AA55_AA55_AA55);

    /// Files array for indexing
    pub const FILES: [Bitboard; 8] = [
        Self::FILE_A,
        Self::FILE_B,
        Self::FILE_C,
        Self::FILE_D,
        Self::FILE_E,
        Self::FILE_F,
        Self::FILE_G,
        Self::FILE_H,
    ];

    /// Ranks array for indexing
    pub const RANKS: [Bitboard; 8] = [
        Self::RANK_1,
        Self::RANK_2,
        Self::RANK_3,
        Self::RANK_4,
        Self::RANK_5,
        Self::RANK_6,
        Self::RANK_7,
        Self::RANK_8,
    ];

    /// Creates a bitboard from a raw u64
    #[inline]
    pub const fn new(bits: u64) -> Self {
        Bitboard(bits)
    }

    /// Returns true if no squares are set
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns true if any squares are set
    #[inline]
    pub const fn is_not_empty(self) -> bool {
        self.0 != 0
    }

    /// Returns true if the given square is set
    #[inline]
    pub const fn contains(self, sq: Square) -> bool {
        (self.0 >> sq.0) & 1 == 1
    }

    /// Sets the given square
    #[inline]
    pub fn set(&mut self, sq: Square) {
        self.0 |= 1u64 << sq.0;
    }

    /// Clears the given square
    #[inline]
    pub fn clear(&mut self, sq: Square) {
        self.0 &= !(1u64 << sq.0);
    }

    /// Toggles the given square
    #[inline]
    pub fn toggle(&mut self, sq: Square) {
        self.0 ^= 1u64 << sq.0;
    }

    /// Returns the number of set squares (population count)
    #[inline]
    pub const fn popcount(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns true if exactly one square is set
    #[inline]
    pub const fn is_single(self) -> bool {
        self.0 != 0 && (self.0 & (self.0 - 1)) == 0
    }

    /// Returns the index of the least significant set bit, or None if empty
    #[inline]
    pub const fn lsb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(Square(self.0.trailing_zeros() as u8))
        }
    }

    /// Returns the index of the most significant set bit, or None if empty
    #[inline]
    pub const fn msb(self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(Square(63 - self.0.leading_zeros() as u8))
        }
    }

    /// Pops and returns the least significant set bit, clearing it
    #[inline]
    pub fn pop_lsb(&mut self) -> Option<Square> {
        if self.0 == 0 {
            return None;
        }
        let sq = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1; // Clear lowest bit
        Some(Square(sq))
    }

    /// Returns an iterator over all set squares
    #[inline]
    pub fn iter(self) -> BitboardIter {
        BitboardIter(self)
    }

    /// Shift north (toward rank 8)
    #[inline]
    pub const fn north(self) -> Bitboard {
        Bitboard(self.0 << 8)
    }

    /// Shift south (toward rank 1)
    #[inline]
    pub const fn south(self) -> Bitboard {
        Bitboard(self.0 >> 8)
    }

    /// Shift east (toward h-file), masking out wraparound
    #[inline]
    pub const fn east(self) -> Bitboard {
        Bitboard((self.0 << 1) & !Self::FILE_A.0)
    }

    /// Shift west (toward a-file), masking out wraparound
    #[inline]
    pub const fn west(self) -> Bitboard {
        Bitboard((self.0 >> 1) & !Self::FILE_H.0)
    }

    /// Shift northeast
    #[inline]
    pub const fn north_east(self) -> Bitboard {
        Bitboard((self.0 << 9) & !Self::FILE_A.0)
    }

    /// Shift northwest
    #[inline]
    pub const fn north_west(self) -> Bitboard {
        Bitboard((self.0 << 7) & !Self::FILE_H.0)
    }

    /// Shift southeast
    #[inline]
    pub const fn south_east(self) -> Bitboard {
        Bitboard((self.0 >> 7) & !Self::FILE_A.0)
    }

    /// Shift southwest
    #[inline]
    pub const fn south_west(self) -> Bitboard {
        Bitboard((self.0 >> 9) & !Self::FILE_H.0)
    }

    /// Returns the file mask for a given file index (0-7)
    #[inline]
    pub const fn file(file: u8) -> Bitboard {
        Bitboard(Self::FILE_A.0 << file)
    }

    /// Returns the rank mask for a given rank index (0-7)
    #[inline]
    pub const fn rank(rank: u8) -> Bitboard {
        Bitboard(Self::RANK_1.0 << (rank * 8))
    }
}

/// Iterator over the set squares in a bitboard
pub struct BitboardIter(Bitboard);

impl Iterator for BitboardIter {
    type Item = Square;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.pop_lsb()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let count = self.0.popcount() as usize;
        (count, Some(count))
    }
}

impl ExactSizeIterator for BitboardIter {}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = BitboardIter;

    fn into_iter(self) -> Self::IntoIter {
        BitboardIter(self)
    }
}

// Bitwise operations
impl BitAnd for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitand(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 & rhs.0)
    }
}

impl BitAndAssign for Bitboard {
    #[inline]
    fn bitand_assign(&mut self, rhs: Bitboard) {
        self.0 &= rhs.0;
    }
}

impl BitOr for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 | rhs.0)
    }
}

impl BitOrAssign for Bitboard {
    #[inline]
    fn bitor_assign(&mut self, rhs: Bitboard) {
        self.0 |= rhs.0;
    }
}

impl BitXor for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn bitxor(self, rhs: Bitboard) -> Bitboard {
        Bitboard(self.0 ^ rhs.0)
    }
}

impl BitXorAssign for Bitboard {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Bitboard) {
        self.0 ^= rhs.0;
    }
}

impl Not for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

impl Shl<u8> for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn shl(self, rhs: u8) -> Bitboard {
        Bitboard(self.0 << rhs)
    }
}

impl Shr<u8> for Bitboard {
    type Output = Bitboard;
    #[inline]
    fn shr(self, rhs: u8) -> Bitboard {
        Bitboard(self.0 >> rhs)
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Bitboard(0x{:016X})", self.0)?;
        for rank in (0..8).rev() {
            write!(f, "{}  ", rank + 1)?;
            for file in 0..8 {
                let sq = Square::from_coords(file, rank).unwrap();
                if self.contains(sq) {
                    write!(f, "X ")?;
                } else {
                    write!(f, ". ")?;
                }
            }
            writeln!(f)?;
        }
        writeln!(f, "   a b c d e f g h")
    }
}

impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = Square::from_coords(file, rank).unwrap();
                if self.contains(sq) {
                    write!(f, "X ")?;
                } else {
                    write!(f, ". ")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitboard_empty() {
        assert!(Bitboard::EMPTY.is_empty());
        assert!(!Bitboard::ALL.is_empty());
    }

    #[test]
    fn test_bitboard_contains() {
        let bb = Square::E4.bitboard();
        assert!(bb.contains(Square::E4));
        assert!(!bb.contains(Square::E5));
    }

    #[test]
    fn test_bitboard_set_clear() {
        let mut bb = Bitboard::EMPTY;
        bb.set(Square::A1);
        assert!(bb.contains(Square::A1));
        bb.clear(Square::A1);
        assert!(!bb.contains(Square::A1));
    }

    #[test]
    fn test_bitboard_popcount() {
        assert_eq!(Bitboard::EMPTY.popcount(), 0);
        assert_eq!(Bitboard::ALL.popcount(), 64);
        assert_eq!(Bitboard::FILE_A.popcount(), 8);
        assert_eq!(Bitboard::RANK_1.popcount(), 8);
        assert_eq!(Square::E4.bitboard().popcount(), 1);
    }

    #[test]
    fn test_bitboard_iter() {
        let bb = Square::A1.bitboard() | Square::H8.bitboard();
        let squares: Vec<_> = bb.iter().collect();
        assert_eq!(squares.len(), 2);
        assert_eq!(squares[0], Square::A1);
        assert_eq!(squares[1], Square::H8);
    }

    #[test]
    fn test_bitboard_pop_lsb() {
        let mut bb = Square::A1.bitboard() | Square::H8.bitboard();
        assert_eq!(bb.pop_lsb(), Some(Square::A1));
        assert_eq!(bb.pop_lsb(), Some(Square::H8));
        assert_eq!(bb.pop_lsb(), None);
    }

    #[test]
    fn test_bitboard_shifts() {
        let a1 = Square::A1.bitboard();
        assert_eq!(a1.north(), Square::A2.bitboard());
        assert_eq!(a1.east(), Square::B1.bitboard());

        let h8 = Square::H8.bitboard();
        assert_eq!(h8.south(), Square::H7.bitboard());
        assert_eq!(h8.west(), Square::G8.bitboard());

        // Test wraparound prevention
        let h1 = Square::H1.bitboard();
        assert!(h1.east().is_empty()); // Should not wrap to a1

        let a8 = Square::A8.bitboard();
        assert!(a8.west().is_empty()); // Should not wrap to h8
    }

    #[test]
    fn test_file_rank_masks() {
        // Check file A has correct squares
        assert!(Bitboard::FILE_A.contains(Square::A1));
        assert!(Bitboard::FILE_A.contains(Square::A8));
        assert!(!Bitboard::FILE_A.contains(Square::B1));

        // Check rank 1 has correct squares
        assert!(Bitboard::RANK_1.contains(Square::A1));
        assert!(Bitboard::RANK_1.contains(Square::H1));
        assert!(!Bitboard::RANK_1.contains(Square::A2));
    }

    #[test]
    fn test_bitboard_operations() {
        let a = Bitboard::FILE_A;
        let b = Bitboard::RANK_1;

        // Intersection should be just a1
        let intersection = a & b;
        assert_eq!(intersection.popcount(), 1);
        assert!(intersection.contains(Square::A1));

        // Union should have 15 squares (8 + 8 - 1)
        let union = a | b;
        assert_eq!(union.popcount(), 15);
    }

    #[test]
    fn test_is_single() {
        assert!(Square::E4.bitboard().is_single());
        assert!(!Bitboard::EMPTY.is_single());
        assert!(!(Square::A1.bitboard() | Square::H8.bitboard()).is_single());
    }
}
