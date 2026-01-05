//! Pre-computed attack tables for efficient move generation.
//!
//! This module provides attack bitboards for all piece types:
//! - Pawn, Knight, King: Simple lookup tables indexed by square
//! - Bishop, Rook, Queen: Magic bitboard tables for sliding piece attacks

use crate::{Bitboard, Color, Square};

/// Magic numbers for rook attacks (pre-computed, from well-known sources)
const ROOK_MAGICS: [u64; 64] = [
    0x0080001020400080,
    0x0040001000200040,
    0x0080081000200080,
    0x0080040800100080,
    0x0080020400080080,
    0x0080010200040080,
    0x0080008001000200,
    0x0080002040800100,
    0x0000800020400080,
    0x0000400020005000,
    0x0000801000200080,
    0x0000800800100080,
    0x0000800400080080,
    0x0000800200040080,
    0x0000800100020080,
    0x0000800040800100,
    0x0000208000400080,
    0x0000404000201000,
    0x0000808010002000,
    0x0000808008001000,
    0x0000808004000800,
    0x0000808002000400,
    0x0000010100020004,
    0x0000020000408104,
    0x0000208080004000,
    0x0000200040005000,
    0x0000100080200080,
    0x0000080080100080,
    0x0000040080080080,
    0x0000020080040080,
    0x0000010080800200,
    0x0000800080004100,
    0x0000204000800080,
    0x0000200040401000,
    0x0000100080802000,
    0x0000080080801000,
    0x0000040080800800,
    0x0000020080800400,
    0x0000020001010004,
    0x0000800040800100,
    0x0000204000808000,
    0x0000200040008080,
    0x0000100020008080,
    0x0000080010008080,
    0x0000040008008080,
    0x0000020004008080,
    0x0000010002008080,
    0x0000004081020004,
    0x0000204000800080,
    0x0000200040008080,
    0x0000100020008080,
    0x0000080010008080,
    0x0000040008008080,
    0x0000020004008080,
    0x0000800100020080,
    0x0000800041000080,
    0x00FFFCDDFCED714A,
    0x007FFCDDFCED714A,
    0x003FFFCDFFD88096,
    0x0000040810002101,
    0x0001000204080011,
    0x0001000204000801,
    0x0001000082000401,
    0x0001FFFAABFAD1A2,
];

/// Magic numbers for bishop attacks (pre-computed, from well-known sources)
const BISHOP_MAGICS: [u64; 64] = [
    0x0002020202020200,
    0x0002020202020000,
    0x0004010202000000,
    0x0004040080000000,
    0x0001104000000000,
    0x0000821040000000,
    0x0000410410400000,
    0x0000104104104000,
    0x0000040404040400,
    0x0000020202020200,
    0x0000040102020000,
    0x0000040400800000,
    0x0000011040000000,
    0x0000008210400000,
    0x0000004104104000,
    0x0000002082082000,
    0x0004000808080800,
    0x0002000404040400,
    0x0001000202020200,
    0x0000800802004000,
    0x0000800400A00000,
    0x0000200100884000,
    0x0000400082082000,
    0x0000200041041000,
    0x0002080010101000,
    0x0001040008080800,
    0x0000208004010400,
    0x0000404004010200,
    0x0000840000802000,
    0x0000404002011000,
    0x0000808001041000,
    0x0000404000820800,
    0x0001041000202000,
    0x0000820800101000,
    0x0000104400080800,
    0x0000020080080080,
    0x0000404040040100,
    0x0000808100020100,
    0x0001010100020800,
    0x0000808080010400,
    0x0000820820004000,
    0x0000410410002000,
    0x0000082088001000,
    0x0000002011000800,
    0x0000080100400400,
    0x0001010101000200,
    0x0002020202000400,
    0x0001010101000200,
    0x0000410410400000,
    0x0000208208200000,
    0x0000002084100000,
    0x0000000020880000,
    0x0000001002020000,
    0x0000040408020000,
    0x0004040404040000,
    0x0002020202020000,
    0x0000104104104000,
    0x0000002082082000,
    0x0000000020841000,
    0x0000000000208800,
    0x0000000010020200,
    0x0000000404080200,
    0x0000040404040400,
    0x0002020202020200,
];

/// Number of bits in the index for rook attacks at each square
const ROOK_BITS: [u8; 64] = [
    12, 11, 11, 11, 11, 11, 11, 12, 11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11, 11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11, 12, 11, 11, 11, 11, 11, 11, 12,
];

/// Number of bits in the index for bishop attacks at each square
const BISHOP_BITS: [u8; 64] = [
    6, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 5, 5, 5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5, 5, 5, 7, 7, 7, 7, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 5, 5, 5, 5, 5, 5, 6,
];

/// Entry for a magic bitboard lookup
struct MagicEntry {
    mask: Bitboard,
    magic: u64,
    shift: u8,
    offset: usize,
}

/// Pre-computed attack tables (initialized at startup)
pub struct AttackTables {
    /// Pawn attacks indexed by [color][square]
    pawn_attacks: [[Bitboard; 64]; 2],
    /// Knight attacks indexed by [square]
    knight_attacks: [Bitboard; 64],
    /// King attacks indexed by [square]
    king_attacks: [Bitboard; 64],
    /// Rook magic entries indexed by [square]
    rook_magics: [MagicEntry; 64],
    /// Bishop magic entries indexed by [square]
    bishop_magics: [MagicEntry; 64],
    /// Rook attack table (shared, indexed by magic)
    rook_attacks: Vec<Bitboard>,
    /// Bishop attack table (shared, indexed by magic)
    bishop_attacks: Vec<Bitboard>,
}

impl AttackTables {
    /// Initialize all attack tables
    fn init() -> Self {
        let pawn_attacks = init_pawn_attacks();
        let knight_attacks = init_knight_attacks();
        let king_attacks = init_king_attacks();

        let (rook_magics, rook_attacks) = init_rook_magics();
        let (bishop_magics, bishop_attacks) = init_bishop_magics();

        AttackTables {
            pawn_attacks,
            knight_attacks,
            king_attacks,
            rook_magics,
            bishop_magics,
            rook_attacks,
            bishop_attacks,
        }
    }

    /// Get pawn attacks from a square for a given color
    #[inline]
    pub fn pawn_attacks(&self, sq: Square, color: Color) -> Bitboard {
        self.pawn_attacks[color.index()][sq.index()]
    }

    /// Get knight attacks from a square
    #[inline]
    pub fn knight_attacks(&self, sq: Square) -> Bitboard {
        self.knight_attacks[sq.index()]
    }

    /// Get king attacks from a square
    #[inline]
    pub fn king_attacks(&self, sq: Square) -> Bitboard {
        self.king_attacks[sq.index()]
    }

    /// Get rook attacks from a square given the occupancy
    #[inline]
    pub fn rook_attacks(&self, sq: Square, occupancy: Bitboard) -> Bitboard {
        let entry = &self.rook_magics[sq.index()];
        let index = magic_index(entry, occupancy);
        self.rook_attacks[index]
    }

    /// Get bishop attacks from a square given the occupancy
    #[inline]
    pub fn bishop_attacks(&self, sq: Square, occupancy: Bitboard) -> Bitboard {
        let entry = &self.bishop_magics[sq.index()];
        let index = magic_index(entry, occupancy);
        self.bishop_attacks[index]
    }

    /// Get queen attacks from a square given the occupancy
    #[inline]
    pub fn queen_attacks(&self, sq: Square, occupancy: Bitboard) -> Bitboard {
        self.rook_attacks(sq, occupancy) | self.bishop_attacks(sq, occupancy)
    }
}

/// Compute the index into the attack table using magic multiplication
#[inline]
fn magic_index(entry: &MagicEntry, occupancy: Bitboard) -> usize {
    let blockers = (occupancy & entry.mask).0;
    let hash = blockers.wrapping_mul(entry.magic);
    let index = (hash >> entry.shift) as usize;
    entry.offset + index
}

/// Initialize pawn attack tables
fn init_pawn_attacks() -> [[Bitboard; 64]; 2] {
    let mut attacks = [[Bitboard::EMPTY; 64]; 2];

    for sq in Square::all() {
        let bb = sq.bitboard();

        // White pawns attack diagonally up
        attacks[Color::White.index()][sq.index()] = bb.north_east() | bb.north_west();

        // Black pawns attack diagonally down
        attacks[Color::Black.index()][sq.index()] = bb.south_east() | bb.south_west();
    }

    attacks
}

/// Initialize knight attack tables
fn init_knight_attacks() -> [Bitboard; 64] {
    let mut attacks = [Bitboard::EMPTY; 64];

    for sq in Square::all() {
        let bb = sq.bitboard();
        let mut attack = Bitboard::EMPTY;

        // Knight moves: 2 squares in one direction, 1 in perpendicular
        // Using shifts with file masks to prevent wraparound

        // Up 2, left 1
        attack |= Bitboard((bb.0 << 15) & !Bitboard::FILE_H.0);
        // Up 2, right 1
        attack |= Bitboard((bb.0 << 17) & !Bitboard::FILE_A.0);
        // Up 1, left 2
        attack |= Bitboard((bb.0 << 6) & !(Bitboard::FILE_G.0 | Bitboard::FILE_H.0));
        // Up 1, right 2
        attack |= Bitboard((bb.0 << 10) & !(Bitboard::FILE_A.0 | Bitboard::FILE_B.0));
        // Down 1, left 2
        attack |= Bitboard((bb.0 >> 10) & !(Bitboard::FILE_G.0 | Bitboard::FILE_H.0));
        // Down 1, right 2
        attack |= Bitboard((bb.0 >> 6) & !(Bitboard::FILE_A.0 | Bitboard::FILE_B.0));
        // Down 2, left 1
        attack |= Bitboard((bb.0 >> 17) & !Bitboard::FILE_H.0);
        // Down 2, right 1
        attack |= Bitboard((bb.0 >> 15) & !Bitboard::FILE_A.0);

        attacks[sq.index()] = attack;
    }

    attacks
}

/// Initialize king attack tables
fn init_king_attacks() -> [Bitboard; 64] {
    let mut attacks = [Bitboard::EMPTY; 64];

    for sq in Square::all() {
        let bb = sq.bitboard();
        let mut attack = Bitboard::EMPTY;

        // King moves: one square in any direction
        attack |= bb.north();
        attack |= bb.south();
        attack |= bb.east();
        attack |= bb.west();
        attack |= bb.north_east();
        attack |= bb.north_west();
        attack |= bb.south_east();
        attack |= bb.south_west();

        attacks[sq.index()] = attack;
    }

    attacks
}

/// Generate the occupancy mask for a rook on a given square
/// This excludes the edges (because blockers on edges don't affect anything beyond)
fn rook_mask(sq: Square) -> Bitboard {
    let mut mask = Bitboard::EMPTY;
    let file = sq.file() as i8;
    let rank = sq.rank() as i8;

    // North (exclude rank 8)
    for r in (rank + 1)..7 {
        mask.set(Square::from_coords(file as u8, r as u8).unwrap());
    }
    // South (exclude rank 1)
    for r in 1..rank {
        mask.set(Square::from_coords(file as u8, r as u8).unwrap());
    }
    // East (exclude file H)
    for f in (file + 1)..7 {
        mask.set(Square::from_coords(f as u8, rank as u8).unwrap());
    }
    // West (exclude file A)
    for f in 1..file {
        mask.set(Square::from_coords(f as u8, rank as u8).unwrap());
    }

    mask
}

/// Generate the occupancy mask for a bishop on a given square
fn bishop_mask(sq: Square) -> Bitboard {
    let mut mask = Bitboard::EMPTY;
    let file = sq.file() as i8;
    let rank = sq.rank() as i8;

    // Northeast diagonal
    let (mut f, mut r) = (file + 1, rank + 1);
    while f < 7 && r < 7 {
        mask.set(Square::from_coords(f as u8, r as u8).unwrap());
        f += 1;
        r += 1;
    }

    // Northwest diagonal
    let (mut f, mut r) = (file - 1, rank + 1);
    while f > 0 && r < 7 {
        mask.set(Square::from_coords(f as u8, r as u8).unwrap());
        f -= 1;
        r += 1;
    }

    // Southeast diagonal
    let (mut f, mut r) = (file + 1, rank - 1);
    while f < 7 && r > 0 {
        mask.set(Square::from_coords(f as u8, r as u8).unwrap());
        f += 1;
        r -= 1;
    }

    // Southwest diagonal
    let (mut f, mut r) = (file - 1, rank - 1);
    while f > 0 && r > 0 {
        mask.set(Square::from_coords(f as u8, r as u8).unwrap());
        f -= 1;
        r -= 1;
    }

    mask
}

/// Generate rook attacks for a given square and occupancy (slow, for table init)
fn rook_attacks_slow(sq: Square, occupancy: Bitboard) -> Bitboard {
    let mut attacks = Bitboard::EMPTY;
    let file = sq.file() as i8;
    let rank = sq.rank() as i8;

    // North
    for r in (rank + 1)..8 {
        let s = Square::from_coords(file as u8, r as u8).unwrap();
        attacks.set(s);
        if occupancy.contains(s) {
            break;
        }
    }
    // South
    for r in (0..rank).rev() {
        let s = Square::from_coords(file as u8, r as u8).unwrap();
        attacks.set(s);
        if occupancy.contains(s) {
            break;
        }
    }
    // East
    for f in (file + 1)..8 {
        let s = Square::from_coords(f as u8, rank as u8).unwrap();
        attacks.set(s);
        if occupancy.contains(s) {
            break;
        }
    }
    // West
    for f in (0..file).rev() {
        let s = Square::from_coords(f as u8, rank as u8).unwrap();
        attacks.set(s);
        if occupancy.contains(s) {
            break;
        }
    }

    attacks
}

/// Generate bishop attacks for a given square and occupancy (slow, for table init)
fn bishop_attacks_slow(sq: Square, occupancy: Bitboard) -> Bitboard {
    let mut attacks = Bitboard::EMPTY;
    let file = sq.file() as i8;
    let rank = sq.rank() as i8;

    // Northeast
    let (mut f, mut r) = (file + 1, rank + 1);
    while f < 8 && r < 8 {
        let s = Square::from_coords(f as u8, r as u8).unwrap();
        attacks.set(s);
        if occupancy.contains(s) {
            break;
        }
        f += 1;
        r += 1;
    }

    // Northwest
    let (mut f, mut r) = (file - 1, rank + 1);
    while f >= 0 && r < 8 {
        let s = Square::from_coords(f as u8, r as u8).unwrap();
        attacks.set(s);
        if occupancy.contains(s) {
            break;
        }
        f -= 1;
        r += 1;
    }

    // Southeast
    let (mut f, mut r) = (file + 1, rank - 1);
    while f < 8 && r >= 0 {
        let s = Square::from_coords(f as u8, r as u8).unwrap();
        attacks.set(s);
        if occupancy.contains(s) {
            break;
        }
        f += 1;
        r -= 1;
    }

    // Southwest
    let (mut f, mut r) = (file - 1, rank - 1);
    while f >= 0 && r >= 0 {
        let s = Square::from_coords(f as u8, r as u8).unwrap();
        attacks.set(s);
        if occupancy.contains(s) {
            break;
        }
        f -= 1;
        r -= 1;
    }

    attacks
}

/// Generate all possible occupancy variations for a mask
fn generate_occupancies(mask: Bitboard) -> Vec<Bitboard> {
    let bits: Vec<Square> = mask.iter().collect();
    let n = bits.len();
    let mut occupancies = Vec::with_capacity(1 << n);

    // Iterate through all subsets of the mask
    for i in 0..(1u64 << n) {
        let mut occ = Bitboard::EMPTY;
        for (j, &sq) in bits.iter().enumerate() {
            if (i >> j) & 1 == 1 {
                occ.set(sq);
            }
        }
        occupancies.push(occ);
    }

    occupancies
}

/// Initialize rook magic bitboard tables
fn init_rook_magics() -> ([MagicEntry; 64], Vec<Bitboard>) {
    let mut entries: [MagicEntry; 64] = std::array::from_fn(|_| MagicEntry {
        mask: Bitboard::EMPTY,
        magic: 0,
        shift: 0,
        offset: 0,
    });

    // Calculate total table size
    let mut total_size = 0;
    for bits in ROOK_BITS {
        total_size += 1 << bits;
    }

    let mut attacks = vec![Bitboard::EMPTY; total_size];
    let mut offset = 0;

    for sq_idx in 0..64 {
        let sq = Square::new_unchecked(sq_idx as u8);
        let mask = rook_mask(sq);
        let bits = ROOK_BITS[sq_idx];
        let magic = ROOK_MAGICS[sq_idx];
        let shift = 64 - bits;
        let table_size = 1 << bits;

        entries[sq_idx] = MagicEntry {
            mask,
            magic,
            shift,
            offset,
        };

        // Generate all occupancy variations and fill the attack table
        let occupancies = generate_occupancies(mask);
        for occ in occupancies {
            let attack = rook_attacks_slow(sq, occ);
            let hash = (occ.0 & mask.0).wrapping_mul(magic);
            let index = (hash >> shift) as usize;
            attacks[offset + index] = attack;
        }

        offset += table_size;
    }

    (entries, attacks)
}

/// Initialize bishop magic bitboard tables
fn init_bishop_magics() -> ([MagicEntry; 64], Vec<Bitboard>) {
    let mut entries: [MagicEntry; 64] = std::array::from_fn(|_| MagicEntry {
        mask: Bitboard::EMPTY,
        magic: 0,
        shift: 0,
        offset: 0,
    });

    // Calculate total table size
    let mut total_size = 0;
    for bits in BISHOP_BITS {
        total_size += 1 << bits;
    }

    let mut attacks = vec![Bitboard::EMPTY; total_size];
    let mut offset = 0;

    for sq_idx in 0..64 {
        let sq = Square::new_unchecked(sq_idx as u8);
        let mask = bishop_mask(sq);
        let bits = BISHOP_BITS[sq_idx];
        let magic = BISHOP_MAGICS[sq_idx];
        let shift = 64 - bits;
        let table_size = 1 << bits;

        entries[sq_idx] = MagicEntry {
            mask,
            magic,
            shift,
            offset,
        };

        // Generate all occupancy variations and fill the attack table
        let occupancies = generate_occupancies(mask);
        for occ in occupancies {
            let attack = bishop_attacks_slow(sq, occ);
            let hash = (occ.0 & mask.0).wrapping_mul(magic);
            let index = (hash >> shift) as usize;
            attacks[offset + index] = attack;
        }

        offset += table_size;
    }

    (entries, attacks)
}

// Global attack tables, initialized once
use std::sync::OnceLock;
static ATTACKS: OnceLock<AttackTables> = OnceLock::new();

/// Get a reference to the global attack tables
pub fn attacks() -> &'static AttackTables {
    ATTACKS.get_or_init(AttackTables::init)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pawn_attacks() {
        let tables = attacks();

        // White pawn on e4 attacks d5 and f5
        let e4_attacks = tables.pawn_attacks(Square::E4, Color::White);
        assert!(e4_attacks.contains(Square::D5));
        assert!(e4_attacks.contains(Square::F5));
        assert!(!e4_attacks.contains(Square::E5));

        // Black pawn on e5 attacks d4 and f4
        let e5_attacks = tables.pawn_attacks(Square::E5, Color::Black);
        assert!(e5_attacks.contains(Square::D4));
        assert!(e5_attacks.contains(Square::F4));

        // Edge cases: a-file pawn
        let a4_attacks = tables.pawn_attacks(Square::A4, Color::White);
        assert!(a4_attacks.contains(Square::B5));
        assert!(!a4_attacks.contains(Square::A5)); // Can't attack own file
        assert_eq!(a4_attacks.popcount(), 1);

        // Edge cases: h-file pawn
        let h4_attacks = tables.pawn_attacks(Square::H4, Color::White);
        assert!(h4_attacks.contains(Square::G5));
        assert_eq!(h4_attacks.popcount(), 1);
    }

    #[test]
    fn test_knight_attacks() {
        let tables = attacks();

        // Knight on e4 should attack 8 squares
        let e4_attacks = tables.knight_attacks(Square::E4);
        assert_eq!(e4_attacks.popcount(), 8);
        assert!(e4_attacks.contains(Square::D6));
        assert!(e4_attacks.contains(Square::F6));
        assert!(e4_attacks.contains(Square::G5));
        assert!(e4_attacks.contains(Square::G3));
        assert!(e4_attacks.contains(Square::F2));
        assert!(e4_attacks.contains(Square::D2));
        assert!(e4_attacks.contains(Square::C3));
        assert!(e4_attacks.contains(Square::C5));

        // Knight on a1 should attack 2 squares
        let a1_attacks = tables.knight_attacks(Square::A1);
        assert_eq!(a1_attacks.popcount(), 2);
        assert!(a1_attacks.contains(Square::B3));
        assert!(a1_attacks.contains(Square::C2));

        // Knight on h8 should attack 2 squares
        let h8_attacks = tables.knight_attacks(Square::H8);
        assert_eq!(h8_attacks.popcount(), 2);
    }

    #[test]
    fn test_king_attacks() {
        let tables = attacks();

        // King on e4 should attack 8 squares
        let e4_attacks = tables.king_attacks(Square::E4);
        assert_eq!(e4_attacks.popcount(), 8);
        assert!(e4_attacks.contains(Square::D5));
        assert!(e4_attacks.contains(Square::E5));
        assert!(e4_attacks.contains(Square::F5));
        assert!(e4_attacks.contains(Square::D4));
        assert!(e4_attacks.contains(Square::F4));
        assert!(e4_attacks.contains(Square::D3));
        assert!(e4_attacks.contains(Square::E3));
        assert!(e4_attacks.contains(Square::F3));

        // King on a1 should attack 3 squares
        let a1_attacks = tables.king_attacks(Square::A1);
        assert_eq!(a1_attacks.popcount(), 3);
        assert!(a1_attacks.contains(Square::A2));
        assert!(a1_attacks.contains(Square::B2));
        assert!(a1_attacks.contains(Square::B1));
    }

    #[test]
    fn test_rook_attacks_empty_board() {
        let tables = attacks();

        // Rook on e4 with empty board
        let e4_attacks = tables.rook_attacks(Square::E4, Bitboard::EMPTY);
        assert_eq!(e4_attacks.popcount(), 14); // 7 on file + 7 on rank
        assert!(e4_attacks.contains(Square::E1));
        assert!(e4_attacks.contains(Square::E8));
        assert!(e4_attacks.contains(Square::A4));
        assert!(e4_attacks.contains(Square::H4));
    }

    #[test]
    fn test_rook_attacks_with_blockers() {
        let tables = attacks();

        // Rook on e4 with blockers on e6 and c4
        let blockers = Square::E6.bitboard() | Square::C4.bitboard();
        let e4_attacks = tables.rook_attacks(Square::E4, blockers);

        // Should see e5, e6 (blocker) but not e7, e8
        assert!(e4_attacks.contains(Square::E5));
        assert!(e4_attacks.contains(Square::E6));
        assert!(!e4_attacks.contains(Square::E7));
        assert!(!e4_attacks.contains(Square::E8));

        // Should see d4, c4 (blocker) but not b4, a4
        assert!(e4_attacks.contains(Square::D4));
        assert!(e4_attacks.contains(Square::C4));
        assert!(!e4_attacks.contains(Square::B4));
        assert!(!e4_attacks.contains(Square::A4));

        // Should still see full range in other directions
        assert!(e4_attacks.contains(Square::E3));
        assert!(e4_attacks.contains(Square::E2));
        assert!(e4_attacks.contains(Square::E1));
        assert!(e4_attacks.contains(Square::F4));
        assert!(e4_attacks.contains(Square::G4));
        assert!(e4_attacks.contains(Square::H4));
    }

    #[test]
    fn test_bishop_attacks_empty_board() {
        let tables = attacks();

        // Bishop on e4 with empty board
        let e4_attacks = tables.bishop_attacks(Square::E4, Bitboard::EMPTY);
        assert_eq!(e4_attacks.popcount(), 13);
        assert!(e4_attacks.contains(Square::D5));
        assert!(e4_attacks.contains(Square::C6));
        assert!(e4_attacks.contains(Square::B7));
        assert!(e4_attacks.contains(Square::A8));
        assert!(e4_attacks.contains(Square::F5));
        assert!(e4_attacks.contains(Square::H7));
    }

    #[test]
    fn test_bishop_attacks_with_blockers() {
        let tables = attacks();

        // Bishop on e4 with blocker on c6
        let blockers = Square::C6.bitboard();
        let e4_attacks = tables.bishop_attacks(Square::E4, blockers);

        // Should see d5, c6 (blocker) but not b7, a8
        assert!(e4_attacks.contains(Square::D5));
        assert!(e4_attacks.contains(Square::C6));
        assert!(!e4_attacks.contains(Square::B7));
        assert!(!e4_attacks.contains(Square::A8));
    }

    #[test]
    fn test_queen_attacks() {
        let tables = attacks();

        // Queen on e4 with empty board should attack like rook + bishop
        let e4_attacks = tables.queen_attacks(Square::E4, Bitboard::EMPTY);
        assert_eq!(e4_attacks.popcount(), 27); // 14 (rook) + 13 (bishop)
    }
}
