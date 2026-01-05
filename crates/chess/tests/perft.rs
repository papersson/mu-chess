//! Perft (Performance Test) for validating move generation correctness.
//!
//! Perft counts the number of leaf nodes at a given depth, which must match
//! known correct values to verify move generation is working correctly.

use muzero_chess::Position;

/// Count all leaf nodes at a given depth
fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = pos.legal_moves();

    if depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0;
    for mv in moves {
        let new_pos = pos.make_move(mv);
        nodes += perft(&new_pos, depth - 1);
    }
    nodes
}

/// Perft with divide - shows node count for each initial move
/// Useful for debugging discrepancies
#[allow(dead_code)]
fn perft_divide(pos: &Position, depth: u32) -> u64 {
    let moves = pos.legal_moves();
    let mut total = 0;

    for mv in &moves {
        let new_pos = pos.make_move(*mv);
        let count = perft(&new_pos, depth - 1);
        println!("{}: {}", mv, count);
        total += count;
    }

    println!("\nTotal: {}", total);
    total
}

// =============================================================================
// Starting Position Tests
// =============================================================================

#[test]
fn test_perft_starting_depth_1() {
    let pos = Position::starting();
    assert_eq!(perft(&pos, 1), 20);
}

#[test]
fn test_perft_starting_depth_2() {
    let pos = Position::starting();
    assert_eq!(perft(&pos, 2), 400);
}

#[test]
fn test_perft_starting_depth_3() {
    let pos = Position::starting();
    assert_eq!(perft(&pos, 3), 8_902);
}

#[test]
fn test_perft_starting_depth_4() {
    let pos = Position::starting();
    assert_eq!(perft(&pos, 4), 197_281);
}

#[test]
fn test_perft_starting_depth_5() {
    let pos = Position::starting();
    // This is the critical validation criterion!
    assert_eq!(perft(&pos, 5), 4_865_609);
}

// =============================================================================
// Kiwipete Position Tests
// Famous test position with many edge cases
// =============================================================================

fn kiwipete() -> Position {
    Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        .unwrap()
}

#[test]
fn test_perft_kiwipete_depth_1() {
    let pos = kiwipete();
    assert_eq!(perft(&pos, 1), 48);
}

#[test]
fn test_perft_kiwipete_depth_2() {
    let pos = kiwipete();
    assert_eq!(perft(&pos, 2), 2_039);
}

#[test]
fn test_perft_kiwipete_depth_3() {
    let pos = kiwipete();
    assert_eq!(perft(&pos, 3), 97_862);
}

#[test]
fn test_perft_kiwipete_depth_4() {
    let pos = kiwipete();
    assert_eq!(perft(&pos, 4), 4_085_603);
}

// =============================================================================
// Position 3 - Tests en passant
// =============================================================================

fn position3() -> Position {
    Position::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap()
}

#[test]
fn test_perft_position3_depth_1() {
    let pos = position3();
    assert_eq!(perft(&pos, 1), 14);
}

#[test]
fn test_perft_position3_depth_2() {
    let pos = position3();
    assert_eq!(perft(&pos, 2), 191);
}

#[test]
fn test_perft_position3_depth_3() {
    let pos = position3();
    assert_eq!(perft(&pos, 3), 2_812);
}

#[test]
fn test_perft_position3_depth_4() {
    let pos = position3();
    assert_eq!(perft(&pos, 4), 43_238);
}

#[test]
fn test_perft_position3_depth_5() {
    let pos = position3();
    assert_eq!(perft(&pos, 5), 674_624);
}

// =============================================================================
// Position 4 - Tests many edge cases
// =============================================================================

fn position4() -> Position {
    Position::from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1").unwrap()
}

#[test]
fn test_perft_position4_depth_1() {
    let pos = position4();
    assert_eq!(perft(&pos, 1), 6);
}

#[test]
fn test_perft_position4_depth_2() {
    let pos = position4();
    assert_eq!(perft(&pos, 2), 264);
}

#[test]
fn test_perft_position4_depth_3() {
    let pos = position4();
    assert_eq!(perft(&pos, 3), 9_467);
}

#[test]
fn test_perft_position4_depth_4() {
    let pos = position4();
    assert_eq!(perft(&pos, 4), 422_333);
}

// =============================================================================
// Position 5 - Alternative starting position test
// =============================================================================

fn position5() -> Position {
    Position::from_fen("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8").unwrap()
}

#[test]
fn test_perft_position5_depth_1() {
    let pos = position5();
    assert_eq!(perft(&pos, 1), 44);
}

#[test]
fn test_perft_position5_depth_2() {
    let pos = position5();
    assert_eq!(perft(&pos, 2), 1_486);
}

#[test]
fn test_perft_position5_depth_3() {
    let pos = position5();
    assert_eq!(perft(&pos, 3), 62_379);
}

#[test]
fn test_perft_position5_depth_4() {
    let pos = position5();
    assert_eq!(perft(&pos, 4), 2_103_487);
}

// =============================================================================
// Position 6 - Promotion tests
// =============================================================================

fn position6() -> Position {
    Position::from_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10")
        .unwrap()
}

#[test]
fn test_perft_position6_depth_1() {
    let pos = position6();
    assert_eq!(perft(&pos, 1), 46);
}

#[test]
fn test_perft_position6_depth_2() {
    let pos = position6();
    assert_eq!(perft(&pos, 2), 2_079);
}

#[test]
fn test_perft_position6_depth_3() {
    let pos = position6();
    assert_eq!(perft(&pos, 3), 89_890);
}

#[test]
fn test_perft_position6_depth_4() {
    let pos = position6();
    assert_eq!(perft(&pos, 4), 3_894_594);
}
