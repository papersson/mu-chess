//! Tests verifying MCTS plays tic-tac-toe perfectly.
//!
//! Perfect play in tic-tac-toe means:
//! - Never losing against any opponent
//! - Always exploiting opponent mistakes
//! - Drawing against another perfect player

use muzero_core::Game;
use muzero_mcts::{games::TicTacToe, Mcts, MctsConfig, RolloutEvaluator};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Create an MCTS instance with the given seed.
fn create_mcts(seed: u64, simulations: usize) -> Mcts<TicTacToe, RolloutEvaluator<ChaCha8Rng>, ChaCha8Rng> {
    let config = MctsConfig::with_simulations(simulations);
    let rng = ChaCha8Rng::seed_from_u64(seed);
    let evaluator = RolloutEvaluator::new(rng.clone(), 50);
    Mcts::new(config, evaluator, rng)
}

/// Test that MCTS never loses at tic-tac-toe against a random opponent.
///
/// This tests the core requirement from SPEC.md §8 P2:
/// "MCTS MUST solve tic-tac-toe perfectly"
#[test]
fn test_mcts_never_loses_as_x() {
    let game = TicTacToe;

    // Play 50 games with MCTS as X (first player)
    for seed in 0..50 {
        let mut mcts = create_mcts(seed, 1000);
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 1000);

        let mut state = game.initial_state();
        let mut move_count = 0;

        while !game.is_terminal(&state) {
            let is_x_turn = move_count % 2 == 0;

            if is_x_turn {
                // MCTS plays
                let result = mcts.search(&game, &state);
                state = game.apply(&state, result.best_action);
            } else {
                // Random opponent plays
                let actions = game.legal_actions(&state);
                let idx = (rand::Rng::gen::<usize>(&mut rng)) % actions.len();
                state = game.apply(&state, actions[idx]);
            }
            move_count += 1;
        }

        // MCTS (X) should not lose
        // outcome is +1 if player who just moved won
        // X loses if O made the winning move (even move_count, outcome +1)
        let outcome = game.outcome(&state).unwrap();
        let x_lost_game = move_count % 2 == 0 && outcome > 0.5;

        assert!(
            !x_lost_game,
            "MCTS (X) lost game with seed {}. Final state after {} moves:\n{}",
            seed,
            move_count,
            state
        );
    }
}

/// Test that MCTS never loses as O (second player) against a random opponent.
#[test]
fn test_mcts_never_loses_as_o() {
    let game = TicTacToe;

    // Play 50 games with MCTS as O (second player)
    for seed in 0..50 {
        let mut mcts = create_mcts(seed, 1000);
        let mut rng = ChaCha8Rng::seed_from_u64(seed + 2000);

        let mut state = game.initial_state();
        let mut move_count = 0;

        while !game.is_terminal(&state) {
            let is_x_turn = move_count % 2 == 0;

            if is_x_turn {
                // Random opponent plays
                let actions = game.legal_actions(&state);
                let idx = (rand::Rng::gen::<usize>(&mut rng)) % actions.len();
                state = game.apply(&state, actions[idx]);
            } else {
                // MCTS plays
                let result = mcts.search(&game, &state);
                state = game.apply(&state, result.best_action);
            }
            move_count += 1;
        }

        // MCTS (O) should not lose
        // O loses if X made the winning move (move_count is odd, outcome is +1)
        let o_lost_game = move_count % 2 == 1 && game.outcome(&state).unwrap() > 0.5;

        assert!(
            !o_lost_game,
            "MCTS (O) lost game with seed {}. Final state after {} moves:\n{}",
            seed,
            move_count,
            state
        );
    }
}

/// Test that two MCTS players always draw.
/// Uses more simulations since both players need to be strong.
#[test]
fn test_mcts_vs_mcts_always_draws() {
    let game = TicTacToe;

    // Use 2000 simulations for strong play, run 10 games to keep test fast
    for seed in 0..10 {
        let mut mcts_x = create_mcts(seed, 2000);
        let mut mcts_o = create_mcts(seed + 500, 2000);

        let mut state = game.initial_state();
        let mut move_count = 0;

        while !game.is_terminal(&state) {
            let is_x_turn = move_count % 2 == 0;

            let result = if is_x_turn {
                mcts_x.search(&game, &state)
            } else {
                mcts_o.search(&game, &state)
            };

            state = game.apply(&state, result.best_action);
            move_count += 1;
        }

        // Should be a draw
        let outcome = game.outcome(&state).unwrap();
        assert!(
            outcome.abs() < 0.5,
            "MCTS vs MCTS should draw, but got outcome {} with seed {}. Final state:\n{}",
            outcome,
            seed,
            state
        );
    }
}

/// Test that MCTS finds winning moves in won positions.
#[test]
fn test_mcts_finds_winning_move() {
    let game = TicTacToe;
    let mut mcts = create_mcts(42, 500);

    // Set up a position where X can win immediately:
    // X _ X
    // O O _
    // _ _ _
    // X at 0, 2; O at 3, 4; X to move at 1 wins
    let mut state = game.initial_state();
    state = game.apply(&state, muzero_mcts::games::TicTacToeAction(0)); // X
    state = game.apply(&state, muzero_mcts::games::TicTacToeAction(3)); // O
    state = game.apply(&state, muzero_mcts::games::TicTacToeAction(2)); // X
    state = game.apply(&state, muzero_mcts::games::TicTacToeAction(4)); // O

    let result = mcts.search(&game, &state);

    assert_eq!(
        result.best_action,
        muzero_mcts::games::TicTacToeAction(1),
        "MCTS should find winning move at cell 1"
    );
}

/// Test that MCTS blocks opponent's winning move.
#[test]
fn test_mcts_blocks_winning_move() {
    let game = TicTacToe;
    let mut mcts = create_mcts(42, 500);

    // Set up a position where O must block:
    // X X _
    // O _ _
    // _ _ _
    // X at 0, 1; O at 3; O to move must block at 2
    let mut state = game.initial_state();
    state = game.apply(&state, muzero_mcts::games::TicTacToeAction(0)); // X
    state = game.apply(&state, muzero_mcts::games::TicTacToeAction(3)); // O
    state = game.apply(&state, muzero_mcts::games::TicTacToeAction(1)); // X

    let result = mcts.search(&game, &state);

    assert_eq!(
        result.best_action,
        muzero_mcts::games::TicTacToeAction(2),
        "MCTS should block X's winning move at cell 2"
    );
}

/// Test determinism: same seed produces identical games.
/// This verifies SPEC.md §7 INV-3: Same seed → identical game
#[test]
fn test_mcts_deterministic() {
    let game = TicTacToe;

    let play_game = |seed: u64| -> Vec<muzero_mcts::games::TicTacToeAction> {
        let mut mcts = create_mcts(seed, 100);
        let mut state = game.initial_state();
        let mut moves = Vec::new();

        while !game.is_terminal(&state) {
            let result = mcts.search(&game, &state);
            moves.push(result.best_action);
            state = game.apply(&state, result.best_action);
        }
        moves
    };

    let game1 = play_game(12345);
    let game2 = play_game(12345);

    assert_eq!(
        game1, game2,
        "Same seed should produce identical game sequences"
    );
}

/// Test that MCTS handles the starting position correctly.
/// First player (X) should be able to force at least a draw.
#[test]
fn test_mcts_starting_position_value() {
    let game = TicTacToe;
    let mut mcts = create_mcts(42, 1000);

    let state = game.initial_state();
    let result = mcts.search(&game, &state);

    // Value should be >= 0 (first player can force at least a draw)
    assert!(
        result.root_value >= -0.3,
        "Starting position value should be >= -0.3 (draw or better), got {}",
        result.root_value
    );

    // Policy should have some mass on reasonable opening moves
    // (corners and center are generally good openings)
    let corner_and_center: f32 = [0usize, 2, 4, 6, 8]
        .iter()
        .map(|&i| result.policy[i])
        .sum();

    assert!(
        corner_and_center > 0.3,
        "Opening policy should favor corners/center, but total was {}",
        corner_and_center
    );
}
