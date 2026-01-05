//! Property-based tests for MCTS implementation.
//!
//! These tests verify the invariants from SPEC.md §7 and §10.2:
//! - INV-1: Policy sums to 1.0 (±1e-5)
//! - INV-2: Value in [-1, 1]
//! - P3: Policy valid
//! - P4: Determinism

use muzero_core::Game;
use muzero_mcts::{games::TicTacToe, Mcts, MctsConfig, RolloutEvaluator};
use proptest::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Tolerance for policy sum validation (from SPEC.md §7 INV-1)
const POLICY_SUM_TOLERANCE: f32 = 1e-5;

// =============================================================================
// Strategies for generating test inputs
// =============================================================================

/// Generate a random seed for MCTS
fn arb_seed() -> impl Strategy<Value = u64> {
    any::<u64>()
}

/// Generate a random number of simulations (10-200 for fast tests)
fn arb_simulations() -> impl Strategy<Value = usize> {
    10usize..200
}

/// Generate a random move number (0-9 for tic-tac-toe)
fn arb_move_number() -> impl Strategy<Value = usize> {
    0usize..9
}

/// Generate a random tic-tac-toe position by making random moves
fn arb_tictactoe_position() -> impl Strategy<Value = (
    <TicTacToe as Game>::State,
    usize, // move number
)> {
    arb_move_number().prop_flat_map(|num_moves| {
        arb_seed().prop_map(move |seed| {
            let game = TicTacToe;
            let mut state = game.initial_state();
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let mut actual_moves = 0;

            for _ in 0..num_moves {
                if game.is_terminal(&state) {
                    break;
                }
                let actions = game.legal_actions(&state);
                if actions.is_empty() {
                    break;
                }
                let idx = rand::Rng::gen_range(&mut rng, 0..actions.len());
                state = game.apply(&state, actions[idx]);
                actual_moves += 1;
            }

            (state, actual_moves)
        })
    })
}

// =============================================================================
// INV-1: Policy sums to 1.0 (SPEC.md §7, §10.2 P3)
// =============================================================================

proptest! {
    /// INV-1/P3: MCTS policy should sum to 1.0 (±1e-5) for any non-terminal position
    #[test]
    fn prop_policy_sums_to_one(
        seed in arb_seed(),
        simulations in arb_simulations(),
        (state, _) in arb_tictactoe_position()
    ) {
        let game = TicTacToe;

        // Skip terminal positions (no policy for terminal states)
        if game.is_terminal(&state) {
            return Ok(());
        }

        let config = MctsConfig::with_simulations(simulations);
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let evaluator = RolloutEvaluator::new(rng.clone(), 20);
        let mut mcts = Mcts::new(config, evaluator, rng);

        let result = mcts.search(&game, &state);

        // Policy should sum to ~1.0
        let policy_sum: f32 = result.policy.iter().sum();
        prop_assert!(
            (policy_sum - 1.0).abs() < POLICY_SUM_TOLERANCE,
            "Policy sum {} is not 1.0 (tolerance {})",
            policy_sum,
            POLICY_SUM_TOLERANCE
        );
    }

    /// Policy should only have non-zero values for legal actions
    #[test]
    fn prop_policy_only_legal_actions(
        seed in arb_seed(),
        simulations in arb_simulations(),
        (state, _) in arb_tictactoe_position()
    ) {
        let game = TicTacToe;

        if game.is_terminal(&state) {
            return Ok(());
        }

        let config = MctsConfig::with_simulations(simulations);
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let evaluator = RolloutEvaluator::new(rng.clone(), 20);
        let mut mcts = Mcts::new(config, evaluator, rng);

        let result = mcts.search(&game, &state);
        let legal_actions = game.legal_actions(&state);

        // Check that policy is zero for illegal actions
        for i in 0..game.num_actions() {
            let is_legal = legal_actions.iter().any(|a| game.action_to_index(*a) == i);
            if !is_legal {
                prop_assert!(
                    result.policy[i] == 0.0,
                    "Policy has non-zero value {} for illegal action index {}",
                    result.policy[i],
                    i
                );
            }
        }
    }

    /// All policy values should be non-negative
    #[test]
    fn prop_policy_non_negative(
        seed in arb_seed(),
        simulations in arb_simulations(),
        (state, _) in arb_tictactoe_position()
    ) {
        let game = TicTacToe;

        if game.is_terminal(&state) {
            return Ok(());
        }

        let config = MctsConfig::with_simulations(simulations);
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let evaluator = RolloutEvaluator::new(rng.clone(), 20);
        let mut mcts = Mcts::new(config, evaluator, rng);

        let result = mcts.search(&game, &state);

        for (i, &p) in result.policy.iter().enumerate() {
            prop_assert!(
                p >= 0.0,
                "Policy has negative value {} at index {}",
                p,
                i
            );
        }
    }
}

// =============================================================================
// INV-2: Value in [-1, 1] (SPEC.md §7)
// =============================================================================

proptest! {
    /// INV-2: Root value should be in range [-1, 1]
    #[test]
    fn prop_value_in_range(
        seed in arb_seed(),
        simulations in arb_simulations(),
        (state, _) in arb_tictactoe_position()
    ) {
        let game = TicTacToe;

        if game.is_terminal(&state) {
            return Ok(());
        }

        let config = MctsConfig::with_simulations(simulations);
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let evaluator = RolloutEvaluator::new(rng.clone(), 20);
        let mut mcts = Mcts::new(config, evaluator, rng);

        let result = mcts.search(&game, &state);

        prop_assert!(
            result.root_value >= -1.0 && result.root_value <= 1.0,
            "Root value {} is outside range [-1, 1]",
            result.root_value
        );
    }
}

// =============================================================================
// INV-3 / P4: Determinism (SPEC.md §7, §10.2)
// =============================================================================

proptest! {
    /// INV-3/P4: Same seed should produce identical results
    #[test]
    fn prop_deterministic(
        seed in arb_seed(),
        simulations in arb_simulations(),
        (state, _) in arb_tictactoe_position()
    ) {
        let game = TicTacToe;

        if game.is_terminal(&state) {
            return Ok(());
        }

        // Run MCTS twice with same seed
        let run_mcts = || {
            let config = MctsConfig::with_simulations(simulations);
            let rng = ChaCha8Rng::seed_from_u64(seed);
            let evaluator = RolloutEvaluator::new(rng.clone(), 20);
            let mut mcts = Mcts::new(config, evaluator, rng);
            mcts.search(&game, &state)
        };

        let result1 = run_mcts();
        let result2 = run_mcts();

        // Results should be identical
        prop_assert_eq!(result1.best_action, result2.best_action);
        prop_assert_eq!(result1.visit_counts, result2.visit_counts);
        prop_assert_eq!(result1.policy, result2.policy);

        // Float comparison with tolerance
        prop_assert!(
            (result1.root_value - result2.root_value).abs() < 1e-6,
            "Root values differ: {} vs {}",
            result1.root_value,
            result2.root_value
        );
    }
}

// =============================================================================
// Additional property tests
// =============================================================================

proptest! {
    /// Best action should correspond to the action with highest visit count
    #[test]
    fn prop_best_action_is_max_visits(
        seed in arb_seed(),
        simulations in arb_simulations(),
        (state, _) in arb_tictactoe_position()
    ) {
        let game = TicTacToe;

        if game.is_terminal(&state) {
            return Ok(());
        }

        let config = MctsConfig::with_simulations(simulations);
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let evaluator = RolloutEvaluator::new(rng.clone(), 20);
        let mut mcts = Mcts::new(config, evaluator, rng);

        let result = mcts.search(&game, &state);

        // Find the action with max visits
        let max_visits_action = result.visit_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(a, _)| *a);

        prop_assert_eq!(
            Some(result.best_action),
            max_visits_action,
            "Best action {:?} doesn't match action with max visits {:?}",
            result.best_action,
            max_visits_action
        );
    }

    /// Visit counts should be consistent with policy (higher visits = higher probability)
    #[test]
    fn prop_visit_counts_match_policy(
        seed in arb_seed(),
        simulations in 50usize..200, // Need enough simulations for meaningful distribution
        (state, _) in arb_tictactoe_position()
    ) {
        let game = TicTacToe;

        if game.is_terminal(&state) {
            return Ok(());
        }

        let config = MctsConfig::with_simulations(simulations);
        let rng = ChaCha8Rng::seed_from_u64(seed);
        let evaluator = RolloutEvaluator::new(rng.clone(), 20);
        let mut mcts = Mcts::new(config, evaluator, rng);

        let result = mcts.search(&game, &state);

        let total_visits: u32 = result.visit_counts.iter().map(|(_, c)| *c).sum();
        if total_visits == 0 {
            return Ok(());
        }

        // Check that policy matches normalized visit counts
        for (action, count) in &result.visit_counts {
            let idx = game.action_to_index(*action);
            let expected_prob = *count as f32 / total_visits as f32;
            let actual_prob = result.policy[idx];

            prop_assert!(
                (expected_prob - actual_prob).abs() < 1e-5,
                "Policy mismatch for action {:?}: expected {}, got {}",
                action,
                expected_prob,
                actual_prob
            );
        }
    }
}
