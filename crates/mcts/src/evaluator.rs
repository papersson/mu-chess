//! Evaluation abstraction for MCTS.
//!
//! The `Evaluator` trait allows swapping between different evaluation strategies:
//! - Phase 2: `RolloutEvaluator` uses random playouts
//! - Phase 3+: Neural network provides (policy, value)

use muzero_core::Game;
use rand::Rng;
use std::cell::RefCell;

/// Evaluation result: prior policy + value estimate.
#[derive(Clone, Debug)]
pub struct Evaluation {
    /// Prior probability for each action index.
    /// Length should equal `game.num_actions()`.
    pub policy: Vec<f32>,

    /// Value estimate from this position (from perspective of player to move).
    /// Should be in range [-1, 1].
    pub value: f32,
}

/// Trait for evaluating game positions.
///
/// This abstraction allows MCTS to work with different evaluation strategies.
/// In Phase 2, we use random rollouts. In later phases, a neural network
/// will provide the policy and value estimates.
pub trait Evaluator<G: Game> {
    /// Evaluate a position, returning prior policy and value estimate.
    ///
    /// The policy should be a distribution over all action indices (summing to ~1.0),
    /// with non-zero probability only for legal actions.
    ///
    /// The value should be from the perspective of the player to move,
    /// in the range [-1, 1] where +1 is winning and -1 is losing.
    fn evaluate(&self, game: &G, state: &G::State) -> Evaluation;
}

/// Evaluator using uniform prior and random rollouts.
///
/// This is the simplest evaluation strategy:
/// - Policy: uniform distribution over legal actions
/// - Value: result of a random playout from the position
pub struct RolloutEvaluator<R: Rng> {
    /// Random number generator (wrapped in RefCell for interior mutability).
    rng: RefCell<R>,

    /// Maximum depth for random rollouts.
    max_rollout_depth: usize,
}

impl<R: Rng> RolloutEvaluator<R> {
    /// Create a new rollout evaluator.
    ///
    /// # Arguments
    /// * `rng` - Random number generator for rollouts
    /// * `max_rollout_depth` - Maximum moves in a random playout
    pub fn new(rng: R, max_rollout_depth: usize) -> Self {
        Self {
            rng: RefCell::new(rng),
            max_rollout_depth,
        }
    }

    /// Perform a random rollout from the given state.
    ///
    /// Returns the game outcome from the perspective of the player
    /// who was to move at the start of the rollout.
    fn rollout<G: Game>(&self, game: &G, initial_state: &G::State) -> f32 {
        let mut state = initial_state.clone();
        let mut depth = 0;

        while !game.is_terminal(&state) && depth < self.max_rollout_depth {
            let legal_actions = game.legal_actions(&state);
            if legal_actions.is_empty() {
                break;
            }

            // Random move
            let idx = self.rng.borrow_mut().gen_range(0..legal_actions.len());
            let action = legal_actions[idx];
            state = game.apply(&state, action);
            depth += 1;
        }

        // Get terminal value if game ended
        if let Some(outcome) = game.outcome(&state) {
            // outcome is from perspective of player who just moved
            // We want it from perspective of initial player to move
            // If depth is odd, initial player just moved → return outcome
            // If depth is even, opponent just moved → return -outcome
            if depth % 2 == 1 {
                outcome
            } else {
                -outcome
            }
        } else {
            // Game didn't end in rollout - return 0 (draw estimate)
            0.0
        }
    }
}

impl<G: Game, R: Rng> Evaluator<G> for RolloutEvaluator<R> {
    fn evaluate(&self, game: &G, state: &G::State) -> Evaluation {
        let legal_actions = game.legal_actions(state);
        let num_actions = game.num_actions();

        // Uniform prior over legal actions
        let mut policy = vec![0.0; num_actions];
        if !legal_actions.is_empty() {
            let prior = 1.0 / legal_actions.len() as f32;
            for action in &legal_actions {
                policy[game.action_to_index(*action)] = prior;
            }
        }

        // Random rollout for value estimate
        let value = self.rollout(game, state);

        Evaluation { policy, value }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::ChaCha8Rng;
    use rand::SeedableRng;

    // Simple test game: counting to 3 (whoever says 3 wins)
    #[derive(Clone)]
    struct CountingGame;

    #[derive(Clone, PartialEq, Eq)]
    struct CountingState {
        count: u8,
        current_player: u8, // 0 or 1
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    struct CountingAction(u8); // 1 or 2

    impl Game for CountingGame {
        type State = CountingState;
        type Action = CountingAction;
        type Observation = ();

        fn initial_state(&self) -> Self::State {
            CountingState {
                count: 0,
                current_player: 0,
            }
        }

        fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
            if state.count >= 3 {
                Vec::new()
            } else if state.count == 2 {
                // Can only say 1 (reaching 3)
                vec![CountingAction(1)]
            } else {
                // Can say 1 or 2
                vec![CountingAction(1), CountingAction(2)]
            }
        }

        fn apply(&self, state: &Self::State, action: Self::Action) -> Self::State {
            CountingState {
                count: state.count + action.0,
                current_player: 1 - state.current_player,
            }
        }

        fn is_terminal(&self, state: &Self::State) -> bool {
            state.count >= 3
        }

        fn outcome(&self, state: &Self::State) -> Option<f32> {
            if state.count >= 3 {
                // Player who just moved reached 3 and wins
                Some(1.0)
            } else {
                None
            }
        }

        fn observe(&self, _state: &Self::State) -> Self::Observation {}

        fn action_to_index(&self, action: Self::Action) -> usize {
            (action.0 - 1) as usize
        }

        fn index_to_action(&self, index: usize) -> Option<Self::Action> {
            match index {
                0 => Some(CountingAction(1)),
                1 => Some(CountingAction(2)),
                _ => None,
            }
        }

        fn num_actions(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_rollout_evaluator_policy() {
        let rng = ChaCha8Rng::seed_from_u64(42);
        let evaluator = RolloutEvaluator::new(rng, 10);
        let game = CountingGame;
        let state = game.initial_state();

        let eval = evaluator.evaluate(&game, &state);

        // Policy should be uniform over 2 legal actions
        assert_eq!(eval.policy.len(), 2);
        assert!((eval.policy[0] - 0.5).abs() < 1e-5);
        assert!((eval.policy[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_rollout_evaluator_value_range() {
        let rng = ChaCha8Rng::seed_from_u64(42);
        let evaluator = RolloutEvaluator::new(rng, 10);
        let game = CountingGame;
        let state = game.initial_state();

        let eval = evaluator.evaluate(&game, &state);

        // Value should be in [-1, 1]
        assert!(eval.value >= -1.0 && eval.value <= 1.0);
    }
}
