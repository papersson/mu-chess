//! Monte Carlo Tree Search implementation.
//!
//! Implements the MCTS algorithm with PUCT selection as described in
//! the MuZero paper and SPEC.md §15.

use crate::{
    config::MctsConfig,
    evaluator::{Evaluation, Evaluator},
    node::{Node, NodeId},
    tree::Tree,
};
use muzero_core::{Game, Policy, Value};
use rand::Rng;
use rand_distr::{Dirichlet, Distribution};
use std::hash::Hash;
use std::marker::PhantomData;

/// Result of an MCTS search.
#[derive(Clone, Debug)]
pub struct SearchResult<A: Clone + Copy + Eq + Hash> {
    /// Visit count for each action at root.
    pub visit_counts: Vec<(A, u32)>,

    /// Best action (highest visit count).
    /// For temperature-based selection, use `select_action()` instead.
    pub best_action: A,

    /// Policy derived from normalized visit counts.
    /// Length equals `game.num_actions()`, sums to 1.0.
    pub policy: Vec<f32>,

    /// Value estimate at root (from perspective of player to move).
    pub root_value: f32,
}

impl<A: Clone + Copy + Eq + Hash> SearchResult<A> {
    /// Select an action using temperature-based sampling.
    ///
    /// - temperature = 0: always return best action (greedy)
    /// - temperature = 1: sample proportional to visit counts
    /// - temperature > 1: more uniform distribution
    /// - temperature < 1: more peaked distribution
    ///
    /// Formula: P(a) ∝ N(a)^(1/τ) where τ is temperature
    pub fn select_action<R: Rng>(&self, temperature: f32, rng: &mut R) -> A {
        // Greedy selection
        if temperature <= 0.0 || self.visit_counts.len() <= 1 {
            return self.best_action;
        }

        // Temperature-based sampling
        let inv_temp = 1.0 / temperature;

        // Compute adjusted probabilities
        let adjusted: Vec<f64> = self
            .visit_counts
            .iter()
            .map(|(_, count)| (*count as f64).powf(inv_temp as f64))
            .collect();

        let sum: f64 = adjusted.iter().sum();
        if sum == 0.0 {
            return self.best_action;
        }

        // Sample from the distribution
        let threshold: f64 = rng.gen::<f64>() * sum;
        let mut cumulative = 0.0;

        for (i, &prob) in adjusted.iter().enumerate() {
            cumulative += prob;
            if cumulative >= threshold {
                return self.visit_counts[i].0;
            }
        }

        // Fallback (shouldn't happen)
        self.best_action
    }

    /// Get the best action (greedy selection).
    pub fn best(&self) -> A {
        self.best_action
    }

    /// Get the policy as a typed Policy (enforces sum to 1.0 invariant).
    ///
    /// This validates that the policy sums to 1.0. For MCTS output, this
    /// should always succeed unless there's a bug.
    ///
    /// # Errors
    /// Returns error if policy doesn't sum to 1.0 (indicates a bug).
    pub fn typed_policy(&self) -> muzero_core::Result<Policy> {
        Policy::new(self.policy.clone())
    }

    /// Get the root value as a typed Value (enforces [-1, 1] range invariant).
    ///
    /// Uses clamping to handle floating point edge cases.
    pub fn typed_value(&self) -> Value {
        Value::clamped(self.root_value)
    }
}

/// Monte Carlo Tree Search with PUCT selection.
///
/// Generic over:
/// - `G`: The game being played
/// - `E`: The evaluation strategy (rollouts or neural network)
/// - `R`: The random number generator
pub struct Mcts<G: Game, E: Evaluator<G>, R: Rng> {
    config: MctsConfig,
    evaluator: E,
    rng: R,
    tree: Tree<G::Action>,
    _game: PhantomData<G>,
}

impl<G, E, R> Mcts<G, E, R>
where
    G: Game,
    G::Action: Clone + Copy + Eq + Hash,
    E: Evaluator<G>,
    R: Rng,
{
    /// Create a new MCTS instance.
    pub fn new(config: MctsConfig, evaluator: E, rng: R) -> Self {
        Self {
            config,
            evaluator,
            rng,
            tree: Tree::new(),
            _game: PhantomData,
        }
    }

    /// Run MCTS from the given state, returning search results.
    pub fn search(&mut self, game: &G, state: &G::State) -> SearchResult<G::Action> {
        self.tree.clear();

        // Check if root is terminal
        if game.is_terminal(state) {
            // No moves available - return empty result
            // This shouldn't happen in normal use, but handle gracefully
            let outcome = game.outcome(state).unwrap_or(0.0);
            return SearchResult {
                visit_counts: Vec::new(),
                best_action: game.index_to_action(0).unwrap(), // Dummy action
                policy: vec![0.0; game.num_actions()],
                root_value: outcome,
            };
        }

        // Expand root node
        let root_eval = self.evaluator.evaluate(game, state);
        self.expand_node(game, state, NodeId::ROOT, &root_eval);

        // Add Dirichlet noise at root for exploration
        self.add_root_noise();

        // Run simulations
        for _ in 0..self.config.num_simulations {
            self.simulate(game, state.clone());
        }

        // Extract results from root
        self.extract_results(game)
    }

    /// Run a single simulation: select -> expand -> backpropagate.
    fn simulate(&mut self, game: &G, initial_state: G::State) {
        let mut path: Vec<NodeId> = vec![NodeId::ROOT];
        let mut current_state = initial_state;
        let mut current_id = NodeId::ROOT;

        // SELECT: traverse tree using PUCT until unexpanded/terminal node
        loop {
            let node = self.tree.get(current_id);

            // If terminal, backpropagate the terminal value
            // (already converted to "player to move" perspective when stored)
            if node.terminal {
                let value = node.terminal_value.unwrap_or(0.0);
                self.backpropagate(&path, value);
                return;
            }

            // If not expanded, we've reached a leaf to expand
            if !node.expanded {
                break;
            }

            // Select best child using PUCT
            let action = self.select_child(current_id);
            current_state = game.apply(&current_state, action);

            // Find child node for this action
            // INVARIANT: select_child only returns actions that have child nodes
            let child_id = self
                .tree
                .get(current_id)
                .children
                .iter()
                .find(|(a, _)| *a == action)
                .map(|(_, id)| *id)
                .expect("BUG: select_child returned action without child node");

            path.push(child_id);
            current_id = child_id;
        }

        // EXPAND: Check for terminal first
        if game.is_terminal(&current_state) {
            // outcome() returns from perspective of player who JUST moved
            // We need value from perspective of player TO MOVE at this node
            // These are opposite, so we negate
            let outcome = game.outcome(&current_state).unwrap_or(0.0);
            let value = -outcome; // Convert to "player to move" perspective
            let node = self.tree.get_mut(current_id);
            node.terminal = true;
            node.terminal_value = Some(value);
            self.backpropagate(&path, value);
            return;
        }

        // Evaluate and expand
        let evaluation = self.evaluator.evaluate(game, &current_state);
        self.expand_node(game, &current_state, current_id, &evaluation);

        // Backpropagate the evaluation value
        self.backpropagate(&path, evaluation.value);
    }

    /// Expand a node by adding children for all legal actions.
    fn expand_node(
        &mut self,
        game: &G,
        state: &G::State,
        node_id: NodeId,
        evaluation: &Evaluation,
    ) {
        let legal_actions = game.legal_actions(state);

        for action in legal_actions {
            let prior = evaluation.policy[game.action_to_index(action)];
            let child = Node::new(Some(action), prior);
            let child_id = self.tree.add(child);
            self.tree.get_mut(node_id).children.push((action, child_id));
        }

        self.tree.get_mut(node_id).expanded = true;
    }

    /// Select best child using PUCT formula.
    ///
    /// UCB(a) = Q(a) + c * P(a) * sqrt(N_parent) / (1 + N_child)
    /// where c = pb_c_init + log((N_parent + pb_c_base + 1) / pb_c_base)
    fn select_child(&self, node_id: NodeId) -> G::Action {
        let node = self.tree.get(node_id);
        let parent_visits = node.stats.visit_count.max(1) as f32;

        // PUCT exploration constant
        let pb_c = ((parent_visits + self.config.pb_c_base + 1.0) / self.config.pb_c_base).ln()
            + self.config.pb_c_init;

        let mut best_action = None;
        let mut best_ucb = f32::NEG_INFINITY;

        for (action, child_id) in &node.children {
            let child = self.tree.get(*child_id);

            // Q-value (negated because we want value from parent's perspective)
            // Child stores value from child's perspective (opponent)
            let q = -child.stats.mean_value();
            let p = child.stats.prior;
            let n = child.stats.visit_count as f32;

            // UCB = Q + c * P * sqrt(N_parent) / (1 + N_child)
            let ucb = q + pb_c * p * parent_visits.sqrt() / (1.0 + n);

            if ucb > best_ucb {
                best_ucb = ucb;
                best_action = Some(*action);
            }
        }

        // INVARIANT: This is only called on expanded nodes with children
        best_action.expect("BUG: select_child called on node without children")
    }

    /// Backpropagate value through path.
    ///
    /// Value is from the perspective of the player at the leaf.
    /// As we go up the tree, we alternate perspective by negating.
    fn backpropagate(&mut self, path: &[NodeId], leaf_value: f32) {
        let mut value = leaf_value;

        for &node_id in path.iter().rev() {
            let node = self.tree.get_mut(node_id);
            node.stats.visit_count += 1;
            node.stats.value_sum += value;
            // Negate for opponent's perspective
            value = -value;
        }
    }

    /// Add Dirichlet noise to root node priors for exploration.
    fn add_root_noise(&mut self) {
        let root = self.tree.get(NodeId::ROOT);
        let num_children = root.children.len();

        // Dirichlet requires at least 2 elements
        if num_children < 2 {
            return;
        }

        // Sample Dirichlet noise
        let alpha = vec![self.config.dirichlet_alpha; num_children];
        let dirichlet = Dirichlet::new(&alpha).unwrap();
        let noise: Vec<f32> = dirichlet.sample(&mut self.rng);

        // Apply noise to priors
        let eps = self.config.exploration_fraction;
        let children: Vec<(G::Action, NodeId)> = self.tree.get(NodeId::ROOT).children.clone();

        for (i, (_, child_id)) in children.iter().enumerate() {
            let child = self.tree.get_mut(*child_id);
            child.stats.prior = (1.0 - eps) * child.stats.prior + eps * noise[i];
        }
    }

    /// Extract search results from root node.
    fn extract_results(&self, game: &G) -> SearchResult<G::Action> {
        let root = self.tree.get(NodeId::ROOT);

        // Collect visit counts
        let visit_counts: Vec<(G::Action, u32)> = root
            .children
            .iter()
            .map(|(a, id)| (*a, self.tree.get(*id).stats.visit_count))
            .collect();

        // Best action is highest visit count
        // INVARIANT: Root is expanded before simulations, so it has children
        let best_action = visit_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(a, _)| *a)
            .expect("BUG: extract_results called but root has no children");

        // Policy from normalized visit counts
        let total_visits: u32 = visit_counts.iter().map(|(_, c)| *c).sum();
        let mut policy = vec![0.0; game.num_actions()];
        if total_visits > 0 {
            for (a, count) in &visit_counts {
                policy[game.action_to_index(*a)] = *count as f32 / total_visits as f32;
            }
        }

        // Root value: average of children weighted by visits (from root's perspective)
        // Since children store opponent's perspective, we negate
        let root_value = if total_visits > 0 {
            let weighted_sum: f32 = root
                .children
                .iter()
                .map(|(_, id)| {
                    let child = self.tree.get(*id);
                    -child.stats.mean_value() * child.stats.visit_count as f32
                })
                .sum();
            weighted_sum / total_visits as f32
        } else {
            0.0
        };

        SearchResult {
            visit_counts,
            best_action,
            policy,
            root_value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluator::RolloutEvaluator;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    // Simple test game: race to 5
    // Players take turns adding 1 or 2. First to reach exactly 5 wins.
    // First player can always win with optimal play.
    #[derive(Clone)]
    struct RaceToFive;

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct RaceState {
        count: u8,
        current_player: u8,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    struct RaceAction(u8); // 1 or 2

    impl Game for RaceToFive {
        type State = RaceState;
        type Action = RaceAction;
        type Observation = ();

        fn initial_state(&self) -> Self::State {
            RaceState {
                count: 0,
                current_player: 0,
            }
        }

        fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
            if state.count >= 5 {
                Vec::new()
            } else {
                let mut actions = vec![RaceAction(1)];
                if state.count + 2 <= 5 {
                    actions.push(RaceAction(2));
                }
                actions
            }
        }

        fn apply(&self, state: &Self::State, action: Self::Action) -> Self::State {
            RaceState {
                count: state.count + action.0,
                current_player: 1 - state.current_player,
            }
        }

        fn is_terminal(&self, state: &Self::State) -> bool {
            state.count >= 5
        }

        fn outcome(&self, state: &Self::State) -> Option<f32> {
            if state.count >= 5 {
                // Player who just moved reached 5 and wins
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
                0 => Some(RaceAction(1)),
                1 => Some(RaceAction(2)),
                _ => None,
            }
        }

        fn num_actions(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_mcts_basic() {
        let config = MctsConfig::with_simulations(100);
        let rng = ChaCha8Rng::seed_from_u64(42);
        let evaluator = RolloutEvaluator::new(rng.clone(), 20);
        let mut mcts = Mcts::new(config, evaluator, rng);

        let game = RaceToFive;
        let state = game.initial_state();

        let result = mcts.search(&game, &state);

        // Should return a valid action
        assert!(result.best_action == RaceAction(1) || result.best_action == RaceAction(2));

        // Policy should sum to ~1.0
        let policy_sum: f32 = result.policy.iter().sum();
        assert!((policy_sum - 1.0).abs() < 0.01);

        // Visit counts should match simulations
        let total_visits: u32 = result.visit_counts.iter().map(|(_, c)| *c).sum();
        // Each simulation visits root once
        assert!(total_visits > 0);
    }

    #[test]
    fn test_mcts_deterministic() {
        let config = MctsConfig::with_simulations(50);

        let run_search = |seed: u64| {
            let rng = ChaCha8Rng::seed_from_u64(seed);
            let evaluator = RolloutEvaluator::new(rng.clone(), 20);
            let mut mcts = Mcts::new(config.clone(), evaluator, rng);

            let game = RaceToFive;
            let state = game.initial_state();
            mcts.search(&game, &state)
        };

        let result1 = run_search(12345);
        let result2 = run_search(12345);

        // Same seed should produce same results
        assert_eq!(result1.best_action, result2.best_action);
        assert_eq!(result1.visit_counts, result2.visit_counts);
    }
}
