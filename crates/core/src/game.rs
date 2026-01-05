use std::hash::Hash;

/// A game abstraction for MuZero planning.
///
/// This trait defines the interface that any game must implement to be
/// compatible with the MuZero algorithm. It is designed to be game-agnostic,
/// supporting chess, tic-tac-toe, and other two-player zero-sum games.
pub trait Game: Clone + Send + Sync {
    /// The game state (e.g., chess position)
    type State: Clone + Send;

    /// A game action (e.g., chess move)
    type Action: Clone + Copy + Send + Eq + Hash;

    /// The observation format for the neural network
    type Observation;

    /// Returns the initial game state
    fn initial_state(&self) -> Self::State;

    /// Returns all legal actions from the given state
    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action>;

    /// Applies an action, returning a new state (immutable operation)
    fn apply(&self, state: &Self::State, action: Self::Action) -> Self::State;

    /// Returns true if the game has ended (checkmate, stalemate, draw, etc.)
    fn is_terminal(&self, state: &Self::State) -> bool;

    /// Returns the game outcome from the perspective of the player who just moved:
    /// - `Some(1.0)` if that player won
    /// - `Some(-1.0)` if that player lost
    /// - `Some(0.0)` for a draw
    /// - `None` if the game is not terminal
    fn outcome(&self, state: &Self::State) -> Option<f32>;

    /// Converts game state to neural network observation format
    fn observe(&self, state: &Self::State) -> Self::Observation;

    /// Maps an action to a flat index for the policy vector
    fn action_to_index(&self, action: Self::Action) -> usize;

    /// Maps a flat index back to an action, returning None if invalid
    fn index_to_action(&self, index: usize) -> Option<Self::Action>;

    /// Total number of possible action indices (size of policy vector)
    fn num_actions(&self) -> usize;
}
