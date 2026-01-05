//! Tic-tac-toe implementation for MCTS validation.
//!
//! Tic-tac-toe is a solved game where perfect play always results in a draw.
//! This makes it ideal for validating MCTS correctness:
//! - MCTS should never lose against any opponent
//! - Two MCTS players should always draw
//! - MCTS should exploit opponent mistakes

use muzero_core::Game;
use std::fmt;
use std::hash::Hash;

/// Tic-tac-toe player.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Player {
    X,
    O,
}

impl Player {
    /// Get the opposing player.
    pub fn opposite(self) -> Self {
        match self {
            Player::X => Player::O,
            Player::O => Player::X,
        }
    }
}

impl fmt::Display for Player {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Player::X => write!(f, "X"),
            Player::O => write!(f, "O"),
        }
    }
}

/// Tic-tac-toe board state.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct TicTacToeState {
    /// Board: 9 cells, indexed 0-8 (row-major).
    /// ```text
    /// 0 | 1 | 2
    /// ---------
    /// 3 | 4 | 5
    /// ---------
    /// 6 | 7 | 8
    /// ```
    board: [Option<Player>; 9],

    /// Current player to move.
    current: Player,

    /// Cached winner (if any).
    winner: Option<Player>,
}

impl TicTacToeState {
    /// Create a new empty board with X to move.
    pub fn new() -> Self {
        Self {
            board: [None; 9],
            current: Player::X,
            winner: None,
        }
    }

    /// Get the current player to move.
    pub fn current_player(&self) -> Player {
        self.current
    }

    /// Get the winner, if any.
    pub fn winner(&self) -> Option<Player> {
        self.winner
    }

    /// Get the piece at a cell, if any.
    pub fn get(&self, cell: usize) -> Option<Player> {
        self.board.get(cell).copied().flatten()
    }

    /// Check for a winner on the current board.
    fn check_winner(&self) -> Option<Player> {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2], // top row
            [3, 4, 5], // middle row
            [6, 7, 8], // bottom row
            [0, 3, 6], // left column
            [1, 4, 7], // center column
            [2, 5, 8], // right column
            [0, 4, 8], // main diagonal
            [2, 4, 6], // anti-diagonal
        ];

        for line in LINES {
            if let Some(player) = self.board[line[0]] {
                if self.board[line[1]] == Some(player) && self.board[line[2]] == Some(player) {
                    return Some(player);
                }
            }
        }
        None
    }

    /// Check if the board is full (draw if no winner).
    fn is_full(&self) -> bool {
        self.board.iter().all(|c| c.is_some())
    }
}

impl Default for TicTacToeState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TicTacToeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..3 {
            if row > 0 {
                writeln!(f, "-----------")?;
            }
            for col in 0..3 {
                if col > 0 {
                    write!(f, " | ")?;
                }
                let cell = row * 3 + col;
                match self.board[cell] {
                    Some(Player::X) => write!(f, " X ")?,
                    Some(Player::O) => write!(f, " O ")?,
                    None => write!(f, "   ")?,
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Tic-tac-toe action (cell index 0-8).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TicTacToeAction(pub u8);

impl TicTacToeAction {
    /// Get the row (0-2).
    pub fn row(self) -> u8 {
        self.0 / 3
    }

    /// Get the column (0-2).
    pub fn col(self) -> u8 {
        self.0 % 3
    }
}

impl fmt::Display for TicTacToeAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.row(), self.col())
    }
}

/// Tic-tac-toe game implementation.
#[derive(Clone, Debug)]
pub struct TicTacToe;

impl Game for TicTacToe {
    type State = TicTacToeState;
    type Action = TicTacToeAction;
    type Observation = [f32; 18]; // 9 cells x 2 (X plane, O plane)

    fn initial_state(&self) -> Self::State {
        TicTacToeState::new()
    }

    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
        state
            .board
            .iter()
            .enumerate()
            .filter(|(_, cell)| cell.is_none())
            .map(|(i, _)| TicTacToeAction(i as u8))
            .collect()
    }

    fn apply(&self, state: &Self::State, action: Self::Action) -> Self::State {
        let mut new_state = state.clone();
        new_state.board[action.0 as usize] = Some(state.current);
        new_state.current = state.current.opposite();
        new_state.winner = new_state.check_winner();
        new_state
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        state.winner.is_some() || state.is_full()
    }

    fn outcome(&self, state: &Self::State) -> Option<f32> {
        if let Some(winner) = state.winner {
            // Return from perspective of player who just moved.
            // If X won and it's now O's turn, X just moved and won.
            if winner != state.current {
                Some(1.0) // Player who just moved won
            } else {
                // This shouldn't happen in a valid game
                Some(-1.0)
            }
        } else if state.is_full() {
            Some(0.0) // Draw
        } else {
            None // Game not over
        }
    }

    fn observe(&self, state: &Self::State) -> Self::Observation {
        let mut obs = [0.0f32; 18];
        for (i, cell) in state.board.iter().enumerate() {
            match cell {
                Some(Player::X) => obs[i] = 1.0,
                Some(Player::O) => obs[i + 9] = 1.0,
                None => {}
            }
        }
        obs
    }

    fn action_to_index(&self, action: Self::Action) -> usize {
        action.0 as usize
    }

    fn index_to_action(&self, index: usize) -> Option<Self::Action> {
        if index < 9 {
            Some(TicTacToeAction(index as u8))
        } else {
            None
        }
    }

    fn num_actions(&self) -> usize {
        9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let game = TicTacToe;
        let state = game.initial_state();

        assert_eq!(state.current_player(), Player::X);
        assert!(state.winner().is_none());
        assert!(!game.is_terminal(&state));
    }

    #[test]
    fn test_legal_moves_empty_board() {
        let game = TicTacToe;
        let state = game.initial_state();
        let actions = game.legal_actions(&state);

        assert_eq!(actions.len(), 9);
    }

    #[test]
    fn test_legal_moves_partial_board() {
        let game = TicTacToe;
        let mut state = game.initial_state();

        // Play X in center
        state = game.apply(&state, TicTacToeAction(4));
        let actions = game.legal_actions(&state);

        assert_eq!(actions.len(), 8);
        assert!(!actions.contains(&TicTacToeAction(4)));
    }

    #[test]
    fn test_apply_move() {
        let game = TicTacToe;
        let state = game.initial_state();

        let new_state = game.apply(&state, TicTacToeAction(0));

        assert_eq!(new_state.get(0), Some(Player::X));
        assert_eq!(new_state.current_player(), Player::O);
    }

    #[test]
    fn test_x_wins_top_row() {
        let game = TicTacToe;
        let mut state = game.initial_state();

        // X plays 0, 1, 2 (top row)
        // O plays 3, 4
        state = game.apply(&state, TicTacToeAction(0)); // X
        state = game.apply(&state, TicTacToeAction(3)); // O
        state = game.apply(&state, TicTacToeAction(1)); // X
        state = game.apply(&state, TicTacToeAction(4)); // O
        state = game.apply(&state, TicTacToeAction(2)); // X wins

        assert!(game.is_terminal(&state));
        assert_eq!(state.winner(), Some(Player::X));

        // X just moved and won, outcome should be +1
        let outcome = game.outcome(&state).unwrap();
        assert_eq!(outcome, 1.0);
    }

    #[test]
    fn test_o_wins_diagonal() {
        let game = TicTacToe;
        let mut state = game.initial_state();

        // O plays 2, 4, 6 (anti-diagonal)
        // X plays 0, 1, 3
        state = game.apply(&state, TicTacToeAction(0)); // X
        state = game.apply(&state, TicTacToeAction(2)); // O
        state = game.apply(&state, TicTacToeAction(1)); // X
        state = game.apply(&state, TicTacToeAction(4)); // O
        state = game.apply(&state, TicTacToeAction(3)); // X
        state = game.apply(&state, TicTacToeAction(6)); // O wins

        assert!(game.is_terminal(&state));
        assert_eq!(state.winner(), Some(Player::O));

        // O just moved and won, outcome should be +1
        let outcome = game.outcome(&state).unwrap();
        assert_eq!(outcome, 1.0);
    }

    #[test]
    fn test_draw() {
        let game = TicTacToe;
        let mut state = game.initial_state();

        // Classic draw game:
        // X O X
        // X X O
        // O X O
        let moves = [0, 1, 2, 4, 3, 5, 7, 6, 8];
        for (i, &cell) in moves.iter().enumerate() {
            state = game.apply(&state, TicTacToeAction(cell));
        }

        assert!(game.is_terminal(&state));
        assert!(state.winner().is_none());
        assert_eq!(game.outcome(&state), Some(0.0));
    }

    #[test]
    fn test_observation() {
        let game = TicTacToe;
        let mut state = game.initial_state();

        state = game.apply(&state, TicTacToeAction(0)); // X at 0
        state = game.apply(&state, TicTacToeAction(4)); // O at 4

        let obs = game.observe(&state);

        // X plane: position 0 should be 1
        assert_eq!(obs[0], 1.0);
        assert_eq!(obs[1], 0.0);

        // O plane: position 4 should be 1
        assert_eq!(obs[9 + 4], 1.0);
        assert_eq!(obs[9], 0.0);
    }

    #[test]
    fn test_action_index_roundtrip() {
        let game = TicTacToe;

        for i in 0..9 {
            let action = TicTacToeAction(i);
            let index = game.action_to_index(action);
            let recovered = game.index_to_action(index).unwrap();
            assert_eq!(action, recovered);
        }
    }

    #[test]
    fn test_display() {
        let game = TicTacToe;
        let mut state = game.initial_state();

        state = game.apply(&state, TicTacToeAction(0)); // X
        state = game.apply(&state, TicTacToeAction(4)); // O

        let display = format!("{}", state);
        assert!(display.contains("X"));
        assert!(display.contains("O"));
    }
}
