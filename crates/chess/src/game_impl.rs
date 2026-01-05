//! Implementation of the MuZero Game trait for Chess.

use muzero_core::Game;

use crate::{Color, GameResult, Move, Position};

/// Chess game implementation for MuZero.
#[derive(Clone)]
pub struct Chess;

impl Game for Chess {
    type State = Position;
    type Action = Move;
    // For now, observation is empty - proper encoding comes in later phases
    type Observation = Vec<f32>;

    fn initial_state(&self) -> Position {
        Position::starting()
    }

    fn legal_actions(&self, state: &Position) -> Vec<Move> {
        state.legal_moves()
    }

    fn apply(&self, state: &Position, action: Move) -> Position {
        state.make_move(action)
    }

    fn is_terminal(&self, state: &Position) -> bool {
        state.is_terminal()
    }

    fn outcome(&self, state: &Position) -> Option<f32> {
        state.outcome().map(|result| match result {
            GameResult::WhiteWins => {
                // Return +1 if the side that just moved won
                // After a terminal position, it's the opponent's turn
                // So if White won, Black was about to move, meaning White just moved
                if state.side_to_move() == Color::Black {
                    1.0
                } else {
                    -1.0
                }
            }
            GameResult::BlackWins => {
                if state.side_to_move() == Color::White {
                    1.0
                } else {
                    -1.0
                }
            }
            GameResult::Draw(_) => 0.0,
        })
    }

    fn observe(&self, _state: &Position) -> Vec<f32> {
        // Placeholder - proper observation encoding for neural network in later phases
        Vec::new()
    }

    fn action_to_index(&self, action: Move) -> usize {
        // Simple encoding: raw 16-bit move value
        // This gives 65536 possible indices
        // Proper AlphaZero-style encoding (8x8x73=4672) in later phases
        action.raw() as usize
    }

    fn index_to_action(&self, index: usize) -> Option<Move> {
        if index < 65536 {
            Some(Move(index as u16))
        } else {
            None
        }
    }

    fn num_actions(&self) -> usize {
        65536
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_trait_basic() {
        let game = Chess;
        let state = game.initial_state();

        // Initial position is not terminal
        assert!(!game.is_terminal(&state));
        assert!(game.outcome(&state).is_none());

        // Should have 20 legal moves
        let actions = game.legal_actions(&state);
        assert_eq!(actions.len(), 20);
    }

    #[test]
    fn test_game_apply_move() {
        let game = Chess;
        let state = game.initial_state();
        let actions = game.legal_actions(&state);

        // Apply first legal move
        let new_state = game.apply(&state, actions[0]);

        // New state should have different side to move
        assert_ne!(state.side_to_move(), new_state.side_to_move());
    }

    #[test]
    fn test_game_checkmate() {
        let game = Chess;
        // Scholar's mate - black is checkmated
        let state = Position::from_fen(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
        )
        .unwrap();

        assert!(game.is_terminal(&state));

        // White won, so from Black's perspective (the side about to move), it's -1
        // But our outcome returns from the perspective of the side that just moved
        // In this case, white just moved and won, so outcome should be +1 for white
        // But wait - it's black to move in this position after white just moved
        // So outcome for the side that just moved (white) should be +1
        let outcome = game.outcome(&state).unwrap();
        assert_eq!(outcome, 1.0); // White just moved and won
    }
}
