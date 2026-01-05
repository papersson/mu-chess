//! Test game implementations for MCTS validation.
//!
//! These games are used to verify MCTS correctness before
//! applying it to more complex domains like chess.

pub mod tictactoe;

pub use tictactoe::{TicTacToe, TicTacToeAction, TicTacToeState, Player};
