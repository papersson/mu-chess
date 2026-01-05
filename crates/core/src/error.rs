use std::io;
use thiserror::Error;

/// Errors that can occur in the MuZero system
#[derive(Error, Debug)]
pub enum MuZeroError {
    #[error("Invalid FEN string: {0}")]
    InvalidFen(String),

    #[error("Invalid square index: {0}")]
    InvalidSquare(u8),

    #[error("Invalid move: {0}")]
    InvalidMove(String),

    #[error("No legal moves available")]
    NoLegalMoves,

    #[error("Game is not terminal")]
    NotTerminal,

    #[error("Invalid policy: {0}")]
    InvalidPolicy(String),

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
}

/// Convenience Result type for MuZero operations
pub type Result<T> = std::result::Result<T, MuZeroError>;
