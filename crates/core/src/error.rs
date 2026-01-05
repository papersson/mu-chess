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
}

/// Convenience Result type for MuZero operations
pub type Result<T> = std::result::Result<T, MuZeroError>;
