//! MuZero Chess - Chess implementation with bitboard representation
//!
//! This crate implements the chess rules using bitboards for efficient
//! move generation and position evaluation.

mod attacks;
mod bitboard;
mod game_impl;
mod movegen;
mod moves;
mod piece;
mod position;
mod square;

pub use attacks::{attacks, AttackTables};
pub use bitboard::{Bitboard, BitboardIter};
pub use game_impl::Chess;
pub use moves::{Move, MoveFlags};
pub use piece::{Color, Piece, PieceType};
pub use position::{CastlingRights, DrawReason, GameResult, Position};
pub use square::Square;
