//! MuZero Core - Game abstractions and common types
//!
//! This crate provides the core `Game` trait that defines the interface
//! for any game to be compatible with the MuZero algorithm.

mod error;
mod game;

pub use error::{MuZeroError, Result};
pub use game::Game;
