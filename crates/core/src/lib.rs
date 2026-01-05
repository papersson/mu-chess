//! MuZero Core - Game abstractions and common types
//!
//! This crate provides the core `Game` trait that defines the interface
//! for any game to be compatible with the MuZero algorithm.
//!
//! # Types
//!
//! - [`Game`] - Trait for game implementations
//! - [`Policy`] - Probability distribution over actions (sums to 1.0)
//! - [`Value`] - Game value estimate in [-1, 1]

mod error;
mod game;
mod types;

pub use error::{MuZeroError, Result};
pub use game::Game;
pub use types::{Policy, Value};
