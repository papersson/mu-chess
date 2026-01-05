//! Monte Carlo Tree Search for MuZero.
//!
//! This crate provides a generic MCTS implementation that can be used with
//! any game implementing the `muzero_core::Game` trait.
//!
//! # Features
//!
//! - **Generic**: Works with any `Game` implementation
//! - **PUCT Selection**: Uses the PUCT formula for action selection
//! - **Evaluator Abstraction**: Supports different evaluation strategies
//!   (rollouts for Phase 2, neural networks for later phases)
//! - **Dirichlet Noise**: Adds exploration noise at the root node
//! - **Temperature Sampling**: Supports temperature-based action selection
//! - **Typed Results**: Optional typed Policy/Value outputs with invariant enforcement
//!
//! # Example
//!
//! ```
//! use muzero_mcts::{Mcts, MctsConfig, RolloutEvaluator, games::TicTacToe};
//! use muzero_core::Game;
//! use rand::SeedableRng;
//! use rand_chacha::ChaCha8Rng;
//!
//! let game = TicTacToe;
//! let state = game.initial_state();
//!
//! let config = MctsConfig::with_simulations(100);
//! let rng = ChaCha8Rng::seed_from_u64(42);
//! let evaluator = RolloutEvaluator::new(rng.clone(), 20);
//! let mut mcts = Mcts::new(config, evaluator, rng);
//!
//! let result = mcts.search(&game, &state);
//! println!("Best action: {:?}", result.best_action);
//! println!("Root value: {}", result.root_value);
//!
//! // Get typed Policy/Value with invariant enforcement
//! let policy = result.typed_policy().expect("valid policy");
//! let value = result.typed_value();
//! ```

pub mod config;
pub mod evaluator;
pub mod games;
mod node;
pub mod search;
mod tree;

pub use config::MctsConfig;
pub use evaluator::{Evaluation, Evaluator, RolloutEvaluator};
pub use search::{Mcts, SearchResult};
