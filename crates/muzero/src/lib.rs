//! ONNX Runtime inference for MuZero.
//!
//! This crate provides neural network inference using ONNX Runtime,
//! implementing the `Evaluator` trait from `muzero_mcts` for use in MCTS.

mod inference;
mod evaluator;

pub use inference::{OnnxModel, HiddenState, InferenceResult, RecurrentResult};
pub use evaluator::NeuralEvaluator;
