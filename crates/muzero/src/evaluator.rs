//! Neural network evaluator implementing the MCTS Evaluator trait.

use crate::inference::OnnxModel;
use muzero_chess::Chess;
use muzero_core::Game;
use muzero_mcts::{Evaluation, Evaluator};
use std::path::Path;

/// Neural network evaluator using ONNX Runtime.
///
/// Implements the `Evaluator` trait for use with MCTS.
pub struct NeuralEvaluator {
    model: OnnxModel,
}

impl NeuralEvaluator {
    /// Create a new neural evaluator from ONNX model files.
    ///
    /// # Arguments
    /// * `initial_path` - Path to initial_inference.onnx
    /// * `recurrent_path` - Path to recurrent_inference.onnx
    /// * `hidden_dim` - Hidden state dimension (default 256)
    pub fn new(
        initial_path: impl AsRef<Path>,
        recurrent_path: impl AsRef<Path>,
        hidden_dim: usize,
    ) -> anyhow::Result<Self> {
        let model = OnnxModel::load(initial_path, recurrent_path, hidden_dim)?;
        Ok(Self { model })
    }

    /// Create from a directory containing the ONNX models.
    ///
    /// Expects `initial_inference.onnx` and `recurrent_inference.onnx` in the directory.
    pub fn from_directory(dir: impl AsRef<Path>, hidden_dim: usize) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        let initial_path = dir.join("initial_inference.onnx");
        let recurrent_path = dir.join("recurrent_inference.onnx");
        Self::new(initial_path, recurrent_path, hidden_dim)
    }
}

impl Evaluator<Chess> for NeuralEvaluator {
    fn evaluate(
        &self,
        game: &Chess,
        state: &<Chess as Game>::State,
    ) -> Evaluation {
        evaluate_impl(self, game, state)
    }
}

impl Evaluator<Chess> for &NeuralEvaluator {
    fn evaluate(
        &self,
        game: &Chess,
        state: &<Chess as Game>::State,
    ) -> Evaluation {
        evaluate_impl(*self, game, state)
    }
}

fn evaluate_impl(
    evaluator: &NeuralEvaluator,
    game: &Chess,
    state: &<Chess as Game>::State,
) -> Evaluation {
    let observation = game.observe(state);

    // Run initial inference
    let result = evaluator
        .model
        .initial_inference(&observation)
        .expect("Initial inference failed");

    // The policy from ONNX is already softmax probabilities over 65536 actions
    // We need to mask illegal moves and renormalize
    let legal_actions = game.legal_actions(state);
    let mut masked_policy = vec![0.0; game.num_actions()];

    // Extract probabilities for legal actions
    let mut sum = 0.0;
    for action in &legal_actions {
        let idx = game.action_to_index(*action);
        let prob = result.policy[idx];
        masked_policy[idx] = prob;
        sum += prob;
    }

    // Renormalize to sum to 1.0
    if sum > 0.0 {
        for action in &legal_actions {
            let idx = game.action_to_index(*action);
            masked_policy[idx] /= sum;
        }
    } else {
        // Fallback to uniform if all legal moves have zero probability
        let uniform = 1.0 / legal_actions.len() as f32;
        for action in &legal_actions {
            let idx = game.action_to_index(*action);
            masked_policy[idx] = uniform;
        }
    }

    Evaluation {
        policy: masked_policy,
        value: result.value,
    }
}

#[cfg(test)]
mod tests {
    // Tests require actual ONNX models
    // See integration tests
}
