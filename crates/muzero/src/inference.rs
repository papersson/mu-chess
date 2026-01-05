//! ONNX Runtime session management for MuZero inference.

use anyhow::{Context, Result};
use ndarray::Array4;
use ort::{
    session::Session,
    value::Value,
};
use std::cell::RefCell;
use std::path::Path;

/// Hidden state tensor from the representation network.
#[derive(Clone)]
pub struct HiddenState {
    /// Hidden state data (batch, channels, height, width)
    data: Array4<f32>,
}

impl HiddenState {
    /// Create a new hidden state from raw data.
    pub fn new(data: Array4<f32>) -> Self {
        Self { data }
    }

    /// Get the raw data.
    pub fn data(&self) -> &Array4<f32> {
        &self.data
    }
}

/// Result from initial_inference.
pub struct InferenceResult {
    pub hidden_state: HiddenState,
    pub policy: Vec<f32>,
    pub value: f32,
}

/// Result from recurrent_inference.
pub struct RecurrentResult {
    pub hidden_state: HiddenState,
    pub reward: f32,
    pub policy: Vec<f32>,
    pub value: f32,
}

/// ONNX model wrapper for MuZero inference.
pub struct OnnxModel {
    initial_session: RefCell<Session>,
    recurrent_session: RefCell<Session>,
    hidden_dim: usize,
}

impl OnnxModel {
    /// Load ONNX models from files.
    ///
    /// # Arguments
    /// * `initial_path` - Path to initial_inference.onnx
    /// * `recurrent_path` - Path to recurrent_inference.onnx
    /// * `hidden_dim` - Hidden state dimension (default 256)
    pub fn load(
        initial_path: impl AsRef<Path>,
        recurrent_path: impl AsRef<Path>,
        hidden_dim: usize,
    ) -> Result<Self> {
        let initial_session = Session::builder()?
            .commit_from_file(initial_path.as_ref())
            .with_context(|| {
                format!(
                    "Failed to load initial_inference model from {:?}",
                    initial_path.as_ref()
                )
            })?;

        let recurrent_session = Session::builder()?
            .commit_from_file(recurrent_path.as_ref())
            .with_context(|| {
                format!(
                    "Failed to load recurrent_inference model from {:?}",
                    recurrent_path.as_ref()
                )
            })?;

        Ok(Self {
            initial_session: RefCell::new(initial_session),
            recurrent_session: RefCell::new(recurrent_session),
            hidden_dim,
        })
    }

    /// Run initial inference on an observation.
    ///
    /// # Arguments
    /// * `observation` - Flattened observation (21 * 64 = 1344 floats)
    ///
    /// # Returns
    /// Initial inference result with hidden state, policy, and value.
    pub fn initial_inference(&self, observation: &[f32]) -> Result<InferenceResult> {
        assert_eq!(
            observation.len(),
            21 * 64,
            "Observation must have 21*64=1344 elements"
        );

        // Create input as (shape, Vec<T>) tuple
        let shape: Vec<i64> = vec![1, 21, 8, 8];
        let obs_value = Value::from_array((shape, observation.to_vec()))?;

        // Run inference
        let mut session = self.initial_session.borrow_mut();
        let outputs = session.run(ort::inputs![obs_value])?;

        // Extract outputs - output order: hidden_state, policy, value
        let hidden_state = self.extract_hidden_state(&outputs, 0)?;
        let policy = self.extract_policy(&outputs, 1)?;
        let value = self.extract_value(&outputs, 2)?;

        Ok(InferenceResult {
            hidden_state,
            policy,
            value,
        })
    }

    /// Run recurrent inference on a hidden state and action.
    ///
    /// # Arguments
    /// * `hidden_state` - Current hidden state
    /// * `action` - Action index (0-65535)
    ///
    /// # Returns
    /// Recurrent inference result with new hidden state, reward, policy, and value.
    pub fn recurrent_inference(
        &self,
        hidden_state: &HiddenState,
        action: u16,
    ) -> Result<RecurrentResult> {
        // Create input tensors using (shape, Vec<T>) tuples
        let hidden_shape: Vec<i64> = vec![1, self.hidden_dim as i64, 8, 8];
        let hidden_data: Vec<f32> = hidden_state.data.iter().copied().collect();
        let hidden_value = Value::from_array((hidden_shape, hidden_data))?;

        // Action as (1,) tensor
        let action_shape: Vec<i64> = vec![1];
        let action_data: Vec<i64> = vec![action as i64];
        let action_value = Value::from_array((action_shape, action_data))?;

        // Run inference
        let mut session = self.recurrent_session.borrow_mut();
        let outputs = session.run(ort::inputs![hidden_value, action_value])?;

        // Extract outputs - output order: next_hidden_state, reward, policy, value
        let next_hidden_state = self.extract_hidden_state(&outputs, 0)?;
        let reward = self.extract_scalar(&outputs, 1)?;
        let policy = self.extract_policy(&outputs, 2)?;
        let value = self.extract_value(&outputs, 3)?;

        Ok(RecurrentResult {
            hidden_state: next_hidden_state,
            reward,
            policy,
            value,
        })
    }

    /// Extract hidden state from output.
    fn extract_hidden_state(
        &self,
        outputs: &ort::session::SessionOutputs,
        index: usize,
    ) -> Result<HiddenState> {
        let output_names: Vec<_> = outputs.keys().collect();
        let name = output_names
            .get(index)
            .context("Missing hidden state output")?;

        let tensor = outputs
            .get(*name)
            .context("Failed to get hidden state tensor")?;

        let (_shape, data) = tensor.try_extract_tensor::<f32>()?;

        // Convert to owned Array4
        let expected_size = 1 * self.hidden_dim * 8 * 8;
        assert_eq!(
            data.len(),
            expected_size,
            "Hidden state size mismatch: expected {}, got {}",
            expected_size,
            data.len()
        );

        let array4 = Array4::from_shape_vec((1, self.hidden_dim, 8, 8), data.to_vec())
            .context("Failed to reshape hidden state")?;

        Ok(HiddenState::new(array4))
    }

    /// Extract policy from output.
    fn extract_policy(
        &self,
        outputs: &ort::session::SessionOutputs,
        index: usize,
    ) -> Result<Vec<f32>> {
        let output_names: Vec<_> = outputs.keys().collect();
        let name = output_names.get(index).context("Missing policy output")?;

        let tensor = outputs
            .get(*name)
            .context("Failed to get policy tensor")?;

        let (_, data) = tensor.try_extract_tensor::<f32>()?;
        Ok(data.to_vec())
    }

    /// Extract scalar value from output.
    fn extract_value(
        &self,
        outputs: &ort::session::SessionOutputs,
        index: usize,
    ) -> Result<f32> {
        self.extract_scalar(outputs, index)
    }

    /// Extract scalar from output.
    fn extract_scalar(
        &self,
        outputs: &ort::session::SessionOutputs,
        index: usize,
    ) -> Result<f32> {
        let output_names: Vec<_> = outputs.keys().collect();
        let name = output_names.get(index).context("Missing scalar output")?;

        let tensor = outputs
            .get(*name)
            .context("Failed to get scalar tensor")?;

        let (_, data) = tensor.try_extract_tensor::<f32>()?;
        data.first()
            .copied()
            .context("Empty scalar output")
    }

    /// Get the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }
}

#[cfg(test)]
mod tests {
    // Tests would require actual ONNX models
    // See integration tests in the selfplay crate
}
