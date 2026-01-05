//! MuZero domain types with enforced invariants.
//!
//! These types ensure critical invariants are maintained at the type level:
//! - Policy: probability distribution summing to 1.0
//! - Value: game value in range [-1, 1]

use crate::{MuZeroError, Result};

/// Tolerance for policy sum validation.
const POLICY_SUM_TOLERANCE: f32 = 1e-5;

/// A probability distribution over actions.
///
/// Invariant: All values are non-negative and sum to 1.0 (±1e-5).
///
/// # Example
/// ```
/// use muzero_core::Policy;
///
/// let policy = Policy::new(vec![0.3, 0.5, 0.2]).unwrap();
/// assert!((policy.sum() - 1.0).abs() < 1e-5);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Policy(Vec<f32>);

impl Policy {
    /// Create a new policy from a probability distribution.
    ///
    /// # Errors
    /// Returns `MuZeroError::InvalidPolicy` if:
    /// - Any value is negative
    /// - Values don't sum to 1.0 (±1e-5)
    /// - Vector is empty
    pub fn new(probs: Vec<f32>) -> Result<Self> {
        if probs.is_empty() {
            return Err(MuZeroError::InvalidPolicy(
                "policy cannot be empty".to_string(),
            ));
        }

        // Check for negative values
        if probs.iter().any(|&p| p < 0.0) {
            return Err(MuZeroError::InvalidPolicy(
                "policy contains negative values".to_string(),
            ));
        }

        // Check sum
        let sum: f32 = probs.iter().sum();
        if (sum - 1.0).abs() > POLICY_SUM_TOLERANCE {
            return Err(MuZeroError::InvalidPolicy(format!(
                "policy sum {} is not 1.0 (tolerance {})",
                sum, POLICY_SUM_TOLERANCE
            )));
        }

        Ok(Self(probs))
    }

    /// Create a policy from raw values, normalizing them to sum to 1.0.
    ///
    /// # Errors
    /// Returns error if any value is negative or all values are zero.
    pub fn from_unnormalized(values: Vec<f32>) -> Result<Self> {
        if values.is_empty() {
            return Err(MuZeroError::InvalidPolicy(
                "policy cannot be empty".to_string(),
            ));
        }

        if values.iter().any(|&v| v < 0.0) {
            return Err(MuZeroError::InvalidPolicy(
                "policy contains negative values".to_string(),
            ));
        }

        let sum: f32 = values.iter().sum();
        if sum == 0.0 {
            return Err(MuZeroError::InvalidPolicy(
                "cannot normalize: all values are zero".to_string(),
            ));
        }

        let normalized: Vec<f32> = values.iter().map(|&v| v / sum).collect();
        Ok(Self(normalized))
    }

    /// Create a uniform policy over the given number of actions.
    ///
    /// # Errors
    /// Returns error if num_actions is zero.
    pub fn uniform(num_actions: usize) -> Result<Self> {
        if num_actions == 0 {
            return Err(MuZeroError::InvalidPolicy(
                "cannot create uniform policy with 0 actions".to_string(),
            ));
        }

        let prob = 1.0 / num_actions as f32;
        Ok(Self(vec![prob; num_actions]))
    }

    /// Get the probability at the given index.
    pub fn get(&self, index: usize) -> Option<f32> {
        self.0.get(index).copied()
    }

    /// Get the probability at the given index, returning 0 if out of bounds.
    pub fn get_or_zero(&self, index: usize) -> f32 {
        self.0.get(index).copied().unwrap_or(0.0)
    }

    /// Get the number of actions in this policy.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the policy is empty (should never be true for valid policies).
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the sum of all probabilities (should be ~1.0).
    pub fn sum(&self) -> f32 {
        self.0.iter().sum()
    }

    /// Get the index of the maximum probability.
    pub fn argmax(&self) -> usize {
        self.0
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get the underlying vector (consumes self).
    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }

    /// Get a reference to the underlying slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }

    /// Iterate over the probabilities.
    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.0.iter()
    }
}

impl std::ops::Index<usize> for Policy {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IntoIterator for Policy {
    type Item = f32;
    type IntoIter = std::vec::IntoIter<f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

/// A game value estimate.
///
/// Invariant: Value is in range [-1, 1] where:
/// - +1 means the current player is winning
/// - -1 means the current player is losing
/// - 0 means a draw or equal position
///
/// # Example
/// ```
/// use muzero_core::Value;
///
/// let value = Value::new(0.5).unwrap();
/// assert!(value.get() >= -1.0 && value.get() <= 1.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Value(f32);

impl Value {
    /// Create a new value.
    ///
    /// # Errors
    /// Returns `MuZeroError::InvalidValue` if the value is outside [-1, 1].
    pub fn new(value: f32) -> Result<Self> {
        if !(-1.0..=1.0).contains(&value) {
            return Err(MuZeroError::InvalidValue(format!(
                "value {} is outside range [-1, 1]",
                value
            )));
        }
        Ok(Self(value))
    }

    /// Create a value by clamping to [-1, 1].
    ///
    /// Use this when you have a value that might be slightly outside
    /// the valid range due to floating point errors.
    pub fn clamped(value: f32) -> Self {
        Self(value.clamp(-1.0, 1.0))
    }

    /// Create a value from tanh output (always in [-1, 1]).
    pub fn from_tanh(value: f32) -> Self {
        // tanh output is always in (-1, 1), but clamp for safety
        Self(value.clamp(-1.0, 1.0))
    }

    /// Value for a win.
    pub const WIN: Self = Self(1.0);

    /// Value for a loss.
    pub const LOSS: Self = Self(-1.0);

    /// Value for a draw.
    pub const DRAW: Self = Self(0.0);

    /// Get the underlying value.
    pub fn get(self) -> f32 {
        self.0
    }

    /// Negate the value (for opponent's perspective).
    pub fn negate(self) -> Self {
        Self(-self.0)
    }

    /// Check if this represents a win (value > 0.5).
    pub fn is_winning(self) -> bool {
        self.0 > 0.5
    }

    /// Check if this represents a loss (value < -0.5).
    pub fn is_losing(self) -> bool {
        self.0 < -0.5
    }

    /// Check if this represents a draw-ish position (|value| <= 0.5).
    pub fn is_drawn(self) -> bool {
        self.0.abs() <= 0.5
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.3}", self.0)
    }
}

impl From<Value> for f32 {
    fn from(v: Value) -> f32 {
        v.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Policy tests
    #[test]
    fn test_policy_new_valid() {
        let policy = Policy::new(vec![0.3, 0.5, 0.2]).unwrap();
        assert_eq!(policy.len(), 3);
        assert!((policy.sum() - 1.0).abs() < POLICY_SUM_TOLERANCE);
    }

    #[test]
    fn test_policy_new_invalid_sum() {
        let result = Policy::new(vec![0.3, 0.3, 0.3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_policy_new_negative() {
        let result = Policy::new(vec![0.5, -0.2, 0.7]);
        assert!(result.is_err());
    }

    #[test]
    fn test_policy_new_empty() {
        let result = Policy::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_policy_from_unnormalized() {
        let policy = Policy::from_unnormalized(vec![1.0, 2.0, 1.0]).unwrap();
        assert!((policy[0] - 0.25).abs() < 1e-5);
        assert!((policy[1] - 0.50).abs() < 1e-5);
        assert!((policy[2] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_policy_uniform() {
        let policy = Policy::uniform(4).unwrap();
        assert_eq!(policy.len(), 4);
        for i in 0..4 {
            assert!((policy[i] - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_policy_argmax() {
        let policy = Policy::new(vec![0.1, 0.6, 0.3]).unwrap();
        assert_eq!(policy.argmax(), 1);
    }

    // Value tests
    #[test]
    fn test_value_new_valid() {
        assert!(Value::new(0.0).is_ok());
        assert!(Value::new(1.0).is_ok());
        assert!(Value::new(-1.0).is_ok());
        assert!(Value::new(0.5).is_ok());
    }

    #[test]
    fn test_value_new_invalid() {
        assert!(Value::new(1.1).is_err());
        assert!(Value::new(-1.1).is_err());
        assert!(Value::new(f32::INFINITY).is_err());
    }

    #[test]
    fn test_value_clamped() {
        assert_eq!(Value::clamped(1.5).get(), 1.0);
        assert_eq!(Value::clamped(-1.5).get(), -1.0);
        assert_eq!(Value::clamped(0.5).get(), 0.5);
    }

    #[test]
    fn test_value_negate() {
        assert_eq!(Value::new(0.5).unwrap().negate().get(), -0.5);
        assert_eq!(Value::WIN.negate().get(), -1.0);
    }

    #[test]
    fn test_value_is_winning_losing() {
        assert!(Value::new(0.8).unwrap().is_winning());
        assert!(!Value::new(0.3).unwrap().is_winning());
        assert!(Value::new(-0.8).unwrap().is_losing());
        assert!(!Value::new(-0.3).unwrap().is_losing());
    }

    #[test]
    fn test_value_constants() {
        assert_eq!(Value::WIN.get(), 1.0);
        assert_eq!(Value::LOSS.get(), -1.0);
        assert_eq!(Value::DRAW.get(), 0.0);
    }
}
