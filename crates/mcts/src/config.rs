//! MCTS configuration parameters.
//!
//! These parameters control the behavior of the Monte Carlo Tree Search algorithm.
//! Default values are taken from SPEC.md §6.

/// MCTS configuration parameters.
#[derive(Clone, Debug)]
pub struct MctsConfig {
    /// Number of simulations per search.
    pub num_simulations: usize,

    /// Dirichlet noise alpha (for root exploration).
    /// Higher values = more uniform noise, lower = more concentrated.
    pub dirichlet_alpha: f32,

    /// Fraction of prior replaced with Dirichlet noise at root.
    /// 0 = no exploration noise, 1 = pure noise.
    pub exploration_fraction: f32,

    /// PUCT exploration constant base.
    /// Part of the formula: c = pb_c_init + log((N + pb_c_base + 1) / pb_c_base)
    pub pb_c_base: f32,

    /// PUCT exploration constant init.
    /// Part of the formula: c = pb_c_init + log((N + pb_c_base + 1) / pb_c_base)
    pub pb_c_init: f32,

    /// Temperature for action selection during training.
    /// - 0.0: always pick highest visit count (greedy)
    /// - 1.0: sample proportional to visit counts
    /// - >1.0: more uniform distribution
    /// - <1.0: more peaked distribution
    pub temperature: f32,

    /// Move number at which to drop temperature to 0 (greedy).
    /// This encourages exploration early in the game and exploitation later.
    /// Set to 0 to always use the configured temperature.
    pub temperature_drop_move: usize,
}

impl Default for MctsConfig {
    fn default() -> Self {
        // Values from SPEC.md §6 and config.toml
        Self {
            num_simulations: 800,
            dirichlet_alpha: 0.3,
            exploration_fraction: 0.25,
            pb_c_base: 19652.0,
            pb_c_init: 1.25,
            temperature: 1.0,
            temperature_drop_move: 30,
        }
    }
}

impl MctsConfig {
    /// Create a new config with the specified number of simulations.
    pub fn with_simulations(num_simulations: usize) -> Self {
        Self {
            num_simulations,
            ..Default::default()
        }
    }

    /// Create a config for evaluation (greedy action selection).
    pub fn for_evaluation(num_simulations: usize) -> Self {
        Self {
            num_simulations,
            temperature: 0.0,
            temperature_drop_move: 0,
            // No exploration noise for evaluation
            exploration_fraction: 0.0,
            ..Default::default()
        }
    }

    /// Get the effective temperature for a given move number.
    pub fn effective_temperature(&self, move_number: usize) -> f32 {
        if self.temperature_drop_move > 0 && move_number >= self.temperature_drop_move {
            0.0
        } else {
            self.temperature
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MctsConfig::default();
        assert_eq!(config.num_simulations, 800);
        assert!((config.dirichlet_alpha - 0.3).abs() < 1e-5);
        assert!((config.exploration_fraction - 0.25).abs() < 1e-5);
        assert!((config.pb_c_base - 19652.0).abs() < 1e-5);
        assert!((config.pb_c_init - 1.25).abs() < 1e-5);
        assert!((config.temperature - 1.0).abs() < 1e-5);
        assert_eq!(config.temperature_drop_move, 30);
    }

    #[test]
    fn test_with_simulations() {
        let config = MctsConfig::with_simulations(100);
        assert_eq!(config.num_simulations, 100);
        // Other values should be default
        assert!((config.dirichlet_alpha - 0.3).abs() < 1e-5);
    }

    #[test]
    fn test_for_evaluation() {
        let config = MctsConfig::for_evaluation(100);
        assert_eq!(config.num_simulations, 100);
        assert_eq!(config.temperature, 0.0);
        assert_eq!(config.exploration_fraction, 0.0);
    }

    #[test]
    fn test_effective_temperature() {
        let config = MctsConfig::default();

        // Before drop move
        assert!((config.effective_temperature(0) - 1.0).abs() < 1e-5);
        assert!((config.effective_temperature(29) - 1.0).abs() < 1e-5);

        // At and after drop move
        assert_eq!(config.effective_temperature(30), 0.0);
        assert_eq!(config.effective_temperature(50), 0.0);
    }

    #[test]
    fn test_effective_temperature_no_drop() {
        let mut config = MctsConfig::default();
        config.temperature_drop_move = 0;

        // Should always use configured temperature
        assert!((config.effective_temperature(0) - 1.0).abs() < 1e-5);
        assert!((config.effective_temperature(100) - 1.0).abs() < 1e-5);
    }
}
