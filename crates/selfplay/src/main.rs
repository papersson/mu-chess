//! Self-play game generation for MuZero training.
//!
//! Generates chess games using MCTS and saves them in MessagePack format
//! for the Python training pipeline.

use anyhow::{Context, Result};
use clap::Parser;
use muzero_chess::Chess;
use muzero_core::Game;
use muzero_inference::NeuralEvaluator;
use muzero_mcts::{Mcts, MctsConfig, RolloutEvaluator, SearchResult};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Instant;

/// Self-play game generator for MuZero Chess.
#[derive(Parser)]
#[command(name = "selfplay")]
#[command(about = "Generate self-play games for MuZero training")]
struct Cli {
    /// Number of games to generate.
    #[arg(short, long, default_value = "10")]
    games: usize,

    /// Output directory for game files.
    #[arg(short, long, default_value = "data/games")]
    output: PathBuf,

    /// Number of MCTS simulations per move.
    #[arg(short, long, default_value = "50")]
    simulations: usize,

    /// Random seed for reproducibility.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Temperature for move selection (1.0 = proportional to visit counts).
    #[arg(short, long, default_value = "1.0")]
    temperature: f32,

    /// Move number after which temperature drops to 0 (greedy).
    #[arg(long, default_value = "30")]
    temperature_drop: usize,

    /// Maximum rollout depth for evaluator (only used without --model).
    #[arg(long, default_value = "50")]
    rollout_depth: usize,

    /// Directory containing ONNX models (initial_inference.onnx, recurrent_inference.onnx).
    /// If not specified, uses random rollout evaluator.
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Hidden dimension for neural network (default: 256).
    #[arg(long, default_value = "256")]
    hidden_dim: usize,
}

/// A single step in a game trajectory.
#[derive(Serialize, Deserialize, Debug)]
struct GameStep {
    /// Observation tensor (21 planes × 64 squares = 1344 floats).
    observation: Vec<f32>,

    /// Action index (0-65535 for chess moves).
    action: u16,

    /// MCTS policy as sparse map: {action_index: probability}.
    mcts_policy: HashMap<u16, f32>,

    /// Immediate reward (usually 0 for chess).
    reward: f32,
}

/// A complete game trajectory.
#[derive(Serialize, Deserialize, Debug)]
struct GameRecord {
    /// Sequence of game steps.
    steps: Vec<GameStep>,

    /// Game outcome: +1 (white wins), -1 (black wins), 0 (draw).
    outcome: f32,

    /// Optional metadata.
    metadata: HashMap<String, serde_json::Value>,
}

/// Generate a single game using MCTS.
fn generate_game(
    game: &Chess,
    config: &MctsConfig,
    seed: u64,
    temperature: f32,
    temperature_drop: usize,
    rollout_depth: usize,
) -> GameRecord {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let evaluator = RolloutEvaluator::new(ChaCha8Rng::seed_from_u64(seed.wrapping_add(1)), rollout_depth);
    let mut mcts = Mcts::new(config.clone(), evaluator, rng.clone());

    let mut steps = Vec::new();
    let mut state = game.initial_state();
    let mut move_number = 0;

    while !game.is_terminal(&state) {
        // Get observation before making the move
        let observation = game.observe(&state);

        // Run MCTS search
        let result: SearchResult<_> = mcts.search(game, &state);

        // Select action based on temperature
        let temp = if move_number < temperature_drop {
            temperature
        } else {
            0.0 // Greedy after temperature drop
        };
        let action = result.select_action(temp, &mut rng);

        // Convert to sparse policy (only legal moves)
        let mcts_policy = sparse_policy(&result, game);

        steps.push(GameStep {
            observation,
            action: game.action_to_index(action) as u16,
            mcts_policy,
            reward: 0.0, // Chess has no intermediate rewards
        });

        state = game.apply(&state, action);
        move_number += 1;
    }

    // Get game outcome
    let outcome = game.outcome(&state).unwrap_or(0.0);

    // Metadata
    let mut metadata = HashMap::new();
    metadata.insert("seed".to_string(), serde_json::json!(seed));
    metadata.insert("moves".to_string(), serde_json::json!(move_number));

    GameRecord {
        steps,
        outcome,
        metadata,
    }
}

/// Generate a single game using neural network evaluator.
fn generate_game_neural(
    game: &Chess,
    config: &MctsConfig,
    evaluator: &NeuralEvaluator,
    seed: u64,
    temperature: f32,
    temperature_drop: usize,
) -> GameRecord {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut mcts = Mcts::new(config.clone(), evaluator, rng.clone());

    let mut steps = Vec::new();
    let mut state = game.initial_state();
    let mut move_number = 0;

    while !game.is_terminal(&state) {
        // Get observation before making the move
        let observation = game.observe(&state);

        // Run MCTS search
        let result: SearchResult<_> = mcts.search(game, &state);

        // Select action based on temperature
        let temp = if move_number < temperature_drop {
            temperature
        } else {
            0.0 // Greedy after temperature drop
        };
        let action = result.select_action(temp, &mut rng);

        // Convert to sparse policy (only legal moves)
        let mcts_policy = sparse_policy(&result, game);

        steps.push(GameStep {
            observation,
            action: game.action_to_index(action) as u16,
            mcts_policy,
            reward: 0.0,
        });

        state = game.apply(&state, action);
        move_number += 1;
    }

    let outcome = game.outcome(&state).unwrap_or(0.0);

    let mut metadata = HashMap::new();
    metadata.insert("seed".to_string(), serde_json::json!(seed));
    metadata.insert("moves".to_string(), serde_json::json!(move_number));
    metadata.insert("evaluator".to_string(), serde_json::json!("neural"));

    GameRecord {
        steps,
        outcome,
        metadata,
    }
}

/// Convert SearchResult policy to sparse format (only include non-zero probabilities).
fn sparse_policy(
    result: &SearchResult<muzero_chess::Move>,
    game: &Chess,
) -> HashMap<u16, f32> {
    let total: u32 = result.visit_counts.iter().map(|(_, c)| *c).sum();
    if total == 0 {
        return HashMap::new();
    }
    result
        .visit_counts
        .iter()
        .filter(|(_, count)| *count > 0)
        .map(|(action, count)| {
            let prob = *count as f32 / total as f32;
            (game.action_to_index(*action) as u16, prob)
        })
        .collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Create output directory
    fs::create_dir_all(&cli.output)
        .with_context(|| format!("Failed to create output directory: {:?}", cli.output))?;

    let evaluator_type = if cli.model.is_some() { "neural" } else { "rollout" };
    println!("Generating {} games with {} simulations/move ({})", cli.games, cli.simulations, evaluator_type);
    println!("Output directory: {:?}", cli.output);
    println!("Seed: {}", cli.seed);

    let start = Instant::now();

    // MCTS configuration
    let config = MctsConfig::with_simulations(cli.simulations);
    let game = Chess;

    // Generate games based on evaluator type
    let games: Vec<GameRecord> = if let Some(model_dir) = &cli.model {
        // Neural evaluator mode (sequential)
        println!("Loading neural network from {:?}", model_dir);
        let evaluator = NeuralEvaluator::from_directory(model_dir, cli.hidden_dim)
            .with_context(|| format!("Failed to load neural network from {:?}", model_dir))?;

        println!("Generating games sequentially with neural evaluator...");
        (0..cli.games)
            .map(|i| {
                let game_seed = cli.seed.wrapping_add(i as u64 * 1000);
                if i > 0 && i % 10 == 0 {
                    println!("  Generated {} games...", i);
                }
                generate_game_neural(
                    &game,
                    &config,
                    &evaluator,
                    game_seed,
                    cli.temperature,
                    cli.temperature_drop,
                )
            })
            .collect()
    } else {
        // Rollout evaluator mode (parallel)
        (0..cli.games)
            .into_par_iter()
            .map(|i| {
                let game_seed = cli.seed.wrapping_add(i as u64 * 1000);
                generate_game(
                    &game,
                    &config,
                    game_seed,
                    cli.temperature,
                    cli.temperature_drop,
                    cli.rollout_depth,
                )
            })
            .collect()
    };

    // Save each game to a separate MessagePack file
    for (i, game_record) in games.iter().enumerate() {
        let filename = cli.output.join(format!("game_{:06}.msgpack", i));
        let file = File::create(&filename)
            .with_context(|| format!("Failed to create file: {:?}", filename))?;
        let mut writer = BufWriter::new(file);
        // Use named fields to serialize structs as maps (not arrays)
        rmp_serde::encode::write_named(&mut writer, game_record)
            .with_context(|| format!("Failed to serialize game {}", i))?;
    }

    let elapsed = start.elapsed();
    let total_moves: usize = games.iter().map(|g| g.steps.len()).sum();
    let avg_moves = total_moves as f64 / cli.games as f64;

    println!("\nCompleted in {:.2}s", elapsed.as_secs_f64());
    println!("Games generated: {}", cli.games);
    println!("Total moves: {}", total_moves);
    println!("Average game length: {:.1} moves", avg_moves);
    println!("Files saved to: {:?}", cli.output);

    // Print outcome distribution
    let white_wins = games.iter().filter(|g| g.outcome > 0.5).count();
    let black_wins = games.iter().filter(|g| g.outcome < -0.5).count();
    let draws = games.iter().filter(|g| g.outcome.abs() <= 0.5).count();
    println!("\nOutcomes: White wins: {}, Black wins: {}, Draws: {}", white_wins, black_wins, draws);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_game() {
        let game = Chess;
        let config = MctsConfig::with_simulations(10); // Low for testing
        let record = generate_game(&game, &config, 42, 1.0, 30, 20);

        // Game should have steps
        assert!(!record.steps.is_empty());

        // Each step should have valid observation
        for step in &record.steps {
            assert_eq!(step.observation.len(), 21 * 64);
            assert!(!step.mcts_policy.is_empty());
        }

        // Outcome should be valid
        assert!(record.outcome >= -1.0 && record.outcome <= 1.0);
    }

    #[test]
    fn test_sparse_policy() {
        let game = Chess;
        let config = MctsConfig::with_simulations(10);
        let rng = ChaCha8Rng::seed_from_u64(42);
        let evaluator = RolloutEvaluator::new(rng.clone(), 20);
        let mut mcts = Mcts::new(config, evaluator, rng);

        let state = game.initial_state();
        let result = mcts.search(&game, &state);
        let policy = sparse_policy(&result, &game);

        // Should have some entries
        assert!(!policy.is_empty());

        // Probabilities should sum to ~1.0
        let sum: f32 = policy.values().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
}
