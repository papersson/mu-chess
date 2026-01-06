//! Self-play game generation and evaluation for MuZero training.
//!
//! Generates chess games using MCTS and saves them in MessagePack format
//! for the Python training pipeline. Also supports evaluation against
//! baseline opponents.

mod minimax;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use minimax::MinimaxEvaluator;
use muzero_chess::{Chess, Color, Move};
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

/// MuZero Chess self-play and evaluation tool.
#[derive(Parser)]
#[command(name = "muzero-selfplay")]
#[command(about = "Generate self-play games and evaluate MuZero models")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate self-play games for training.
    Generate {
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
    },

    /// Evaluate model against baseline opponent.
    Evaluate {
        /// Directory containing ONNX models.
        #[arg(short, long)]
        model: PathBuf,

        /// Number of games to play.
        #[arg(short, long, default_value = "100")]
        games: usize,

        /// Minimax search depth.
        #[arg(short, long, default_value = "3")]
        depth: usize,

        /// Number of MCTS simulations per move.
        #[arg(short, long, default_value = "100")]
        simulations: usize,

        /// Hidden dimension for neural network.
        #[arg(long, default_value = "256")]
        hidden_dim: usize,

        /// Random seed for reproducibility.
        #[arg(long, default_value = "42")]
        seed: u64,
    },
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

/// Evaluation results.
struct EvaluationResult {
    muzero_wins: usize,
    minimax_wins: usize,
    draws: usize,
    total_games: usize,
}

impl EvaluationResult {
    fn win_rate(&self) -> f32 {
        self.muzero_wins as f32 / self.total_games as f32
    }
}

/// Generate a single game using MCTS with rollout evaluator.
fn generate_game(
    game: &Chess,
    config: &MctsConfig,
    seed: u64,
    temperature: f32,
    temperature_drop: usize,
    rollout_depth: usize,
) -> GameRecord {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let evaluator =
        RolloutEvaluator::new(ChaCha8Rng::seed_from_u64(seed.wrapping_add(1)), rollout_depth);
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
fn sparse_policy(result: &SearchResult<Move>, game: &Chess) -> HashMap<u16, f32> {
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

/// Play a single game between MuZero (with MCTS) and Minimax.
fn play_evaluation_game(
    game: &Chess,
    muzero_evaluator: &NeuralEvaluator,
    minimax: &MinimaxEvaluator,
    mcts_config: &MctsConfig,
    muzero_plays_white: bool,
    seed: u64,
) -> f32 {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut mcts = Mcts::new(mcts_config.clone(), muzero_evaluator, rng.clone());

    let mut state = game.initial_state();
    let mut move_count = 0;
    const MAX_MOVES: usize = 500; // Prevent infinite games

    while !game.is_terminal(&state) && move_count < MAX_MOVES {
        let is_white_turn = state.side_to_move() == Color::White;
        let is_muzero_turn = is_white_turn == muzero_plays_white;

        let mv = if is_muzero_turn {
            // MuZero's turn - use MCTS
            let result = mcts.search(game, &state);
            result.select_action(0.0, &mut rng) // Greedy for evaluation
        } else {
            // Minimax's turn
            minimax.best_move(game, &state).unwrap_or_else(|| {
                // Fallback: pick first legal move
                game.legal_actions(&state)[0]
            })
        };

        state = game.apply(&state, mv);
        move_count += 1;
    }

    // Get outcome (from White's perspective for consistency)
    if let Some(outcome) = game.outcome(&state) {
        // outcome is from perspective of player who just moved
        // Convert to White's perspective
        if state.side_to_move() == Color::Black {
            // White just moved
            outcome
        } else {
            // Black just moved
            -outcome
        }
    } else {
        0.0 // Draw (including timeout)
    }
}

/// Run evaluation: MuZero vs Minimax.
fn run_evaluation(
    model_dir: &PathBuf,
    num_games: usize,
    depth: usize,
    simulations: usize,
    hidden_dim: usize,
    seed: u64,
) -> Result<EvaluationResult> {
    println!("Loading neural network from {:?}", model_dir);
    let neural_evaluator = NeuralEvaluator::from_directory(model_dir, hidden_dim)
        .with_context(|| format!("Failed to load neural network from {:?}", model_dir))?;

    let game = Chess;
    let minimax = MinimaxEvaluator::new(depth);
    let mcts_config = MctsConfig::with_simulations(simulations);

    let mut muzero_wins = 0;
    let mut minimax_wins = 0;
    let mut draws = 0;

    println!(
        "\nPlaying {} games: MuZero (MCTS {} sims) vs Minimax (depth {})",
        num_games, simulations, depth
    );
    println!("================================================");

    for i in 0..num_games {
        // Alternate colors for fairness
        let muzero_plays_white = i % 2 == 0;
        let game_seed = seed.wrapping_add(i as u64 * 1000);

        let outcome = play_evaluation_game(
            &game,
            &neural_evaluator,
            &minimax,
            &mcts_config,
            muzero_plays_white,
            game_seed,
        );

        // Determine winner
        // outcome is from White's perspective
        if outcome > 0.5 {
            // White wins
            if muzero_plays_white {
                muzero_wins += 1;
            } else {
                minimax_wins += 1;
            }
        } else if outcome < -0.5 {
            // Black wins
            if muzero_plays_white {
                minimax_wins += 1;
            } else {
                muzero_wins += 1;
            }
        } else {
            draws += 1;
        }

        // Progress update
        if (i + 1) % 10 == 0 || i + 1 == num_games {
            println!(
                "Game {}/{}: MuZero {} - {} Minimax ({} draws)",
                i + 1,
                num_games,
                muzero_wins,
                minimax_wins,
                draws
            );
        }
    }

    Ok(EvaluationResult {
        muzero_wins,
        minimax_wins,
        draws,
        total_games: num_games,
    })
}

/// Run the generate command.
fn cmd_generate(
    games: usize,
    output: PathBuf,
    simulations: usize,
    seed: u64,
    temperature: f32,
    temperature_drop: usize,
    rollout_depth: usize,
    model: Option<PathBuf>,
    hidden_dim: usize,
) -> Result<()> {
    // Create output directory
    fs::create_dir_all(&output)
        .with_context(|| format!("Failed to create output directory: {:?}", output))?;

    let evaluator_type = if model.is_some() { "neural" } else { "rollout" };
    println!(
        "Generating {} games with {} simulations/move ({})",
        games, simulations, evaluator_type
    );
    println!("Output directory: {:?}", output);
    println!("Seed: {}", seed);

    let start = Instant::now();

    // MCTS configuration
    let config = MctsConfig::with_simulations(simulations);
    let game = Chess;

    // Generate games based on evaluator type
    let game_records: Vec<GameRecord> = if let Some(model_dir) = &model {
        // Neural evaluator mode (sequential)
        println!("Loading neural network from {:?}", model_dir);
        let evaluator = NeuralEvaluator::from_directory(model_dir, hidden_dim)
            .with_context(|| format!("Failed to load neural network from {:?}", model_dir))?;

        println!("Generating games sequentially with neural evaluator...");
        (0..games)
            .map(|i| {
                let game_seed = seed.wrapping_add(i as u64 * 1000);
                if i > 0 && i % 10 == 0 {
                    println!("  Generated {} games...", i);
                }
                generate_game_neural(&game, &config, &evaluator, game_seed, temperature, temperature_drop)
            })
            .collect()
    } else {
        // Rollout evaluator mode (parallel)
        (0..games)
            .into_par_iter()
            .map(|i| {
                let game_seed = seed.wrapping_add(i as u64 * 1000);
                generate_game(&game, &config, game_seed, temperature, temperature_drop, rollout_depth)
            })
            .collect()
    };

    // Save each game to a separate MessagePack file
    for (i, game_record) in game_records.iter().enumerate() {
        let filename = output.join(format!("game_{:06}.msgpack", i));
        let file =
            File::create(&filename).with_context(|| format!("Failed to create file: {:?}", filename))?;
        let mut writer = BufWriter::new(file);
        // Use named fields to serialize structs as maps (not arrays)
        rmp_serde::encode::write_named(&mut writer, game_record)
            .with_context(|| format!("Failed to serialize game {}", i))?;
    }

    let elapsed = start.elapsed();
    let total_moves: usize = game_records.iter().map(|g| g.steps.len()).sum();
    let avg_moves = total_moves as f64 / games as f64;

    println!("\nCompleted in {:.2}s", elapsed.as_secs_f64());
    println!("Games generated: {}", games);
    println!("Total moves: {}", total_moves);
    println!("Average game length: {:.1} moves", avg_moves);
    println!("Files saved to: {:?}", output);

    // Print outcome distribution
    let white_wins = game_records.iter().filter(|g| g.outcome > 0.5).count();
    let black_wins = game_records.iter().filter(|g| g.outcome < -0.5).count();
    let draws = game_records.iter().filter(|g| g.outcome.abs() <= 0.5).count();
    println!(
        "\nOutcomes: White wins: {}, Black wins: {}, Draws: {}",
        white_wins, black_wins, draws
    );

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            games,
            output,
            simulations,
            seed,
            temperature,
            temperature_drop,
            rollout_depth,
            model,
            hidden_dim,
        } => cmd_generate(
            games,
            output,
            simulations,
            seed,
            temperature,
            temperature_drop,
            rollout_depth,
            model,
            hidden_dim,
        ),

        Commands::Evaluate {
            model,
            games,
            depth,
            simulations,
            hidden_dim,
            seed,
        } => {
            let result = run_evaluation(&model, games, depth, simulations, hidden_dim, seed)?;

            println!("\n================================================");
            println!("FINAL RESULTS");
            println!("================================================");
            println!("MuZero wins:  {} ({:.1}%)", result.muzero_wins, result.muzero_wins as f32 / result.total_games as f32 * 100.0);
            println!("Minimax wins: {} ({:.1}%)", result.minimax_wins, result.minimax_wins as f32 / result.total_games as f32 * 100.0);
            println!("Draws:        {} ({:.1}%)", result.draws, result.draws as f32 / result.total_games as f32 * 100.0);
            println!("------------------------------------------------");
            println!("Win rate: {:.1}%", result.win_rate() * 100.0);

            if result.win_rate() >= 0.7 {
                println!("\n SUCCESS: MuZero achieves >70% win rate!");
            } else {
                println!("\n Target not yet reached (need >70% win rate)");
            }

            Ok(())
        }
    }
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
