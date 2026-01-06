#!/bin/bash
# MuZero Chess Training Loop
#
# This script orchestrates the full training loop:
# 1. Generate self-play games
# 2. Train the network
# 3. Export to ONNX
# 4. Evaluate against minimax
# 5. Repeat until target win rate is achieved

set -e

# Configuration
ITERATIONS=${1:-5}
GAMES_PER_ITER=${GAMES_PER_ITER:-100}
SIMULATIONS=${SIMULATIONS:-100}
TRAIN_STEPS=${TRAIN_STEPS:-5000}
EVAL_GAMES=${EVAL_GAMES:-50}
MINIMAX_DEPTH=${MINIMAX_DEPTH:-3}
TARGET_WIN_RATE=${TARGET_WIN_RATE:-70}

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "========================================"
echo "MuZero Chess Training Loop"
echo "========================================"
echo "Iterations:      $ITERATIONS"
echo "Games/iteration: $GAMES_PER_ITER"
echo "MCTS sims:       $SIMULATIONS"
echo "Train steps:     $TRAIN_STEPS"
echo "Eval games:      $EVAL_GAMES"
echo "Target win rate: ${TARGET_WIN_RATE}%"
echo "========================================"
echo ""

# Create data directory if it doesn't exist
mkdir -p data/games
mkdir -p checkpoints

for iter in $(seq 1 "$ITERATIONS"); do
    echo ""
    echo "========================================"
    echo "ITERATION $iter / $ITERATIONS"
    echo "========================================"

    # 1. Generate games
    echo ""
    echo "Step 1: Generating self-play games..."
    echo "----------------------------------------"

    if [ "$iter" -eq 1 ] && [ ! -f "checkpoints/initial_inference.onnx" ]; then
        echo "First iteration: Using rollout evaluator (random playouts)"
        cargo run -p muzero-selfplay --release -- generate \
            --games "$GAMES_PER_ITER" \
            --simulations "$SIMULATIONS" \
            --output "data/games/iter_$iter"
    else
        echo "Using trained neural network"
        cargo run -p muzero-selfplay --release -- generate \
            --games "$GAMES_PER_ITER" \
            --simulations "$SIMULATIONS" \
            --model checkpoints \
            --output "data/games/iter_$iter"
    fi

    # 2. Train
    echo ""
    echo "Step 2: Training network..."
    echo "----------------------------------------"
    cd training
    uv run python -m muzero.cli train \
        --steps "$TRAIN_STEPS" \
        --checkpoint-interval 1000 \
        --log-interval 100
    cd "$PROJECT_ROOT"

    # 3. Export ONNX
    echo ""
    echo "Step 3: Exporting ONNX model..."
    echo "----------------------------------------"
    cd training
    uv run python -m muzero.cli export --verify
    cd "$PROJECT_ROOT"

    # 4. Evaluate
    echo ""
    echo "Step 4: Evaluating against minimax (depth $MINIMAX_DEPTH)..."
    echo "----------------------------------------"

    # Capture evaluation output
    EVAL_OUTPUT=$(cargo run -p muzero-selfplay --release -- evaluate \
        --model checkpoints \
        --games "$EVAL_GAMES" \
        --depth "$MINIMAX_DEPTH" \
        --simulations "$SIMULATIONS" 2>&1)

    echo "$EVAL_OUTPUT"

    # Extract win rate (look for "Win rate: XX.X%")
    WIN_RATE=$(echo "$EVAL_OUTPUT" | grep -o "Win rate: [0-9.]*" | grep -o "[0-9.]*" | tail -1)

    if [ -n "$WIN_RATE" ]; then
        echo ""
        echo "----------------------------------------"
        echo "Iteration $iter complete. Win rate: ${WIN_RATE}%"

        # Check if target reached (compare as integers after removing decimal)
        WIN_RATE_INT=$(echo "$WIN_RATE" | cut -d. -f1)
        if [ "$WIN_RATE_INT" -ge "$TARGET_WIN_RATE" ]; then
            echo ""
            echo "========================================"
            echo "TARGET ACHIEVED!"
            echo "Win rate ${WIN_RATE}% >= ${TARGET_WIN_RATE}%"
            echo "========================================"
            exit 0
        fi
    else
        echo "Warning: Could not parse win rate from output"
    fi
done

echo ""
echo "========================================"
echo "Training complete after $ITERATIONS iterations"
echo "Target win rate not yet reached."
echo "Consider running more iterations or adjusting hyperparameters."
echo "========================================"
