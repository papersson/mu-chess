# MuZero Chess — Implementation Prompt

Read `SPEC.md` first. It contains all architectural decisions.

## Overview

| Component | Stack |
|-----------|-------|
| Chess engine, MCTS, inference | Rust (bitboards, rayon, ort) |
| Training, replay buffer | Python (PyTorch, uv) |
| Communication | MessagePack files |
| Goal | Beat minimax (depth 2-3) via pure self-play |

## Implementation Phases

| Phase | Deliverable | Validation | Status |
|-------|-------------|------------|--------|
| 1 | Rust workspace + chess engine | Perft depth 5 = 4,865,609 | ✅ Done |
| 2 | Generic MCTS + tic-tac-toe | MCTS plays TTT perfectly | ✅ Done |
| 3 | Python training pipeline | Loss decreases on synthetic data | ✅ Done |
| 4 | ONNX export + Rust inference | Full loop works | 🔲 Next |
| 5 | Training + evaluation | Beats minimax >70% | 🔲 |

**Do NOT proceed until current phase validates.**

---

## Phase 4 Requirements

Phase 4 connects training to self-play. Key deliverables:

### 4.1 Rust Side (`crates/muzero/`)
- ONNX Runtime inference wrapper
- `NeuralEvaluator` implementing the `Evaluator` trait
- Load `initial_inference.onnx` and `recurrent_inference.onnx`

### 4.2 Rust Side (`crates/selfplay/`)
- Parallel game generation with Rayon
- Write games to `data/games/*.msgpack`
- Observation encoding matching Python's 21-plane format

### 4.3 Deferred from Phase 3
- **Legal moves masking**: `utils.apply_legal_moves_mask()` exists but not used in training
- **Priority replay sampling**: `priority_alpha` config exists, uniform sampling implemented
- **Real game data**: Training validated on synthetic data

### 4.4 Validation
```bash
# Generate games with random/rollout evaluator
cargo run -p selfplay -- --games 100 --output data/games/

# Verify Python can load them
cd training && uv run python -c "from muzero.replay import ReplayBuffer; rb = ReplayBuffer('data/games'); print(rb.load_games())"

# Train on real games
uv run python train.py train -s 1000

# Export ONNX
uv run python train.py export --verify

# Verify Rust can load ONNX
cargo test -p muzero
```

---

## Rust Workflow

**NEVER manually edit Cargo.toml deps. Use `cargo add`.**

```bash
# Setup workspace
cargo new --lib crates/core && cargo new --lib crates/chess
cargo new --lib crates/mcts && cargo new --lib crates/muzero
cargo new crates/selfplay  # binary

# Add deps (examples)
cargo add -p chess thiserror serde --features derive
cargo add -p chess --dev proptest
cargo add -p mcts rand rand_chacha
cargo add -p muzero ort ndarray
cargo add -p selfplay rayon anyhow clap --features derive
```

**Dev loop:**
```bash
cargo check                      # Fast error check
cargo fmt                        # Format
cargo clippy --fix --allow-dirty # Lint + autofix
cargo test                       # Test
cargo test -p chess perft        # Specific test
```

---

## Python Workflow

**Use `uv` for deps, `uvx` for tools (don't install ruff/mypy as deps).**

```bash
# Setup (already done)
cd training && uv sync

# Run code
uv run python train.py test      # Test network shapes
uv run python train.py train     # Train
uv run python train.py export    # Export to ONNX
```

**Dev loop:**
```bash
uvx ruff format .        # Format
uvx ruff check --fix .   # Lint + autofix
uvx mypy src/            # Type check
uv run pytest            # Test
```

**pyproject.toml tools config:**
```toml
[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP"]

[tool.mypy]
strict = true
```

---

## Data Format (MessagePack)

```rust
#[derive(Serialize, Deserialize)]
struct GameRecord {
    steps: Vec<GameStep>,
    outcome: f32,  // +1/-1/0
    metadata: HashMap<String, Value>,
}

#[derive(Serialize, Deserialize)]
struct GameStep {
    observation: Vec<f32>,     // 21*8*8 = 1344 floats
    action: u16,               // Move in 16-bit format
    mcts_policy: HashMap<u16, f32>,  // Sparse: {action: prob}
    reward: f32,               // Usually 0 for chess
}
```

### Observation Encoding (21 planes × 8×8)
```
Planes 0-5:   White pieces (P, N, B, R, Q, K)
Planes 6-11:  Black pieces (P, N, B, R, Q, K)
Plane 12:     Side to move (all 1s if white)
Planes 13-16: Castling rights (KQkq)
Plane 17:     En passant file
Plane 18:     Halfmove clock (normalized /100)
Plane 19:     Fullmove number (normalized /200)
Plane 20:     All ones (bias plane)
```

---

## Key Constraints

| Constraint | Enforcement |
|------------|-------------|
| Policy sums to 1.0 | Type wrapper + softmax |
| Value in [-1, 1] | Type wrapper + tanh |
| Same seed → same game | Inject all RNG |
| All moves legal | Mask at policy output |
| No human chess knowledge | Pure self-play |

**Priority: Correctness > Understanding > Speed**

---

## Invariants to Property Test

- FEN round-trip: `parse(fen(pos)) == pos`
- All generated moves are legal (no self-check)
- Policy valid distribution
- Deterministic replay with seed

---

## Project Layout

```
mu-chess/
├── crates/
│   ├── core/      # Game trait, Policy/Value types
│   ├── chess/     # Chess engine (bitboards, perft)
│   ├── mcts/      # Generic MCTS + tic-tac-toe
│   ├── muzero/    # ONNX inference (Phase 4)
│   └── selfplay/  # Game generation (Phase 4)
├── training/      # PyTorch training pipeline
│   ├── src/muzero/
│   └── train.py
├── data/games/    # MessagePack game files
└── checkpoints/   # ONNX models
```

---

## Start Phase 4

```bash
cd /Users/patrikpersson/Code/sandbox/mu-chess

# 1. Implement observation encoding in chess crate
# 2. Implement selfplay binary with MessagePack output
# 3. Generate test games
# 4. Train on real data
# 5. Implement ONNX inference in muzero crate
# 6. Full loop: self-play → train → export → inference
```
