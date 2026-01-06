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
| 4 | ONNX export + Rust inference | Full loop works | ✅ Done |
| 5 | Training + evaluation | Beats minimax >70% | 🔨 Code Complete |

**Do NOT proceed until current phase validates.**

---

## Phase 4 (Completed)

Phase 4 connected training to self-play:

- ✅ `crates/muzero/`: ONNX inference with `NeuralEvaluator`
- ✅ `crates/selfplay/`: Parallel game generation with MessagePack output
- ✅ `crates/chess/src/observation.rs`: 21-plane encoding
- ✅ Full loop: selfplay → train → export → inference

---

## Phase 5 Requirements

Phase 5 completes the training loop and validates against minimax.

### 5.1 CRITICAL: Legal Moves Masking (SPEC P3)

**Problem:** Training and inference are inconsistent:
- Training: Softmax over all 65536 actions (illegal get probability mass)
- Inference: Rust masks illegal moves post-hoc, renormalizes

**Per SPEC.md P3:** "Illegal moves MUST be impossible (masking)"

**Fix in `training/src/muzero/trainer.py`:**

```python
def _policy_loss(
    self, policy_logits: torch.Tensor, target_policy: torch.Tensor
) -> torch.Tensor:
    # Derive legal mask from target (target > 0 means legal)
    legal_mask = (target_policy > 0).float()

    # Apply mask: set illegal logits to -inf before softmax
    masked_logits = policy_logits.clone()
    masked_logits[legal_mask == 0] = float("-inf")

    # Cross-entropy on masked logits
    log_probs = F.log_softmax(masked_logits, dim=-1)
    return -torch.sum(target_policy * log_probs, dim=-1).mean()
```

**This ensures:**
1. Network learns to put zero probability on illegal moves
2. Training matches inference behavior
3. Policy sums to 1.0 over legal moves only (INV-1)

### 5.2 Priority Replay Sampling

Currently: Uniform sampling from replay buffer.
Required: Prioritized sampling based on TD error or similar.

**Modify `training/src/muzero/replay.py`:**
- Track priority for each position
- Sample proportional to priority^alpha
- Use `config.replay.priority_alpha`

### 5.3 Minimax Baseline

Implement a simple minimax opponent:
- Material counting evaluation
- Depth 2-3 search
- No pruning needed for baseline

**Location:** `crates/selfplay/src/minimax.rs` or separate `crates/minimax/`

### 5.4 Evaluation Pipeline

**Add to `training/train.py`:**
```bash
uv run python train.py evaluate --opponent minimax --games 100
```

**Logic:**
1. Load trained ONNX model
2. Play games: MuZero vs Minimax
3. Report win/loss/draw statistics
4. Target: >70% wins for MuZero

### 5.5 Training Loop Integration

Full training loop:
1. Generate games with current model (or random for bootstrap)
2. Train on generated games
3. Export ONNX
4. Evaluate against minimax
5. Repeat until >70% win rate

### 5.6 Validation

```bash
# 1. Generate bootstrap games with rollout evaluator
cargo run -p muzero-selfplay --release -- generate --games 500 --simulations 100 --output data/games/bootstrap

# 2. Train with legal moves masking
cd training && uv run python -m muzero.cli train -s 5000

# 3. Export ONNX
uv run python -m muzero.cli export --verify

# 4. Generate games with trained model
cargo run -p muzero-selfplay --release -- generate --games 100 --simulations 200 --model checkpoints --output data/games/iter1

# 5. Evaluate against minimax
cargo run -p muzero-selfplay --release -- evaluate --model checkpoints --games 100 --depth 3
# Expected: >70% win rate

# 6. Or run the full training loop script
./scripts/train_loop.sh 5
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
cargo test -p muzero-chess perft # Specific test
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
| Policy sums to 1.0 | Mask illegal → softmax (training & inference) |
| Value in [-1, 1] | tanh output |
| Same seed → same game | Inject all RNG |
| All moves legal | Mask at policy output (CRITICAL) |
| No human chess knowledge | Pure self-play |

**Priority: Correctness > Understanding > Speed**

---

## Invariants to Property Test

- FEN round-trip: `parse(fen(pos)) == pos`
- All generated moves are legal (no self-check)
- Policy valid distribution (sums to 1.0 over legal moves)
- Deterministic replay with seed

---

## Project Layout

```
mu-chess/
├── crates/
│   ├── core/      # Game trait, Policy/Value types
│   ├── chess/     # Chess engine (bitboards, perft)
│   ├── mcts/      # Generic MCTS + tic-tac-toe
│   ├── muzero/    # ONNX inference
│   └── selfplay/  # Game generation
├── training/      # PyTorch training pipeline
│   ├── src/muzero/
│   └── train.py
├── data/games/    # MessagePack game files
└── checkpoints/   # ONNX models
```

---

## Start Phase 5

```bash
cd /Users/patrikpersson/Code/sandbox/mu-chess

# 1. Fix legal moves masking in trainer.py
# 2. Generate bootstrap games
# 3. Train with fixed masking
# 4. Implement minimax baseline
# 5. Evaluate: MuZero vs minimax
# 6. Iterate until >70% win rate
```
