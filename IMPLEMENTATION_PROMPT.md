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

| Phase | Deliverable | Validation |
|-------|-------------|------------|
| 1 | Rust workspace + chess engine | Perft depth 5 = 4,865,609 |
| 2 | Generic MCTS + tic-tac-toe | MCTS plays TTT perfectly |
| 3 | Python training pipeline | Loss decreases on random data |
| 4 | ONNX export + Rust inference | Full loop works |
| 5 | Training + evaluation | Beats minimax >70% |

**Do NOT proceed until current phase validates.**

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

**Useful tools:**
```bash
cargo install cargo-watch cargo-nextest
cargo watch -x 'test -p chess'   # Auto-test on save
```

---

## Python Workflow

**Use `uv` for deps, `uvx` for tools (don't install ruff/mypy as deps).**

```bash
# Setup
cd python && uv init --python 3.11
uv add torch numpy msgpack onnx onnxruntime
uv add --dev pytest

# Run code
uv run train.py
uv run -m muzero.trainer
```

**Dev loop:**
```bash
uvx ruff format .        # Format
uvx ruff check --fix .   # Lint + autofix
uvx mypy muzero/         # Type check
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
}

#[derive(Serialize, Deserialize)]
struct GameStep {
    observation: Vec<f32>,
    action: u16,
    policy: Vec<f32>,
    value: f32,
}
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

## Start

```bash
cd /Users/patrikpersson/Code/sandbox/chess
# Create workspace, implement Game trait, then chess with bitboards
# Validate with perft before proceeding to MCTS
```
