# SPEC: MuZero Chess

**Version:** 1.0
**Status:** Approved
**Last Updated:** 2026-01-05

---

## 1. PURPOSE

A learning project to achieve full mastery of MuZero:
- Ability to implement game AI for new domains
- Deep understanding of WHY each component works
- Practical debugging skills for RL systems

Chess serves as the concrete domain with minimal human priors—everything learned from self-play.

---

## 2. SCOPE

### IN SCOPE
- Standard chess (FIDE rules)
- MuZero algorithm with learned dynamics
- Single-machine training (M3 dev, GPU VM scaling)
- Self-play training loop with parallel game generation
- Configurable hyperparameters for experimentation
- Game-agnostic abstractions via Rust traits (future games)

### OUT OF SCOPE
- Distributed training / multi-GPU clusters (future consideration)
- Chess variants (Chess960, Crazyhouse)
- Superhuman play / beating Stockfish
- Production deployment (UCI protocol, web APIs)
- Opening books or endgame tablebases

---

## 3. ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────┐
│                            RUST                                      │
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │ Game Trait   │    │    MCTS      │    │  Inference (ort)   │    │
│  │ impl: Chess  │───▶│   Planner    │◀──▶│  ONNX Runtime      │    │
│  │ (bitboards)  │    │   (Rayon)    │    │  (MPS / CUDA)      │    │
│  └──────────────┘    └──────────────┘    └────────────────────┘    │
│         │                   │                      ▲                │
│         ▼                   ▼                      │                │
│  ┌────────────────────────────────────┐   ┌───────────────────┐    │
│  │   Self-Play Engine (parallel)      │   │ model.onnx        │    │
│  │   → games/*.msgpack                │   │ (checkpoint)      │    │
│  └────────────────────────────────────┘   └───────────────────┘    │
│                                                    ▲                │
└────────────────────────────────────────────────────│────────────────┘
                           file I/O (MessagePack)    │
┌────────────────────────────────────────────────────│────────────────┐
│                      PYTHON (uv)                   │                │
│                                                    │                │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  PyTorch Training (MPS / CUDA)                             │    │
│  │  - ReplayBuffer: load games from disk                      │    │
│  │  - Train: h(), g(), f() networks                           │    │
│  │  - Export: → ONNX                                          │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. KEY DECISIONS

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Board representation | Bitboards | Performance |
| Legal move handling | Mask at output | Standard MuZero; guaranteed legality |
| Rust ↔ Python | File-based (MessagePack) | Simplicity, debuggability |
| Model format | ONNX via `ort` | GPU support, cross-platform |
| Concurrency | Rayon | Simple parallelism for self-play |
| Config format | TOML + env overrides | Rust-native, type-safe |
| Python tooling | uv | Modern, fast |
| Error handling | thiserror + anyhow | Layered, idiomatic |
| Development | M3 local + GPU VM scaling | Flexible |

---

## 5. DOMAIN TYPES

### 5.1 Game Abstraction (Rust)

```rust
pub trait Game: Clone + Send + Sync {
    type State: Clone + Send;
    type Action: Clone + Copy + Send + Eq + Hash;
    type Observation;

    fn initial_state(&self) -> Self::State;
    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action>;
    fn apply(&self, state: &Self::State, action: Self::Action) -> Self::State;
    fn is_terminal(&self, state: &Self::State) -> bool;
    fn outcome(&self, state: &Self::State) -> Option<f32>;
    fn observe(&self, state: &Self::State) -> Self::Observation;
    fn action_to_index(&self, action: Self::Action) -> usize;
    fn index_to_action(&self, index: usize) -> Option<Self::Action>;
    fn num_actions(&self) -> usize;
}
```

### 5.2 Chess Types

```rust
pub struct Square(u8);           // 0-63, validated
pub struct Move(u16);            // from(6) | to(6) | flags(4)
pub struct Position { ... }      // Bitboards + game state

pub enum GameResult {
    WhiteWins,
    BlackWins,
    Draw(DrawReason),
}
```

### 5.3 MuZero Types

```rust
pub struct Policy(Vec<f32>);     // Sum to 1.0 (enforced)
pub struct Value(f32);           // In [-1, 1] (enforced)
pub struct HiddenState(Tensor);  // Configurable dimension
```

---

## 6. CONFIGURATION

```toml
# config.toml

[network]
hidden_dim = 256
num_res_blocks = 16

[mcts]
num_simulations = 800
dirichlet_alpha = 0.3
exploration_fraction = 0.25
pb_c_base = 19652
pb_c_init = 1.25

[training]
batch_size = 1024
learning_rate = 0.001
weight_decay = 0.0001
unroll_steps = 5
td_steps = 10
discount = 1.0

[selfplay]
games_per_iteration = 100
num_workers = 4
temperature = 1.0
temperature_drop_move = 30

[replay]
buffer_size = 100000
priority_alpha = 1.0

[paths]
data_dir = "data/games"
checkpoint_dir = "checkpoints"
config_file = "config.toml"
```

**Override via environment:** `MUZERO_MCTS_NUM_SIMULATIONS=100`

---

## 7. INVARIANTS

| ID | Property | Verification |
|----|----------|--------------|
| INV-1 | Policy sums to 1.0 (±1e-5) | Property test + type enforcement |
| INV-2 | Value in [-1, 1] | tanh + type wrapper |
| INV-3 | Same seed → identical game | Deterministic simulation test |
| INV-4 | `parse(to_fen(pos)) == pos` | Property test |
| INV-5 | All generated moves are legal | Property test |
| INV-6 | Exactly one king per side | Property test |

---

## 8. CONSTRAINTS

### PRIORITY ORDER
1. **Correctness** — Chess rules, MCTS logic, training math
2. **Understanding** — Clear code over clever optimization
3. **Performance** — Speed matters but not at expense of above

### POSITIVE (MUST)

| ID | Constraint |
|----|------------|
| P1 | Chess engine MUST pass perft tests |
| P2 | MCTS MUST solve tic-tac-toe perfectly |
| P3 | Illegal moves MUST be impossible (masking) |
| P4 | All randomness MUST be seedable |
| P5 | Config MUST be serializable |

### NEGATIVE (MUST NOT)

| ID | Constraint |
|----|------------|
| N1 | MUST NOT use human chess knowledge |
| N2 | MUST NOT have hidden global state |
| N3 | MUST NOT panic on valid inputs |

---

## 9. SUCCESS CRITERIA

**Minimum Viable:** Trained model beats material-counting minimax (depth 2-3) in >70% of games.

---

## 10. VERIFICATION

### 10.1 Unit Tests (Given/When/Then)

| ID | Scenario | Given | When | Then |
|----|----------|-------|------|------|
| U1 | Starting moves | Starting position | Generate moves | 20 legal moves |
| U2 | Perft depth 5 | Starting position | Count nodes | 4,865,609 |
| U3 | MCTS tic-tac-toe | Empty board | 1000 simulations | Optimal play |

### 10.2 Property Tests

| ID | Property | Assertion |
|----|----------|-----------|
| P1 | FEN round-trip | `parse(fen(pos)) == pos` |
| P2 | Legal moves | No move leaves king in check |
| P3 | Policy valid | `sum(policy) ≈ 1.0` |
| P4 | Determinism | Same seed → same game |

### 10.3 Manual Verification

| ID | Check | Expected |
|----|-------|----------|
| M1 | Play against model | Legal, sensible moves |
| M2 | Watch self-play | Chess-like development |
| M3 | Known positions | High policy on winning moves |
| M4 | Training curves | Loss decreasing |

---

## 11. PROJECT STRUCTURE

```
muzero-chess/
├── Cargo.toml                 # Workspace
├── config.toml                # Default configuration
├── crates/
│   ├── core/                  # Game trait + utilities
│   ├── chess/                 # Chess implementation
│   │   ├── src/
│   │   │   ├── bitboard.rs
│   │   │   ├── movegen.rs
│   │   │   ├── position.rs
│   │   │   └── lib.rs
│   │   └── tests/perft.rs
│   ├── mcts/                  # Generic MCTS
│   ├── muzero/                # Inference (ort)
│   └── selfplay/              # Binary: parallel game gen
├── python/
│   ├── pyproject.toml         # uv project
│   ├── muzero/
│   │   ├── networks.py        # PyTorch models
│   │   ├── trainer.py
│   │   ├── replay.py
│   │   └── export.py          # → ONNX
│   └── train.py               # Entry point
├── data/games/                # MessagePack game files
└── checkpoints/               # ONNX models
```

---

## 12. ERROR HANDLING

```rust
#[derive(Error, Debug)]
pub enum MuZeroError {
    #[error("Invalid FEN: {0}")]
    InvalidFen(String),

    #[error("ONNX inference failed: {0}")]
    Inference(#[from] ort::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Deserialization error: {0}")]
    Deserialize(#[from] rmp_serde::decode::Error),
}
```

**Policy:** Result for recoverable errors, panic for invariant violations (bugs).

---

## 13. HOST ENVIRONMENT

| Environment | Use | Backend |
|-------------|-----|---------|
| MacBook M3 | Development, small experiments | MPS (Metal) |
| GPU VM | Full training runs | CUDA |

**Requirements:**
- Rust 1.75+
- Python 3.11+ (via uv)
- ONNX Runtime with appropriate backend

---

## 14. FUTURE CONSIDERATIONS

### 14.1 Performance Optimizations

| Optimization | When to Consider | Complexity |
|--------------|------------------|------------|
| Batched inference | GPU underutilized | Medium |
| Shared memory (Rust↔Python) | File I/O bottleneck | High |
| CUDA graphs | Inference latency | Medium |
| Quantization (INT8) | Model too large | Low |

### 14.2 Algorithm Improvements

| Improvement | Description |
|-------------|-------------|
| Gumbel MuZero | Better exploration, fewer simulations needed |
| Sampled MuZero | For larger action spaces |
| Stochastic MuZero | Handle stochastic games |
| EfficientZero | Sample efficiency improvements |

### 14.3 Scaling

| Scale | Approach |
|-------|----------|
| Multi-GPU | DataParallel in PyTorch |
| Distributed self-play | Redis/RabbitMQ for game queue |
| Cluster training | Ray or Horovod |

### 14.4 Additional Games

| Game | Notes |
|------|-------|
| Tic-tac-toe | Already planned for MCTS testing |
| Connect 4 | Simple, good for validation |
| Go (9x9) | Test scalability |
| Atari | Tests MuZero on non-board games |

### 14.5 Tooling

| Tool | Purpose |
|------|---------|
| TensorBoard | Training visualization |
| Weights & Biases | Experiment tracking |
| Lichess API | External evaluation |

---

## 15. APPENDIX: MUZERO ALGORITHM REFERENCE

### What MuZero Learns

| Component | Function | Input → Output |
|-----------|----------|----------------|
| Representation h() | Encode observation | observation → hidden state |
| Dynamics g() | Predict transition | (hidden, action) → (next hidden, reward) |
| Prediction f() | Evaluate state | hidden → (policy, value) |

### Training Loss

```
L = Σₜ [ L_policy(πₜ, pₜ) + L_value(zₜ, vₜ) + L_reward(uₜ, rₜ) ] + c‖θ‖²

Where:
- πₜ = MCTS policy (visit counts normalized)
- pₜ = predicted policy
- zₜ = actual game outcome (bootstrapped with n-step returns)
- vₜ = predicted value
- uₜ = actual reward
- rₜ = predicted reward
```

### MCTS with Learned Model

```
1. At root: hidden = h(observation)
2. For each simulation:
   a. Select: traverse tree using PUCT until leaf
   b. Expand: (policy, value) = f(hidden)
   c. If not at max depth:
      (next_hidden, reward) = g(hidden, action)
   d. Backpropagate value
3. Return: visit count distribution as policy
```

---

## CHANGELOG

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-05 | Initial approved specification |
