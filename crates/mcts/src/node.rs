//! MCTS node types for tree storage.
//!
//! Uses arena allocation with indices for cache locality and simpler memory management.

use std::hash::Hash;

/// Index into the node arena.
///
/// This is a lightweight handle that references a node in the tree.
/// Using indices instead of pointers avoids Rc/RefCell overhead.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeId(pub(crate) usize);

impl NodeId {
    /// The root node is always at index 0.
    pub const ROOT: NodeId = NodeId(0);
}

/// Statistics for a single MCTS node.
#[derive(Clone, Debug)]
pub struct NodeStats {
    /// Number of times this node was visited during search.
    pub visit_count: u32,

    /// Sum of values from all visits (for computing Q-value).
    pub value_sum: f32,

    /// Prior probability from the policy network.
    pub prior: f32,
}

impl NodeStats {
    /// Create new stats with the given prior probability.
    pub fn new(prior: f32) -> Self {
        Self {
            visit_count: 0,
            value_sum: 0.0,
            prior,
        }
    }

    /// Mean value (Q-value) for this node.
    ///
    /// Returns 0.0 if the node has never been visited.
    pub fn mean_value(&self) -> f32 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f32
        }
    }
}

/// A node in the MCTS tree.
///
/// Each node represents a game state and stores statistics about
/// the search results from that state.
#[derive(Clone, Debug)]
pub struct Node<A: Clone + Copy + Eq + Hash> {
    /// Action that led to this node (None for root).
    /// Kept for debugging and potential future use (path reconstruction).
    #[allow(dead_code)]
    pub action: Option<A>,

    /// Node statistics (visits, value, prior).
    pub stats: NodeStats,

    /// Children: (action, node_id) pairs.
    pub children: Vec<(A, NodeId)>,

    /// Whether this node has been expanded (children generated).
    pub expanded: bool,

    /// Whether this node represents a terminal game state.
    pub terminal: bool,

    /// Terminal value if this is a terminal state.
    pub terminal_value: Option<f32>,
}

impl<A: Clone + Copy + Eq + Hash> Node<A> {
    /// Create a new unexpanded node.
    pub fn new(action: Option<A>, prior: f32) -> Self {
        Self {
            action,
            stats: NodeStats::new(prior),
            children: Vec::new(),
            expanded: false,
            terminal: false,
            terminal_value: None,
        }
    }

    /// Create the root node.
    pub fn root() -> Self {
        Self::new(None, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_stats_mean_value() {
        let mut stats = NodeStats::new(0.5);

        // Unvisited node has Q = 0
        assert_eq!(stats.mean_value(), 0.0);

        // After visits
        stats.visit_count = 2;
        stats.value_sum = 1.5;
        assert!((stats.mean_value() - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_node_creation() {
        let node: Node<u8> = Node::new(Some(42), 0.3);
        assert_eq!(node.action, Some(42));
        assert!((node.stats.prior - 0.3).abs() < 1e-5);
        assert!(!node.expanded);
        assert!(!node.terminal);
    }

    #[test]
    fn test_root_node() {
        let root: Node<u8> = Node::root();
        assert_eq!(root.action, None);
        assert_eq!(root.stats.prior, 1.0);
    }
}
