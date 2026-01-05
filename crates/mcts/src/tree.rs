//! Arena-allocated MCTS tree.
//!
//! Using a Vec<Node> with indices provides better cache locality
//! and simpler ownership compared to Rc<RefCell<Node>>.

use crate::node::{Node, NodeId};
use std::hash::Hash;

/// Arena-allocated MCTS tree.
///
/// Nodes are stored in a contiguous vector and referenced by index.
/// This provides:
/// - Better cache locality
/// - No runtime borrow checking overhead
/// - Simple tree clearing/reuse
#[derive(Debug)]
pub struct Tree<A: Clone + Copy + Eq + Hash> {
    nodes: Vec<Node<A>>,
}

impl<A: Clone + Copy + Eq + Hash> Tree<A> {
    /// Create a new tree with an empty root node.
    pub fn new() -> Self {
        let root = Node::root();
        Self { nodes: vec![root] }
    }

    /// Get a reference to a node by ID.
    ///
    /// # Panics
    /// Panics if the NodeId is invalid.
    pub fn get(&self, id: NodeId) -> &Node<A> {
        &self.nodes[id.0]
    }

    /// Get a mutable reference to a node by ID.
    ///
    /// # Panics
    /// Panics if the NodeId is invalid.
    pub fn get_mut(&mut self, id: NodeId) -> &mut Node<A> {
        &mut self.nodes[id.0]
    }

    /// Add a new node to the tree, returning its ID.
    pub fn add(&mut self, node: Node<A>) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Clear the tree for reuse, keeping only a fresh root.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.push(Node::root());
    }

    /// Get the number of nodes in the tree.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tree is empty (should never be true as root always exists).
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get the root node.
    #[allow(dead_code)]
    pub fn root(&self) -> &Node<A> {
        self.get(NodeId::ROOT)
    }

    /// Get a mutable reference to the root node.
    #[allow(dead_code)]
    pub fn root_mut(&mut self) -> &mut Node<A> {
        self.get_mut(NodeId::ROOT)
    }
}

impl<A: Clone + Copy + Eq + Hash> Default for Tree<A> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_creation() {
        let tree: Tree<u8> = Tree::new();
        assert_eq!(tree.len(), 1); // Root node
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_tree_add_node() {
        let mut tree: Tree<u8> = Tree::new();
        let node = Node::new(Some(42), 0.5);
        let id = tree.add(node);

        assert_eq!(id.0, 1); // After root
        assert_eq!(tree.get(id).action, Some(42));
    }

    #[test]
    fn test_tree_clear() {
        let mut tree: Tree<u8> = Tree::new();

        // Add some nodes
        tree.add(Node::new(Some(1), 0.5));
        tree.add(Node::new(Some(2), 0.3));
        assert_eq!(tree.len(), 3);

        // Clear
        tree.clear();
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.root().action, None);
        assert!(!tree.root().expanded);
    }

    #[test]
    fn test_tree_modification() {
        let mut tree: Tree<u8> = Tree::new();

        // Modify root
        tree.root_mut().expanded = true;
        tree.root_mut().stats.visit_count = 10;

        assert!(tree.root().expanded);
        assert_eq!(tree.root().stats.visit_count, 10);
    }
}
