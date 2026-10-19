"""
Hamiltonian Cycle Dataset Generator for ILP/LFIO Experiments

This module generates graph datasets for learning Hamiltonian Cycle properties.
The Hamiltonian Cycle problem asks: given a graph, does there exist a cycle
that visits each vertex exactly once?

For ILP experiments, we represent graphs using logical predicates:
- vertex(V): V is a vertex in the graph
- edge(U, V): There is an edge between U and V
- has_cycle: The graph has a Hamiltonian cycle

The learning task: predict whether a graph has a Hamiltonian cycle based on
its structural properties.
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import itertools
import networkx as nx


@dataclass
class Graph:
    """Represents a graph for ILP learning."""
    n_vertices: int
    edges: Set[Tuple[int, int]]  # Undirected edges
    adjacency: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.adjacency is None:
            self.adjacency = np.zeros((self.n_vertices, self.n_vertices), dtype=int)
            for u, v in self.edges:
                self.adjacency[u, v] = 1
                self.adjacency[v, u] = 1

    def get_edges_list(self) -> List[Tuple[int, int]]:
        """Return sorted list of edges."""
        return sorted(list(self.edges))

    def get_degree_sequence(self) -> List[int]:
        """Return degree of each vertex."""
        return [sum(self.adjacency[i, :]) for i in range(self.n_vertices)]

    def is_complete(self) -> bool:
        """Check if graph is complete."""
        return len(self.edges) == self.n_vertices * (self.n_vertices - 1) // 2

    def density(self) -> float:
        """Graph density (0 to 1)."""
        max_edges = self.n_vertices * (self.n_vertices - 1) / 2
        return len(self.edges) / max_edges if max_edges > 0 else 0


class HamiltonianCycleSolver:
    """
    Solver for Hamiltonian Cycle problem.
    Uses backtracking for small graphs.
    """

    def __init__(self, max_vertices: int = 10):
        self.max_vertices = max_vertices

    def has_hamiltonian_cycle(self, graph: Graph) -> bool:
        """Check if graph has a Hamiltonian cycle using backtracking."""
        if graph.n_vertices > self.max_vertices:
            # For larger graphs, use heuristic
            return self._heuristic_check(graph)

        # Backtracking algorithm
        n = graph.n_vertices
        adj = graph.adjacency

        # Try starting from each vertex
        for start in range(n):
            path = [start]
            visited = {start}

            if self._dfs_hamiltonian(graph, path, visited, start):
                return True

        return False

    def _dfs_hamiltonian(self, graph: Graph, path: List[int],
                        visited: Set[int], start: int) -> bool:
        """DFS helper for Hamiltonian cycle finding."""
        n = graph.n_vertices

        # Base case: all vertices visited
        if len(path) == n:
            # Check if there's an edge back to start
            return graph.adjacency[path[-1], start] == 1

        current = path[-1]

        # Try all unvisited neighbors
        for neighbor in range(graph.n_vertices):
            if neighbor not in visited and graph.adjacency[current, neighbor] == 1:
                path.append(neighbor)
                visited.add(neighbor)

                if self._dfs_hamiltonian(graph, path, visited, start):
                    return True

                path.pop()
                visited.remove(neighbor)

        return False

    def _heuristic_check(self, graph: Graph) -> bool:
        """Heuristic check for larger graphs."""
        # Necessary conditions:
        # 1. Minimum degree >= 2
        degrees = graph.get_degree_sequence()
        if min(degrees) < 2:
            return False

        # 2. Graph must be connected
        if not self._is_connected(graph):
            return False

        # Dirac's theorem: if n >= 3 and min_degree >= n/2, then Hamiltonian
        n = graph.n_vertices
        if n >= 3 and min(degrees) >= n / 2:
            return True

        # Ore's theorem: if for all non-adjacent u,v: deg(u) + deg(v) >= n
        degrees = graph.get_degree_sequence()
        for u in range(n):
            for v in range(u + 1, n):
                if graph.adjacency[u, v] == 0:  # Not adjacent
                    if degrees[u] + degrees[v] < n:
                        return False

        return True  # Ore's condition satisfied

    def _is_connected(self, graph: Graph) -> bool:
        """Check if graph is connected using BFS."""
        if graph.n_vertices == 0:
            return True

        visited = set()
        queue = [0]
        visited.add(0)

        while queue:
            current = queue.pop(0)
            for neighbor in range(graph.n_vertices):
                if graph.adjacency[current, neighbor] == 1 and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == graph.n_vertices


class HamiltonianCycleGenerator:
    """
    Generate datasets for Hamiltonian Cycle learning.
    """

    def __init__(self, n_vertices: int = 6):
        self.n_vertices = n_vertices
        self.solver = HamiltonianCycleSolver(max_vertices=n_vertices + 2)

    def generate_random_graph(self, edge_probability: float = 0.5) -> Graph:
        """Generate a random Erdős–Rényi graph."""
        edges = set()

        for u in range(self.n_vertices):
            for v in range(u + 1, self.n_vertices):
                if np.random.random() < edge_probability:
                    edges.add((u, v))

        return Graph(self.n_vertices, edges)

    def generate_complete_graph(self) -> Graph:
        """Generate a complete graph (always has Hamiltonian cycle)."""
        edges = set()
        for u in range(self.n_vertices):
            for v in range(u + 1, self.n_vertices):
                edges.add((u, v))
        return Graph(self.n_vertices, edges)

    def generate_cycle_graph(self) -> Graph:
        """Generate a simple cycle graph."""
        edges = set()
        for i in range(self.n_vertices):
            edges.add((i, (i + 1) % self.n_vertices))
        return Graph(self.n_vertices, edges)

    def generate_star_graph(self) -> Graph:
        """Generate a star graph (no Hamiltonian cycle for n > 3)."""
        edges = set()
        center = 0
        for i in range(1, self.n_vertices):
            edges.add((center, i))
        return Graph(self.n_vertices, edges)

    def generate_path_graph(self) -> Graph:
        """Generate a path graph (no Hamiltonian cycle)."""
        edges = set()
        for i in range(self.n_vertices - 1):
            edges.add((i, i + 1))
        return Graph(self.n_vertices, edges)

    def generate_bipartite_graph(self, p: float = 0.5) -> Graph:
        """Generate random bipartite graph."""
        half = self.n_vertices // 2
        edges = set()

        for u in range(half):
            for v in range(half, self.n_vertices):
                if np.random.random() < p:
                    edges.add((u, v))

        return Graph(self.n_vertices, edges)

    def generate_dataset(self, n_samples: int = 1000,
                        balanced: bool = True,
                        edge_prob_range: Tuple[float, float] = (0.1, 0.9)) -> List[Graph]:
        """
        Generate a dataset of graphs.

        Args:
            n_samples: Number of graphs to generate
            balanced: Balance positive and negative examples
            edge_prob_range: Range of edge probabilities to sample from

        Returns:
            List of graphs with labels
        """
        graphs = []

        if balanced:
            # Generate equal positive and negative examples
            for _ in range(n_samples // 2):
                # Try to generate positive example
                edge_prob = np.random.uniform(0.5, 0.9)  # Higher density -> more likely Hamiltonian
                graph = self.generate_random_graph(edge_prob)
                if self.solver.has_hamiltonian_cycle(graph):
                    graphs.append((graph, 1))
                else:
                    # Try again with complete graph
                    graphs.append((self.generate_complete_graph(), 1))

            for _ in range(n_samples // 2):
                # Try to generate negative example
                edge_prob = np.random.uniform(0.1, 0.5)  # Lower density
                graph = self.generate_random_graph(edge_prob)
                if not self.solver.has_hamiltonian_cycle(graph):
                    graphs.append((graph, 0))
                else:
                    # Use path graph (definitely no cycle)
                    graphs.append((self.generate_path_graph(), 0))
        else:
            # Random generation
            for _ in range(n_samples):
                edge_prob = np.random.uniform(*edge_prob_range)
                graph = self.generate_random_graph(edge_prob)
                label = 1 if self.solver.has_hamiltonian_cycle(graph) else 0
                graphs.append((graph, label))

        return graphs


class GraphFeatureExtractor:
    """
    Extract logical features from graphs for ILP learning.
    """

    @staticmethod
    def extract_boolean_features(graph: Graph) -> np.ndarray:
        """
        Extract Boolean features from a graph.

        Features include:
        - Basic properties: degree conditions, connectivity
        - Local structure: triangles, paths
        - Global properties: density, completeness
        """
        features = []
        n = graph.n_vertices
        adj = graph.adjacency
        degrees = graph.get_degree_sequence()

        # Degree-related features
        features.append(min(degrees) >= 2)  # All vertices have degree >= 2
        features.append(max(degrees) >= n - 1)  # Has a vertex connected to all others
        features.append(sum(degrees) >= 2 * n)  # Total degree threshold

        # Density-based features
        density = graph.density()
        features.append(density >= 0.5)  # Dense graph
        features.append(density >= 0.7)  # Very dense graph
        features.append(graph.is_complete())  # Complete graph

        # Connectivity
        # Check if graph is connected via BFS
        if n > 0:
            visited = {0}
            queue = [0]
            while queue:
                current = queue.pop(0)
                for neighbor in range(n):
                    if adj[current, neighbor] == 1 and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            features.append(len(visited) == n)  # Is connected
        else:
            features.append(True)

        # Local structure: Check for specific patterns
        # Has triangle
        has_triangle = False
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 1:
                    for k in range(j + 1, n):
                        if adj[i, k] == 1 and adj[j, k] == 1:
                            has_triangle = True
                            break
                if has_triangle:
                    break
            if has_triangle:
                break
        features.append(has_triangle)

        # Has vertex of degree 2 (necessary but not sufficient)
        features.append(any(d == 2 for d in degrees))

        # Minimum degree condition
        features.append(min(degrees) >= n // 2)  # Dirac's condition

        # Ore's condition approximation: sum of degrees of non-adjacent pairs
        satisfies_ore = True
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0 and degrees[i] + degrees[j] < n:
                    satisfies_ore = False
                    break
            if not satisfies_ore:
                break
        features.append(satisfies_ore)

        # Path-based features
        has_path_of_length = [False] * (n + 1)
        for length in range(2, min(n + 1, 5)):  # Check paths up to length 4
            for start in range(n):
                if GraphFeatureExtractor._has_path_of_length(adj, start, length):
                    has_path_of_length[length] = True
                    break
        features.extend(has_path_of_length[2:5])  # Add path existence features

        # Regularity check (all degrees equal)
        features.append(len(set(degrees)) == 1)

        # Bipartite approximation
        # Graph is bipartite if no odd cycles
        is_bipartite = True
        if n > 0:
            color = {0: 0}
            queue = [0]
            while queue:
                current = queue.pop(0)
                for neighbor in range(n):
                    if adj[current, neighbor] == 1:
                        if neighbor not in color:
                            color[neighbor] = 1 - color[current]
                            queue.append(neighbor)
                        elif color[neighbor] == color[current]:
                            is_bipartite = False
                            break
        features.append(not is_bipartite)  # Has odd cycle

        return np.array(features, dtype=int)

    @staticmethod
    def _has_path_of_length(adj: np.ndarray, start: int, length: int) -> bool:
        """Check if there's a simple path of given length from start."""
        n = adj.shape[0]

        def dfs(current, visited, path_len):
            if path_len == length:
                return True
            for neighbor in range(n):
                if adj[current, neighbor] == 1 and neighbor not in visited:
                    visited.add(neighbor)
                    if dfs(neighbor, visited, path_len + 1):
                        return True
                    visited.remove(neighbor)
            return False

        return dfs(start, {start}, 0)

    @staticmethod
    def get_feature_names(n_vertices: int) -> List[str]:
        """Get names of extracted features."""
        return [
            "min_degree_ge_2",
            "has_hub_vertex",
            "total_degree_ge_2n",
            "dense_graph",
            "very_dense_graph",
            "is_complete",
            "is_connected",
            "has_triangle",
            "has_degree_2_vertex",
            "dirac_condition",
            "ore_condition",
            "has_path_2",
            "has_path_3",
            "has_path_4",
            "is_regular",
            "has_odd_cycle"
        ]

    @staticmethod
    def extract_literal_features(graph: Graph) -> List[Tuple[str, tuple]]:
        """
        Extract graph as ILP literals.

        Returns a list of literals like:
        - vertex(v)
        - edge(u, v)
        - degree(v, d)
        """
        literals = []
        n = graph.n_vertices
        degrees = graph.get_degree_sequence()

        # Vertex literals
        for v in range(n):
            literals.append(("vertex", (v,)))

        # Edge literals
        for u, v in graph.edges:
            literals.append(("edge", (u, v)))

        # Degree literals
        for v, d in enumerate(degrees):
            literals.append(("degree", (v, d)))

        return literals


class HamiltonianCycleDataset:
    """
    Complete dataset for Hamiltonian Cycle learning.
    """

    def __init__(self, n_vertices: int = 6, n_samples: int = 1000):
        self.n_vertices = n_vertices
        self.n_samples = n_samples
        self.generator = HamiltonianCycleGenerator(n_vertices)
        self.extractor = GraphFeatureExtractor()

        self.graphs: List[Tuple[Graph, int]] = []
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    def generate(self, balanced: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate the complete dataset.

        Returns:
            X: Feature matrix
            y: Labels (1 = has Hamiltonian cycle, 0 = does not)
            feature_names: Names of features
        """
        print(f"Generating {self.n_samples} graphs with {self.n_vertices} vertices...")

        self.graphs = self.generator.generate_dataset(
            n_samples=self.n_samples,
            balanced=balanced
        )

        # Extract features
        self.feature_names = self.extractor.get_feature_names(self.n_vertices)
        X_list = []
        y_list = []

        for graph, label in self.graphs:
            features = self.extractor.extract_boolean_features(graph)
            X_list.append(features)
            y_list.append(label)

        self.X = np.array(X_list)
        self.y = np.array(y_list)

        print(f"Generated {len(self.graphs)} graphs")
        print(f"Positive examples: {np.sum(self.y)}")
        print(f"Negative examples: {len(self.y) - np.sum(self.y)}")
        print(f"Feature dimensions: {self.X.shape[1]}")

        return self.X, self.y, self.feature_names

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if self.X is None:
            return {}

        pos_mask = self.y == 1
        neg_mask = self.y == 0

        stats = {
            "total_samples": len(self.y),
            "positive_samples": np.sum(pos_mask),
            "negative_samples": np.sum(neg_mask),
            "feature_dim": self.X.shape[1],
            "positive_feature_means": self.X[pos_mask].mean(axis=0),
            "negative_feature_means": self.X[neg_mask].mean(axis=0),
        }

        return stats

    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()

        print("\n=== Hamiltonian Cycle Dataset Statistics ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Positive (has cycle): {stats['positive_samples']}")
        print(f"Negative (no cycle): {stats['negative_samples']}")
        print(f"Feature dimensions: {stats['feature_dim']}")

        print("\nFeature means by class:")
        print(f"{'Feature':<25} {'Positive':>10} {'Negative':>10} {'Diff':>10}")
        print("-" * 60)

        for i, name in enumerate(self.feature_names):
            pos_mean = stats['positive_feature_means'][i]
            neg_mean = stats['negative_feature_means'][i]
            diff = pos_mean - neg_mean
            print(f"{name:<25} {pos_mean:>10.3f} {neg_mean:>10.3f} {diff:>10.3f}")

    def to_prolog(self, graph_idx: int) -> str:
        """
        Convert a graph to Prolog facts.

        Example output:
        vertex(0). vertex(1). ...
        edge(0, 1). edge(1, 2). ...
        has_cycle.  (if the graph has Hamiltonian cycle)
        """
        if graph_idx >= len(self.graphs):
            return ""

        graph, label = self.graphs[graph_idx]
        prolog = []

        # Vertices
        for v in range(graph.n_vertices):
            prolog.append(f"vertex({v}).")

        # Edges
        for u, v in sorted(graph.edges):
            prolog.append(f"edge({u}, {v}).")

        # Label
        if label == 1:
            prolog.append("has_cycle.")

        return "\n".join(prolog)

    def save_to_file(self, filename: str):
        """Save dataset to file."""
        if self.X is None:
            self.generate()

        np.savez(filename, X=self.X, y=self.y)
        print(f"Dataset saved to {filename}")

    def load_from_file(self, filename: str):
        """Load dataset from file."""
        data = np.load(filename)
        self.X = data['X']
        self.y = data['y']
        print(f"Dataset loaded from {filename}")


# ============================================================
# Specialized Graph Generators for Specific Properties
# ============================================================

class SpecializedGraphGenerator:
    """
    Generate graphs with specific properties for testing LFIO.
    """

    @staticmethod
    def generate_dirac_graphs(n: int) -> List[Graph]:
        """Generate graphs satisfying Dirac's condition (min_degree >= n/2)."""
        graphs = []
        # Dirac: n >= 3 and min_degree >= n/2 guarantees Hamiltonian cycle
        for _ in range(50):
            edges = set()
            min_degree = int(np.ceil(n / 2))

            # Start with a cycle to ensure Hamiltonian
            for i in range(n):
                edges.add((i, (i + 1) % n))

            # Add random edges to satisfy min degree condition
            for v in range(n):
                current_degree = sum(1 for u, w in edges if u == v or w == v)
                while current_degree < min_degree:
                    other = np.random.randint(0, n)
                    if other != v and (v, other) not in edges and (other, v) not in edges:
                        edges.add((min(v, other), max(v, other)))
                        current_degree += 1

            graphs.append(Graph(n, edges))

        return graphs

    @staticmethod
    def generate_non_hamiltonian_graphs(n: int) -> List[Graph]:
        """Generate graphs that definitely don't have Hamiltonian cycles."""
        graphs = []

        # Star graphs (no cycle for n > 3)
        if n > 3:
            edges = set()
            for i in range(1, n):
                edges.add((0, i))
            graphs.append(Graph(n, edges))

        # Path graphs
        edges = set()
        for i in range(n - 1):
            edges.add((i, i + 1))
        graphs.append(Graph(n, edges))

        # Disconnected graph
        half = n // 2
        edges = set()
        for i in range(half - 1):
            edges.add((i, i + 1))
        for i in range(half, n - 1):
            edges.add((i, i + 1))
        graphs.append(Graph(n, edges))

        # Graph with vertex of degree 1
        edges = set()
        edges.add((0, 1))  # Vertex 1 has degree 1
        for i in range(2, n):
            next_v = (i + 1) if i < n - 1 else 2
            edges.add((i, next_v))
        graphs.append(Graph(n, edges))

        # Sparse bipartite graph (unlikely to have Hamiltonian cycle)
        edges = set()
        for i in range(half):
            edges.add((i, half))  # Connect left side to one vertex on right
        for i in range(half + 1, n):
            edges.add((half, i))
        graphs.append(Graph(n, edges))

        return graphs


# ============================================================
# Main Functions and Tests
# ============================================================

def create_hamiltonian_cycle_dataset(n_vertices: int = 6,
                                    n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create a Hamiltonian Cycle dataset.

    Args:
        n_vertices: Number of vertices in graphs (6-8 recommended)
        n_samples: Total number of samples

    Returns:
        X: Feature matrix
        y: Labels
        feature_names: Names of features
    """
    dataset = HamiltonianCycleDataset(n_vertices=n_vertices, n_samples=n_samples)
    return dataset.generate(balanced=True)


if __name__ == "__main__":
    print("=== Hamiltonian Cycle Dataset Generator ===\n")

    # Test with small graphs
    print("Testing with 6-vertex graphs...\n")
    dataset = HamiltonianCycleDataset(n_vertices=6, n_samples=500)
    X, y, feature_names = dataset.generate()

    dataset.print_statistics()

    # Show some example graphs
    print("\n=== Example Graphs ===")
    for i in range(min(5, len(dataset.graphs))):
        graph, label = dataset.graphs[i]
        print(f"\nGraph {i+1} (Label: {'Has Cycle' if label else 'No Cycle'}):")
        print(f"  Vertices: {graph.n_vertices}")
        print(f"  Edges: {sorted(graph.edges)}")
        print(f"  Degrees: {graph.get_degree_sequence()}")
        print(f"  Density: {graph.density():.2f}")

    # Test with specialized generators
    print("\n=== Specialized Graphs ===")

    print("\nDirac Graphs (guaranteed Hamiltonian):")
    dirac_graphs = SpecializedGraphGenerator.generate_dirac_graphs(6)
    for g in dirac_graphs[:3]:
        print(f"  Edges: {len(g.edges)}, Density: {g.density():.2f}")

    print("\nNon-Hamiltonian Graphs:")
    non_ham_graphs = SpecializedGraphGenerator.generate_non_hamiltonian_graphs(6)
    for g in non_ham_graphs:
        print(f"  Type: {len(g.edges)} edges, Density: {g.density():.2f}")

    # Test LFIO on this dataset
    print("\n=== Testing LFIO on Hamiltonian Cycle Data ===")

    import sys
    sys.path.insert(0, '.')
    from lfio_learner import LFIOBoolean

    lfio = LFIOBoolean(min_coverage=10, min_precision=0.8)
    lfio.fit(X, y, feature_names)
    lfio.print_rules()

    # Evaluate
    predictions = lfio.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nLFIO Accuracy: {accuracy:.3f}")
