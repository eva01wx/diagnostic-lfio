"""
N-Queens Dataset Generator for ILP/LFIO Experiments

The N-Queens problem: place N queens on an N×N chessboard so that no two queens
threaten each other. Queens attack along rows, columns, and diagonals.

For ILP learning, we generate board configurations and extract logical features
that characterize whether the configuration is a valid solution.

Features include:
- Row/column conflicts
- Diagonal conflicts (both main and anti-diagonals)
- Queen count and distribution
- Conflict counts and patterns
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
import itertools


@dataclass
class Chessboard:
    """Represents an N×N chessboard with queens placed."""
    n: int  # Board size (N-Queens)
    positions: List[Tuple[int, int]]  # Queen positions as (row, col)

    def __post_init__(self):
        # Create board representation
        self.board = np.zeros((self.n, self.n), dtype=int)
        for row, col in self.positions:
            if 0 <= row < self.n and 0 <= col < self.n:
                self.board[row, col] = 1

    def is_valid_solution(self) -> bool:
        """Check if this is a valid N-Queens solution (no conflicts)."""
        return (len(self.positions) == self.n
                and all(0 <= r < self.n and 0 <= c < self.n for r, c in self.positions)
                and self.count_conflicts() == 0)

    def count_conflicts(self) -> int:
        """Count total number of queen pairs attacking each other."""
        conflicts = 0
        for i, (r1, c1) in enumerate(self.positions):
            for j, (r2, c2) in enumerate(self.positions):
                if i < j:
                    if self.queens_attack(r1, c1, r2, c2):
                        conflicts += 1
        return conflicts

    @staticmethod
    def queens_attack(r1: int, c1: int, r2: int, c2: int) -> bool:
        """Check if queens at (r1,c1) and (r2,c2) attack each other."""
        # Same row
        if r1 == r2:
            return True
        # Same column
        if c1 == c2:
            return True
        # Same diagonal
        if abs(r1 - r2) == abs(c1 - c2):
            return True
        return False

    def get_row_counts(self) -> List[int]:
        """Number of queens in each row."""
        return [int(np.sum(self.board[i, :])) for i in range(self.n)]

    def get_col_counts(self) -> List[int]:
        """Number of queens in each column."""
        return [int(np.sum(self.board[:, i])) for i in range(self.n)]

    def get_queen_count(self) -> int:
        """Total number of queens on board."""
        return len(self.positions)

    def has_multiple_queens_row(self) -> bool:
        """Check if any row has multiple queens."""
        return any(c > 1 for c in self.get_row_counts())

    def has_multiple_queens_col(self) -> bool:
        """Check if any column has multiple queens."""
        return any(c > 1 for c in self.get_col_counts())


class NQueensSolver:
    """Solver for N-Queens problem using backtracking."""

    def __init__(self, n: int):
        self.n = n
        self.solutions = []

    def find_all_solutions(self) -> List[Chessboard]:
        """Find all solutions to N-Queens problem."""
        self.solutions = []
        self._solve(0, [])
        return [Chessboard(self.n, sol) for sol in self.solutions]

    def _solve(self, row: int, positions: List[Tuple[int, int]]):
        """Backtracking solver."""
        if row == self.n:
            self.solutions.append(positions.copy())
            return

        for col in range(self.n):
            # Check if placing queen at (row, col) is safe
            if self._is_safe(positions, row, col):
                positions.append((row, col))
                self._solve(row + 1, positions)
                positions.pop()

    def _is_safe(self, positions: List[Tuple[int, int]], row: int, col: int) -> bool:
        """Check if position is safe given existing queens."""
        for r, c in positions:
            if c == col or abs(r - row) == abs(c - col):
                return False
        return True

    def count_solutions(self) -> int:
        """Count total solutions (without storing them all)."""
        return self._count_solutions_recursive(0, set(), set(), set())

    def _count_solutions_recursive(self, row: int, cols: Set[int],
                                   diag1: Set[int], diag2: Set[int]) -> int:
        """Count solutions using symmetry."""
        if row == self.n:
            return 1

        count = 0
        for col in range(self.n):
            d1 = row - col  # Main diagonal
            d2 = row + col  # Anti-diagonal

            if col not in cols and d1 not in diag1 and d2 not in diag2:
                cols.add(col)
                diag1.add(d1)
                diag2.add(d2)

                count += self._count_solutions_recursive(row + 1, cols, diag1, diag2)

                cols.remove(col)
                diag1.remove(d1)
                diag2.remove(d2)

        return count


class NQueensGenerator:
    """Generate N-Queens datasets for ILP learning."""

    def __init__(self, n: int = 8):
        self.n = n
        self.solver = NQueensSolver(n)

    def generate_random_board(self, n_queens: Optional[int] = None) -> Chessboard:
        """Generate a random board with N queens placed randomly."""
        if n_queens is None:
            n_queens = self.n

        positions = []
        available = [(r, c) for r in range(self.n) for c in range(self.n)]

        for _ in range(min(n_queens, len(available))):
            idx = np.random.randint(0, len(available))
            positions.append(available.pop(idx))

        return Chessboard(self.n, positions)

    def generate_solution(self) -> Chessboard:
        """Generate a valid N-Queens solution."""
        solutions = self.solver.find_all_solutions()
        if solutions:
            return np.random.choice(solutions)
        raise ValueError(f"No complete solution exists for {self.n}-Queens")

    def generate_partial_solution(self, max_rows: int) -> Chessboard:
        """Generate a partial solution (queens placed in first k rows safely)."""
        if max_rows >= self.n:
            return self.generate_solution()

        positions = []
        cols = set()
        diag1 = set()
        diag2 = set()

        for row in range(max_rows):
            # Find safe columns
            safe_cols = []
            for col in range(self.n):
                d1 = row - col
                d2 = row + col
                if col not in cols and d1 not in diag1 and d2 not in diag2:
                    safe_cols.append(col)

            if safe_cols:
                col = np.random.choice(safe_cols)
                positions.append((row, col))
                cols.add(col)
                diag1.add(d1)
                diag2.add(d2)

        return Chessboard(self.n, positions)

    def generate_conflicting_board(self, min_conflicts: int = 1) -> Chessboard:
        """Generate a board with at least min_conflicts."""
        while True:
            board = self.generate_random_board()
            if board.count_conflicts() >= min_conflicts:
                return board

    def generate_dataset(self, n_samples: int = 1000,
                        balanced: bool = True,
                        solution_ratio: float = 0.5) -> List[Tuple[Chessboard, int]]:
        """
        Generate dataset of boards with labels.

        Args:
            n_samples: Total number of samples
            balanced: Balance positive and negative examples
            solution_ratio: Target ratio of valid solutions

        Returns:
            List of (board, label) where label=1 means valid solution
        """
        boards = []

        if balanced:
            n_pos = n_samples // 2
            n_neg = n_samples - n_pos

            # Generate valid solutions (positive examples)
            print(f"Generating {n_pos} valid solutions...")
            for _ in range(n_pos):
                boards.append((self.generate_solution(), 1))

            # Generate invalid boards (negative examples)
            print(f"Generating {n_neg} invalid boards...")
            for _ in range(n_neg):
                # Generate boards with conflicts
                if np.random.random() < 0.5:
                    boards.append((self.generate_conflicting_board(min_conflicts=1), 0))
                else:
                    board = self.generate_random_board(n_queens=self.n)
                    while board.is_valid_solution():
                        board = self.generate_random_board(n_queens=self.n)
                    boards.append((board, 0))

        else:
            for _ in range(n_samples):
                board = self.generate_random_board()
                label = 1 if board.is_valid_solution() else 0
                boards.append((board, label))

        return boards


class NQueensFeatureExtractor:
    """Extract logical features from chessboard configurations."""

    @staticmethod
    def extract_boolean_features(board: Chessboard) -> np.ndarray:
        """
        Extract Boolean features for ILP learning.

        Features capture:
        - Row/column conflicts
        - Diagonal conflicts
        - Queen distribution
        - Validity conditions
        """
        features = []
        n = board.n
        positions = board.positions

        # Basic counts
        n_queens = len(positions)
        row_counts = board.get_row_counts()
        col_counts = board.get_col_counts()

        # === Conflict-based features ===

        # Row conflicts: multiple queens in same row
        has_row_conflict = any(c > 1 for c in row_counts)
        features.append(int(not has_row_conflict))  # no_row_conflict

        # Column conflicts: multiple queens in same column
        has_col_conflict = any(c > 1 for c in col_counts)
        features.append(int(not has_col_conflict))  # no_col_conflict

        # Diagonal conflicts
        has_diag_conflict, diag_count = NQueensFeatureExtractor._count_diagonal_conflicts(positions)
        features.append(int(not has_diag_conflict))  # no_diag_conflict

        # === Specific conflict counts ===

        # Exactly N queens (required for N-Queens)
        features.append(int(n_queens == n))  # correct_queen_count

        # At least N queens
        features.append(int(n_queens >= n))  # enough_queens

        # No row has > 1 queen
        features.append(int(max(row_counts) <= 1))  # max_one_per_row

        # No column has > 1 queen
        features.append(int(max(col_counts) <= 1))  # max_one_per_col

        # === Distribution features ===

        # Queens are spread across different rows
        unique_rows = len(set(r for r, _ in positions))
        features.append(int(unique_rows == n))  # all_rows_used

        # Queens are spread across different columns
        unique_cols = len(set(c for _, c in positions))
        features.append(int(unique_cols == n))  # all_cols_used

        # === Pairwise safety features ===

        # All queens are safe from each other
        all_safe = True
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                r1, c1 = positions[i]
                r2, c2 = positions[j]
                if Chessboard.queens_attack(r1, c1, r2, c2):
                    all_safe = False
                    break
            if not all_safe:
                break
        features.append(int(all_safe))  # all_queens_safe

        # === Specific pattern features ===

        # Queens form a permutation (one per row and column)
        is_permutation = (max(row_counts) <= 1 and max(col_counts) <= 1 and n_queens == n)
        features.append(int(is_permutation))  # is_valid_permutation

        # Check specific diagonal patterns for small N
        if n <= 8:
            # No two queens on main diagonal
            main_diag_conflict = False
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    r1, c1 = positions[i]
                    r2, c2 = positions[j]
                    if r1 - c1 == r2 - c2:  # Same main diagonal
                        main_diag_conflict = True
                        break
                if main_diag_conflict:
                    break
            features.append(int(not main_diag_conflict))  # no_main_diag_conflict

            # No two queens on anti-diagonal
            anti_diag_conflict = False
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    r1, c1 = positions[i]
                    r2, c2 = positions[j]
                    if r1 + c1 == r2 + c2:  # Same anti-diagonal
                        anti_diag_conflict = True
                        break
                if anti_diag_conflict:
                    break
            features.append(int(not anti_diag_conflict))  # no_anti_diag_conflict

        # === Density features ===

        # Board is not too crowded (for partial solutions)
        density = n_queens / (n * n)
        features.append(int(density <= 0.5))  # reasonable_density

        # === Symmetry features (for 4-queens and 8-queens) ===

        if n == 4:
            # Special case: check known solution patterns
            # Valid 4-queens solutions: (1,2), (2,4), (3,1), (4,3) or rotations
            known_solutions = [
                {(0, 1), (1, 3), (2, 0), (3, 2)},  # One of the 2 solutions
                {(0, 2), (1, 0), (2, 3), (3, 1)},  # The other solution
            ]
            matches_known = any(set(positions).issuperset(sol) for sol in known_solutions)
            features.append(int(matches_known))  # matches_known_pattern

        # === Corner/edge features ===

        # Queens in corners
        corner_queens = sum(1 for r, c in positions if (r in [0, n-1] and c in [0, n-1]))
        features.append(int(corner_queens <= 1))  # not_too_many_corners

        return np.array(features, dtype=int)

    @staticmethod
    def _count_diagonal_conflicts(positions: List[Tuple[int, int]]) -> Tuple[bool, int]:
        """Count diagonal conflicts."""
        diag1_set = set()  # Main diagonals (r-c)
        diag2_set = set()  # Anti-diagonals (r+c)

        conflicts = 0
        for r, c in positions:
            d1 = r - c
            d2 = r + c

            if d1 in diag1_set:
                conflicts += 1
            diag1_set.add(d1)

            if d2 in diag2_set:
                conflicts += 1
            diag2_set.add(d2)

        return conflicts > 0, conflicts

    @staticmethod
    def get_feature_names(n: int) -> List[str]:
        """Get names of extracted features."""
        names = [
            "no_row_conflict",
            "no_col_conflict",
            "no_diag_conflict",
            "correct_queen_count",
            "enough_queens",
            "max_one_per_row",
            "max_one_per_col",
            "all_rows_used",
            "all_cols_used",
            "all_queens_safe",
            "is_valid_permutation",
        ]

        if n <= 8:
            names.extend([
                "no_main_diag_conflict",
                "no_anti_diag_conflict",
            ])

        if n == 4:
            names.append("matches_known_pattern")

        names.extend([
            "reasonable_density",
            "not_too_many_corners",
        ])

        return names

    @staticmethod
    def extract_literal_features(board: Chessboard) -> List[Tuple[str, tuple]]:
        """
        Extract board as ILP literals.

        Returns literals like:
        - queen(row, col)
        - threatens(row1, col1, row2, col2)
        """
        literals = []

        # Queen positions
        for r, c in board.positions:
            literals.append(("queen", (r, c)))

        # Threat relationships
        for i, (r1, c1) in enumerate(board.positions):
            for j, (r2, c2) in enumerate(board.positions):
                if i < j and Chessboard.queens_attack(r1, c1, r2, c2):
                    literals.append(("threatens", (r1, c1, r2, c2)))

        return literals


class NQueensDataset:
    """Complete dataset for N-Queens learning."""

    def __init__(self, n: int = 8, n_samples: int = 1000):
        self.n = n
        self.n_samples = n_samples
        self.generator = NQueensGenerator(n)
        self.extractor = NQueensFeatureExtractor()

        self.boards: List[Tuple[Chessboard, int]] = []
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    def generate(self, balanced: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate the complete dataset."""
        print(f"Generating N-Queens dataset (N={self.n}, samples={self.n_samples})...")

        self.boards = self.generator.generate_dataset(
            n_samples=self.n_samples,
            balanced=balanced
        )

        # Extract features
        self.feature_names = self.extractor.get_feature_names(self.n)
        X_list = []
        y_list = []

        for board, label in self.boards:
            features = self.extractor.extract_boolean_features(board)
            X_list.append(features)
            y_list.append(label)

        self.X = np.array(X_list)
        self.y = np.array(y_list)

        pos_count = np.sum(self.y)
        print(f"Generated {len(self.boards)} boards")
        print(f"Valid solutions: {pos_count}")
        print(f"Invalid boards: {len(self.y) - pos_count}")
        print(f"Feature dimensions: {self.X.shape[1]}")

        return self.X, self.y, self.feature_names

    def print_statistics(self):
        """Print dataset statistics."""
        if self.X is None:
            return

        pos_mask = self.y == 1
        neg_mask = self.y == 0

        print("\n=== N-Queens Dataset Statistics ===")
        print(f"N (board size): {self.n}")
        print(f"Total samples: {len(self.y)}")
        print(f"Valid solutions: {np.sum(pos_mask)}")
        print(f"Invalid boards: {np.sum(neg_mask)}")
        print(f"Feature dimensions: {self.X.shape[1]}")

        print("\nFeature means by class:")
        print(f"{'Feature':<30} {'Valid':>10} {'Invalid':>10} {'Diff':>10}")
        print("-" * 65)

        pos_mean = self.X[pos_mask].mean(axis=0) if np.any(pos_mask) else np.zeros(self.X.shape[1])
        neg_mean = self.X[neg_mask].mean(axis=0) if np.any(neg_mask) else np.zeros(self.X.shape[1])

        for i, name in enumerate(self.feature_names):
            diff = pos_mean[i] - neg_mean[i]
            print(f"{name:<30} {pos_mean[i]:>10.3f} {neg_mean[i]:>10.3f} {diff:>10.3f}")

    def visualize_board(self, board_idx: int = 0):
        """Print a visual representation of a board."""
        if board_idx >= len(self.boards):
            return

        board, label = self.boards[board_idx]
        status = "Valid Solution" if label else "Invalid"

        print(f"\nBoard {board_idx + 1} ({status}):")
        print("  " + " ".join(str(i).rjust(2) for i in range(self.n)))
        for r in range(self.n):
            print(f"{r}: ", end="")
            for c in range(self.n):
                if (r, c) in board.positions:
                    print(" Q ", end="")
                else:
                    print(" . ", end="")
            print()

    def to_prolog(self, board_idx: int) -> str:
        """Convert board to Prolog facts."""
        if board_idx >= len(self.boards):
            return ""

        board, label = self.boards[board_idx]
        prolog = []

        for r, c in board.positions:
            prolog.append(f"queen({r}, {c}).")

        if label == 1:
            prolog.append("valid_solution.")

        return "\n".join(prolog)


def create_nqueens_dataset(n: int = 8, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create an N-Queens dataset.

    Args:
        n: Board size (4, 8, or 12 recommended)
        n_samples: Total number of samples

    Returns:
        X: Feature matrix
        y: Labels (1 = valid solution, 0 = invalid)
        feature_names: Names of features
    """
    dataset = NQueensDataset(n=n, n_samples=n_samples)
    return dataset.generate(balanced=True)


# ============================================================
# Specialized N-Queens Datasets
# ============================================================

class SpecializedNQueensGenerator:
    """Generate specialized N-Queens configurations."""

    @staticmethod
    def generate_n_queens_variants(n: int = 4) -> List[Tuple[Chessboard, str]]:
        """Generate various N-Queens board types with descriptions."""
        variants = []

        # Valid solution
        solver = NQueensSolver(n)
        solutions = solver.find_all_solutions()
        if solutions:
            variants.append((solutions[0], "valid_solution"))

        # Two queens in same row
        positions = [(0, 0), (0, 1)]
        if n > 2:
            positions.extend([(i, i) for i in range(2, n)])
        variants.append((Chessboard(n, positions), "row_conflict"))

        # Two queens in same column
        positions = [(0, 0), (1, 0)]
        if n > 2:
            positions.extend([(i, i % n) for i in range(2, n)])
        variants.append((Chessboard(n, positions), "column_conflict"))

        # Diagonal conflict
        positions = [(0, 0), (1, 1)]
        if n > 2:
            positions.extend([(i, (i + 1) % n) for i in range(2, n)])
        variants.append((Chessboard(n, positions), "diagonal_conflict"))

        # Too few queens
        variants.append((Chessboard(n, [(0, 0), (1, 2), (2, 4)]), "too_few_queens"))

        # Too many queens in one area
        if n >= 4:
            positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
            variants.append((Chessboard(n, positions), "crowded_area"))

        return variants


# ============================================================
# Main Functions and Tests
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.lfio_learner import LFIOBoolean
    from src.lfio_extended import LFIOBooleanWithNegation

    print("=== N-Queens Dataset Generator ===\n")

    # Test with 4-queens (smaller, faster)
    print("Testing with 4-Queens...\n")
    dataset = NQueensDataset(n=4, n_samples=500)
    X, y, feature_names = dataset.generate()
    dataset.print_statistics()

    # Show example boards
    print("\n=== Example Boards ===")
    for i in range(min(3, len(dataset.boards))):
        dataset.visualize_board(i)

    # Test specialized variants
    print("\n=== Specialized Variants ===")
    variants = SpecializedNQueensGenerator.generate_n_queens_variants(4)
    for board, desc in variants:
        print(f"\n{desc}:")
        print(f"  Positions: {sorted(board.positions)}")
        print(f"  Valid: {board.is_valid_solution()}")

    # Test LFIO
    print("\n=== Testing LFIO on N-Queens Data ===")
    lfio = LFIOBoolean(min_coverage=10, min_precision=0.9)
    lfio.fit(X, y, feature_names)
    lfio.print_rules()

    predictions = lfio.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nLFIO Accuracy: {accuracy:.3f}")

    # Test with negation
    print("\n=== Testing LFIO with Negation ===")
    lfio_neg = LFIOBooleanWithNegation(min_coverage=5, min_precision=0.85, allow_negation=True)
    lfio_neg.fit(X, y, feature_names)
    lfio_neg.print_rules()

    predictions_neg = lfio_neg.predict(X)
    accuracy_neg = np.mean(predictions_neg == y)
    print(f"\nLFIO+Negation Accuracy: {accuracy_neg:.3f}")

    # Test with 8-queens (classic problem)
    print("\n\n=== 8-Queens Dataset ===")
    dataset_8 = NQueensDataset(n=8, n_samples=500)
    X8, y8, names8 = dataset_8.generate()

    print("\nMost predictive features:")
    pos_mean = X8[y8 == 1].mean(axis=0)
    neg_mean = X8[y8 == 0].mean(axis=0)
    diffs = np.abs(pos_mean - neg_mean)
    sorted_idx = np.argsort(diffs)[::-1]

    for i in sorted_idx[:5]:
        print(f"  {names8[i]}: diff={diffs[i]:.3f}")

    # LFIO on 8-queens
    print("\nLFIO on 8-Queens:")
    lfio_8 = LFIOBoolean(min_coverage=10, min_precision=0.9)
    lfio_8.fit(X8, y8, names8)
    lfio_8.print_rules()
