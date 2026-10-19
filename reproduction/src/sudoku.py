"""
Sudoku Dataset Generator for ILP/LFIO Experiments

Sudoku is a logic puzzle where you fill a 9×9 grid with digits 1-9 so that:
- Each row contains all digits 1-9 exactly once
- Each column contains all digits 1-9 exactly once
- Each 3×3 subgrid (box) contains all digits 1-9 exactly once

For ILP learning, we generate grid configurations and extract logical features
that characterize whether the configuration is valid or leads to a solution.

Features include:
- Row/column/box completeness
- Duplicate detection
- Missing value counts
- Constraint satisfaction patterns
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import itertools


@dataclass
class SudokuGrid:
    """Represents a Sudoku grid."""
    grid: np.ndarray  # 9×9 grid, 0 represents empty cell
    size: int = 9  # Can be 4 (4×4 Sudoku) or 9 (standard)

    def __post_init__(self):
        if self.grid.shape != (self.size, self.size):
            raise ValueError(f"Grid must be {self.size}×{self.size}")

    @staticmethod
    def empty(size: int = 9) -> 'SudokuGrid':
        """Create an empty Sudoku grid."""
        return SudokuGrid(np.zeros((size, size), dtype=int), size)

    @staticmethod
    def from_list(grid_list: List[List[int]]) -> 'SudokuGrid':
        """Create a SudokuGrid from a 2D list."""
        grid = np.array(grid_list, dtype=int)
        size = grid.shape[0]
        return SudokuGrid(grid, size)

    def is_valid_placement(self, row: int, col: int, value: int) -> bool:
        """Check if placing value at (row, col) is valid."""
        # Check row
        if value in self.grid[row, :]:
            return False

        # Check column
        if value in self.grid[:, col]:
            return False

        # Check subgrid (box)
        box_size = int(self.size ** 0.5)
        box_row = (row // box_size) * box_size
        box_col = (col // box_size) * box_size
        if value in self.grid[box_row:box_row + box_size, box_col:box_col + box_size]:
            return False

        return True

    def count_violations(self) -> int:
        """Count total constraint violations."""
        violations = 0

        # Row violations
        for r in range(self.size):
            row_vals = [v for v in self.grid[r, :] if v != 0]
            violations += len(row_vals) - len(set(row_vals))

        # Column violations
        for c in range(self.size):
            col_vals = [v for v in self.grid[:, c] if v != 0]
            violations += len(col_vals) - len(set(col_vals))

        # Box violations
        box_size = int(self.size ** 0.5)
        for br in range(0, self.size, box_size):
            for bc in range(0, self.size, box_size):
                box_vals = []
                for r in range(br, br + box_size):
                    for c in range(bc, bc + box_size):
                        v = self.grid[r, c]
                        if v != 0:
                            box_vals.append(v)
                violations += len(box_vals) - len(set(box_vals))

        return violations

    def is_valid(self) -> bool:
        """Check if current grid has no violations."""
        return self.count_violations() == 0

    def is_complete(self) -> bool:
        """Check if grid is completely filled."""
        return np.all(self.grid != 0)

    def is_solved(self) -> bool:
        """Check if grid is a valid complete solution."""
        return self.is_complete() and self.is_valid()

    def get_filled_count(self) -> int:
        """Count number of filled cells."""
        return np.sum(self.grid != 0)

    def get_completion_percentage(self) -> float:
        """Percentage of cells filled."""
        return self.get_filled_count() / (self.size * self.size)

    def get_row_completeness(self) -> List[int]:
        """Number of filled cells in each row."""
        return [int(np.sum(self.grid[r, :] != 0)) for r in range(self.size)]

    def get_col_completeness(self) -> List[int]:
        """Number of filled cells in each column."""
        return [int(np.sum(self.grid[:, c] != 0)) for c in range(self.size)]

    def get_box_completeness(self) -> List[int]:
        """Number of filled cells in each box."""
        box_size = int(self.size ** 0.5)
        completeness = []

        for br in range(0, self.size, box_size):
            for bc in range(0, self.size, box_size):
                count = 0
                for r in range(br, br + box_size):
                    for c in range(bc, bc + box_size):
                        if self.grid[r, c] != 0:
                            count += 1
                completeness.append(count)

        return completeness

    def has_row_duplicates(self) -> List[bool]:
        """Check which rows have duplicate values."""
        dup_rows = []
        for r in range(self.size):
            row_vals = [v for v in self.grid[r, :] if v != 0]
            dup_rows.append(len(row_vals) != len(set(row_vals)))
        return dup_rows

    def has_col_duplicates(self) -> List[bool]:
        """Check which columns have duplicate values."""
        dup_cols = []
        for c in range(self.size):
            col_vals = [v for v in self.grid[:, c] if v != 0]
            dup_cols.append(len(col_vals) != len(set(col_vals)))
        return dup_cols

    def has_box_duplicates(self) -> List[bool]:
        """Check which boxes have duplicate values."""
        box_size = int(self.size ** 0.5)
        dup_boxes = []

        for br in range(0, self.size, box_size):
            for bc in range(0, self.size, box_size):
                box_vals = []
                for r in range(br, br + box_size):
                    for c in range(bc, bc + box_size):
                        v = self.grid[r, c]
                        if v != 0:
                            box_vals.append(v)
                dup_boxes.append(len(box_vals) != len(set(box_vals)))

        return dup_boxes

    def __str__(self) -> str:
        """String representation of the grid."""
        lines = []
        box_size = int(self.size ** 0.5)
        separator = "+" + "+".join(["-" * (box_size * 2 + 1)] * box_size) + "+"

        for r in range(self.size):
            if r % box_size == 0:
                lines.append(separator)

            row_str = "|"
            for c in range(self.size):
                if c % box_size == 0:
                    row_str += " "
                val = self.grid[r, c]
                row_str += str(val) if val != 0 else "."
                row_str += " "
            row_str += "|"
            lines.append(row_str)

        if r % box_size == box_size - 1:
            lines.append(separator)

        return "\n".join(lines)


class SudokuSolver:
    """Solver for Sudoku using backtracking."""

    def __init__(self, size: int = 9):
        self.size = size
        self.box_size = int(size ** 0.5)

    def solve(self, grid: SudokuGrid) -> Optional[SudokuGrid]:
        """Solve the Sudoku puzzle. Returns solution or None."""
        # Find empty cell
        empty = self._find_empty(grid)
        if not empty:
            return grid  # Solved!

        row, col = empty

        # Try digits 1 to size
        for num in range(1, self.size + 1):
            if grid.is_valid_placement(row, col, num):
                grid.grid[row, col] = num

                if self.solve(grid):
                    return grid

                grid.grid[row, col] = 0  # Backtrack

        return None  # No solution found

    def _find_empty(self, grid: SudokuGrid) -> Optional[Tuple[int, int]]:
        """Find an empty cell."""
        for r in range(self.size):
            for c in range(self.size):
                if grid.grid[r, c] == 0:
                    return (r, c)
        return None

    def has_unique_solution(self, grid: SudokuGrid) -> bool:
        """Check if puzzle has exactly one solution."""
        solutions = self._count_solutions(grid, max_solutions=2)
        return solutions == 1

    def _count_solutions(self, grid: SudokuGrid, max_solutions: int = 2) -> int:
        """Count solutions (up to max_solutions)."""
        empty = self._find_empty(grid)
        if not empty:
            return 1  # Found a solution

        row, col = empty
        count = 0

        for num in range(1, self.size + 1):
            if grid.is_valid_placement(row, col, num):
                grid.grid[row, col] = num
                count += self._count_solutions(grid, max_solutions - count)
                grid.grid[row, col] = 0

                if count >= max_solutions:
                    return count

        return count


class SudokuGenerator:
    """Generate Sudoku datasets for ILP learning."""

    def __init__(self, size: int = 9):
        self.size = size
        self.box_size = int(size ** 0.5)
        self.solver = SudokuSolver(size)

    def generate_solved_grid(self) -> SudokuGrid:
        """Generate a completely solved valid Sudoku grid."""
        if self.size == 4:
            # For 4×4, use predefined solutions (faster and guaranteed)
            return self._generate_4x4_solution()
        else:
            # For 9×9, use the solver
            grid = SudokuGrid.empty(self.size)
            return self._generate_solution(grid)

    def _generate_4x4_solution(self) -> SudokuGrid:
        """Generate a valid 4×4 solution using known patterns."""
        import random

        # Base 4×4 solutions
        solutions = [
            # Solution 1
            [[1, 2, 3, 4],
             [3, 4, 1, 2],
             [2, 1, 4, 3],
             [4, 3, 2, 1]],
            # Solution 2 (swapped 1<->2, 3<->4)
            [[2, 1, 4, 3],
             [4, 3, 2, 1],
             [1, 2, 3, 4],
             [3, 4, 1, 2]],
            # Solution 3 (rotated)
            [[1, 3, 2, 4],
             [2, 4, 1, 3],
             [3, 1, 4, 2],
             [4, 2, 3, 1]],
        ]

        base = np.array(random.choice(solutions), dtype=int)

        # Apply random transformations for variety
        # Swap digits randomly
        digits = [1, 2, 3, 4]
        np.random.shuffle(digits)
        mapping = {i+1: digits[i] for i in range(4)}

        transformed = base.copy()
        for r in range(4):
            for c in range(4):
                transformed[r, c] = mapping[base[r, c]]

        return SudokuGrid(transformed, size=4)

    def _generate_solution(self, grid: SudokuGrid) -> SudokuGrid:
        """Generate a solution by filling diagonal boxes then solving."""
        # Fill diagonal boxes (independent)
        box_size = self.box_size
        for i in range(0, self.size, box_size):
            self._fill_box(grid, i, i)

        # Solve the rest
        return self.solver.solve(grid)

    def _fill_box(self, grid: SudokuGrid, row: int, col: int):
        """Fill a box with random valid numbers."""
        box_size = self.box_size
        nums = list(range(1, self.size + 1))
        np.random.shuffle(nums)

        idx = 0
        for r in range(row, row + box_size):
            for c in range(col, col + box_size):
                grid.grid[r, c] = nums[idx]
                idx += 1

    def generate_puzzle(self, remove_count: int = 40) -> SudokuGrid:
        """Generate a Sudoku puzzle by removing cells from a solution."""
        solution = self.generate_solved_grid()
        puzzle = SudokuGrid(solution.grid.copy(), self.size)

        # Remove cells while maintaining unique solution
        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        np.random.shuffle(cells)

        removed = 0
        for r, c in cells:
            if removed >= remove_count:
                break

            backup = puzzle.grid[r, c]
            puzzle.grid[r, c] = 0

            # Check if still has unique solution
            if self.solver.has_unique_solution(puzzle):
                removed += 1
            else:
                puzzle.grid[r, c] = backup  # Put it back

        return puzzle

    def generate_invalid_grid(self, violation_type: str = "row") -> SudokuGrid:
        """Generate an invalid grid with specific violation type."""
        if violation_type not in {"row", "column", "box"}:
            raise ValueError("violation_type must be row, column, or box")
        # Match the positive generator's occupancy. Sparse negatives previously
        # made completeness a shortcut, and insertion could silently do nothing.
        grid = self.generate_partial_solution(fill_percentage=0.7)
        if violation_type == "row":
            groups = [[(r, c) for c in range(self.size)] for r in range(self.size)]
        elif violation_type == "column":
            groups = [[(r, c) for r in range(self.size)] for c in range(self.size)]
        else:
            groups = [[(r, c)
                       for r in range(br, br + self.box_size)
                       for c in range(bc, bc + self.box_size)]
                      for br in range(0, self.size, self.box_size)
                      for bc in range(0, self.size, self.box_size)]
        filled_groups = [[cell for cell in group if grid.grid[cell] != 0] for group in groups]
        eligible = [group for group in filled_groups if len(group) >= 2]
        group = eligible[np.random.randint(len(eligible))]
        first, second = np.random.choice(len(group), size=2, replace=False)
        grid.grid[group[second]] = grid.grid[group[first]]
        return grid

    def generate_partial_solution(self, fill_percentage: float = 0.5) -> SudokuGrid:
        """Generate a partially filled valid grid."""
        solution = self.generate_solved_grid()
        partial = SudokuGrid(solution.grid.copy(), self.size)

        # Remove random cells
        n_cells = self.size * self.size
        n_remove = int(n_cells * (1 - fill_percentage))

        cells = [(r, c) for r in range(self.size) for c in range(self.size)]
        np.random.shuffle(cells)

        for r, c in cells[:n_remove]:
            partial.grid[r, c] = 0

        return partial


class SudokuDatasetGenerator:
    """Generate Sudoku datasets for ILP learning."""

    def __init__(self, size: int = 9):
        self.size = size
        self.generator = SudokuGenerator(size)

    def generate_dataset(self, n_samples: int = 1000,
                        balanced: bool = True,
                        solution_ratio: float = 0.5) -> List[Tuple[SudokuGrid, int]]:
        """
        Generate dataset of Sudoku grids.

        Args:
            n_samples: Total number of samples
            balanced: Balance valid and invalid examples
            solution_ratio: Ratio of valid/invalid in balanced mode

        Returns:
            List of (grid, label) where label=1 means valid/solvable
        """
        grids = []

        if balanced:
            n_valid = n_samples // 2
            n_invalid = n_samples - n_valid

            print(f"Generating {n_valid} valid Sudoku grids...")
            for _ in range(n_valid):
                # Generate valid partial solution
                grid = self.generator.generate_partial_solution(fill_percentage=0.7)
                grids.append((grid, 1))

            print(f"Generating {n_invalid} invalid Sudoku grids...")
            violation_types = ["row", "column", "box"]
            for _ in range(n_invalid):
                v_type = np.random.choice(violation_types)
                grid = self.generator.generate_invalid_grid(violation_type=v_type)
                grids.append((grid, 0))

        else:
            for _ in range(n_samples):
                # Random mix
                if np.random.random() < 0.5:
                    grid = self.generator.generate_partial_solution()
                    label = 1
                else:
                    v_type = np.random.choice(["row", "column", "box"])
                    grid = self.generator.generate_invalid_grid(violation_type=v_type)
                    label = 0
                grids.append((grid, label))

        return grids


class SudokuFeatureExtractor:
    """Extract logical features from Sudoku grids."""

    @staticmethod
    def extract_boolean_features(grid: SudokuGrid) -> np.ndarray:
        """
        Extract Boolean features for ILP learning.

        Features capture:
        - Row/column/box violations
        - Completeness metrics
        - Distribution patterns
        """
        features = []
        n = grid.size

        # === Violation detection ===

        # Row duplicates
        has_row_dups = any(grid.has_row_duplicates())
        features.append(int(not has_row_dups))  # no_row_duplicates

        # Column duplicates
        has_col_dups = any(grid.has_col_duplicates())
        features.append(int(not has_col_dups))  # no_col_duplicates

        # Box duplicates
        has_box_dups = any(grid.has_box_duplicates())
        features.append(int(not has_box_dups))  # no_box_duplicates

        # No violations at all
        no_violations = int(grid.is_valid())
        features.append(no_violations)

        # === Completeness features ===

        # Percentage complete
        completion = grid.get_completion_percentage()
        features.append(int(completion >= 0.5))  # mostly_complete
        features.append(int(completion >= 0.8))  # nearly_complete
        features.append(int(completion == 1.0))  # fully_complete

        # All rows have some values
        row_comp = grid.get_row_completeness()
        features.append(int(all(c > 0 for c in row_comp)))  # all_rows_filled

        # All columns have some values
        col_comp = grid.get_col_completeness()
        features.append(int(all(c > 0 for c in col_comp)))  # all_cols_filled

        # All boxes have some values
        box_comp = grid.get_box_completeness()
        features.append(int(all(c > 0 for c in box_comp)))  # all_boxes_filled

        # === Distribution features ===

        # Even distribution across rows
        row_min = min(row_comp)
        row_max = max(row_comp)
        features.append(int(row_max - row_min <= 2))  # even_row_distribution

        # Even distribution across columns
        col_min = min(col_comp)
        col_max = max(col_comp)
        features.append(int(col_max - col_min <= 2))  # even_col_distribution

        # === Specific patterns ===

        # Count of cells with each value (for 9×9 Sudoku)
        if n == 9:
            value_counts = [np.sum(grid.grid == v) for v in range(1, 10)]
            # No value appears more than 9 times (impossible in valid Sudoku)
            features.append(int(all(c <= 9 for c in value_counts)))

            # Each value appears at least once in complete grid
            if grid.is_complete():
                features.append(int(all(c == 9 for c in value_counts)))
            else:
                features.append(int(all(c > 0 for c in value_counts)))  # all_values_used

        # === Center box features ===

        if n == 9:
            # Center box (4,4) to (6,6) is filled
            center_sum = np.sum(grid.grid[3:6, 3:6] != 0)
            features.append(int(center_sum >= 5))  # center_box_filled

        # === Structural validity ===

        # Check if grid can potentially be solved
        # (no obvious contradictions)
        potentially_solvable = True
        if not grid.is_valid():
            potentially_solvable = False
        features.append(int(potentially_solvable))

        return np.array(features, dtype=int)

    @staticmethod
    def get_feature_names(size: int = 9) -> List[str]:
        """Get names of extracted features."""
        names = [
            "no_row_duplicates",
            "no_col_duplicates",
            "no_box_duplicates",
            "no_violations",
            "mostly_complete",
            "nearly_complete",
            "fully_complete",
            "all_rows_filled",
            "all_cols_filled",
            "all_boxes_filled",
            "even_row_distribution",
            "even_col_distribution",
        ]

        if size == 9:
            names.extend([
                "no_value_overcount",
                "all_values_used",
                "center_box_filled",
            ])

        names.append("potentially_solvable")

        return names


class SudokuDataset:
    """Complete Sudoku dataset for ILP."""

    def __init__(self, size: int = 9, n_samples: int = 1000):
        self.size = size
        self.n_samples = n_samples
        self.generator = SudokuDatasetGenerator(size)
        self.extractor = SudokuFeatureExtractor()

        self.grids: List[Tuple[SudokuGrid, int]] = []
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

    def generate(self, balanced: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate the complete dataset."""
        print(f"Generating Sudoku dataset (size={self.size}, samples={self.n_samples})...")

        self.grids = self.generator.generate_dataset(
            n_samples=self.n_samples,
            balanced=balanced
        )

        # Extract features
        self.feature_names = self.extractor.get_feature_names(self.size)
        X_list = []
        y_list = []

        for grid, label in self.grids:
            features = self.extractor.extract_boolean_features(grid)
            X_list.append(features)
            y_list.append(label)

        self.X = np.array(X_list)
        self.y = np.array(y_list)

        pos_count = np.sum(self.y)
        print(f"Generated {len(self.grids)} grids")
        print(f"Valid grids: {pos_count}")
        print(f"Invalid grids: {len(self.y) - pos_count}")
        print(f"Feature dimensions: {self.X.shape[1]}")

        return self.X, self.y, self.feature_names

    def print_statistics(self):
        """Print dataset statistics."""
        if self.X is None:
            return

        pos_mask = self.y == 1
        neg_mask = self.y == 0

        print("\n=== Sudoku Dataset Statistics ===")
        print(f"Size: {self.size}×{self.size}")
        print(f"Total samples: {len(self.y)}")
        print(f"Valid grids: {np.sum(pos_mask)}")
        print(f"Invalid grids: {np.sum(neg_mask)}")
        print(f"Feature dimensions: {self.X.shape[1]}")

        print("\nFeature means by class:")
        print(f"{'Feature':<30} {'Valid':>10} {'Invalid':>10} {'Diff':>10}")
        print("-" * 65)

        pos_mean = self.X[pos_mask].mean(axis=0) if np.any(pos_mask) else np.zeros(self.X.shape[1])
        neg_mean = self.X[neg_mask].mean(axis=0) if np.any(neg_mask) else np.zeros(self.X.shape[1])

        for i, name in enumerate(self.feature_names):
            diff = pos_mean[i] - neg_mean[i]
            print(f"{name:<30} {pos_mean[i]:>10.3f} {neg_mean[i]:>10.3f} {diff:>10.3f}")

    def visualize_grid(self, grid_idx: int = 0):
        """Print a visual representation of a grid."""
        if grid_idx >= len(self.grids):
            return

        grid, label = self.grids[grid_idx]
        status = "Valid" if label else "Invalid"

        print(f"\nGrid {grid_idx + 1} ({status}):")
        print(grid)

    def to_prolog(self, grid_idx: int) -> str:
        """Convert grid to Prolog facts."""
        if grid_idx >= len(self.grids):
            return ""

        grid, label = self.grids[grid_idx]
        prolog = []

        for r in range(self.size):
            for c in range(self.size):
                v = grid.grid[r, c]
                if v != 0:
                    prolog.append(f"value({r}, {c}, {v}).")

        if label == 1:
            prolog.append("valid_grid.")

        return "\n".join(prolog)


# ============================================================
# 4×4 Sudoku (Simpler for testing)
# ============================================================

class Sudoku4x4:
    """Specialized generator for 4×4 Sudoku (easier/faster)."""

    @staticmethod
    def create_grid(values: str) -> SudokuGrid:
        """Create a 4×4 grid from a string representation.
        Format: "1234.34.2..4." where . is empty
        """
        grid_list = []
        for r in range(4):
            row = []
            for c in range(4):
                char = values[r * 4 + c]
                row.append(int(char) if char != '.' else 0)
            grid_list.append(row)

        return SudokuGrid(np.array(grid_list, dtype=int), size=4)

    @staticmethod
    def generate_valid() -> SudokuGrid:
        """Generate a valid 4×4 Sudoku solution."""
        # Valid 4×4 solutions (known patterns)
        solutions = [
            [[1, 2, 3, 4],
             [3, 4, 1, 2],
             [2, 1, 4, 3],
             [4, 3, 2, 1]],
            [[1, 2, 3, 4],
             [3, 4, 1, 2],
             [4, 3, 2, 1],
             [2, 1, 4, 3]],
            [[1, 3, 2, 4],
             [4, 2, 1, 3],
             [3, 1, 4, 2],
             [2, 4, 3, 1]],
        ]
        return SudokuGrid(np.array(np.random.choice(solutions)), size=4)

    @staticmethod
    def generate_invalid_row_dup() -> SudokuGrid:
        """Generate invalid grid with row duplicate."""
        grid = np.array([
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=int)
        return SudokuGrid(grid, size=4)

    @staticmethod
    def generate_invalid_col_dup() -> SudokuGrid:
        """Generate invalid grid with column duplicate."""
        grid = np.array([
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=int)
        return SudokuGrid(grid, size=4)

    @staticmethod
    def generate_invalid_box_dup() -> SudokuGrid:
        """Generate invalid grid with box duplicate."""
        grid = np.array([
            [1, 2, 0, 0],
            [2, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=int)
        return SudokuGrid(grid, size=4)


def create_sudoku_dataset(size: int = 4, n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create a Sudoku dataset.

    Args:
        size: Grid size (4 for 4×4, 9 for standard)
        n_samples: Total number of samples

    Returns:
        X: Feature matrix
        y: Labels (1 = valid, 0 = invalid)
        feature_names: Names of features
    """
    dataset = SudokuDataset(size=size, n_samples=n_samples)
    return dataset.generate(balanced=True)


# ============================================================
# Main Functions and Tests
# ============================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from src.lfio_learner import LFIOBoolean
    from src.lfio_extended import LFIOBooleanWithNegation

    print("=== Sudoku Dataset Generator ===\n")

    # Test with 4×4 Sudoku first (faster)
    print("Testing with 4×4 Sudoku...\n")
    dataset_4 = SudokuDataset(size=4, n_samples=500)
    X4, y4, names4 = dataset_4.generate()
    dataset_4.print_statistics()

    # Show example grids
    print("\n=== Example 4×4 Grids ===")
    dataset_4.visualize_grid(0)  # Valid
    dataset_4.visualize_grid(250)  # Invalid

    # Show specialized variants
    print("\n=== Specialized 4×4 Variants ===")

    valid = Sudoku4x4.generate_valid()
    print(f"\nValid solution:\n{valid}")

    invalid_row = Sudoku4x4.generate_invalid_row_dup()
    print(f"\nInvalid (row dup):\n{invalid_row}")
    print(f"Violations: {invalid_row.count_violations()}")

    # Test LFIO
    print("\n=== Testing LFIO on 4×4 Sudoku ===")
    lfio_4 = LFIOBoolean(min_coverage=10, min_precision=0.9)
    lfio_4.fit(X4, y4, names4)
    lfio_4.print_rules()

    predictions_4 = lfio_4.predict(X4)
    accuracy_4 = np.mean(predictions_4 == y4)
    print(f"\nLFIO Accuracy (4×4): {accuracy_4:.3f}")

    # Test with negation
    print("\n=== LFIO with Negation (4×4) ===")
    lfio_neg_4 = LFIOBooleanWithNegation(min_coverage=5, min_precision=0.85, allow_negation=True)
    lfio_neg_4.fit(X4, y4, names4)
    lfio_neg_4.print_rules()

    predictions_neg_4 = lfio_neg_4.predict(X4)
    accuracy_neg_4 = np.mean(predictions_neg_4 == y4)
    print(f"\nLFIO+Negation Accuracy (4×4): {accuracy_neg_4:.3f}")

    # Test with 9×9 (slower)
    print("\n\n=== 9×9 Sudoku Dataset ===")
    dataset_9 = SudokuDataset(size=9, n_samples=300)
    X9, y9, names9 = dataset_9.generate()

    print("\nMost predictive features:")
    pos_mean = X9[y9 == 1].mean(axis=0)
    neg_mean = X9[y9 == 0].mean(axis=0)
    diffs = np.abs(pos_mean - neg_mean)
    sorted_idx = np.argsort(diffs)[::-1]

    for i in sorted_idx[:5]:
        print(f"  {names9[i]}: diff={diffs[i]:.3f}")

    print("\nLFIO on 9×9:")
    lfio_9 = LFIOBoolean(min_coverage=10, min_precision=0.9)
    lfio_9.fit(X9, y9, names9)
    lfio_9.print_rules()
