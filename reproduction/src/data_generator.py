"""Generate synthetic data with known logical patterns for rule extraction."""

import numpy as np
import pandas as pd
from typing import Tuple, Callable, Dict


class SyntheticDataGenerator:
    """Generate data based on logical rules."""

    def __init__(self, n_samples: int = 1000, n_features: int = 5, noise: float = 0.0):
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise

    def generate_boolean_dataset(self, rule_type: str = "and") -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Generate binary classification data based on logical rules.

        Args:
            rule_type: Type of logical rule ('and', 'or', 'xor', 'majority', 'complex')

        Returns:
            X: Features (binary)
            y: Labels (binary)
            rule_description: Human-readable description of the rule
        """
        X = np.random.randint(0, 2, size=(self.n_samples, self.n_features))

        if rule_type == "and":
            y = np.all(X, axis=1).astype(int)
            rule_description = f"y = 1 if ALL features are 1, else 0"

        elif rule_type == "or":
            y = np.any(X, axis=1).astype(int)
            rule_description = f"y = 1 if ANY feature is 1, else 0"

        elif rule_type == "xor":
            y = (np.sum(X, axis=1) % 2).astype(int)
            rule_description = f"y = 1 if ODD number of features are 1, else 0"

        elif rule_type == "majority":
            y = (np.sum(X, axis=1) > self.n_features / 2).astype(int)
            rule_description = f"y = 1 if MAJORITY of features are 1, else 0"

        elif rule_type == "complex":
            # Complex rule: (x1 AND x2) OR (x3 AND NOT x4)
            y = ((X[:, 0] & X[:, 1]) | (X[:, 2] & ~X[:, 3])).astype(int)
            rule_description = f"y = 1 if (x1 AND x2) OR (x3 AND NOT x4), else 0"

        elif rule_type == "nested":
            # Nested rule: (x1 OR x2) AND (x3 OR x4) AND x5
            y = ((X[:, 0] | X[:, 1]) & (X[:, 2] | X[:, 3]) & X[:, 4]).astype(int)
            rule_description = f"y = 1 if (x1 OR x2) AND (x3 OR x4) AND x5, else 0"

        else:
            raise ValueError(f"Unknown rule type: {rule_type}")

        # Add noise if specified
        if self.noise > 0:
            flip_mask = np.random.random(self.n_samples) < self.noise
            y[flip_mask] = 1 - y[flip_mask]

        return X, y, rule_description

    def generate_continuous_dataset(self, rule_type: str = "threshold") -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Generate continuous feature data with threshold-based rules.

        Args:
            rule_type: Type of rule ('threshold', 'rectangle', 'circle')

        Returns:
            X: Features (continuous)
            y: Labels (binary)
            rule_description: Human-readable description
        """
        X = np.random.randn(self.n_samples, self.n_features)

        if rule_type == "threshold":
            # Simple threshold rule
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
            rule_description = f"y = 1 if x1 + x2 > 0, else 0"

        elif rule_type == "rectangle":
            # Rectangle rule: feature in specific range
            y = ((X[:, 0] > -0.5) & (X[:, 0] < 0.5) &
                 (X[:, 1] > -0.5) & (X[:, 1] < 0.5)).astype(int)
            rule_description = f"y = 1 if -0.5 < x1 < 0.5 AND -0.5 < x2 < 0.5, else 0"

        elif rule_type == "circle":
            # Circle rule: inside unit circle
            y = (np.sum(X[:, :2] ** 2, axis=1) < 1.0).astype(int)
            rule_description = f"y = 1 if x1^2 + x2^2 < 1, else 0"

        elif rule_type == "diagonal":
            # Diagonal decision boundary
            y = ((X[:, 0] - X[:, 1]) > 0).astype(int)
            rule_description = f"y = 1 if x1 > x2, else 0"

        else:
            raise ValueError(f"Unknown rule type: {rule_type}")

        return X, y, rule_description

    def create_feature_names(self) -> list:
        """Create feature names for better interpretability."""
        return [f"x{i+1}" for i in range(self.n_features)]


def load_custom_rule_dataset(rule_fn: Callable, n_samples: int = 1000,
                              n_features: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate data using a custom rule function.

    Args:
        rule_fn: Function that takes X and returns y
        n_samples: Number of samples
        n_features: Number of features

    Returns:
        X: Features
        y: Labels
    """
    X = np.random.randint(0, 2, size=(n_samples, n_features))
    y = rule_fn(X)
    return X, y


if __name__ == "__main__":
    # Test the generator
    generator = SyntheticDataGenerator(n_samples=100, n_features=5)

    print("=== Boolean Rules ===")
    for rule_type in ["and", "or", "xor", "majority", "complex", "nested"]:
        X, y, desc = generator.generate_boolean_dataset(rule_type)
        print(f"\n{rule_type.upper()}:")
        print(f"  Rule: {desc}")
        print(f"  Class balance: {np.mean(y):.2f} positive")
        print(f"  Sample X: {X[:3]}")
        print(f"  Sample y: {y[:3]}")

    print("\n=== Continuous Rules ===")
    for rule_type in ["threshold", "rectangle", "circle", "diagonal"]:
        X, y, desc = generator.generate_continuous_dataset(rule_type)
        print(f"\n{rule_type.upper()}:")
        print(f"  Rule: {desc}")
        print(f"  Class balance: {np.mean(y):.2f} positive")
