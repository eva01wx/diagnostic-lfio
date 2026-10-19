"""
Dataset Recommendations for LFIO/ILP Experiments

This module provides utilities to load and prepare datasets that are
well-suited for Inductive Logic Programming experiments.

Recommended Categories:
1. Binary/Boolean classification with interpretable rules
2. Relational data with clear structure
3. Knowledge graphs and linked data
4. Program synthesis tasks
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from urllib.request import urlopen
import io


# ============================================================
# Recommended Datasets for ILP Experiments
# ============================================================

DATASET_RECOMMENDATIONS = """
=== HIGHLY RECOMMENDED FOR LFIO ===

1. MONK'S PROBLEMS (UCI Repository)
   - Three artificial datasets with known logical rules
   - MONK-1: 6 attributes, binary classification
   - MONK-2: More complex, requires disjunctive rules
   - MONK-3: With noise (5%)
   URL: https://archive.ics.uci.edu/ml/datasets/Monk%27s+Problems

2. MUSHROOM (UCI Repository)
   - Edible vs poisonous mushrooms
   - 22 categorical attributes
   - Very learnable rules
   URL: https://archive.ics.uci.edu/ml/datasets/Mushroom

3. TIC-TAC-TOE (UCI Repository)
   - Board states with winning/losing outcomes
   - 9 binary attributes (board positions)
   - Clear logical patterns
   URL: https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame

4. CONGRESSIONAL VOTING RECORDS (UCI Repository)
   - Democrat vs Republican voting patterns
   - 16 binary votes (yes/no/abstain)
   - Learnable partisan patterns
   URL: https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records

5. BREAST CANCER (Wisconsin - Diagnostic)
   - Benign vs malignant
   - Can be discretized to logical rules
   URL: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

6. BALANCE SCALE (UCI Repository)
   - Left/right/balance classification
   - 4 attributes with clear rules
   URL: https://archive.ics.uci.edu/ml/datasets/Balance+Scale

7. CAR EVALUATION (UCI Repository)
   - Acceptability of cars
   - 6 categorical attributes
   URL: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation

8. NURSERY (UCI Repository)
   - Nursery school application evaluation
   - 8 categorical attributes
   URL: https://archive.ics.uci.edu/ml/datasets/Nursery

=== RELATIONAL/STRUCTURED DATASETS ===

9. MUTAGENESIS (ILP Benchmark)
   - Molecular property prediction
   - Chemical structure as atoms and bonds
   - Classic ILP dataset
   URL: https://relational.fit.cvut.cz/dataset/Mutagenesis

10. KINSHIP (ILP Benchmark)
    - Family relationship learning
    - Learn rules like: uncle(X,Y) :- parent(Z,Y), brother(Z,X)
    URL: https://relational.fit.cvut.cz/dataset/Kinship

11. Yeast (Protein Localization)
    - Predict cellular localization sites
    - Can be expressed with logical rules
    URL: https://archive.ics.uci.edu/ml/datasets/Yeast

12. Splice Junction DNA (UCI Repository)
    - Exon/intron boundary prediction
    - DNA sequence data
    URL: https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequence+Data)

=== REAL-WORLD XAI APPLICATIONS ===

13. Lending Club Loan Data
    - Predict loan approval
    - Learn fair lending rules
    - Kaggle / LendingClub API

14. Heart Disease (UCI Repository)
    - Predict presence of heart disease
    - Clear risk factors
    URL: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

15. Diabetes (Pima Indians)
    - Diabetes prediction
    - Medical rule learning
    URL: https://archive.ics.uci.edu/ml/datasets/Pima+Indians+Diabetes
"""


def print_dataset_recommendations():
    """Print detailed dataset recommendations."""
    print(DATASET_RECOMMENDATIONS)


# ============================================================
# Dataset Loaders
# ============================================================

class ILPDataset:
    """
    Loader and preprocessor for ILP-suitable datasets.
    """

    @staticmethod
    def load_monks(dataset_id: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
        """
        Load one of the MONK's problems.

        Args:
            dataset_id: 1, 2, or 3

        Returns:
            X, y, feature_names, rule_description
        """
        # MONK's problem attributes
        feature_names = [
            'shape1', 'shape2',  # head_shape, body_shape
            'is_floating', 'has_tie'
        ]

        # These are simplified - actual Monk's has 6 attributes
        # For full implementation, we'd load from UCI

        if dataset_id == 1:
            # MONK-1 rule: (head_shape=round) OR (body_shape=round AND has_tie=yes)
            n = 124
            X = np.random.randint(0, 3, size=(n, 4))
            y = ((X[:, 0] == 0) | ((X[:, 1] == 0) & (X[:, 3] == 1))).astype(int)
            rule = "y=1 if (head_shape=round) OR (body_shape=round AND has_tie=yes)"

        elif dataset_id == 2:
            # MONK-2: Exactly two of the attributes have value 1
            n = 169
            X = np.random.randint(0, 3, size=(n, 4))
            y = (np.sum(X == 1, axis=1) == 2).astype(int)
            rule = "y=1 if exactly two attributes equal 1"

        else:  # MONK-3
            # MONK-3: (shape!=round) OR (body_shape!=round AND has_tie=yes)
            n = 122
            X = np.random.randint(0, 3, size=(n, 4))
            y = ((X[:, 0] != 0) | ((X[:, 1] != 0) & (X[:, 3] == 1))).astype(int)
            rule = "y=1 if (head_shape!=round) OR (body_shape!=round AND has_tie=yes)"

        return X, y, feature_names, rule

    @staticmethod
    def load_tic_tac_toe() -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load Tic-Tac-Toe endgame dataset.

        Board positions represented as 9 attributes (top-left to bottom-right).
        Values: x (player X), o (player O), b (blank)
        Target: player X wins (True) or loses (False)

        Returns:
            X (one-hot encoded), y, feature_names
        """
        # Generate synthetic tic-tac-toe data for demonstration
        n = 500
        feature_names = []

        # 9 positions, each can be x, o, or b
        X_raw = np.random.choice(['x', 'o', 'b'], size=(n, 9))

        # One-hot encode: 9 positions * 3 possible values = 27 features
        X = np.zeros((n, 27))
        for i in range(9):
            for j, val in enumerate(['x', 'o', 'b']):
                col_idx = i * 3 + j
                X[:, col_idx] = (X_raw[:, i] == val).astype(int)
                feature_names.append(f'pos{i}_{val}')

        # Simple win condition: x in top row
        y = ((X_raw[:, 0] == 'x') & (X_raw[:, 1] == 'x') & (X_raw[:, 2] == 'x')).astype(int)

        return X, y, feature_names

    @staticmethod
    def load_mushroom() -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load Mushroom dataset (edible vs poisonous).

        Key attributes:
        - cap-shape, cap-surface, cap-color
        - odor (very predictive)
        - gill-attachment, gill-size, gill-color
        - stalk-surface-above-ring, stalk-surface-below-ring
        - spore-print-color
        - population, habitat

        Returns:
            X (encoded), y, feature_names
        """
        # Generate synthetic mushroom data for demonstration
        n = 1000

        # Key rule: mushrooms with almond or no odor are edible
        # Those with foul, creosote, fishy, spicy, pungent are poisonous
        feature_names = [
            'cap_shape_bell', 'cap_shape_conical', 'cap_shape_convex',
            'cap_shape_flat', 'cap_shape_knobbled', 'cap_shape_sunken',
            'odor_almond', 'odor_anise', 'odor_creosote', 'odor_fishy',
            'odor_foul', 'odor_musty', 'odor_none', 'odor_pungent', 'odor_spicy'
        ]

        X = np.random.randint(0, 2, size=(n, len(feature_names)))

        # Rule: edible if (odor_almond OR odor_anise OR odor_none)
        # AND not (foul, fishy, etc.)
        y = ((X[:, 6] == 1) | (X[:, 7] == 1) | (X[:, 12] == 1)).astype(int)

        return X, y, feature_names

    @staticmethod
    def load_congressional_voting() -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load Congressional Voting Records dataset.

        16 votes on various issues (yes/no/unknown)
        Target: Democrat vs Republican

        Key patterns:
        - Republicans tend to vote yes on budget resolution, etc.
        - Democrats tend to vote yes on education spending, etc.

        Returns:
            X (encoded), y, feature_names
        """
        n = 435

        feature_names = [
            # Republican-leaning votes (historically)
            'voted_yes_budget', 'voted_yes_physician_fee',
            'voted_yes_elsalvador', 'voted_yes_religious_groups',
            'voted_yes_anti_satellite', 'voted_yes_aid_contras',
            'voted_yes_mx_missile', 'voted_yes_immigration',
            'voted_yes_synfuels', 'voted_yes_education',
            'voted_yes_superfund', 'voted_yes_crime',
            'voted_yes_duty_free', 'voted_yes_export_admin',
            # Democrat-leaning
            'handicapped_infants', 'water_project'
        ]

        # Generate synthetic voting data
        X = np.random.randint(0, 3, size=(n, len(feature_names)))  # 0=no, 1=yes, 2=unknown
        X[X == 2] = 0  # Treat unknown as no for simplicity

        # Simplified party affiliation rule
        # Republican if voted yes on budget AND physician fee
        y = ((X[:, 0] == 1) & (X[:, 1] == 1)).astype(int)

        return X, y, feature_names

    @staticmethod
    def load_balance_scale() -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load Balance Scale dataset.

        Attributes:
        - Left weight, Left distance
        - Right weight, Right distance

        Rule: Balance if L*dist_L == R*dist_R
        Target: L (left heavy), B (balanced), R (right heavy)

        Returns:
            X, y (binary), feature_names, rule_description
        """
        n = 625

        # Generate synthetic balance scale data
        left_weight = np.random.randint(1, 6, n)
        left_dist = np.random.randint(1, 6, n)
        right_weight = np.random.randint(1, 6, n)
        right_dist = np.random.randint(1, 6, n)

        # Binary classification: left-heavy vs not-left-heavy
        y = (left_weight * left_dist > right_weight * right_dist).astype(int)

        X = np.column_stack([left_weight, left_dist, right_weight, right_dist])
        feature_names = ['left_weight', 'left_distance', 'right_weight', 'right_distance']
        rule = "left_heavy if (left_weight * left_dist) > (right_weight * right_dist)"

        return X, y, feature_names, rule


def create_binary_ilp_dataset(rule_type: str = "mixed",
                              n_samples: int = 1000,
                              n_features: int = 10) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """
    Create a synthetic binary dataset with learnable logical rules.

    Args:
        rule_type: Type of logical rule
        - "conjunction": AND of conditions
        - "disjunction": OR of conditions
        - "dnf": Disjunctive Normal Form (OR of ANDs)
        - "mixed": Mix of positive and negative conditions
        - "hierarchical": Hierarchical rules
        n_samples: Number of samples
        n_features: Number of features

    Returns:
        X, y, feature_names, rule_description
    """
    feature_names = [f"x{i}" for i in range(n_features)]
    X = np.random.randint(0, 2, size=(n_samples, n_features))

    if rule_type == "conjunction":
        # y = 1 if x1=1 AND x2=1 AND x3=1
        y = (X[:, 0] & X[:, 1] & X[:, 2]).astype(int)
        rule = f"y=1 if {feature_names[0]}=1 AND {feature_names[1]}=1 AND {feature_names[2]}=1"

    elif rule_type == "disjunction":
        # y = 1 if x1=1 OR x2=1 OR x3=1
        y = (X[:, 0] | X[:, 1] | X[:, 2]).astype(int)
        rule = f"y=1 if {feature_names[0]}=1 OR {feature_names[1]}=1 OR {feature_names[2]}=1"

    elif rule_type == "dnf":
        # y = 1 if (x1=1 AND x2=1) OR (x3=1 AND x4=1) OR (x5=0 AND x6=0)
        y = ((X[:, 0] & X[:, 1]) | (X[:, 2] & X[:, 3]) |
             ((1 - X[:, 4]) & (1 - X[:, 5]))).astype(int)
        rule = (f"y=1 if ({feature_names[0]}=1 AND {feature_names[1]}=1) OR "
                f"({feature_names[2]}=1 AND {feature_names[3]}=1) OR "
                f"({feature_names[4]}=0 AND {feature_names[5]}=0)")

    elif rule_type == "mixed":
        # Mix of positive and negative: y = 1 if x1=1 AND x2=0 AND NOT x3=1
        y = (X[:, 0] & (1 - X[:, 1]) & (1 - X[:, 2])).astype(int)
        rule = f"y=1 if {feature_names[0]}=1 AND {feature_names[1]}=0 AND NOT {feature_names[2]}=1"

    elif rule_type == "hierarchical":
        # y = 1 if (x1=1 AND (x2=1 OR x3=1)) OR (x4=0 AND x5=1)
        y = ((X[:, 0] & (X[:, 1] | X[:, 2])) |
             ((1 - X[:, 3]) & X[:, 4])).astype(int)
        rule = (f"y=1 if ({feature_names[0]}=1 AND ({feature_names[1]}=1 OR {feature_names[2]}=1)) OR "
                f"({feature_names[3]}=0 AND {feature_names[4]}=1)")

    else:
        raise ValueError(f"Unknown rule_type: {rule_type}")

    return X, y, feature_names, rule


def load_uci_dataset_url(url: str) -> pd.DataFrame:
    """
    Load a dataset directly from UCI repository URL.

    Args:
        url: Direct URL to the dataset file

    Returns:
        DataFrame with the dataset
    """
    try:
        response = urlopen(url)
        data = pd.read_csv(io.StringIO(response.read().decode('utf-8')))
        return data
    except Exception as e:
        print(f"Failed to load dataset from {url}: {e}")
        return None


def discretize_continuous_features(X: np.ndarray, n_bins: int = 3,
                                   method: str = "quantile") -> np.ndarray:
    """
    Discretize continuous features for ILP.

    Args:
        X: Continuous feature matrix
        n_bins: Number of bins
        method: "quantile" or "uniform" or "kmeans"

    Returns:
        Discretized feature matrix (integers)
    """
    from sklearn.preprocessing import KBinsDiscretizer

    encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=method)
    X_disc = encoder.fit_transform(X)
    return X_disc.astype(int)


# ============================================================
# Dataset Evaluator
# ============================================================

def evaluate_dataset_suitability(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """
    Evaluate how suitable a dataset is for ILP.

    Returns metrics:
    - class_balance: Ratio of positive class
    - feature_entropy: Average entropy of features
    - rule_complexity_estimate: Estimated minimum rule complexity
    - ilp_suitability_score: Overall suitability (0-1)
    """
    metrics = {}

    # Class balance
    metrics['class_balance'] = np.mean(y)

    # Feature entropy (lower is better for ILP)
    from scipy.stats import entropy
    feature_entropies = []
    for j in range(X.shape[1]):
        values, counts = np.unique(X[:, j], return_counts=True)
        feat_entropy = entropy(counts / len(X))
        feature_entropies.append(feat_entropy)
    metrics['avg_feature_entropy'] = np.mean(feature_entropies)

    # Estimated rule complexity (based on correlation)
    correlations = np.array([np.corrcoef(X[:, j], y)[0, 1] if X[:, j].std() > 0 else 0
                            for j in range(X.shape[1])])
    metrics['num_predictive_features'] = np.sum(np.abs(correlations) > 0.1)

    # Overall suitability
    balance_score = 1 - abs(metrics['class_balance'] - 0.5) * 2  # Best at 0.5
    entropy_score = 1 - min(metrics['avg_feature_entropy'] / 2, 1)
    predictive_score = min(metrics['num_predictive_features'] / X.shape[1], 1)

    metrics['ilp_suitability_score'] = (balance_score + entropy_score + predictive_score) / 3

    return metrics


# ============================================================
# Usage Examples
# ============================================================

if __name__ == "__main__":
    print("=== ILP Dataset Recommendations ===\n")
    print_dataset_recommendations()

    print("\n\n=== Testing Dataset Loaders ===\n")

    # Test MONK's problem
    X, y, names, rule = ILPDataset.load_monks(1)
    print(f"MONK-1: {X.shape}, {np.sum(y)} positive")
    print(f"Rule: {rule}")

    # Test Tic-Tac-Toe
    X, y, names = ILPDataset.load_tic_tac_toe()
    print(f"\nTic-Tac-Toe: {X.shape}, {np.sum(y)} positive")

    # Test Mushroom
    X, y, names = ILPDataset.load_mushroom()
    print(f"\nMushroom: {X.shape}, {np.sum(y)} positive")

    # Test Congressional Voting
    X, y, names = ILPDataset.load_congressional_voting()
    print(f"\nCongressional Voting: {X.shape}, {np.sum(y)} positive")

    # Test Balance Scale
    X, y, names, rule = ILPDataset.load_balance_scale()
    print(f"\nBalance Scale: {X.shape}, {np.sum(y)} positive")
    print(f"Rule: {rule}")

    # Test custom DNF dataset
    X, y, names, rule = create_binary_ilp_dataset("dnf", n_samples=1000)
    print(f"\nCustom DNF: {X.shape}, {np.sum(y)} positive")
    print(f"Rule: {rule}")

    # Evaluate suitability
    print("\n=== Dataset Suitability Evaluation ===")
    metrics = evaluate_dataset_suitability(X, y)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
