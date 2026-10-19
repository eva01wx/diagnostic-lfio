"""Extract logical rules from black box models."""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from typing import List, Dict, Tuple, Optional
import re


class Rule:
    """Represents a single logical rule."""

    def __init__(self, conditions: List[str], outcome: int, support: int, confidence: float):
        self.conditions = conditions  # List of condition strings
        self.outcome = outcome
        self.support = support  # Number of samples covered
        self.confidence = confidence  # Accuracy on covered samples

    def __str__(self):
        if not self.conditions:
            return f"IF True THEN y = {self.outcome} (conf: {self.confidence:.2f}, support: {self.support})"
        return f"IF {' AND '.join(self.conditions)} THEN y = {self.outcome} (conf: {self.confidence:.2f}, support: {self.support})"

    def to_dict(self):
        return {
            "conditions": self.conditions,
            "outcome": int(self.outcome),
            "support": int(self.support),
            "confidence": float(self.confidence)
        }


class DecisionTreeRuleExtractor:
    """Extract rules from a decision tree surrogate model."""

    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.feature_names = None

    def fit_surrogate(self, X: np.ndarray, y_blackbox: np.ndarray,
                      feature_names: Optional[List[str]] = None) -> DecisionTreeClassifier:
        """
        Fit a decision tree to mimic the black box model.

        Args:
            X: Input features
            y_blackbox: Predictions from the black box model
            feature_names: Names of features for readable rules

        Returns:
            Fitted decision tree
        """
        self.feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]

        self.tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        self.tree.fit(X, y_blackbox)

        # Compute fidelity (agreement with black box)
        fidelity = np.mean(self.tree.predict(X) == y_blackbox)
        print(f"Surrogate tree fidelity: {fidelity:.3f}")

        return self.tree

    def extract_rules(self) -> List[Rule]:
        """
        Extract rules from the fitted decision tree.

        Returns:
            List of Rule objects
        """
        if self.tree is None:
            raise ValueError("Tree not fitted. Call fit_surrogate first.")

        tree_ = self.tree.tree_
        feature_name = [
            self.feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        rules = []

        def recurse(node, conditions):
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                # Left child (feature <= threshold)
                left_conditions = conditions + [f"{name} <= {threshold:.2f}"]
                recurse(tree_.children_left[node], left_conditions)

                # Right child (feature > threshold)
                right_conditions = conditions + [f"{name} > {threshold:.2f}"]
                recurse(tree_.children_right[node], right_conditions)
            else:
                # Leaf node - create rule
                value = tree_.value[node][0]
                outcome = np.argmax(value)
                support = int(tree_.n_node_samples[node])
                confidence = value[outcome] / value.sum()

                rules.append(Rule(conditions, outcome, support, confidence))

        recurse(0, [])
        return rules

    def simplify_rules(self, rules: List[Rule], min_confidence: float = 0.9) -> List[Rule]:
        """Filter rules by confidence threshold."""
        return [r for r in rules if r.confidence >= min_confidence]


class RuleFitExtractor:
    """Extract rules using rule-fit-like approach."""

    def __init__(self, max_rules: int = 20, min_support: int = 10):
        self.max_rules = max_rules
        self.min_support = min_support
        self.rules = []

    def extract_rules_from_data(self, X: np.ndarray, y: np.ndarray,
                                 feature_names: Optional[List[str]] = None) -> List[Rule]:
        """
        Extract simple rule candidates from data.

        For binary features, creates rules like "x1 = 1"
        For continuous features, creates threshold rules
        """
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X.shape[1])]

        rules = []
        n_samples = X.shape[0]

        # Check if features are binary
        is_binary = all(np.all(np.isin(X[:, i], [0, 1])) for i in range(X.shape[1]))

        for feat_idx in range(X.shape[1]):
            feat_name = feature_names[feat_idx]
            feat_values = X[:, feat_idx]

            if is_binary:
                # Binary feature rules
                for val in [0, 1]:
                    mask = feat_values == val
                    if mask.sum() >= self.min_support:
                        for outcome in [0, 1]:
                            outcome_mask = y == outcome
                            support = (mask & outcome_mask).sum()
                            if support >= self.min_support:
                                confidence = support / mask.sum()
                                condition = f"{feat_name} = {val}"
                                rules.append(Rule([condition], outcome, support, confidence))
            else:
                # Continuous feature - find best thresholds
                unique_vals = np.unique(feat_values)
                thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

                for thresh in thresholds:
                    # Less than or equal rule
                    mask = feat_values <= thresh
                    if mask.sum() >= self.min_support:
                        for outcome in [0, 1]:
                            outcome_mask = y == outcome
                            support = (mask & outcome_mask).sum()
                            if support >= self.min_support:
                                confidence = support / mask.sum()
                                condition = f"{feat_name} <= {thresh:.2f}"
                                rules.append(Rule([condition], outcome, support, confidence))

                    # Greater than rule
                    mask = feat_values > thresh
                    if mask.sum() >= self.min_support:
                        for outcome in [0, 1]:
                            outcome_mask = y == outcome
                            support = (mask & outcome_mask).sum()
                            if support >= self.min_support:
                                confidence = support / mask.sum()
                                condition = f"{feat_name} > {thresh:.2f}"
                                rules.append(Rule([condition], outcome, support, confidence))

        # Create conjunctive rules (pairs of conditions)
        if len(rules) > 1:
            combined_rules = []
            for i, r1 in enumerate(rules):
                for r2 in rules[i+1:]:
                    # Combine conditions from different features
                    feats1 = set(c.split()[0] for c in r1.conditions)
                    feats2 = set(c.split()[0] for c in r2.conditions)
                    if feats1.isdisjoint(feats2):
                        # Check overlap in data
                        combined_conditions = r1.conditions + r2.conditions
                        # For simplicity, estimate support (would need actual data check)
                        estimated_support = min(r1.support, r2.support) // 2
                        if estimated_support >= self.min_support:
                            avg_confidence = (r1.confidence + r2.confidence) / 2
                            combined_rules.append(Rule(
                                combined_conditions, r1.outcome,
                                estimated_support, avg_confidence
                            ))
            rules.extend(combined_rules)

        # Sort by confidence and support
        rules.sort(key=lambda r: (r.confidence * r.support / n_samples), reverse=True)

        return rules[:self.max_rules]


class BooleanRuleExtractor:
    """Specialized extractor for boolean/binary data."""

    def __init__(self):
        self.rules = []

    def extract_dnf_rules(self, X: np.ndarray, y: np.ndarray,
                          feature_names: Optional[List[str]] = None) -> List[str]:
        """
        Extract DNF (Disjunctive Normal Form) rules for positive class.

        Returns list of rule strings like "(x1 AND x2) OR (x3 AND x4)"
        """
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X.shape[1])]

        positive_samples = X[y == 1]
        negative_samples = X[y == 0]

        if len(positive_samples) == 0:
            return ["False (no positive samples)"]

        # Find minimal conjuncts that cover positive but not negative samples
        rules = []

        # Start with single positive samples as rules
        for pos_sample in positive_samples:
            # Build condition for this exact sample
            conditions = []
            for i, val in enumerate(pos_sample):
                conditions.append(f"{feature_names[i]}={val}")

            # Check if this covers any negative samples
            covers_negative = False
            for neg_sample in negative_samples:
                if all(neg_sample[i] == val for i, val in enumerate(pos_sample)):
                    covers_negative = True
                    break

            if not covers_negative:
                rules.append(" AND ".join(conditions))

        # Try to generalize by removing conditions
        simplified_rules = []
        for rule in rules[:10]:  # Limit to avoid explosion
            conditions = rule.split(" AND ")
            simplified_rules.append(try_generalize(conditions, negative_samples))

        # Combine as DNF
        if simplified_rules:
            return f"({' OR '.join(simplified_rules)})"
        return "No simple rule found"


def try_generalize(conditions: List[str], negative_samples: np.ndarray,
                   max_iterations: int = 100) -> str:
    """Try to remove conditions while still excluding negative samples."""
    current_conditions = conditions.copy()

    for _ in range(max_iterations):
        improved = False
        for i, cond in enumerate(current_conditions):
            # Try removing this condition
            test_conditions = current_conditions[:i] + current_conditions[i+1:]

            # Check if any negative sample satisfies remaining conditions
            valid = True
            for neg_sample in negative_samples:
                if all(check_condition(c, neg_sample) for c in test_conditions):
                    valid = False
                    break

            if valid:
                current_conditions = test_conditions
                improved = True
                break

        if not improved:
            break

    return " AND ".join(current_conditions)


def check_condition(condition: str, sample: np.ndarray) -> bool:
    """Check if a sample satisfies a condition."""
    # Parse condition like "x0=1" or "x1 <= 0.5"
    if "=" in condition and "<" not in condition and ">" not in condition:
        feat, val = condition.split("=")
        feat_idx = int(feat[1:])
        return sample[feat_idx] == int(val)
    elif "<=" in condition:
        feat, val = condition.split("<=")
        feat_idx = int(feat[1:])
        return sample[feat_idx] <= float(val)
    elif ">" in condition:
        feat, val = condition.split(">")
        feat_idx = int(feat[1:])
        return sample[feat_idx] > float(val)
    return True


class RuleExplainer:
    """Main interface for rule extraction and explanation."""

    def __init__(self, method: str = "decision_tree"):
        self.method = method
        self.extractor = None
        self.rules = []

    def extract_from_blackbox(self, X: np.ndarray, blackbox_model,
                              feature_names: Optional[List[str]] = None) -> List[Rule]:
        """
        Extract rules from a black box model's predictions.

        Args:
            X: Input features
            blackbox_model: Model with predict() method
            feature_names: Optional feature names

        Returns:
            List of extracted rules
        """
        # Get black box predictions
        y_pred = blackbox_model.predict(X)

        if self.method == "decision_tree":
            self.extractor = DecisionTreeRuleExtractor(max_depth=5)
            self.extractor.fit_surrogate(X, y_pred, feature_names)
            self.rules = self.extractor.extract_rules()

        elif self.method == "rulefit":
            self.extractor = RuleFitExtractor(max_rules=30)
            self.rules = self.extractor.extract_rules_from_data(X, y_pred, feature_names)

        return self.rules

    def get_rule_summary(self) -> Dict:
        """Get summary statistics of extracted rules."""
        if not self.rules:
            return {}

        return {
            "num_rules": len(self.rules),
            "avg_conditions": np.mean([len(r.conditions) for r in self.rules]),
            "avg_confidence": np.mean([r.confidence for r in self.rules]),
            "total_support": sum([r.support for r in self.rules])
        }

    def print_rules(self, max_rules: int = 10):
        """Print extracted rules in readable format."""
        print(f"\n=== Extracted Rules ({self.method}) ===")
        for i, rule in enumerate(self.rules[:max_rules]):
            print(f"{i+1}. {rule}")

        summary = self.get_rule_summary()
        print(f"\nSummary: {summary}")


def simplify_boolean_rule(rule_string: str) -> str:
    """
    Simplify a boolean rule string.

    Example: "x0=1 AND x1=1 AND x2=1" -> "ALL(x0, x1, x2)"
    """
    # Count conditions
    conditions = rule_string.split(" AND ")

    if len(conditions) == 1:
        return conditions[0]

    # Check if all are x_i=1
    if all(c.endswith("=1") for c in conditions):
        feats = [c.split("=")[0] for c in conditions]
        if len(feats) == len(conditions):
            return f"ALL({', '.join(feats)})"

    # Check if all are x_i=0
    if all(c.endswith("=0") for c in conditions):
        feats = [c.split("=")[0] for c in conditions]
        if len(feats) == len(conditions):
            return f"NONE({', '.join(feats)})"

    return rule_string


if __name__ == "__main__":
    # Test rule extraction
    from data_generator import SyntheticDataGenerator
    from black_box import create_black_box, BlackBoxTrainer

    print("=== Testing Rule Extraction ===")

    # Generate data with known rule
    generator = SyntheticDataGenerator(n_samples=500, n_features=5)
    X, y, true_rule = generator.generate_boolean_dataset(rule_type="complex")
    print(f"True rule: {true_rule}")

    # Train black box
    model = create_black_box(input_dim=5, hidden_dims=[32, 16])
    trainer = BlackBoxTrainer(model)
    trainer.train(X, y, epochs=50)

    # Extract rules
    feature_names = generator.create_feature_names()

    print("\n--- Decision Tree Method ---")
    explainer = RuleExplainer(method="decision_tree")
    rules = explainer.extract_from_blackbox(X, model, feature_names)
    explainer.print_rules()

    print("\n--- RuleFit Method ---")
    explainer2 = RuleExplainer(method="rulefit")
    rules2 = explainer2.extract_from_blackbox(X, model, feature_names)
    explainer2.print_rules()
