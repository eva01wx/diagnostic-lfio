"""Inductive Logic Programming (ILP) based rule learner."""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Callable
from dataclasses import dataclass
import itertools


@dataclass
class ILPRule:
    """Represents an ILP learned rule."""
    antecedent: Callable[[np.ndarray], bool]  # Function that checks if rule applies
    description: str  # Human-readable description
    coverage: float  # Fraction of positive examples covered
    precision: float  # Fraction of covered examples that are positive
    positive_examples: int
    total_examples: int

    def __str__(self):
        return f"{self.description} (coverage: {self.coverage:.2f}, precision: {self.precision:.2f})"


class ILPLearner:
    """
    Inductive Logic Programming learner for extracting logical rules.

    This implements a simplified ILP approach that searches for rules
    that cover positive examples while excluding negative examples.
    """

    def __init__(self, max_rule_length: int = 4, min_coverage: float = 0.1,
                 min_precision: float = 0.9, beam_width: int = 10):
        self.max_rule_length = max_rule_length
        self.min_coverage = min_coverage
        self.min_precision = min_precision
        self.beam_width = beam_width
        self.rules: List[ILPRule] = []
        self.feature_names: Optional[List[str]] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> List[ILPRule]:
        """
        Learn ILP rules from data.

        Args:
            X: Feature matrix (binary or continuous)
            y: Target labels
            feature_names: Names of features

        Returns:
            List of learned rules
        """
        self.feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]
        self.rules = []

        positive_mask = y == 1
        negative_mask = y == 0

        positive_examples = X[positive_mask]
        negative_examples = X[negative_mask]

        print(f"Learning rules from {len(positive_examples)} positive "
              f"and {len(negative_examples)} negative examples")

        # Sequential covering: learn rules iteratively
        uncovered_pos = set(range(len(positive_examples)))

        iteration = 0
        max_iterations = 20

        while uncovered_pos and iteration < max_iterations:
            iteration += 1
            print(f"\nIteration {iteration}: {len(uncovered_pos)} positive examples remaining")

            # Find best rule for remaining positive examples
            best_rule = self._find_best_rule(
                positive_examples[list(uncovered_pos)],
                negative_examples
            )

            if best_rule is None:
                print("No more rules found that meet criteria")
                break

            self.rules.append(best_rule)

            # Mark covered positive examples
            covered_indices = []
            for idx in uncovered_pos:
                if best_rule.antecedent(positive_examples[idx]):
                    covered_indices.append(idx)

            for idx in covered_indices:
                uncovered_pos.remove(idx)

            print(f"  Rule covers {len(covered_indices)} positive examples")

        return self.rules

    def _find_best_rule(self, positive_examples: np.ndarray,
                        negative_examples: np.ndarray) -> Optional[ILPRule]:
        """
        Find the best rule using beam search.

        Returns the rule with best combination of coverage and precision.
        """
        is_binary = all(np.all(np.isin(positive_examples[:, i], [0, 1]))
                       for i in range(positive_examples.shape[1]))

        # Initialize with single literals
        candidates = self._generate_initial_literals(positive_examples, is_binary)

        best_rule = None
        best_score = -1

        for depth in range(self.max_rule_length):
            new_candidates = []

            for candidate_fn, description in candidates:
                # Evaluate candidate
                pos_covered = sum(1 for ex in positive_examples if candidate_fn(ex))
                neg_covered = sum(1 for ex in negative_examples if candidate_fn(ex))

                if pos_covered == 0:
                    continue

                coverage = pos_covered / len(positive_examples)
                precision = pos_covered / (pos_covered + neg_covered) if (pos_covered + neg_covered) > 0 else 0

                # Check if meets minimum criteria
                if coverage >= self.min_coverage and precision >= self.min_precision:
                    score = coverage * precision

                    if score > best_score:
                        best_score = score
                        best_rule = ILPRule(
                            antecedent=candidate_fn,
                            description=description,
                            coverage=coverage,
                            precision=precision,
                            positive_examples=pos_covered,
                            total_examples=pos_covered + neg_covered
                        )

                # Specialize: add more conditions
                if depth < self.max_rule_length - 1:
                    specialized = self._specialize_rule(
                        candidate_fn, description, positive_examples, negative_examples, is_binary
                    )
                    new_candidates.extend(specialized)

            # Keep best candidates (beam search)
            candidates = self._select_best_candidates(new_candidates, self.beam_width)

            if not candidates:
                break

        return best_rule

    def _generate_initial_literals(self, examples: np.ndarray,
                                    is_binary: bool) -> List[Tuple[Callable, str]]:
        """Generate initial literal candidates."""
        candidates = []
        n_features = examples.shape[1]

        if is_binary:
            # For binary features: x_i = 0, x_i = 1
            for i in range(n_features):
                for val in [0, 1]:
                    def make_condition(idx=i, value=val):
                        return lambda x: x[idx] == value
                    candidates.append((make_condition(), f"{self.feature_names[i]} = {val}"))
        else:
            # For continuous features: x_i <= threshold
            for i in range(n_features):
                values = examples[:, i]
                thresholds = np.percentile(values, [25, 50, 75])

                for thresh in thresholds:
                    def make_leq(idx=i, t=thresh):
                        return lambda x: x[idx] <= t
                    def make_gt(idx=i, t=thresh):
                        return lambda x: x[idx] > t

                    candidates.append((make_leq(), f"{self.feature_names[i]} <= {thresh:.2f}"))
                    candidates.append((make_gt(), f"{self.feature_names[i]} > {thresh:.2f}"))

        return candidates

    def _specialize_rule(self, base_fn: Callable, base_desc: str,
                         pos_examples: np.ndarray, neg_examples: np.ndarray,
                         is_binary: bool) -> List[Tuple[Callable, str]]:
        """Specialize a rule by adding more conditions."""
        candidates = []
        n_features = pos_examples.shape[1]

        # Get features already used in description
        used_features = set()
        parts = base_desc.split(" AND ")
        for part in parts:
            feat = part.split()[0]
            used_features.add(feat)

        # Add conditions on unused features
        for i in range(n_features):
            feat_name = self.feature_names[i]
            if feat_name in used_features:
                continue

            if is_binary:
                for val in [0, 1]:
                    def make_new_condition(base=base_fn, idx=i, value=val):
                        return lambda x: base(x) and x[idx] == value
                    new_desc = f"{base_desc} AND {feat_name} = {val}"
                    candidates.append((make_new_condition(), new_desc))
            else:
                values = np.concatenate([pos_examples[:, i], neg_examples[:, i]])
                threshold = np.median(values)

                def make_leq(base=base_fn, idx=i, t=threshold):
                    return lambda x: base(x) and x[idx] <= t
                def make_gt(base=base_fn, idx=i, t=threshold):
                    return lambda x: base(x) and x[idx] > t

                candidates.append((make_leq(), f"{base_desc} AND {feat_name} <= {threshold:.2f}"))
                candidates.append((make_gt(), f"{base_desc} AND {feat_name} > {threshold:.2f}"))

        return candidates

    def _select_best_candidates(self, candidates: List[Tuple[Callable, str]],
                                 k: int) -> List[Tuple[Callable, str]]:
        """Select k best candidates based on coverage of positive examples."""
        # This is a simplified selection - in practice, you'd evaluate on actual data
        return candidates[:k] if len(candidates) > k else candidates

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using learned rules."""
        predictions = np.zeros(len(X), dtype=int)

        for i, example in enumerate(X):
            for rule in self.rules:
                if rule.antecedent(example):
                    predictions[i] = 1
                    break

        return predictions

    def get_rule_dnf(self) -> str:
        """Get rules in Disjunctive Normal Form."""
        if not self.rules:
            return "No rules learned"

        antecedents = [rule.description for rule in self.rules]
        return f"({' OR '.join(antecedents)}) THEN y = 1"

    def print_rules(self):
        """Print learned rules."""
        print("\n=== ILP Learned Rules ===")
        if not self.rules:
            print("No rules learned")
            return

        for i, rule in enumerate(self.rules, 1):
            print(f"{i}. IF {rule.description} THEN y = 1")
            print(f"   Coverage: {rule.coverage:.2%}, Precision: {rule.precision:.2%}")

        print(f"\nDNF Form: {self.get_rule_dnf()}")


class BooleanILPLearner(ILPLearner):
    """Specialized ILP learner for Boolean/binary data."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def learn_boolean_rules(self, X: np.ndarray, y: np.ndarray) -> List[Dict]:
        """
        Learn Boolean rules and return them in a structured format.

        Returns:
            List of dicts with 'terms' (list of (feature, value) pairs)
        """
        self.fit(X, y)

        structured_rules = []
        for rule in self.rules:
            terms = []
            # Parse description to extract terms
            parts = rule.description.split(" AND ")
            for part in parts:
                part = part.strip()
                if "=" in part:
                    feat, val = part.split("=")
                    terms.append((feat.strip(), int(val.strip())))
                elif "<=" in part:
                    feat, val = part.split("<=")
                    terms.append((feat.strip(), f"<={float(val.strip())}"))
                elif ">" in part:
                    feat, val = part.split(">")
                    terms.append((feat.strip(), f">{float(val.strip())}"))

            structured_rules.append({
                "terms": terms,
                "coverage": rule.coverage,
                "precision": rule.precision
            })

        return structured_rules

    def to_python_expression(self, rule_idx: int = None) -> str:
        """
        Convert learned rules to Python expression.

        Args:
            rule_idx: If specified, only convert that rule; otherwise convert all as OR

        Returns:
            Python expression string
        """
        if not self.rules:
            return "False"

        def rule_to_expr(rule: ILPRule) -> str:
            # Replace " AND " with " and "
            expr = rule.description.replace(" AND ", " and ")
            # Fix feature names to be valid Python
            for i, name in enumerate(self.feature_names or []):
                expr = expr.replace(name, f"x[{i}]")
            return expr

        if rule_idx is not None:
            return rule_to_expr(self.rules[rule_idx])

        # Combine all rules with OR
        rule_exprs = [rule_to_expr(rule) for rule in self.rules]
        return f"({' or '.join(rule_exprs)})"


def compare_rules(ilp_rules: List[ILPRule], dt_rules: List) -> Dict:
    """Compare ILP rules with decision tree rules."""
    return {
        "ilp_count": len(ilp_rules),
        "dt_count": len(dt_rules),
        "ilp_avg_length": np.mean([len(r.description.split(" AND ")) for r in ilp_rules]) if ilp_rules else 0,
        "ilp_avg_precision": np.mean([r.precision for r in ilp_rules]) if ilp_rules else 0
    }


if __name__ == "__main__":
    # Test ILP learner
    from data_generator import SyntheticDataGenerator

    print("=== Testing ILP Learner ===")

    # Generate boolean data with complex rule
    generator = SyntheticDataGenerator(n_samples=500, n_features=5)
    X, y, true_rule = generator.generate_boolean_dataset(rule_type="complex")
    feature_names = generator.create_feature_names()

    print(f"True rule: {true_rule}")
    print(f"Features: {feature_names}")

    # Learn with ILP
    ilp = ILPLearner(max_rule_length=4, min_coverage=0.05, min_precision=0.9)
    ilp.fit(X, y, feature_names)
    ilp.print_rules()

    # Evaluate accuracy
    predictions = ilp.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nILP Rule Accuracy: {accuracy:.3f}")

    # Try with BooleanILPLearner
    print("\n=== Boolean ILP Learner ===")
    boolean_ilp = BooleanILPLearner(max_rule_length=3)
    structured_rules = boolean_ilp.learn_boolean_rules(X, y)

    print("Structured Rules:")
    for i, rule in enumerate(structured_rules, 1):
        print(f"{i}. {rule}")

    print(f"\nPython Expression:\n{boolean_ilp.to_python_expression()}")
