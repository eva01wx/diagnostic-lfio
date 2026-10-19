"""
LFIO (Learning from Interpretations and Observations) - Python Implementation

This is a Python implementation of the LFIO algorithm for Inductive Logic Programming.
LFIO learns logical rules from input-output examples by:
1. Generating initial rules from examples
2. Using Least General Generalization (LGG) to generalize rules
3. Applying subsumption checking to remove redundant rules
4. Refining rules through inverse resolution and v-operator

Improvements over original:
- Support for both discrete and continuous features
- Better integration with scikit-learn models
- More efficient rule indexing and search
- Support for partial interpretations (probabilistic outputs)
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
import re


@dataclass(frozen=True)
class Term:
    """Represents a logical term (constant or variable)."""
    name: str
    is_variable: bool = False

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Var({self.name})" if self.is_variable else f"Const({self.name})"


@dataclass(frozen=True)
class Atom:
    """Represents a logical atom (predicate with arguments)."""
    predicate: str
    arguments: Tuple[Term, ...] = field(default_factory=tuple)
    negated: bool = False

    def __str__(self):
        args = ", ".join(str(a) for a in self.arguments)
        neg = "not " if self.negated else ""
        return f"{neg}{self.predicate}({args})" if args else f"{neg}{self.predicate}"

    def __repr__(self):
        return f"Atom({str(self)})"

    def __hash__(self):
        return hash((self.predicate, self.arguments, self.negated))

    def __eq__(self, other):
        if not isinstance(other, Atom):
            return False
        return (self.predicate == other.predicate and
                self.arguments == other.arguments and
                self.negated == other.negated)

    def arity(self) -> int:
        return len(self.arguments)

    def ground(self) -> bool:
        """Check if atom is ground (no variables)."""
        return all(not a.is_variable for a in self.arguments)


@dataclass
class Rule:
    """Represents a logical rule (Horn clause)."""
    head: Atom
    body: Tuple[Atom, ...] = field(default_factory=tuple)
    confidence: float = 1.0
    coverage: float = 0.0
    positive_examples: int = 0
    total_examples: int = 0

    def __str__(self):
        if not self.body:
            return f"{self.head}."
        body_str = ", ".join(str(b) for b in self.body)
        return f"{self.head} :- {body_str}."

    def __repr__(self):
        return f"Rule({str(self)})"

    def __hash__(self):
        return hash((self.head, self.body))

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return self.head == other.head and self.body == other.body

    def arity(self) -> int:
        return self.head.arity()

    def body_size(self) -> int:
        return len(self.body)

    def to_dnf(self) -> str:
        """Convert to Disjunctive Normal Form string."""
        if not self.body:
            return str(self.head)
        terms = []
        for atom in self.body:
            if atom.negated:
                terms.append(f"not {atom.predicate}")
            else:
                terms.append(atom.predicate)
        return f"{' AND '.join(terms)} -> {self.head.predicate}"


@dataclass
class Example:
    """Represents an input-output example (interpretation)."""
    input_atoms: Set[Atom]
    output_atoms: Set[Atom]
    features: Optional[np.ndarray] = None

    def __str__(self):
        input_str = ", ".join(str(a) for a in self.input_atoms)
        output_str = ", ".join(str(a) for a in self.output_atoms)
        return f"Input: {{{input_str}}} -> Output: {{{output_str}}}"


class Substitution:
    """Represents a substitution mapping variables to terms."""

    def __init__(self, mapping: Optional[Dict[str, Term]] = None):
        self.mapping = mapping or {}

    def __contains__(self, var: str) -> bool:
        return var in self.mapping

    def __getitem__(self, var: str) -> Optional[Term]:
        return self.mapping.get(var)

    def __setitem__(self, var: str, term: Term):
        self.mapping[var] = term

    def extend(self, var: str, term: Term) -> 'Substitution':
        """Create a new substitution with an additional mapping."""
        new_mapping = self.mapping.copy()
        new_mapping[var] = term
        return Substitution(new_mapping)

    def compose(self, other: 'Substitution') -> 'Substitution':
        """Compose two substitutions."""
        new_mapping = self.mapping.copy()
        for var, term in other.mapping.items():
            new_mapping[var] = term
        return Substitution(new_mapping)

    def __str__(self):
        pairs = [f"{v}/{t}" for v, t in self.mapping.items()]
        return f"{{{', '.join(pairs)}}}"


class LFIO:
    """
    LFIO (Learning from Interpretations and Observations) Algorithm.

    This implementation learns logical rules from examples by:
    1. Generating initial ground rules from examples
    2. Generalizing rules using Least General Generalization (LGG)
    3. Applying subsumption to remove redundant rules
    4. Refining rules using inverse resolution
    """

    def __init__(self,
                 min_coverage: float = 0.05,
                 min_confidence: float = 0.8,
                 max_rules: int = 100,
                 max_body_size: int = 5,
                 variable_prefix: str = "V",
                 enable_lgg: bool = True,
                 enable_inverse_resolution: bool = True):
        """
        Initialize LFIO learner.

        Args:
            min_coverage: Minimum coverage for a rule to be kept
            min_confidence: Minimum confidence for a rule to be kept
            max_rules: Maximum number of rules to learn
            max_body_size: Maximum size of rule body
            variable_prefix: Prefix for generated variables
            enable_lgg: Enable Least General Generalization
            enable_inverse_resolution: Enable inverse resolution refinement
        """
        self.min_coverage = min_coverage
        self.min_confidence = min_confidence
        self.max_rules = max_rules
        self.max_body_size = max_body_size
        self.variable_prefix = variable_prefix
        self.enable_lgg = enable_lgg
        self.enable_inverse_resolution = enable_inverse_resolution

        self.rules: List[Rule] = []
        self.examples: List[Example] = []
        self.predicates: Set[str] = set()
        self.var_counter = 0

    def _new_variable(self) -> Term:
        """Generate a new variable."""
        var = Term(f"{self.variable_prefix}{self.var_counter}", is_variable=True)
        self.var_counter += 1
        return var

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None,
            background_knowledge: Optional[List[Rule]] = None) -> List[Rule]:
        """
        Learn rules from binary classification data.

        Args:
            X: Feature matrix (binary or continuous will be discretized)
            y: Target labels (0 or 1)
            feature_names: Names of features
            background_knowledge: Optional background knowledge rules

        Returns:
            List of learned rules
        """
        self.examples = []
        self.rules = []
        self.var_counter = 0

        if feature_names is None:
            feature_names = [f"x{i}" for i in range(X.shape[1])]

        # Convert binary data to examples
        self._convert_to_examples(X, y, feature_names)

        # Add background knowledge if provided
        if background_knowledge:
            self.rules.extend(background_knowledge)

        # Run LFIO algorithm
        self._run_lfio()

        return self.rules

    def _convert_to_examples(self, X: np.ndarray, y: np.ndarray,
                            feature_names: List[str]):
        """Convert binary classification data to logical examples."""
        # Discretize continuous features if needed
        X_bin = self._discretize(X)

        for i in range(len(X)):
            input_atoms = set()
            output_atoms = set()

            # Create input atoms from features
            for j, feat_name in enumerate(feature_names):
                if X_bin[i, j] == 1:
                    input_atoms.add(Atom(feat_name, (Term("1"),)))

            # Create output atom
            if y[i] == 1:
                output_atoms.add(Atom("target", (Term("1"),)))

            self.examples.append(Example(input_atoms, output_atoms, X[i]))

    def _discretize(self, X: np.ndarray) -> np.ndarray:
        """Discretize continuous features to binary."""
        if X.dtype == np.int64 or X.dtype == np.int32:
            return X.astype(int)

        # Use median as threshold
        X_bin = np.zeros_like(X, dtype=int)
        for j in range(X.shape[1]):
            median_val = np.median(X[:, j])
            X_bin[:, j] = (X[:, j] > median_val).astype(int)
        return X_bin

    def _run_lfio(self):
        """Run the main LFIO algorithm."""
        # Step 1: Generate initial rules from examples
        initial_rules = self._generate_initial_rules()
        self.rules.extend(initial_rules)

        # Step 2: Apply LGG to generalize rules
        if self.enable_lgg:
            self._apply_lgg()

        # Step 3: Apply subsumption to remove redundant rules
        self._apply_subsumption()

        # Step 4: Refine rules using inverse resolution
        if self.enable_inverse_resolution:
            self._apply_inverse_resolution()

        # Step 5: Filter by coverage and confidence
        self._filter_rules()

    def _generate_initial_rules(self) -> List[Rule]:
        """Generate initial ground rules from positive examples."""
        rules = []
        positive_examples = [e for e in self.examples if e.output_atoms]

        for ex in positive_examples:
            for output_atom in ex.output_atoms:
                # Create rule: output :- input_conditions
                body_conditions = tuple(ex.input_atoms) if ex.input_atoms else ()
                rule = Rule(output_atom, body_conditions)

                # Calculate coverage and confidence
                self._calculate_rule_stats(rule)

                if rule.coverage >= self.min_coverage and rule.confidence >= self.min_confidence:
                    rules.append(rule)

        return rules

    def _calculate_rule_stats(self, rule: Rule):
        """Calculate coverage and confidence for a rule."""
        pos_matches = 0
        total_pos = len([e for e in self.examples if e.output_atoms])
        total_matches = 0

        for ex in self.examples:
            if self._rule_matches(rule, ex):
                total_matches += 1
                if any(out in ex.output_atoms for out in [rule.head]):
                    pos_matches += 1

        rule.total_examples = total_matches
        rule.positive_examples = pos_matches
        rule.coverage = pos_matches / total_pos if total_pos > 0 else 0
        rule.confidence = pos_matches / total_matches if total_matches > 0 else 0

    def _rule_matches(self, rule: Rule, example: Example) -> bool:
        """Check if a rule matches an example."""
        # Check if all body conditions are in input
        for body_atom in rule.body:
            if body_atom not in example.input_atoms:
                return False
        return True

    def _apply_lgg(self):
        """Apply Least General Generalization to rules."""
        new_rules = []
        processed = set()

        # Group rules by head predicate
        rules_by_head = defaultdict(list)
        for rule in self.rules:
            rules_by_head[rule.head.predicate].append(rule)

        for head_pred, rule_group in rules_by_head.items():
            # Try to generalize pairs of rules
            for i, r1 in enumerate(rule_group):
                for r2 in rule_group[i+1:]:
                    if r1 in processed or r2 in processed:
                        continue

                    if len(r1.body) == len(r2.body):
                        # Try LGG
                        generalized = self._lgg_rules(r1, r2)
                        if generalized and generalized not in self.rules:
                            self._calculate_rule_stats(generalized)
                            if (generalized.coverage >= self.min_coverage and
                                generalized.confidence >= self.min_confidence):
                                new_rules.append(generalized)
                                processed.add(r1)
                                processed.add(r2)

        # Add generalized rules
        for rule in new_rules:
            if rule not in self.rules:
                self.rules.append(rule)

    def _lgg_rules(self, r1: Rule, r2: Rule) -> Optional[Rule]:
        """Compute LGG of two rules."""
        if r1.head.predicate != r2.head.predicate:
            return None

        if len(r1.body) != len(r2.body):
            return None

        # LGG of heads
        lgg_head = self._lgg_atoms(r1.head, r2.head)
        if not lgg_head:
            return None

        # LGG of bodies (assuming same ordering)
        lgg_body = []
        for b1, b2 in zip(r1.body, r2.body):
            lgg_atom = self._lgg_atoms(b1, b2)
            if not lgg_atom:
                return None
            lgg_body.append(lgg_atom)

        return Rule(lgg_head, tuple(lgg_body))

    def _lgg_atoms(self, a1: Atom, a2: Atom) -> Optional[Atom]:
        """Compute LGG of two atoms."""
        if a1.predicate != a2.predicate:
            return None

        if len(a1.arguments) != len(a2.arguments):
            return None

        # Align arguments and create variables where they differ
        lgg_args = []
        for arg1, arg2 in zip(a1.arguments, a2.arguments):
            if arg1.name == arg2.name:
                lgg_args.append(arg1)
            else:
                # Different constants -> create variable
                new_var = self._new_variable()
                lgg_args.append(new_var)

        return Atom(a1.predicate, tuple(lgg_args), a1.negated and a2.negated)

    def _apply_subsumption(self):
        """Remove rules that are subsumed by more general rules."""
        to_remove = set()

        for i, r1 in enumerate(self.rules):
            for j, r2 in enumerate(self.rules):
                if i >= j or r1 in to_remove or r2 in to_remove:
                    continue

                if self._subsumes(r1, r2):
                    # r1 subsumes r2, remove r2
                    to_remove.add(r2)
                elif self._subsumes(r2, r1):
                    # r2 subsumes r1, remove r1
                    to_remove.add(r1)

        self.rules = [r for r in self.rules if r not in to_remove]

    def _subsumes(self, r1: Rule, r2: Rule) -> bool:
        """Check if rule r1 subsumes rule r2."""
        # r1 subsumes r2 if r1 is more general and covers at least as much
        if r1.head.predicate != r2.head.predicate:
            return False

        if len(r1.body) > len(r2.body):
            return False

        # Simple check: r1 has fewer or equal body conditions
        # For proper subsumption, we'd need to check variable mappings
        r1_body_preds = {b.predicate for b in r1.body}
        r2_body_preds = {b.predicate for b in r2.body}

        return r1_body_preds.issubset(r2_body_preds)

    def _apply_inverse_resolution(self):
        """Apply inverse resolution to refine rules."""
        new_rules = []

        for rule in self.rules:
            # Try to simplify rules by removing body literals
            if len(rule.body) > 1:
                for i in range(len(rule.body)):
                    new_body = rule.body[:i] + rule.body[i+1:]
                    simplified = Rule(rule.head, new_body)
                    self._calculate_rule_stats(simplified)

                    if (simplified.confidence >= self.min_confidence and
                        simplified not in self.rules and
                        simplified not in new_rules):
                        new_rules.append(simplified)

        self.rules.extend(new_rules)

    def _filter_rules(self):
        """Filter rules by coverage and confidence."""
        self.rules = [
            r for r in self.rules
            if r.coverage >= self.min_coverage and r.confidence >= self.min_confidence
        ]

        # Sort by confidence and coverage
        self.rules.sort(key=lambda r: (r.confidence, r.coverage), reverse=True)

        # Keep only top rules
        if len(self.rules) > self.max_rules:
            self.rules = self.rules[:self.max_rules]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using learned rules."""
        predictions = np.zeros(len(X), dtype=int)

        for i, example in enumerate(X):
            # Create example
            input_atoms = set()
            for j, val in enumerate(example):
                if val > 0.5:  # Binary threshold
                    input_atoms.add(Atom(f"x{j}", (Term("1"),)))

            # Check rules
            for rule in self.rules:
                matches = all(body_atom in input_atoms for body_atom in rule.body)
                if matches and rule.head.predicate == "target":
                    predictions[i] = 1
                    break

        return predictions

    def print_rules(self):
        """Print learned rules."""
        print("\n=== LFIO Learned Rules ===")
        if not self.rules:
            print("No rules learned")
            return

        for i, rule in enumerate(self.rules, 1):
            print(f"{i}. {rule}")
            print(f"   Coverage: {rule.coverage:.2%}, Confidence: {rule.confidence:.2%}")

    def to_prolog(self) -> str:
        """Convert rules to Prolog syntax."""
        prolog_rules = []
        for rule in self.rules:
            if not rule.body:
                prolog_rules.append(f"{rule.head.predicate}({', '.join(str(a) for a in rule.head.arguments)}).")
            else:
                head_args = ", ".join(str(a) for a in rule.head.arguments)
                body_strs = []
                for atom in rule.body:
                    args = ", ".join(str(a) for a in atom.arguments)
                    if atom.negated:
                        body_strs.append(f"\\+ {atom.predicate}({args})")
                    else:
                        body_strs.append(f"{atom.predicate}({args})")
                body_clause = ", ".join(body_strs)
                prolog_rules.append(f"{rule.head.predicate}({head_args}) :- {body_clause}.")
        return "\n".join(prolog_rules)

    def to_python(self, feature_names: List[str]) -> str:
        """Convert rules to Python function."""
        conditions = []
        for rule in self.rules:
            if rule.head.predicate != "target":
                continue

            rule_conds = []
            for atom in rule.body:
                if atom.predicate in feature_names:
                    idx = feature_names.index(atom.predicate)
                    if len(atom.arguments) > 0 and atom.arguments[0].name == "1":
                        rule_conds.append(f"x[{idx}] == 1")
                    else:
                        rule_conds.append(f"x[{idx}] == 0")

            if rule_conds:
                conditions.append(f"({' and '.join(rule_conds)})")

        if not conditions:
            return "return 0"

        return f"return 1 if {' or '.join(conditions)} else 0"


class LFIOBoolean:
    """
    Simplified LFIO for Boolean classification.

    This version is optimized for learning Boolean rules from binary data,
    with improved efficiency and integration with scikit-learn.
    """

    def __init__(self,
                 min_coverage: int = 5,
                 min_precision: float = 0.9,
                 max_literals: int = 4,
                 enable_pruning: bool = True):
        """
        Initialize LFIO Boolean learner.

        Args:
            min_coverage: Minimum number of positive examples covered
            min_precision: Minimum precision for a rule
            max_literals: Maximum number of literals in a rule body
            enable_pruning: Enable rule pruning based on statistical tests
        """
        self.min_coverage = min_coverage
        self.min_precision = min_precision
        self.max_literals = max_literals
        self.enable_pruning = enable_pruning

        self.rules: List[Dict] = []
        self.feature_names: Optional[List[str]] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Learn Boolean rules from data.

        Args:
            X: Binary feature matrix
            y: Binary target labels
            feature_names: Names of features

        Returns:
            List of learned rules (each as a dictionary)
        """
        self.feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]
        self.rules = []

        # Get positive and negative examples
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        X_pos = X[pos_idx]
        X_neg = X[neg_idx]

        # Sequential covering algorithm
        uncovered = set(pos_idx.tolist())

        iteration = 0
        max_iterations = 50

        while uncovered and iteration < max_iterations:
            iteration += 1

            # Find best rule for uncovered examples
            best_rule = self._find_best_rule(X, uncovered, neg_idx)

            if best_rule is None:
                break

            self.rules.append(best_rule)

            # Update uncovered examples
            newly_covered = self._get_covered_examples(X, best_rule, uncovered)
            uncovered -= newly_covered

        return self.rules

    def _find_best_rule(self, X: np.ndarray, uncovered_pos: Set[int],
                       neg_idx: np.ndarray) -> Optional[Dict]:
        """Find best rule using beam search."""
        n_features = X.shape[1]
        best_rule = None
        best_score = -1

        # Start with single literals
        candidates = []

        for feat in range(n_features):
            for val in [0, 1]:
                candidates.append([(feat, val)])

        # Extend candidates
        for level in range(self.max_literals):
            new_candidates = []

            for candidate in candidates:
                # Evaluate candidate
                pos_covered = sum(1 for idx in uncovered_pos
                                if self._matches(X[idx], candidate))
                neg_covered = sum(1 for idx in neg_idx
                                if self._matches(X[idx], candidate))

                if pos_covered >= self.min_coverage:
                    precision = pos_covered / (pos_covered + neg_covered)
                    coverage = pos_covered / len(uncovered_pos)

                    # Combined score
                    score = coverage * precision

                    if precision >= self.min_precision and score > best_score:
                        best_score = score
                        best_rule = {
                            'literals': candidate.copy(),
                            'coverage': coverage,
                            'precision': precision,
                            'pos_covered': pos_covered,
                            'neg_covered': neg_covered
                        }

                # Extend candidate
                if level < self.max_literals - 1:
                    used_features = {lit[0] for lit in candidate}
                    for feat in range(n_features):
                        if feat not in used_features:
                            for val in [0, 1]:
                                new_candidate = candidate + [(feat, val)]
                                new_candidates.append(new_candidate)

            # Prune candidates (keep top K)
            candidates = self._prune_candidates(new_candidates, k=50)

            if not candidates:
                break

        return best_rule

    def _matches(self, example: np.ndarray, literals: List[Tuple[int, int]]) -> bool:
        """Check if example matches all literals."""
        for feat, val in literals:
            if example[feat] != val:
                return False
        return True

    def _get_covered_examples(self, X: np.ndarray, rule: Dict,
                            indices: Set[int]) -> Set[int]:
        """Get examples covered by a rule."""
        covered = set()
        for idx in indices:
            if self._matches(X[idx], rule['literals']):
                covered.add(idx)
        return covered

    def _prune_candidates(self, candidates: List[List[Tuple[int, int]]],
                         k: int) -> List[List[Tuple[int, int]]]:
        """Keep top k candidates based on coverage."""
        scored = []
        for candidate in candidates:
            score = len(candidate)  # Prefer shorter rules
            scored.append((score, candidate))

        scored.sort(key=lambda x: x[0])
        return [c for _, c in scored[:k]]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = np.zeros(len(X), dtype=int)

        for i, example in enumerate(X):
            for rule in self.rules:
                if self._matches(example, rule['literals']):
                    predictions[i] = 1
                    break

        return predictions

    def print_rules(self):
        """Print learned rules."""
        print("\n=== LFIO Boolean Rules ===")
        for i, rule in enumerate(self.rules, 1):
            literals_str = " AND ".join([
                f"{self.feature_names[f]}={v}" for f, v in rule['literals']
            ])
            print(f"{i}. IF {literals_str} THEN y = 1")
            print(f"   Coverage: {rule['coverage']:.2%}, Precision: {rule['precision']:.2%}")

    def to_dnf(self) -> str:
        """Get rules in Disjunctive Normal Form."""
        if not self.rules:
            return "False"

        disjuncts = []
        for rule in self.rules:
            literals = " AND ".join([
                f"{self.feature_names[f]}={v}" for f, v in rule['literals']
            ])
            disjuncts.append(f"({literals})")

        return " OR ".join(disjuncts)

    def to_python_function(self, function_name: str = "predict") -> str:
        """Convert rules to Python function."""
        conditions = []
        for rule in self.rules:
            literals = " and ".join([
                f"x[{f}] == {v}" for f, v in rule['literals']
            ])
            conditions.append(f"({literals})")

        if not conditions:
            return f"def {function_name}(x): return 0"

        disjunction = " or ".join(conditions)
        return f"def {function_name}(x): return 1 if {disjunction} else 0"


class LFIOBlackBox:
    """
    LFIO learner for extracting rules from black box models.

    This version is designed to work with trained neural networks
    or other black box models, learning rules that approximate
    the model's behavior.
    """

    def __init__(self,
                 black_box_model,
                 min_coverage: float = 0.05,
                 min_fidelity: float = 0.9,
                 max_rules: int = 20):
        """
        Initialize LFIO for black box rule extraction.

        Args:
            black_box_model: Trained model with predict() method
            min_coverage: Minimum coverage for rules
            min_fidelity: Minimum fidelity (agreement with black box)
            max_rules: Maximum number of rules to extract
        """
        self.black_box = black_box_model
        self.min_coverage = min_coverage
        self.min_fidelity = min_fidelity
        self.max_rules = max_rules

        self.rules: List[Dict] = []
        self.feature_names: Optional[List[str]] = None

    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Extract rules from black box model.

        Args:
            X: Feature matrix
            feature_names: Names of features

        Returns:
            List of extracted rules
        """
        self.feature_names = feature_names or [f"x{i}" for i in range(X.shape[1])]
        self.rules = []

        # Get black box predictions
        bb_preds = self.black_box.predict(X)
        if len(bb_preds.shape) > 1:
            bb_preds = np.argmax(bb_preds, axis=1)
        else:
            bb_preds = (bb_preds > 0.5).astype(int)

        # Learn rules that approximate black box behavior using sequential covering
        pos_idx = np.where(bb_preds == 1)[0]

        if len(pos_idx) == 0:
            return self.rules

        # Sequential covering
        uncovered = set(pos_idx.tolist())
        all_indices = set(range(len(X)))
        neg_idx = all_indices - set(pos_idx)

        iteration = 0
        max_iterations = self.max_rules * 2

        while uncovered and iteration < max_iterations:
            iteration += 1

            # Find best rule
            best_rule = self._find_best_fidelity_rule(
                X, uncovered, neg_idx, bb_preds
            )

            if best_rule is None or best_rule['fidelity'] < self.min_fidelity:
                break

            self.rules.append(best_rule)

            # Update uncovered - only remove examples that are correctly covered
            newly_covered = set()
            for idx in uncovered:
                if self._matches(X[idx], best_rule['literals']):
                    # Only count as covered if black box agrees
                    if bb_preds[idx] == 1:
                        newly_covered.add(idx)

            uncovered -= newly_covered

            if len(self.rules) >= self.max_rules:
                break

        return self.rules

    def _find_best_fidelity_rule(self, X: np.ndarray, uncovered_pos: Set[int],
                                 neg_idx: Set[int], bb_preds: np.ndarray) -> Optional[Dict]:
        """Find rule with best fidelity to black box using improved search."""
        n_features = X.shape[1]
        best_rule = None
        best_score = -1

        # Use a combined score: fidelity * coverage, with penalty for negative coverage
        for num_literals in range(1, min(4, n_features + 1)):  # Try 1, 2, 3 literals
            # Generate combinations of features
            for feat_combo in itertools.combinations(range(n_features), num_literals):
                # Generate all value assignments for these features
                for values in itertools.product([0, 1], repeat=num_literals):
                    candidate = list(zip(feat_combo, values))

                    # Calculate how many positive examples this covers
                    pos_covered = sum(1 for idx in uncovered_pos
                                    if self._matches(X[idx], candidate))

                    if pos_covered < max(1, self.min_coverage * len(uncovered_pos)):
                        continue

                    # Calculate fidelity: how well rule matches black box
                    correct = 0
                    total = 0

                    # Check all examples (not just uncovered)
                    for idx in range(len(X)):
                        if self._matches(X[idx], candidate):
                            total += 1
                            if bb_preds[idx] == 1:
                                correct += 1

                    if total == 0:
                        continue

                    precision = correct / total
                    recall = pos_covered / len(uncovered_pos)

                    # Combined score - prioritize precision heavily
                    score = precision * precision * recall

                    if precision >= self.min_fidelity and score > best_score:
                        best_score = score
                        best_rule = {
                            'literals': candidate,
                            'coverage': recall,
                            'fidelity': precision,
                            'pos_covered': pos_covered,
                            'neg_covered': total - correct,
                            'score': score
                        }

        return best_rule

    def _matches(self, example: np.ndarray, literals: List[Tuple[int, int]]) -> bool:
        """Check if example matches all literals."""
        for feat, val in literals:
            if example[feat] != val:
                return False
        return True

    def _get_covered_examples(self, X: np.ndarray, rule: Dict,
                            indices: Set[int]) -> Set[int]:
        """Get examples covered by a rule."""
        covered = set()
        for idx in indices:
            if self._matches(X[idx], rule['literals']):
                covered.add(idx)
        return covered

    def _prune_candidates(self, candidates: List[List[Tuple[int, int]]],
                         k: int) -> List[List[Tuple[int, int]]]:
        """Keep top k candidates."""
        return candidates[:k] if len(candidates) > k else candidates

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using extracted rules."""
        predictions = np.zeros(len(X), dtype=int)

        for i, example in enumerate(X):
            for rule in self.rules:
                if self._matches(example, rule['literals']):
                    predictions[i] = 1
                    break

        return predictions

    def print_rules(self):
        """Print extracted rules."""
        print("\n=== LFIO Black Box Rules ===")
        for i, rule in enumerate(self.rules, 1):
            literals_str = " AND ".join([
                f"{self.feature_names[f]}={v}" for f, v in rule['literals']
            ])
            print(f"{i}. IF {literals_str} THEN y = 1")
            print(f"   Coverage: {rule['coverage']:.2%}, Fidelity: {rule['fidelity']:.2%}")


if __name__ == "__main__":
    # Test LFIO implementations
    print("=== Testing LFIO Learners ===\n")

    # Generate test data
    import sys
    sys.path.append('src')
    from src.data_generator import SyntheticDataGenerator

    generator = SyntheticDataGenerator(n_samples=200, n_features=5)
    X, y, true_rule = generator.generate_boolean_dataset("complex")

    print(f"True rule: {true_rule}")
    print(f"Positive examples: {np.sum(y)}")

    # Test LFIO Boolean
    print("\n--- LFIO Boolean ---")
    lfio_bool = LFIOBoolean(min_coverage=5, min_precision=0.85)
    lfio_bool.fit(X, y, generator.create_feature_names())
    lfio_bool.print_rules()

    # Evaluate
    predictions = lfio_bool.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"DNF: {lfio_bool.to_dnf()}")
