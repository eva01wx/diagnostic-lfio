"""
Extended LFIO (Learning from Interpretations and Observations) with support for
explicit negative literals.

This implementation extends LFIO to handle negative atoms in both input and
output interpretations. It implements:
1. Extended interpretation representation with explicit negative atoms
2. Rule generation with negation in body
3. LGG for atoms with negation
4. Subsumption checking for rules with negative literals
5. Boolean feature-value negation with optional complement canonicalization

Key improvements:
- Proper handling of negative literals in interpretations
- Support for learning rules like: p(X) :- q(X), not r(X)
- Efficient indexing for positive and negative atoms
- Explicit Boolean negation for complete feature vectors

This module does not implement full negation-as-failure or well-founded
semantics. Negated Boolean literals are evaluated by direct inequality.
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
import copy


# ============================================================
# Core Data Structures with Negation Support
# ============================================================

@dataclass(frozen=True, order=True)
class Term:
    """Represents a logical term (constant or variable)."""
    name: str
    is_variable: bool = False

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Var({self.name})" if self.is_variable else f"Const({self.name})"


@dataclass(frozen=True)
class Literal:
    """
    Represents a logical literal (possibly negated atom).

    A literal is either an atom or its negation. This representation
    explicitly tracks whether the literal is negated, which is essential
    for handling negative information in interpretations.
    """
    predicate: str
    arguments: Tuple[Term, ...] = field(default_factory=tuple)
    negated: bool = False

    def __str__(self):
        args = ", ".join(str(a) for a in self.arguments)
        if self.negated:
            return f"not {self.predicate}({args})" if args else f"not {self.predicate}"
        return f"{self.predicate}({args})" if args else self.predicate

    def __repr__(self):
        return f"Literal({str(self)})"

    def __hash__(self):
        return hash((self.predicate, self.arguments, self.negated))

    def __eq__(self, other):
        if not isinstance(other, Literal):
            return False
        return (self.predicate == other.predicate and
                self.arguments == other.arguments and
                self.negated == other.negated)

    def arity(self) -> int:
        return len(self.arguments)

    def ground(self) -> bool:
        """Check if literal is ground (no variables)."""
        return all(not a.is_variable for a in self.arguments)

    def is_positive(self) -> bool:
        """Check if this is a positive literal."""
        return not self.negated

    def is_negative(self) -> bool:
        """Check if this is a negative literal."""
        return self.negated

    def negate(self) -> 'Literal':
        """Return the negation of this literal."""
        return Literal(self.predicate, self.arguments, not self.negated)

    def to_atom(self) -> 'Atom':
        """Convert to Atom (for backward compatibility)."""
        return Atom(self.predicate, self.arguments, self.negated)


@dataclass(frozen=True)
class Atom:
    """Backward compatible Atom class."""
    predicate: str
    arguments: Tuple[Term, ...] = field(default_factory=tuple)
    negated: bool = False

    def to_literal(self) -> Literal:
        """Convert to Literal."""
        return Literal(self.predicate, self.arguments, self.negated)

    def __str__(self):
        args = ", ".join(str(a) for a in self.arguments)
        neg = "not " if self.negated else ""
        return f"{neg}{self.predicate}({args})" if args else f"{neg}{self.predicate}"


@dataclass
class Rule:
    """
    Represents a logical rule (Horn clause) with possible negation in body.

    Rules are of the form: Head :- Body where Body can contain
    both positive and negative literals.

    Example: p(X) :- q(X), not r(X)
    """
    head: Literal
    body: Tuple[Literal, ...] = field(default_factory=tuple)
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

    def has_negation_in_body(self) -> bool:
        """Check if body contains negative literals."""
        return any(lit.negated for lit in self.body)

    def positive_body_literals(self) -> Tuple[Literal, ...]:
        """Get only positive body literals."""
        return tuple(lit for lit in self.body if not lit.negated)

    def negative_body_literals(self) -> Tuple[Literal, ...]:
        """Get only negative body literals."""
        return tuple(lit for lit in self.body if lit.negated)

    def body_size(self) -> int:
        return len(self.body)

    def to_dnf(self) -> str:
        """Convert to Disjunctive Normal Form string."""
        if not self.body:
            return str(self.head)
        terms = []
        for lit in self.body:
            if lit.negated:
                terms.append(f"NOT {lit.predicate}")
            else:
                terms.append(lit.predicate)
        return f"{' AND '.join(terms)} -> {self.head.predicate}"


@dataclass
class Interpretation:
    """
    Represents an interpretation with explicit positive and negative atoms.

    An interpretation consists of:
    - positive: Set of atoms that are true
    - negative: Set of atoms that are explicitly false (using CWA)

    This represents a partial Herbrand interpretation where we explicitly
    track which atoms are known to be false (not just unknown).
    """
    positive: FrozenSet[Literal] = field(default_factory=frozenset)
    negative: FrozenSet[Literal] = field(default_factory=frozenset)
    features: Optional[np.ndarray] = None

    def __str__(self):
        pos_str = ", ".join(str(a) for a in self.positive) if self.positive else "empty"
        neg_str = ", ".join(str(a) for a in self.negative) if self.negative else "empty"
        return f"Pos: {{{pos_str}}}, Neg: {{{neg_str}}}"

    def add_positive(self, literal: Literal) -> 'Interpretation':
        """Return new interpretation with added positive literal."""
        new_pos = set(self.positive)
        new_pos.add(literal)
        # Remove from negative if present (consistency)
        new_neg = set(self.negative) - {literal, literal.negate()}
        return Interpretation(frozenset(new_pos), frozenset(new_neg), self.features)

    def add_negative(self, literal: Literal) -> 'Interpretation':
        """Return new interpretation with added negative literal."""
        new_neg = set(self.negative)
        new_neg.add(literal)
        # Remove from positive if present (consistency)
        new_pos = set(self.positive) - {literal, literal.negate()}
        return Interpretation(frozenset(new_pos), frozenset(new_neg), self.features)

    def satisfies(self, literal: Literal) -> bool:
        """Check if interpretation satisfies a literal."""
        if literal.negated:
            # A negative literal is satisfied if its negation is NOT in positive
            return literal.negate() not in self.positive
        else:
            # A positive literal is satisfied if it's in positive
            return literal in self.positive

    def satisfies_body(self, body: Tuple[Literal, ...]) -> bool:
        """Check if interpretation satisfies all body literals."""
        return all(self.satisfies(lit) for lit in body)

    def consistent(self) -> bool:
        """Check if interpretation is consistent (no atom in both pos and neg)."""
        return self.positive.isdisjoint(self.negative)

    def union(self, other: 'Interpretation') -> 'Interpretation':
        """Combine two interpretations."""
        new_pos = self.positive | other.positive
        new_neg = self.negative | other.negative
        return Interpretation(frozenset(new_pos), frozenset(new_neg), self.features)


@dataclass
class Example:
    """
    Represents an input-output example with interpretations.

    Each example has:
    - input: Interpretation of input predicates
    - output: Interpretation of output predicates to learn
    """
    input: Interpretation
    output: Interpretation

    def __str__(self):
        return f"Input: [{self.input}] -> Output: [{self.output}]"

    def is_consistent(self) -> bool:
        """Check if example is internally consistent."""
        return self.input.consistent() and self.output.consistent()


# ============================================================
# Extended LFIO Algorithm with Negation Support
# ============================================================

class LFIOExtended:
    """
    Extended LFIO that supports negative atoms in interpretations.

    Key features:
    1. Handles both positive and negative literals in input/output
    2. Learns rules with negation in body (stratified rules)
    3. Proper LGG for negated atoms
    4. Subsumption checking for rules with negation
    5. Efficient indexing for positive/negative literals
    """

    def __init__(self,
                 min_coverage: float = 0.05,
                 min_confidence: float = 0.8,
                 max_rules: int = 100,
                 max_body_size: int = 5,
                 allow_negation_in_body: bool = True,
                 max_negative_literals: int = 2):
        """
        Initialize Extended LFIO learner.

        Args:
            min_coverage: Minimum coverage for a rule
            min_confidence: Minimum confidence for a rule
            max_rules: Maximum number of rules to learn
            max_body_size: Maximum size of rule body
            allow_negation_in_body: Allow negative literals in rule body
            max_negative_literals: Maximum negative literals per rule body
        """
        self.min_coverage = min_coverage
        self.min_confidence = min_confidence
        self.max_rules = max_rules
        self.max_body_size = max_body_size
        self.allow_negation_in_body = allow_negation_in_body
        self.max_negative_literals = max_negative_literals

        self.rules: List[Rule] = []
        self.examples: List[Example] = []
        self.var_counter = 0

        # Indexing for efficient lookup
        self.positive_index: Dict[str, Set[int]] = defaultdict(set)
        self.negative_index: Dict[str, Set[int]] = defaultdict(set)

    def _new_variable(self) -> Term:
        """Generate a new variable."""
        var = Term(f"V{self.var_counter}", is_variable=True)
        self.var_counter += 1
        return var

    def fit(self, examples: List[Example],
            background_knowledge: Optional[List[Rule]] = None) -> List[Rule]:
        """
        Learn rules from examples with negative atoms.

        Args:
            examples: List of input-output examples
            background_knowledge: Optional background rules

        Returns:
            List of learned rules
        """
        self.examples = examples
        self.rules = []
        self.var_counter = 0

        # Build indexes
        self._build_indexes()

        # Add background knowledge if provided
        if background_knowledge:
            self.rules.extend(background_knowledge)

        # Run extended LFIO algorithm
        self._run_extended_lfio()

        return self.rules

    def _build_indexes(self):
        """Build indexes for efficient lookup."""
        self.positive_index = defaultdict(set)
        self.negative_index = defaultdict(set)

        for idx, ex in enumerate(self.examples):
            for lit in ex.input.positive:
                self.positive_index[lit.predicate].add(idx)
            for lit in ex.input.negative:
                self.negative_index[lit.predicate].add(idx)

    def _run_extended_lfio(self):
        """Run the main extended LFIO algorithm."""
        # Step 1: Generate initial rules from examples
        initial_rules = self._generate_initial_rules()
        self.rules.extend(initial_rules)

        # Step 2: Apply LGG to generalize rules (with negation support)
        self._apply_lgg_with_negation()

        # Step 3: Apply subsumption to remove redundant rules
        self._apply_subsumption_with_negation()

        # Step 4: Filter and prune
        self._filter_rules()

    def _generate_initial_rules(self) -> List[Rule]:
        """Generate initial ground rules from examples."""
        rules = []

        for ex_idx, example in enumerate(self.examples):
            if not example.output.positive:
                continue

            for head_literal in example.output.positive:
                # Generate rule variants with different body compositions
                rule_variants = self._generate_rule_variants(
                    head_literal, example, ex_idx
                )
                rules.extend(rule_variants)

        return rules

    def _generate_rule_variants(self, head: Literal, example: Example,
                                ex_idx: int) -> List[Rule]:
        """Generate rule variants for a given head and example."""
        variants = []

        # Get available positive and negative literals from input
        pos_literals = list(example.input.positive)
        neg_literals = list(example.input.negative)

        # Generate combinations of body literals
        max_body = min(self.max_body_size,
                      len(pos_literals) + len(neg_literals))

        for body_size in range(max_body + 1):
            # Try different combinations of positive and negative literals
            for num_neg in range(min(self.max_negative_literals, body_size) + 1):
                num_pos = body_size - num_neg

                # Select positive literals
                for pos_combo in itertools.combinations(pos_literals, num_pos):
                    # Select negative literals
                    for neg_combo in itertools.combinations(neg_literals, num_neg):
                        body = tuple(pos_combo) + tuple(
                            lit.negate() for lit in neg_combo
                        )

                        rule = Rule(head, body)
                        self._calculate_rule_stats(rule)

                        if (rule.coverage >= self.min_coverage and
                            rule.confidence >= self.min_confidence):
                            variants.append(rule)

        return variants

    def _calculate_rule_stats(self, rule: Rule):
        """Calculate coverage and confidence for a rule."""
        pos_matches = 0
        total_pos = len([e for e in self.examples if e.output.positive])
        total_matches = 0

        for ex in self.examples:
            if self._rule_matches(rule, ex):
                total_matches += 1
                if rule.head in ex.output.positive:
                    pos_matches += 1

        rule.total_examples = total_matches
        rule.positive_examples = pos_matches
        rule.coverage = pos_matches / total_pos if total_pos > 0 else 0
        rule.confidence = pos_matches / total_matches if total_matches > 0 else 0

    def _rule_matches(self, rule: Rule, example: Example) -> bool:
        """Check if a rule matches an example."""
        return example.input.satisfies_body(rule.body)

    def _apply_lgg_with_negation(self):
        """Apply LGG with support for negated atoms."""
        new_rules = []
        processed = set()

        # Group rules by head predicate
        rules_by_head = defaultdict(list)
        for rule in self.rules:
            rules_by_head[rule.head.predicate].append(rule)

        for head_pred, rule_group in rules_by_head.items():
            for i, r1 in enumerate(rule_group):
                for r2 in rule_group[i+1:]:
                    if r1 in processed or r2 in processed:
                        continue

                    generalized = self._lgg_rules_with_negation(r1, r2)
                    if generalized and generalized not in self.rules:
                        self._calculate_rule_stats(generalized)
                        if (generalized.coverage >= self.min_coverage and
                            generalized.confidence >= self.min_confidence):
                            new_rules.append(generalized)
                            processed.add(r1)
                            processed.add(r2)

        for rule in new_rules:
            if rule not in self.rules:
                self.rules.append(rule)

    def _lgg_rules_with_negation(self, r1: Rule, r2: Rule) -> Optional[Rule]:
        """Compute LGG of two rules with negation support."""
        if r1.head.predicate != r2.head.predicate:
            return None

        # LGG of heads
        lgg_head = self._lgg_literals(r1.head, r2.head)
        if not lgg_head:
            return None

        # Align bodies for LGG
        # Need to pair up body literals with same predicate and negation
        lgg_body = self._lgg_bodies(r1.body, r2.body)
        if lgg_body is None:
            return None

        return Rule(lgg_head, lgg_body)

    def _lgg_bodies(self, body1: Tuple[Literal, ...],
                   body2: Tuple[Literal, ...]) -> Optional[Tuple[Literal, ...]]:
        """Compute LGG of two bodies, handling negation."""
        # Group literals by (predicate, is_negated)
        body1_groups = defaultdict(list)
        body2_groups = defaultdict(list)

        for lit in body1:
            body1_groups[(lit.predicate, lit.negated)].append(lit)

        for lit in body2:
            body2_groups[(lit.predicate, lit.negated)].append(lit)

        lgg_body = []

        # Process common groups
        common_keys = set(body1_groups.keys()) & set(body2_groups.keys())

        for key in common_keys:
            group1 = body1_groups[key]
            group2 = body2_groups[key]

            # Try to pair up literals within groups
            for lit1 in group1:
                for lit2 in group2:
                    lgg_lit = self._lgg_literals(lit1, lit2)
                    if lgg_lit and lgg_lit not in lgg_body:
                        lgg_body.append(lgg_lit)

        # Add literals that are unique to each body (only if they're consistent)
        for key in set(body1_groups.keys()) - common_keys:
            for lit in body1_groups[key]:
                if lit not in lgg_body:
                    lgg_body.append(lit)

        return tuple(lgg_body) if lgg_body else None

    def _lgg_literals(self, l1: Literal, l2: Literal) -> Optional[Literal]:
        """Compute LGG of two literals."""
        if l1.predicate != l2.predicate:
            return None

        if l1.negated != l2.negated:
            # Can't generalize between positive and negative literals
            return None

        if len(l1.arguments) != len(l2.arguments):
            return None

        # LGG of arguments
        lgg_args = []
        for arg1, arg2 in zip(l1.arguments, l2.arguments):
            if arg1.name == arg2.name:
                lgg_args.append(arg1)
            else:
                # Different constants -> create variable
                new_var = self._new_variable()
                lgg_args.append(new_var)

        return Literal(l1.predicate, tuple(lgg_args), l1.negated)

    def _apply_subsumption_with_negation(self):
        """Remove rules subsumed by more general rules (with negation support)."""
        to_remove = set()

        for i, r1 in enumerate(self.rules):
            for j, r2 in enumerate(self.rules):
                if i >= j or r1 in to_remove or r2 in to_remove:
                    continue

                if self._subsumes_with_negation(r1, r2):
                    to_remove.add(r2)
                elif self._subsumes_with_negation(r2, r1):
                    to_remove.add(r1)

        self.rules = [r for r in self.rules if r not in to_remove]

    def _subsumes_with_negation(self, r1: Rule, r2: Rule) -> bool:
        """
        Check if r1 subsumes r2, considering negation.

        r1 subsumes r2 if r1 is more general than r2:
        - Same head predicate
        - r1's body is a subset of r2's body (considering predicate and negation)
        """
        if r1.head.predicate != r2.head.predicate:
            return False

        # Get body predicates with negation status
        r1_body_keys = {(lit.predicate, lit.negated) for lit in r1.body}
        r2_body_keys = {(lit.predicate, lit.negated) for lit in r2.body}

        return r1_body_keys.issubset(r2_body_keys)

    def _filter_rules(self):
        """Filter rules by coverage and confidence."""
        self.rules = [
            r for r in self.rules
            if r.coverage >= self.min_coverage and r.confidence >= self.min_confidence
        ]

        # Sort by confidence and coverage
        self.rules.sort(key=lambda r: (r.confidence, r.coverage), reverse=True)

        if len(self.rules) > self.max_rules:
            self.rules = self.rules[:self.max_rules]

    def print_rules(self):
        """Print learned rules."""
        print("\n=== Extended LFIO Learned Rules ===")
        if not self.rules:
            print("No rules learned")
            return

        for i, rule in enumerate(self.rules, 1):
            print(f"{i}. {rule}")
            print(f"   Coverage: {rule.coverage:.2%}, Confidence: {rule.confidence:.2%}")
            if rule.has_negation_in_body():
                neg_lits = ", ".join(str(l) for l in rule.negative_body_literals())
                print(f"   (Negative literals: {neg_lits})")


# ============================================================
# Boolean LFIO with Negation Support
# ============================================================

class LFIOBooleanWithNegation:
    """
    Boolean LFIO that handles negation in a simpler way.

    This version is optimized for Boolean classification where:
    - Features can be positive (x=1) or negative (x=0)
    - Rules can include negation: IF x1=1 AND NOT x2=1 THEN y=1
    """

    def __init__(self,
                 min_coverage: int = 5,
                 min_precision: float = 0.9,
                 max_literals: int = 4,
                 allow_negation: bool = True,
                 canonicalize: bool = True,
                 prefer_negated_complement: bool = True,
                 backend: str = "bitset"):
        """
        Initialize Boolean LFIO with negation.

        Args:
            min_coverage: Minimum positive examples covered
            min_precision: Minimum precision for a rule
            max_literals: Maximum literals in rule body
            allow_negation: Allow negated literals in body
            canonicalize: Score each Boolean semantic test once instead of
                separately scoring equivalent positive and negated forms
            prefer_negated_complement: Render semantic tests for absent
                Boolean values as NOT x=1. This affects presentation only.
            backend: "bitset" for packed coverage intersections, or "python"
                for the original exhaustive sample-by-sample reference search.
        """
        self.min_coverage = min_coverage
        self.min_precision = min_precision
        self.max_literals = max_literals
        self.allow_negation = allow_negation
        self.canonicalize = canonicalize
        self.prefer_negated_complement = prefer_negated_complement
        if backend not in {"bitset", "python"}:
            raise ValueError("backend must be 'bitset' or 'python'")
        self.backend = backend

        self.rules: List[Dict] = []
        self.feature_names: Optional[List[str]] = None
        self.search_stats: Dict[str, float] = {}

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Learn Boolean rules with possible negation.

        Args:
            X: Binary feature matrix
            y: Binary target labels
            feature_names: Names of features

        Returns:
            List of learned rules
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or not X.shape[0] or not X.shape[1]:
            raise ValueError("X must be a nonempty 2D Boolean matrix")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a 1D array with one label per row")
        if not np.isin(X, [0, 1]).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("X and y must contain only complete Boolean values (0 or 1)")
        if self.min_coverage < 1 or self.max_literals < 1:
            raise ValueError("min_coverage and max_literals must be positive")
        if not 0 <= self.min_precision <= 1:
            raise ValueError("min_precision must be between 0 and 1")
        if feature_names is not None and len(feature_names) != X.shape[1]:
            raise ValueError("feature_names must match the number of columns")
        self.n_features_in_ = X.shape[1]
        self.feature_names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(X.shape[1])]
        self.rules = []
        self.search_stats = {
            'candidate_evaluations': 0,
            'naive_candidate_evaluations': 0,
            'duplicate_reduction': 0.0,
        }

        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        if self.backend == "bitset":
            # Each bit represents one observation, including duplicate rows.
            self._all_bits = (1 << len(X)) - 1
            self._literal_bits = {
                (feat, val): self._pack_mask(X[:, feat] == val)
                for feat in range(X.shape[1]) for val in (0, 1)
            }
            self._negative_bits = self._pack_mask(y == 0)

        # Sequential covering
        uncovered = set(pos_idx.tolist())

        iteration = 0
        max_iterations = 50

        while uncovered and iteration < max_iterations:
            iteration += 1

            best_rule = self._find_best_rule_with_negation(
                X, uncovered, neg_idx, pos_idx
            )

            if best_rule is None:
                break

            self.rules.append(best_rule)

            # Update uncovered
            newly_covered = self._get_covered_examples(X, best_rule, uncovered)
            uncovered -= newly_covered

        naive = self.search_stats['naive_candidate_evaluations']
        evaluated = self.search_stats['candidate_evaluations']
        if naive:
            self.search_stats['duplicate_reduction'] = 1.0 - (evaluated / naive)

        return self.rules

    def _find_best_rule_with_negation(self, X: np.ndarray, uncovered_pos: Set[int],
                                     neg_idx: np.ndarray, pos_idx: np.ndarray) -> Optional[Dict]:
        if self.backend == "bitset":
            return self._find_best_rule_bitset(X.shape[1], uncovered_pos)
        return self._find_best_rule_python(X, uncovered_pos, neg_idx, pos_idx)

    @staticmethod
    def _pack_mask(mask: np.ndarray) -> int:
        return int.from_bytes(np.packbits(mask, bitorder="little").tobytes(), "little")

    def _find_best_rule_bitset(self, n_features: int, uncovered_pos: Set[int]) -> Optional[Dict]:
        """Exact exhaustive search with the reference enumeration/tie order.

        Literal coverage is computed once per fit. Candidate conjunctions and
        positive/negative counts then use integer AND and population count,
        avoiding a Python loop over observations for every candidate.
        """
        uncovered_bits = sum(1 << idx for idx in uncovered_pos)
        best_rule, best_score = None, -1
        choices = 4 if self.allow_negation else 2
        for size in range(1, min(self.max_literals, n_features) + 1):
            for features in itertools.combinations(range(n_features), size):
                self.search_stats['naive_candidate_evaluations'] += choices ** size
                for values in itertools.product((0, 1), repeat=size):
                    forms = (itertools.product((False, True), repeat=size)
                             if not self.canonicalize and self.allow_negation
                             else [(False,) * size])
                    for negations in forms:
                        self.search_stats['candidate_evaluations'] += 1
                        bits = self._all_bits
                        for feat, val, negated in zip(features, values, negations):
                            bits &= self._literal_bits[feat, 1 - val if negated else val]
                        pos_covered = (bits & uncovered_bits).bit_count()
                        if pos_covered < self.min_coverage:
                            continue
                        neg_covered = (bits & self._negative_bits).bit_count()
                        precision = pos_covered / (pos_covered + neg_covered)
                        coverage = pos_covered / len(uncovered_pos)
                        score = coverage * precision
                        if precision >= self.min_precision and score > best_score:
                            best_score = score
                            literals = (self._semantic_literals(features, values)
                                        if self.canonicalize else list(zip(features, values, negations)))
                            best_rule = dict(literals=literals, coverage=coverage,
                                             precision=precision, pos_covered=pos_covered,
                                             neg_covered=neg_covered)
        return best_rule

    def _find_best_rule_python(self, X: np.ndarray, uncovered_pos: Set[int],
                             neg_idx: np.ndarray, pos_idx: np.ndarray) -> Optional[Dict]:
        """Find best rule, allowing negated literals."""
        n_features = X.shape[1]
        best_rule = None
        best_score = -1

        # Each literal can be: (feat, val, negated), where negated=True means
        # require the feature to NOT equal val. Canonicalized search scores each
        # semantic condition x_feat == required_val once, then renders it as a
        # positive or negated-complement literal.
        for num_literals in range(1, min(self.max_literals + 1, n_features + 1)):
            for feat_combo in itertools.combinations(range(n_features), num_literals):
                choices = 4 if self.allow_negation else 2
                self.search_stats['naive_candidate_evaluations'] += choices ** num_literals

                if self.canonicalize:
                    literal_sets = (
                        [self._semantic_literals(feat_combo, required_values)]
                        for required_values in itertools.product([0, 1], repeat=num_literals)
                    )
                else:
                    literal_sets = (
                        [(feat, val, neg) for feat, val, neg in zip(feat_combo, values, negations)]
                        for values in itertools.product([0, 1], repeat=num_literals)
                        for negations in itertools.product([False, True], repeat=num_literals)
                        if self.allow_negation or not any(negations)
                    )

                for literal_wrapper in literal_sets:
                    literals = literal_wrapper[0] if self.canonicalize else literal_wrapper
                    self.search_stats['candidate_evaluations'] += 1

                    # Evaluate rule
                    pos_covered = sum(1 for idx in uncovered_pos
                                    if self._matches_with_negation(X[idx], literals))
                    neg_covered = sum(1 for idx in neg_idx
                                    if self._matches_with_negation(X[idx], literals))

                    if pos_covered >= self.min_coverage:
                        precision = pos_covered / (pos_covered + neg_covered)
                        coverage = pos_covered / len(uncovered_pos)
                        score = coverage * precision

                        if precision >= self.min_precision and score > best_score:
                            best_score = score
                            best_rule = {
                                'literals': literals.copy(),
                                'coverage': coverage,
                                'precision': precision,
                                'pos_covered': pos_covered,
                                'neg_covered': neg_covered
                            }

        return best_rule

    def _semantic_literals(self, feat_combo: Tuple[int, ...],
                           required_values: Tuple[int, ...]) -> List[Tuple[int, int, bool]]:
        """Convert semantic tests x_feat == value into display literals."""
        literals = []
        for feat, required_val in zip(feat_combo, required_values):
            if self.allow_negation and self.prefer_negated_complement and required_val == 0:
                literals.append((feat, 1, True))
            else:
                literals.append((feat, required_val, False))
        return literals

    def _matches_with_negation(self, example: np.ndarray,
                              literals: List[Tuple[int, int, bool]]) -> bool:
        """Check if example matches all literals (with negation support)."""
        for feat, val, negated in literals:
            actual_val = example[feat]
            if negated:
                # Require actual_val != val
                if actual_val == val:
                    return False
            else:
                # Require actual_val == val
                if actual_val != val:
                    return False
        return True

    def _get_covered_examples(self, X: np.ndarray, rule: Dict,
                            indices: Set[int]) -> Set[int]:
        """Get examples covered by a rule."""
        covered = set()
        for idx in indices:
            if self._matches_with_negation(X[idx], rule['literals']):
                covered.add(idx)
        return covered

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X = np.asarray(X)
        if not hasattr(self, "n_features_in_"):
            raise ValueError("fit must be called before predict")
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isin(X, [0, 1]).all():
            raise ValueError("X must be a complete Boolean matrix with the fitted feature count")
        predictions = np.zeros(len(X), dtype=int)

        for rule in self.rules:
            matches = np.ones(len(X), dtype=bool)
            for feat, val, negated in rule['literals']:
                matches &= X[:, feat] != val if negated else X[:, feat] == val
            predictions[matches] = 1

        return predictions

    def print_rules(self):
        """Print learned rules with negation."""
        print("\n=== LFIO Boolean Rules (with Negation) ===")
        for i, rule in enumerate(self.rules, 1):
            literals_str = []
            for feat, val, negated in rule['literals']:
                if negated:
                    literals_str.append(f"NOT {self.feature_names[feat]}={val}")
                else:
                    literals_str.append(f"{self.feature_names[feat]}={val}")
            print(f"{i}. IF {' AND '.join(literals_str)} THEN y = 1")
            print(f"   Coverage: {rule['coverage']:.2%}, Precision: {rule['precision']:.2%}")

    def to_dnf(self) -> str:
        """Get rules in DNF with negation."""
        if not self.rules:
            return "False"

        disjuncts = []
        for rule in self.rules:
            literals = []
            for feat, val, negated in rule['literals']:
                if negated:
                    literals.append(f"NOT {self.feature_names[feat]}={val}")
                else:
                    literals.append(f"{self.feature_names[feat]}={val}")
            disjuncts.append(f"({' AND '.join(literals)})")

        return " OR ".join(disjuncts)

    def to_python_function(self, function_name: str = "predict") -> str:
        """Convert rules to Python function."""
        conditions = []
        for rule in self.rules:
            literals = []
            for feat, val, negated in rule['literals']:
                if negated:
                    literals.append(f"x[{feat}] != {val}")
                else:
                    literals.append(f"x[{feat}] == {val}")
            conditions.append(f"({' and '.join(literals)})")

        if not conditions:
            return f"def {function_name}(x): return 0"

        disjunction = " or ".join(conditions)
        return f"def {function_name}(x): return 1 if {disjunction} else 0"


# ============================================================
# Example Usage and Tests
# ============================================================

def create_test_examples_with_negation() -> List[Example]:
    """
    Create test examples demonstrating negative atom handling.

    True rule: p(X) :- q(X), not r(X)
    """
    examples = []

    # Create terms
    a = Term("a")
    b = Term("b")
    c = Term("c")

    # Example 1: q(a) is true, r(a) is false, so p(a) should be true
    examples.append(Example(
        input=Interpretation(
            positive=frozenset([Literal("q", (a,))]),
            negative=frozenset([Literal("r", (a,))])
        ),
        output=Interpretation(
            positive=frozenset([Literal("p", (a,))]),
            negative=frozenset()
        )
    ))

    # Example 2: q(b) is true, r(b) is false, so p(b) should be true
    examples.append(Example(
        input=Interpretation(
            positive=frozenset([Literal("q", (b,))]),
            negative=frozenset([Literal("r", (b,))])
        ),
        output=Interpretation(
            positive=frozenset([Literal("p", (b,))]),
            negative=frozenset()
        )
    ))

    # Example 3: q(c) is true, r(c) is true, so p(c) should be false
    examples.append(Example(
        input=Interpretation(
            positive=frozenset([Literal("q", (c,)), Literal("r", (c,))]),
            negative=frozenset()
        ),
        output=Interpretation(
            positive=frozenset(),
            negative=frozenset([Literal("p", (c,))])
        )
    ))

    return examples


def test_extended_lfio():
    """Test the extended LFIO with negative atoms."""
    print("=== Testing Extended LFIO with Negative Atoms ===\n")

    examples = create_test_examples_with_negation()

    print("Test Examples:")
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex}")

    # Learn rules
    lfio = LFIOExtended(
        min_coverage=0.3,
        min_confidence=0.8,
        allow_negation_in_body=True
    )

    lfio.fit(examples)
    lfio.print_rules()


def test_boolean_lfio_with_negation():
    """Test Boolean LFIO with negation on synthetic data."""
    print("\n=== Testing Boolean LFIO with Negation ===\n")

    import numpy as np

    # Generate data with rule: y = 1 if (x1=1 AND x2=0) OR (x3=0 AND x4=1)
    n_samples = 500
    X = np.random.randint(0, 2, size=(n_samples, 5))
    y = ((X[:, 0] == 1) & (X[:, 1] == 0) | (X[:, 2] == 0) & (X[:, 3] == 1)).astype(int)

    print(f"True rule: y = 1 if (x1=1 AND x2=0) OR (x3=0 AND x4=1)")
    print(f"Positive examples: {np.sum(y)}/{n_samples}")

    # Learn rules with negation
    lfio = LFIOBooleanWithNegation(
        min_coverage=10,
        min_precision=0.9,
        allow_negation=True
    )

    lfio.fit(X, y, [f"x{i+1}" for i in range(5)])
    lfio.print_rules()

    # Evaluate
    predictions = lfio.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"DNF: {lfio.to_dnf()}")


if __name__ == "__main__":
    test_extended_lfio()
    test_boolean_lfio_with_negation()
