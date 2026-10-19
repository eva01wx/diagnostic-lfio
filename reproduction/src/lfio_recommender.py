"""
LFIO for Recommender Systems (LFIORecSys)

This implementation adapts the LFIO (Learning from Interpretations and Observations)
algorithm to learn recommendation rules from user-item interaction data.

Key features:
1. Learn explainable rules from user-item interactions
2. Support for both explicit (ratings) and implicit (clicks) feedback
3. Handle user features (demographics, preferences) and item features (categories, attributes)
4. Generate human-readable recommendation rules
5. Apply learned rules to new users/items for recommendations

Example rules learned:
- "IF user.age < 30 AND item.category = 'electronics' THEN recommend"
- "IF user.genre_pref = 'action' AND item.genre = 'action' THEN recommend"
- "IF user.history_contains('scifi') AND item.series = 'Star Wars' THEN recommend"
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Optional, Union, Any
from dataclasses import dataclass, field
from collections import defaultdict
import itertools
import gc
from enum import Enum
from tqdm import tqdm


class FeedbackType(Enum):
    """Type of feedback in recommendation."""
    EXPLICIT = "explicit"  # Ratings, reviews
    IMPLICIT = "implicit"  # Clicks, views, purchases


class RuleType(Enum):
    """Type of recommendation rule."""
    USER_BASED = "user_based"  # Rules based on user features
    ITEM_BASED = "item_based"  # Rules based on item features
    INTERACTION = "interaction"  # Rules based on interaction patterns
    HYBRID = "hybrid"  # Combination of above


# ============================================================
# Core Data Structures for Recommender Systems
# ============================================================

@dataclass
class User:
    """Represents a user with features."""
    user_id: str
    features: Dict[str, Any] = field(default_factory=dict)
    history: List[str] = field(default_factory=list)  # Item IDs interacted with

    def get_feature_value(self, feature_name: str) -> Any:
        """Get feature value, return None if not present."""
        return self.features.get(feature_name)

    def has_feature(self, feature_name: str) -> bool:
        """Check if user has a feature."""
        return feature_name in self.features

    def __hash__(self):
        return hash(self.user_id)

    def __eq__(self, other):
        if not isinstance(other, User):
            return False
        return self.user_id == other.user_id


@dataclass
class Item:
    """Represents an item with features."""
    item_id: str
    features: Dict[str, Any] = field(default_factory=dict)

    def get_feature_value(self, feature_name: str) -> Any:
        """Get feature value, return None if not present."""
        return self.features.get(feature_name)

    def has_feature(self, feature_name: str) -> bool:
        """Check if item has a feature."""
        return feature_name in self.features

    def __hash__(self):
        return hash(self.item_id)

    def __eq__(self, other):
        if not isinstance(other, Item):
            return False
        return self.item_id == other.item_id


@dataclass
class Interaction:
    """Represents a user-item interaction."""
    user_id: str
    item_id: str
    rating: Optional[float] = None  # For explicit feedback
    clicked: bool = False  # For implicit feedback
    timestamp: Optional[float] = None

    def is_positive(self, threshold: float = 4.0) -> bool:
        """
        Check if interaction is positive.

        For explicit feedback (ratings), uses threshold (default: 4.0 on 5-point scale).
        Note: Uses >= to include ratings equal to threshold.
        For implicit feedback (clicked), returns True if clicked=True.
        """
        if self.rating is not None:
            return self.rating >= threshold  # Changed to >=
        return self.clicked


@dataclass
class RecRule:
    """
    Represents a recommendation rule.

    A rule has the form:
    IF <user_conditions> AND <item_conditions> AND <interaction_conditions>
    THEN <recommendation_with_confidence>

    Conditions are stored as tuples of (feat, val) pairs for hashability.
    """
    rule_id: str
    user_conditions: tuple = ()  # Tuple of (feat, val) pairs
    item_conditions: tuple = ()  # Tuple of (feat, val) pairs
    interaction_conditions: tuple = ()  # Tuple of (feat, val) pairs
    confidence: float = 0.0
    support: float = 0.0
    lift: float = 0.0
    rule_type: RuleType = RuleType.HYBRID
    positive_coverage: int = 0
    total_coverage: int = 0

    def __hash__(self):
        """Make rule hashable by using rule_id."""
        return hash(self.rule_id)

    def __eq__(self, other):
        """Rules are equal if they have the same id."""
        if not isinstance(other, RecRule):
            return False
        return self.rule_id == other.rule_id

    def __str__(self):
        parts = []

        # User conditions
        if self.user_conditions:
            user_parts = []
            for feat, val in self.user_conditions:
                if isinstance(val, set):
                    user_parts.append(f"user.{feat} IN {val}")
                elif val is not None:
                    user_parts.append(f"user.{feat} = {val}")
            if user_parts:
                parts.append(" AND ".join(user_parts))

        # Item conditions
        if self.item_conditions:
            item_parts = []
            for feat, val in self.item_conditions:
                if isinstance(val, set):
                    item_parts.append(f"item.{feat} IN {val}")
                elif val is not None:
                    item_parts.append(f"item.{feat} = {val}")
            if item_parts:
                parts.append(" AND ".join(item_parts))

        # Interaction conditions
        if self.interaction_conditions:
            inter_parts = []
            for feat, val in self.interaction_conditions:
                if val is not None:
                    inter_parts.append(f"interaction.{feat} = {val}")
            if inter_parts:
                parts.append(" AND ".join(inter_parts))

        condition_str = " AND ".join(parts) if parts else "TRUE"
        return (f"IF {condition_str} THEN recommend\n"
                f"   Confidence: {self.confidence:.3f}, Support: {self.support:.3f}, "
                f"Lift: {self.lift:.3f}, Coverage: {self.positive_coverage}/{self.total_coverage}")

    def matches_user(self, user: User) -> bool:
        """Check if user matches user conditions."""
        for feat, expected in self.user_conditions:
            actual = user.get_feature_value(feat)
            if isinstance(expected, set):
                if actual not in expected:
                    return False
            elif expected is not None:
                if actual != expected:
                    return False
        return True

    def matches_item(self, item: Item) -> bool:
        """Check if item matches item conditions."""
        for feat, expected in self.item_conditions:
            actual = item.get_feature_value(feat)
            if isinstance(expected, set):
                if actual not in expected:
                    return False
            elif expected is not None:
                if actual != expected:
                    return False
        return True

    def matches(self, user: User, item: Item) -> bool:
        """Check if user-item pair matches all conditions."""
        return self.matches_user(user) and self.matches_item(item)

    def get_user_conditions_dict(self) -> Dict:
        """Get user conditions as dictionary for convenience."""
        return dict(self.user_conditions) if self.user_conditions else {}

    def get_item_conditions_dict(self) -> Dict:
        """Get item conditions as dictionary for convenience."""
        return dict(self.item_conditions) if self.item_conditions else {}


@dataclass
class Recommendation:
    """Represents a recommendation with explanation."""
    user_id: str
    item_id: str
    score: float
    explanation: str
    matched_rules: List[str] = field(default_factory=list)


# ============================================================
# LFIO Recommender Algorithm
# ============================================================

class LFIORecSys:
    """
    LFIO adapted for Recommender Systems.

    This algorithm learns recommendation rules from interaction data by:
    1. Mining frequent patterns from positive interactions
    2. Generating candidate rules using association rule mining
    3. Refining rules using confidence, support, and lift metrics
    4. Applying subsumption to remove redundant rules
    5. Generalizing rules using LGG (Least General Generalization)

    The learned rules can explain why items are recommended to users.
    """

    def __init__(self,
                 min_support: float = 0.05,
                 min_confidence: float = 0.7,
                 min_lift: float = 1.0,
                 max_rules: int = 50,
                 max_condition_size: int = 3,
                 enable_generalization: bool = True,
                 feedback_type: FeedbackType = FeedbackType.IMPLICIT,
                 rating_threshold: float = 4.0):
        """
        Initialize LFIO Recommender.

        Args:
            min_support: Minimum support for a rule
            min_confidence: Minimum confidence for a rule
            min_lift: Minimum lift for a rule
            max_rules: Maximum number of rules to keep
            max_condition_size: Maximum number of conditions in rule body
            enable_generalization: Enable rule generalization
            feedback_type: Type of feedback (explicit or implicit)
            rating_threshold: Threshold for positive ratings (default 4.0)
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.max_rules = max_rules
        self.max_condition_size = max_condition_size
        self.enable_generalization = enable_generalization
        self.feedback_type = feedback_type
        self.rating_threshold = rating_threshold

        self.rules: List[RecRule] = []
        self.users: Dict[str, User] = {}
        self.items: Dict[str, Item] = {}
        self.interactions: List[Interaction] = []
        self.positive_interactions: List[Interaction] = []

        # Feature type tracking
        self.user_feature_types: Dict[str, type] = {}
        self.item_feature_types: Dict[str, type] = {}

        # Statistics
        self.total_interactions = 0
        self.positive_interaction_rate = 0.0

    def add_user(self, user: User):
        """Add a user to the system."""
        self.users[user.user_id] = user
        # Update feature types
        for feat, val in user.features.items():
            if val is not None:
                self.user_feature_types[feat] = type(val)

    def add_item(self, item: Item):
        """Add an item to the system."""
        self.items[item.item_id] = item
        # Update feature types
        for feat, val in item.features.items():
            if val is not None:
                self.item_feature_types[feat] = type(val)

    def add_interaction(self, interaction: Interaction):
        """Add an interaction to the system."""
        self.interactions.append(interaction)
        user = self.users.get(interaction.user_id)
        if user is not None and interaction.item_id not in user.history:
            # Both positive and negative ratings identify already seen items.
            user.history.append(interaction.item_id)
        if interaction.is_positive(self.rating_threshold):
            self.positive_interactions.append(interaction)

    def load_data(self,
                  users: List[User],
                  items: List[Item],
                  interactions: List[Interaction]):
        """Load complete dataset."""
        print(f"Loading {len(users)} users, {len(items)} items, {len(interactions)} interactions...")

        for user in tqdm(users, desc="Loading users"):
            self.add_user(user)

        for item in tqdm(items, desc="Loading items"):
            self.add_item(item)

        for interaction in tqdm(interactions, desc="Loading interactions"):
            self.add_interaction(interaction)

        self.total_interactions = len(self.interactions)
        self.positive_interaction_rate = len(self.positive_interactions) / self.total_interactions

        print(f"Loaded. Positive interaction rate: {self.positive_interaction_rate:.2%}")

    def load_from_dataframe(self,
                            df: pd.DataFrame,
                            user_features: List[str],
                            item_features: List[str]):
        """
        Load data from pandas dataframe.

        Args:
            df: DataFrame with columns [user_id, item_id, rating/click, ...features]
            user_features: List of column names that are user features
            item_features: List of column names that are item features
        """
        print("Loading from dataframe...")

        # Extract users and items
        for _, row in df.iterrows():
            user_id = str(row['user_id'])
            item_id = str(row['item_id'])

            # Create/update user
            if user_id not in self.users:
                user_feats = {f: row[f] for f in user_features if f in row and pd.notna(row[f])}
                self.add_user(User(user_id=user_id, features=user_feats))

            # Create/update item
            if item_id not in self.items:
                item_feats = {f: row[f] for f in item_features if f in row and pd.notna(row[f])}
                self.add_item(Item(item_id=item_id, features=item_feats))

            # Create interaction
            rating = row.get('rating')
            clicked = row.get('clicked', False)

            if pd.notna(rating):
                interaction = Interaction(
                    user_id=user_id,
                    item_id=item_id,
                    rating=float(rating),
                    clicked=bool(clicked) if pd.notna(clicked) else rating > 0
                )
            else:
                interaction = Interaction(
                    user_id=user_id,
                    item_id=item_id,
                    clicked=bool(clicked)
                )

            self.add_interaction(interaction)

        self.total_interactions = len(self.interactions)
        self.positive_interaction_rate = len(self.positive_interactions) / self.total_interactions

        print(f"Loaded {len(self.users)} users, {len(self.items)} items, "
              f"{self.total_interactions} interactions")

    def fit(self) -> List[RecRule]:
        """
        Learn recommendation rules from loaded data - memory optimized.

        Returns:
            List of learned rules
        """
        print("\n=== Learning Recommendation Rules ===")

        if not self.positive_interactions:
            print("No positive interactions found!")
            return []

        self.rules = []

        # Step 1: Generate initial rules from positive interactions
        print("Step 1: Generating initial rules...")
        initial_rules = self._generate_initial_rules()
        self.rules.extend(initial_rules)
        print(f"  Generated {len(initial_rules)} initial rules")

        # Free memory after initial rule generation
        del initial_rules
        gc.collect()

        # Step 2: Apply generalization (LGG) if enabled
        if self.enable_generalization and len(self.rules) < 1000:
            print("Step 2: Generalizing rules...")
            self._generalize_rules()
            print(f"  After generalization: {len(self.rules)} rules")
            gc.collect()
        elif self.enable_generalization:
            print(f"Step 2: Skipping generalization (too many rules: {len(self.rules)})")
        else:
            print("Step 2: Generalization disabled")

        # Step 3: Remove redundant rules using subsumption
        print("Step 3: Removing redundant rules...")
        self._apply_subsumption()
        print(f"  After subsumption: {len(self.rules)} rules")

        # Step 4: Filter by quality metrics
        print("Step 4: Filtering by quality metrics...")
        self._filter_rules()
        print(f"  Final: {len(self.rules)} rules")

        # Free unused indices after rule learning
        if hasattr(self, 'user_feat_index'):
            del self.user_feat_index
        if hasattr(self, 'item_feat_index'):
            del self.item_feat_index
        gc.collect()

        return self.rules

    def _generate_initial_rules(self) -> List[RecRule]:
        """Generate initial rules using FP-Growth style mining instead of brute force."""
        # Pre-compute indices for fast lookup
        self._build_interaction_indices()

        # Use frequent itemset mining approach
        rules = []
        rule_id_counter = 0

        # Build itemsets from positive interactions
        positive_itemsets = self._build_positive_itemsets()

        # Generate rules from frequent itemsets
        for itemset, support_count in positive_itemsets.items():
            if support_count < self.min_support * self.total_interactions * self._mining_sampling_fraction:
                continue

            # Split itemset into user and item conditions
            user_conds_list = []
            item_conds_list = []

            for feat, val in itemset:
                if feat.startswith('user_'):
                    user_conds_list.append((feat[5:], val))
                else:
                    item_conds_list.append((feat[5:], val))

            # Create rule
            user_conds = tuple(sorted(user_conds_list))
            item_conds = tuple(sorted(item_conds_list))

            rule = RecRule(
                rule_id=f"rule_{rule_id_counter}",
                user_conditions=user_conds,
                item_conditions=item_conds,
                rule_type=self._determine_rule_type(dict(user_conds), dict(item_conds))
            )

            # Calculate metrics
            if self._calculate_rule_metrics_lazy(rule):
                rules.append(rule)
                rule_id_counter += 1

        return rules

    def _build_positive_itemsets(self) -> Dict[tuple, int]:
        """Build frequent itemsets from positive interactions - memory optimized with sampling."""
        # For large datasets, use sampling to control memory usage
        sample_size = min(50000, len(self.positive_interactions))

        if len(self.positive_interactions) > 100000:
            # Sample positive interactions for very large datasets
            import random
            sampled_interactions = random.Random(42).sample(self.positive_interactions, sample_size)
            print(f"  Sampling {sample_size} positive interactions from {len(self.positive_interactions)} total")
        else:
            sampled_interactions = self.positive_interactions

        self._mining_sampling_fraction = len(sampled_interactions) / max(1, len(self.positive_interactions))
        itemset_counts = defaultdict(int)

        # Limit combinations to avoid combinatorial explosion
        max_user_features = 5  # Limit user features in combinations
        max_item_features = 5  # Limit item features in combinations

        for pos_int in tqdm(sampled_interactions, desc="Mining itemsets"):
            user = self.users.get(pos_int.user_id)
            item = self.items.get(pos_int.item_id)

            if user is None or item is None:
                continue

            # Build itemset for this interaction
            user_feats = []
            item_feats = []
            for feat, val in user.features.items():
                if val is not None:
                    user_feats.append((f'user_{feat}', val))
            for feat, val in item.features.items():
                if val is not None:
                    item_feats.append((f'item_{feat}', val))

            # Limit feature count to avoid explosion
            user_feats = user_feats[:max_user_features]
            item_feats = item_feats[:max_item_features]

            # Generate limited combinations - prioritize single features and simple pairs
            # 1. Single features (both user and item)
            for feat in user_feats:
                itemset_counts[tuple([feat])] += 1
            for feat in item_feats:
                itemset_counts[tuple([feat])] += 1

            # 2. Simple pairs (one user + one item feature)
            if user_feats and item_feats:
                for user_feat in user_feats[:2]:  # Only first 2 user features
                    for item_feat in item_feats[:2]:  # Only first 2 item features
                        itemset = tuple(sorted([user_feat, item_feat]))
                        itemset_counts[itemset] += 1

            # 3. Two-user or two-item features (only for small feature sets)
            if len(user_feats) >= 2:
                for combo in itertools.combinations(user_feats[:3], 2):
                    itemset_counts[tuple(sorted(combo))] += 1
            if len(item_feats) >= 2:
                for combo in itertools.combinations(item_feats[:3], 2):
                    itemset_counts[tuple(sorted(combo))] += 1

        return dict(itemset_counts)

    def _build_interaction_indices(self):
        """Pre-compute indices for fast rule evaluation - memory optimized."""
        # Feature value -> set of interaction indices
        self.user_feat_index = defaultdict(set)
        self.item_feat_index = defaultdict(set)
        self.positive_interaction_set = set()

        # Confidence needs every matching observation, including negative ones.
        # Large datasets use the separate sparse-array experiment engine.
        for idx, interaction in enumerate(tqdm(self.interactions, desc="Building full index")):
            user = self.users.get(interaction.user_id)
            item = self.items.get(interaction.item_id)
            if interaction.is_positive(self.rating_threshold):
                self.positive_interaction_set.add(idx)
            if user:
                for feat, val in user.features.items():
                    if val is not None:
                        self.user_feat_index[(feat, val)].add(idx)
            if item:
                for feat, val in item.features.items():
                    if val is not None:
                        self.item_feat_index[(feat, val)].add(idx)

    def _extract_user_feature_values(self) -> Dict[str, Set]:
        """Extract all unique values for each user feature."""
        feature_values = defaultdict(set)
        for user in self.users.values():
            for feat, val in user.features.items():
                if val is not None:
                    feature_values[feat].add(val)
        return dict(feature_values)

    def _extract_item_feature_values(self) -> Dict[str, Set]:
        """Extract all unique values for each item feature."""
        feature_values = defaultdict(set)
        for item in self.items.values():
            for feat, val in item.features.items():
                if val is not None:
                    feature_values[feat].add(val)
        return dict(feature_values)

    def _generate_rule_variants(self,
                                user: User,
                                item: Item,
                                user_feature_values: Dict[str, Set],
                                item_feature_values: Dict[str, Set],
                                rule_id_counter: int) -> List[RecRule]:
        """Generate rule variants for a specific user-item pair."""
        variants = []

        # Get available features
        user_feats = list(user.features.keys())
        item_feats = list(item.features.keys())

        # Generate combinations
        max_total = min(self.max_condition_size, len(user_feats) + len(item_feats))

        for total_conds in range(max_total + 1):
            for n_user_cond in range(min(total_conds, len(user_feats)) + 1):
                n_item_cond = total_conds - n_user_cond

                if n_item_cond > len(item_feats):
                    continue

                # User feature combinations
                for user_combo in itertools.combinations(user_feats, n_user_cond):
                    # Item feature combinations
                    for item_combo in itertools.combinations(item_feats, n_item_cond):
                        # Create rule - convert to tuples for immutability
                        user_conds_dict = {
                            f: user.get_feature_value(f)
                            for f in user_combo
                            if user.get_feature_value(f) is not None
                        }
                        item_conds_dict = {
                            f: item.get_feature_value(f)
                            for f in item_combo
                            if item.get_feature_value(f) is not None
                        }

                        if not user_conds_dict and not item_conds_dict:
                            continue

                        # Convert to sorted tuples for immutability
                        user_conds = tuple(sorted(user_conds_dict.items()))
                        item_conds = tuple(sorted(item_conds_dict.items()))

                        rule = RecRule(
                            rule_id=f"rule_{rule_id_counter + len(variants)}",
                            user_conditions=user_conds,
                            item_conditions=item_conds,
                            rule_type=self._determine_rule_type(user_conds_dict, item_conds_dict)
                        )

                        # Calculate metrics
                        self._calculate_rule_metrics(rule)

                        # Add if meets quality thresholds
                        if (rule.support >= self.min_support and
                            rule.confidence >= self.min_confidence and
                            rule.lift >= self.min_lift):
                            variants.append(rule)

        return variants

    def _generate_rule_variants_optimized(self,
                                          user: User,
                                          item: Item,
                                          rule_id_counter: int) -> List[RecRule]:
        """Generate rule variants using pre-computed indices with early pruning."""
        variants = []

        user_feats = list(user.features.keys())
        item_feats = list(item.features.keys())

        max_total = min(self.max_condition_size, len(user_feats) + len(item_feats))

        for total_conds in range(1, max_total + 1):
            for n_user_cond in range(min(total_conds, len(user_feats)) + 1):
                n_item_cond = total_conds - n_user_cond

                if n_item_cond > len(item_feats) or n_item_cond < 0:
                    continue

                user_combo_list = self._get_top_combinations(
                    user_feats, n_user_cond, max_combinations=10
                )
                item_combo_list = self._get_top_combinations(
                    item_feats, n_item_cond, max_combinations=10
                )

                for user_combo in user_combo_list:
                    for item_combo in item_combo_list:
                        user_conds_dict = {
                            f: user.get_feature_value(f)
                            for f in user_combo
                            if user.get_feature_value(f) is not None
                        }
                        item_conds_dict = {
                            f: item.get_feature_value(f)
                            for f in item_combo
                            if item.get_feature_value(f) is not None
                        }

                        if not user_conds_dict and not item_conds_dict:
                            continue

                        # Early pruning: estimate coverage before creating rule
                        if not self._check_minimum_coverage(user_conds_dict, item_conds_dict):
                            continue

                        user_conds = tuple(sorted(user_conds_dict.items()))
                        item_conds = tuple(sorted(item_conds_dict.items()))

                        rule = RecRule(
                            rule_id=f"rule_{rule_id_counter + len(variants)}",
                            user_conditions=user_conds,
                            item_conditions=item_conds,
                            rule_type=self._determine_rule_type(user_conds_dict, item_conds_dict)
                        )

                        # Lazy metric calculation
                        if self._calculate_rule_metrics_lazy(rule):
                            variants.append(rule)

        return variants

    def _check_minimum_coverage(self, user_conds: Dict, item_conds: Dict) -> bool:
        """Quick check if rule meets minimum support threshold (early pruning)."""
        # Calculate minimum required matches
        min_required = int(self.min_support * self.total_interactions)

        # Start with all indices
        coverage_size = self.total_interactions

        for feat, val in user_conds.items():
            if isinstance(val, set):
                # Union for set values
                union_size = 0
                for v in val:
                    union_size += len(self.user_feat_index.get((feat, v), set()))
                coverage_size = min(coverage_size, union_size)
            else:
                feat_size = len(self.user_feat_index.get((feat, val), set()))
                coverage_size = min(coverage_size, feat_size)

        for feat, val in item_conds.items():
            if isinstance(val, set):
                union_size = 0
                for v in val:
                    union_size += len(self.item_feat_index.get((feat, v), set()))
                coverage_size = min(coverage_size, union_size)
            else:
                feat_size = len(self.item_feat_index.get((feat, val), set()))
                coverage_size = min(coverage_size, feat_size)

        return coverage_size >= min_required

    def _calculate_rule_metrics_lazy(self, rule: RecRule) -> bool:
        """
        Calculate metrics lazily - stop early if failing thresholds.
        Returns True if rule passes all thresholds.
        """
        matching_indices = set(range(self.total_interactions))

        # Intersect user feature conditions
        for feat, val in rule.user_conditions:
            if isinstance(val, set):
                union_set = set()
                for v in val:
                    union_set.update(self.user_feat_index.get((feat, v), set()))
                matching_indices &= union_set
            else:
                matching_indices &= self.user_feat_index.get((feat, val), set())

        # Intersect item feature conditions
        for feat, val in rule.item_conditions:
            if isinstance(val, set):
                union_set = set()
                for v in val:
                    union_set.update(self.item_feat_index.get((feat, v), set()))
                matching_indices &= union_set
            else:
                matching_indices &= self.item_feat_index.get((feat, val), set())

        if not matching_indices:
            return False

        matches = len(matching_indices)
        rule.total_coverage = matches

        # Check support first (cheapest)
        rule.support = matches / self.total_interactions
        if rule.support < self.min_support:
            return False

        # Check confidence
        positive_matches = len(matching_indices & self.positive_interaction_set)
        rule.positive_coverage = positive_matches
        rule.confidence = positive_matches / matches if matches > 0 else 0

        if rule.confidence < self.min_confidence:
            return False

        # Only calculate lift if passing other thresholds
        if self.positive_interaction_rate > 0:
            rule.lift = rule.confidence / self.positive_interaction_rate
        else:
            rule.lift = float('inf') if rule.confidence > 0 else 0

        return rule.lift >= self.min_lift

    def _rank_features_by_frequency(self, features: List[str], is_user: bool) -> List[str]:
        """Rank features by how often they appear in positive interactions."""
        return features  # Keep original order for now

    def _get_top_combinations(self, items: List, size: int, max_combinations: int = 10) -> List[tuple]:
        """Get top combinations without generating all possibilities."""
        if size == 0:
            return [()]
        if size > len(items):
            return []

        combos = list(itertools.combinations(items, size))
        # If too many, sample to avoid explosion
        if len(combos) > max_combinations:
            import random
            random.seed(42)  # For reproducibility
            random.shuffle(combos)
            return combos[:max_combinations]
        return combos

    def _calculate_rule_metrics_fast(self, rule: RecRule):
        """Calculate metrics using pre-computed indices for O(1) feature lookup."""
        matching_indices = set(range(self.total_interactions))

        # Intersect user feature conditions
        for feat, val in rule.user_conditions:
            if isinstance(val, set):
                # Union for set values
                union_set = set()
                for v in val:
                    union_set.update(self.user_feat_index.get((feat, v), set()))
                matching_indices &= union_set
            else:
                matching_indices &= self.user_feat_index.get((feat, val), set())

        # Intersect item feature conditions
        for feat, val in rule.item_conditions:
            if isinstance(val, set):
                union_set = set()
                for v in val:
                    union_set.update(self.item_feat_index.get((feat, v), set()))
                matching_indices &= union_set
            else:
                matching_indices &= self.item_feat_index.get((feat, val), set())

        if not matching_indices:
            rule.support = 0
            rule.confidence = 0
            rule.lift = 0
            rule.total_coverage = 0
            rule.positive_coverage = 0
            return

        # Count positive matches using set intersection
        positive_matches = len(matching_indices & self.positive_interaction_set)
        matches = len(matching_indices)

        rule.total_coverage = matches
        rule.positive_coverage = positive_matches
        rule.support = matches / self.total_interactions
        rule.confidence = positive_matches / matches if matches > 0 else 0

        if self.positive_interaction_rate > 0:
            rule.lift = rule.confidence / self.positive_interaction_rate
        else:
            rule.lift = float('inf') if rule.confidence > 0 else 0

    def _determine_rule_type(self, user_conds: Dict, item_conds: Dict) -> RuleType:
        """Determine the type of rule based on conditions."""
        has_user = bool(user_conds)
        has_item = bool(item_conds)

        if has_user and has_item:
            return RuleType.HYBRID
        elif has_user:
            return RuleType.USER_BASED
        else:
            return RuleType.ITEM_BASED

    def _calculate_rule_metrics(self, rule: RecRule):
        """Calculate support, confidence, and lift for a rule."""
        matches = 0
        positive_matches = 0

        for interaction in self.interactions:
            user = self.users.get(interaction.user_id)
            item = self.items.get(interaction.item_id)

            if user is None or item is None:
                continue

            if rule.matches(user, item):
                matches += 1
                if interaction.is_positive(self.rating_threshold):
                    positive_matches += 1

        if matches == 0:
            rule.support = 0
            rule.confidence = 0
            rule.lift = 0
            return

        # Support: fraction of all interactions that match
        rule.support = matches / self.total_interactions

        # Confidence: fraction of matching interactions that are positive
        rule.confidence = positive_matches / matches

        # Lift: confidence / baseline positive rate
        if self.positive_interaction_rate > 0:
            rule.lift = rule.confidence / self.positive_interaction_rate
        else:
            rule.lift = float('inf') if rule.confidence > 0 else 0

        rule.total_coverage = matches
        rule.positive_coverage = positive_matches

    def _generalize_rules(self):
        """Generalize rules using LGG - memory optimized with early stopping."""
        # Skip generalization for very large rule sets to avoid memory issues
        if len(self.rules) > 500:
            print(f"  Skipping generalization (too many rules: {len(self.rules)})")
            return

        new_rules = []
        processed = set()
        max_new_rules = 200  # Limit number of generalized rules

        # Group rules by type and sort by confidence
        rules_by_type = defaultdict(list)
        for rule in self.rules:
            rules_by_type[rule.rule_type].append(rule)

        # Sort each group by confidence (best first)
        for rule_type in rules_by_type:
            rules_by_type[rule_type].sort(key=lambda r: r.confidence, reverse=True)

        for rule_type, rule_group in rules_by_type.items():
            # Limit generalization scope
            max_neighbors = min(50, len(rule_group))

            for i, r1 in enumerate(rule_group[:max_neighbors]):
                # Early stop if we have enough new rules
                if len(new_rules) >= max_new_rules:
                    break

                for j in range(i + 1, min(max_neighbors, len(rule_group))):
                    r2 = rule_group[j]

                    if r1 in processed or r2 in processed:
                        continue

                    # Check coverage overlap before attempting generalization
                    if not self._has_sufficient_overlap(r1, r2):
                        continue

                    # Try to generalize
                    generalized = self._lgg_rec_rules(r1, r2)
                    if generalized:
                        # Use lazy metric calculation
                        if self._calculate_rule_metrics_lazy(generalized):
                            if generalized not in self.rules and generalized not in new_rules:
                                new_rules.append(generalized)
                                processed.add(r1)
                                processed.add(r2)

                                # Early stop
                                if len(new_rules) >= max_new_rules:
                                    break

        # Add generalized rules
        self.rules.extend(new_rules)

    def _has_sufficient_overlap(self, r1: RecRule, r2: RecRule, min_overlap: float = 0.3) -> bool:
        """Check if two rules have sufficient coverage overlap to justify generalization."""
        if r1.total_coverage == 0 or r2.total_coverage == 0:
            return False

        # Quick overlap estimation using coverage sizes
        smaller = min(r1.total_coverage, r2.total_coverage)
        larger = max(r1.total_coverage, r2.total_coverage)

        # If one rule is much more specific than the other, skip
        if smaller / larger < min_overlap:
            return False

        return True

    def _lgg_rec_rules(self, r1: RecRule, r2: RecRule) -> Optional[RecRule]:
        """
        Compute LGG of two recommendation rules.

        For similar rules, creates a more general rule by:
        1. Keeping common conditions
        2. Generalizing specific values to sets when they differ
        """
        if r1.rule_type != r2.rule_type:
            return None

        # Convert tuples to dicts for processing
        r1_user_dict = dict(r1.user_conditions)
        r2_user_dict = dict(r2.user_conditions)

        # LGG for user conditions
        lgg_user_conds_dict = {}
        all_user_keys = set(r1_user_dict.keys()) | set(r2_user_dict.keys())

        for key in all_user_keys:
            val1 = r1_user_dict.get(key)
            val2 = r2_user_dict.get(key)

            if val1 is None:
                lgg_user_conds_dict[key] = val2
            elif val2 is None:
                lgg_user_conds_dict[key] = val1
            elif val1 == val2:
                lgg_user_conds_dict[key] = val1
            else:
                # Different values - create set
                if isinstance(val1, set):
                    values = set(val1)
                else:
                    values = {val1}
                if isinstance(val2, set):
                    values.update(val2)
                else:
                    values.add(val2)
                lgg_user_conds_dict[key] = values

        # LGG for item conditions
        r1_item_dict = dict(r1.item_conditions)
        r2_item_dict = dict(r2.item_conditions)

        lgg_item_conds_dict = {}
        all_item_keys = set(r1_item_dict.keys()) | set(r2_item_dict.keys())

        for key in all_item_keys:
            val1 = r1_item_dict.get(key)
            val2 = r2_item_dict.get(key)

            if val1 is None:
                lgg_item_conds_dict[key] = val2
            elif val2 is None:
                lgg_item_conds_dict[key] = val1
            elif val1 == val2:
                lgg_item_conds_dict[key] = val1
            else:
                # Different values - create set
                if isinstance(val1, set):
                    values = set(val1)
                else:
                    values = {val1}
                if isinstance(val2, set):
                    values.update(val2)
                else:
                    values.add(val2)
                lgg_item_conds_dict[key] = values

        # Convert back to sorted tuples
        lgg_user_conds = tuple(sorted(lgg_user_conds_dict.items()))
        lgg_item_conds = tuple(sorted(lgg_item_conds_dict.items()))

        return RecRule(
            rule_id=f"lgg_{r1.rule_id}_{r2.rule_id}",
            user_conditions=lgg_user_conds,
            item_conditions=lgg_item_conds,
            rule_type=r1.rule_type
        )

    def _apply_subsumption(self):
        """Remove redundant rules using hash-based deduplication (O(n) instead of O(n²))."""
        # Use a dictionary to keep the best rule for each condition signature
        best_by_signature = {}

        for rule in self.rules:
            # Create signature from conditions
            sig = self._get_rule_signature(rule)

            if sig not in best_by_signature:
                best_by_signature[sig] = rule
            else:
                # Keep the rule with better combined quality score
                existing = best_by_signature[sig]
                existing_score = existing.confidence * existing.lift * existing.support
                new_score = rule.confidence * rule.lift * rule.support

                # Also prefer fewer conditions (more general)
                existing_conds = len(existing.user_conditions) + len(existing.item_conditions)
                new_conds = len(rule.user_conditions) + len(rule.item_conditions)

                if new_score > existing_score or (new_score == existing_score and new_conds < existing_conds):
                    best_by_signature[sig] = rule

        self.rules = list(best_by_signature.values())

    def _get_rule_signature(self, rule: RecRule) -> tuple:
        """Create a signature for rule comparison that ignores order and handles sets."""
        # Convert conditions to hashable format (replace sets with sorted tuples)
        def make_hashable(conds):
            if not conds:
                return ()
            result = []
            for feat, val in conds:
                if isinstance(val, set):
                    # Convert set to sorted tuple for hashability
                    result.append((feat, tuple(sorted(val, key=str))))
                else:
                    result.append((feat, val))
            return tuple(sorted(result))

        user_sig = make_hashable(rule.user_conditions)
        item_sig = make_hashable(rule.item_conditions)
        return (rule.rule_type, user_sig, item_sig)

    def _subsumes(self, r1: RecRule, r2: RecRule) -> bool:
        """
        Check if r1 subsumes r2 (r1 is more general than r2).

        r1 subsumes r2 if:
        - Same rule type
        - r1's user conditions are subset of r2's
        - r1's item conditions are subset of r2's
        """
        if r1.rule_type != r2.rule_type:
            return False

        # Convert tuples to dicts for easier processing
        r1_user_dict = dict(r1.user_conditions)
        r2_user_dict = dict(r2.user_conditions)
        r1_item_dict = dict(r1.item_conditions)
        r2_item_dict = dict(r2.item_conditions)

        # Check if r1's user conditions are subset of r2's
        for key, val in r1_user_dict.items():
            if key not in r2_user_dict:
                return False
            val2 = r2_user_dict[key]
            if isinstance(val2, set):
                if not isinstance(val, set) and val not in val2:
                    return False
            elif val != val2:
                return False

        # Check if r1's item conditions are subset of r2's
        for key, val in r1_item_dict.items():
            if key not in r2_item_dict:
                return False
            val2 = r2_item_dict[key]
            if isinstance(val2, set):
                if not isinstance(val, set) and val not in val2:
                    return False
            elif val != val2:
                return False

        # r1 is more general if it has fewer or equal conditions
        n1_conds = len(r1.user_conditions) + len(r1.item_conditions)
        n2_conds = len(r2.user_conditions) + len(r2.item_conditions)

        return n1_conds <= n2_conds

    def _filter_rules(self):
        """Filter rules by quality metrics and keep top rules, prioritizing hybrid rules."""
        # Filter by quality
        self.rules = [
            r for r in self.rules
            if (r.support >= self.min_support and
                r.confidence >= self.min_confidence and
                r.lift >= self.min_lift)
        ]

        # Sort by combined score with bonus for hybrid rules (personalization)
        def rule_score(r):
            base_score = r.confidence * r.lift * r.support
            # Bonus for hybrid rules (have both user and item conditions)
            if r.rule_type == RuleType.HYBRID and r.user_conditions and r.item_conditions:
                base_score *= 2.0  # 2x bonus for personalized rules
            return base_score

        self.rules.sort(key=rule_score, reverse=True)

        # Keep only top rules
        if len(self.rules) > self.max_rules:
            self.rules = self.rules[:self.max_rules]

    def recommend(self,
                  user_id: str,
                  top_k: int = 10,
                  exclude_seen: bool = True) -> List[Recommendation]:
        """
        Generate recommendations for a user.

        Args:
            user_id: ID of user to recommend for
            top_k: Number of recommendations to return
            exclude_seen: Whether to exclude items user has already seen

        Returns:
            List of recommendations with scores and explanations
        """
        user = self.users.get(user_id)
        if user is None:
            return []

        seen_items = set(user.history) if exclude_seen else set()
        recommendations = []

        items_to_check = self.items.items()
        for item_id, item in tqdm(items_to_check,
                                 desc=f"Recommending for {user_id}",
                                 leave=False,
                                 total=len(self.items)):
            if item_id in seen_items:
                continue

            # Calculate score based on matching rules
            score = 0.0
            matched_rules = []

            for rule in self.rules:
                if rule.matches(user, item):
                    score += rule.confidence * rule.lift
                    matched_rules.append(rule.rule_id)

            if matched_rules:
                # Generate explanation
                explanation = self._generate_explanation(user, item, matched_rules)

                recommendations.append(Recommendation(
                    user_id=user_id,
                    item_id=item_id,
                    score=score,
                    explanation=explanation,
                    matched_rules=matched_rules
                ))

        # Sort by score
        recommendations.sort(key=lambda r: (-r.score, r.item_id))

        return recommendations[:top_k]

    def _generate_explanation(self,
                             user: User,
                             item: Item,
                             matched_rule_ids: List[str]) -> str:
        """Generate human-readable explanation for recommendation."""
        if not matched_rule_ids:
            return "Recommended based on overall popularity"

        matched_rules = [r for r in self.rules if r.rule_id in matched_rule_ids]
        top_rule = max(matched_rules, key=lambda r: r.confidence * r.lift)

        parts = []

        # User-based explanation
        if top_rule.user_conditions:
            user_parts = []
            for feat, val in top_rule.user_conditions:  # Iterate over tuple
                if isinstance(val, set):
                    user_parts.append(f"your {feat} is in {val}")
                else:
                    user_parts.append(f"your {feat} is {val}")
            if user_parts:
                parts.append("since " + " and ".join(user_parts))

        # Item-based explanation
        if top_rule.item_conditions:
            item_parts = []
            for feat, val in top_rule.item_conditions:  # Iterate over tuple
                if isinstance(val, set):
                    item_parts.append(f"item has {feat} in {val}")
                else:
                    item_parts.append(f"item has {feat} = {val}")
            if item_parts:
                parts.append("and " + " and ".join(item_parts))

        explanation = "Recommended " + " ".join(parts)
        explanation += f" (confidence: {top_rule.confidence:.1%})"

        return explanation

    def explain_recommendation(self,
                                user_id: str,
                                item_id: str) -> Optional[str]:
        """Explain why a specific item would be recommended to a user."""
        user = self.users.get(user_id)
        item = self.items.get(item_id)

        if user is None or item is None:
            return None

        matched_rules = [r for r in self.rules if r.matches(user, item)]

        if not matched_rules:
            return "No specific rules match this user-item combination"

        return self._generate_explanation(user, item, [r.rule_id for r in matched_rules])

    def print_rules(self, max_rules: int = 20):
        """Print learned rules."""
        print(f"\n=== Learned Recommendation Rules (showing top {min(max_rules, len(self.rules))}) ===")
        for i, rule in enumerate(self.rules[:max_rules], 1):
            print(f"\n{i}. {rule}")

    def get_rule_summary(self) -> pd.DataFrame:
        """Get summary of learned rules as DataFrame."""
        data = []
        for rule in self.rules:
            data.append({
                'rule_id': rule.rule_id,
                'type': rule.rule_type.value,
                'user_conditions': len(rule.user_conditions),
                'item_conditions': len(rule.item_conditions),
                'confidence': rule.confidence,
                'support': rule.support,
                'lift': rule.lift,
                'coverage': f"{rule.positive_coverage}/{rule.total_coverage}"
            })
        return pd.DataFrame(data)

    def save_rules(self, filepath: str):
        """Save learned rules to file."""
        import json

        rules_data = []
        for rule in self.rules:
            # Convert tuple conditions to dict for JSON serialization
            user_conds_dict = dict(rule.user_conditions)
            item_conds_dict = dict(rule.item_conditions)
            inter_conds_dict = dict(rule.interaction_conditions)

            # Convert sets to lists for JSON serialization
            for key, val in list(user_conds_dict.items()):
                if isinstance(val, set):
                    user_conds_dict[key] = sorted(val, key=repr)
            for key, val in list(item_conds_dict.items()):
                if isinstance(val, set):
                    item_conds_dict[key] = sorted(val, key=repr)
            for key, val in list(inter_conds_dict.items()):
                if isinstance(val, set):
                    inter_conds_dict[key] = sorted(val, key=repr)

            rule_data = {
                'rule_id': rule.rule_id,
                'user_conditions': user_conds_dict,
                'item_conditions': item_conds_dict,
                'interaction_conditions': inter_conds_dict,
                'confidence': rule.confidence,
                'support': rule.support,
                'lift': rule.lift,
                'rule_type': rule.rule_type.value,
                'positive_coverage': rule.positive_coverage,
                'total_coverage': rule.total_coverage
            }

            rules_data.append(rule_data)

        with open(filepath, 'w') as f:
            json.dump(rules_data, f, indent=2)

        print(f"Saved {len(rules_data)} rules to {filepath}")

    def load_rules(self, filepath: str):
        """Load rules from file."""
        import json

        with open(filepath, 'r') as f:
            rules_data = json.load(f)

        self.rules = []
        for rule_data in rules_data:
            # JSON arrays encode set-valued alternatives in generalized rules.
            def restore_conditions(name):
                return tuple(sorted((feat, set(val) if isinstance(val, list) else val)
                                    for feat, val in rule_data.get(name, {}).items()))

            user_conds = restore_conditions('user_conditions')
            item_conds = restore_conditions('item_conditions')
            inter_conds = restore_conditions('interaction_conditions')

            rule = RecRule(
                rule_id=rule_data['rule_id'],
                user_conditions=user_conds,
                item_conditions=item_conds,
                interaction_conditions=inter_conds,
                confidence=rule_data.get('confidence', 0.0),
                support=rule_data.get('support', 0.0),
                lift=rule_data.get('lift', 0.0),
                rule_type=RuleType(rule_data.get('rule_type', 'hybrid')),
                positive_coverage=rule_data.get('positive_coverage', 0),
                total_coverage=rule_data.get('total_coverage', 0)
            )
            self.rules.append(rule)

        print(f"Loaded {len(self.rules)} rules from {filepath}")


# ============================================================
# Evaluation Metrics for Recommender Systems
# ============================================================

class RecEvaluator:
    """Evaluator for recommender systems."""

    @staticmethod
    def precision_at_k(recommendations: List[Recommendation],
                       relevant_items: Set[str],
                       k: int) -> float:
        """Calculate Precision@K, treating unfilled recommendation slots as misses."""
        if k < 1:
            raise ValueError("k must be positive")
        if not recommendations:
            return 0.0

        rec_at_k = recommendations[:k]
        hits = sum(1 for r in rec_at_k if r.item_id in relevant_items)
        return hits / k

    @staticmethod
    def recall_at_k(recommendations: List[Recommendation],
                    relevant_items: Set[str],
                    k: int) -> float:
        """Calculate Recall@K."""
        if not relevant_items:
            return 0.0

        rec_at_k = recommendations[:k]
        hits = sum(1 for r in rec_at_k if r.item_id in relevant_items)
        return hits / len(relevant_items)

    @staticmethod
    def ndcg_at_k(recommendations: List[Recommendation],
                  relevant_items: Dict[str, float],
                  k: int) -> float:
        """Calculate NDCG@K."""
        if not recommendations:
            return 0.0

        rec_at_k = recommendations[:k]
        dcg = 0.0

        for i, rec in enumerate(rec_at_k):
            if rec.item_id in relevant_items:
                dcg += relevant_items[rec.item_id] / np.log2(i + 2)

        # Ideal DCG
        ideal_relevances = sorted(relevant_items.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

        return dcg / idcg if idcg > 0 else 0.0


# ============================================================
# Example Usage and Demo
# ============================================================

def create_demo_data() -> Tuple[List[User], List[Item], List[Interaction]]:
    """Create demo data for testing LFIORecSys."""
    print("Creating demo dataset...")

    # Create users with features
    users = [
        User(user_id="u1", features={"age": 25, "gender": "M", "pref_genre": "action"}),
        User(user_id="u2", features={"age": 35, "gender": "F", "pref_genre": "drama"}),
        User(user_id="u3", features={"age": 28, "gender": "M", "pref_genre": "action"}),
        User(user_id="u4", features={"age": 42, "gender": "F", "pref_genre": "comedy"}),
        User(user_id="u5", features={"age": 31, "gender": "M", "pref_genre": "sci_fi"}),
        User(user_id="u6", features={"age": 27, "gender": "F", "pref_genre": "action"}),
        User(user_id="u7", features={"age": 38, "gender": "M", "pref_genre": "drama"}),
        User(user_id="u8", features={"age": 29, "gender": "F", "pref_genre": "sci_fi"}),
    ]

    # Create items with features
    items = [
        Item(item_id="i1", features={"genre": "action", "year": 2020, "rating": 8.5}),
        Item(item_id="i2", features={"genre": "drama", "year": 2019, "rating": 7.8}),
        Item(item_id="i3", features={"genre": "action", "year": 2021, "rating": 9.0}),
        Item(item_id="i4", features={"genre": "comedy", "year": 2018, "rating": 7.2}),
        Item(item_id="i5", features={"genre": "sci_fi", "year": 2022, "rating": 8.9}),
        Item(item_id="i6", features={"genre": "action", "year": 2019, "rating": 8.1}),
        Item(item_id="i7", features={"genre": "drama", "year": 2021, "rating": 8.3}),
        Item(item_id="i8", features={"genre": "sci_fi", "year": 2020, "rating": 8.7}),
    ]

    # Create interactions (positive ones)
    interactions = [
        # Users who like action
        Interaction(user_id="u1", item_id="i1", rating=5),
        Interaction(user_id="u1", item_id="i3", rating=4),
        Interaction(user_id="u3", item_id="i1", rating=5),
        Interaction(user_id="u3", item_id="i6", rating=4),
        Interaction(user_id="u6", item_id="i3", rating=5),
        Interaction(user_id="u6", item_id="i1", rating=4),

        # Users who like drama
        Interaction(user_id="u2", item_id="i2", rating=5),
        Interaction(user_id="u2", item_id="i7", rating=4),
        Interaction(user_id="u7", item_id="i2", rating=4),
        Interaction(user_id="u7", item_id="i7", rating=5),

        # Users who like sci_fi
        Interaction(user_id="u5", item_id="i5", rating=5),
        Interaction(user_id="u5", item_id="i8", rating=4),
        Interaction(user_id="u8", item_id="i5", rating=5),
        Interaction(user_id="u8", item_id="i8", rating=4),

        # Cross-genre (action users also like sci_fi)
        Interaction(user_id="u1", item_id="i5", rating=4),
        Interaction(user_id="u3", item_id="i8", rating=4),

        # Comedy for older users
        Interaction(user_id="u4", item_id="i4", rating=4),
        Interaction(user_id="u7", item_id="i4", rating=3),
    ]

    # Add negative interactions for better learning (clicked=False for implicit feedback)
    negative_interactions = [
        Interaction(user_id="u2", item_id="i1", rating=2, clicked=False),  # Drama user doesn't like action
        Interaction(user_id="u4", item_id="i3", rating=1, clicked=False),  # Comedy user doesn't like action
        Interaction(user_id="u1", item_id="i2", rating=2, clicked=False),  # Action user doesn't like drama
        Interaction(user_id="u5", item_id="i4", rating=1, clicked=False),  # Sci-fi user doesn't like comedy
        Interaction(user_id="u6", item_id="i7", rating=2, clicked=False),  # Action user doesn't like drama
    ]

    interactions.extend(negative_interactions)

    print(f"Created {len(users)} users, {len(items)} items, {len(interactions)} interactions")

    return users, items, interactions


def run_demo():
    """Run a demonstration of LFIORecSys."""
    print("\n" + "="*60)
    print("LFIO RECOMMENDER SYSTEM DEMONSTRATION")
    print("="*60)

    # Create demo data
    users, items, interactions = create_demo_data()

    # Initialize LFIO Recommender with adjusted thresholds for demo data
    recsys = LFIORecSys(
        min_support=0.02,  # Lower threshold for small demo dataset
        min_confidence=0.5,  # Lower threshold
        min_lift=1.0,  # Lower threshold
        max_rules=30,
        max_condition_size=2,  # Limit condition size for small dataset
        enable_generalization=True
    )

    # Load data
    recsys.load_data(users, items, interactions)

    # Learn rules
    recsys.fit()

    # Print learned rules
    recsys.print_rules(max_rules=15)

    # Generate recommendations
    print("\n" + "="*60)
    print("GENERATING RECOMMENDATIONS")
    print("="*60)

    for user_id in ["u1", "u2", "u5"]:
        print(f"\n--- Recommendations for user {user_id} ---")
        print(f"User features: {recsys.users[user_id].features}")

        recommendations = recsys.recommend(user_id, top_k=3)

        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. Item {rec.item_id}: {rec.explanation}")
            print(f"   Score: {rec.score:.3f}")

    # Show rule summary
    print("\n" + "="*60)
    print("RULE SUMMARY")
    print("="*60)

    summary_df = recsys.get_rule_summary()
    print(summary_df.to_string(index=False))

    # Explain specific recommendation
    print("\n" + "="*60)
    print("EXPLAIN SPECIFIC RECOMMENDATION")
    print("="*60)

    explanation = recsys.explain_recommendation("u1", "i5")
    print(f"Why recommend item i5 to user u1?")
    print(f"  {explanation}")

    return recsys


if __name__ == "__main__":
    run_demo()
