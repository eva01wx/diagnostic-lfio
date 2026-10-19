"""
MovieLens LFIO Recommender Experiment

This script runs the LFIO recommender on the MovieLens dataset.
It demonstrates learning recommendation rules from real movie rating data.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import json
import gc
import hashlib
import time
from tqdm import tqdm

from src.lfio_recommender import (
    LFIORecSys, User, Item, Interaction,
    RecEvaluator, FeedbackType, Recommendation
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DATASET_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'dataset'))


def resolve_movielens_path(data_path: str) -> str:
    """Resolve MovieLens paths, preferring ../dataset for bare dataset names."""
    candidates = []

    if os.path.isabs(data_path):
        candidates.append(data_path)
    else:
        has_directory = os.path.dirname(data_path) not in ("", ".")
        if not has_directory:
            candidates.append(os.path.join(DEFAULT_DATASET_ROOT, data_path))

        candidates.extend([
            os.path.abspath(data_path),
            os.path.join(PROJECT_ROOT, data_path),
        ])

        if has_directory:
            candidates.append(os.path.join(DEFAULT_DATASET_ROOT, data_path))

    checked = []
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in checked:
            continue
        checked.append(candidate)

        movies_path = os.path.join(candidate, "movies.csv")
        ratings_path = os.path.join(candidate, "ratings.csv")
        if os.path.isfile(movies_path) and os.path.isfile(ratings_path):
            return candidate

    checked_paths = "\n  ".join(checked)
    raise FileNotFoundError(
        "Could not find MovieLens dataset. Expected movies.csv and ratings.csv in one of:\n"
        f"  {checked_paths}"
    )


class MovieLensExperiment:
    """Run LFIO experiments on MovieLens dataset."""

    def __init__(self, data_path: str = "ml-latest-small"):
        self.data_path = resolve_movielens_path(data_path)
        self.movies_df = None
        self.ratings_df = None
        self.test_ratings_df = None
        self.split_metadata = {}
        self.evaluation_rows = []
        self.users: Dict[str, User] = {}
        self.items: Dict[str, Item] = {}
        self.interactions: List[Interaction] = []

        # Experiment settings
        self.rating_threshold = 4.0  # Ratings >= 4 are considered positive
        self.min_interactions = 5   # Minimum interactions per user/item
        self.max_users = 500        # Limit users for faster experimentation
        self.max_items = 500        # Limit items for faster experimentation

    def load_data(self, chunksize: int = 100000):
        """Load MovieLens data using chunked reading for memory efficiency."""
        print("="*60)
        print("LOADING MOVIELENS DATASET")
        print("="*60)

        # Load movies (usually smaller, can load at once)
        movies_path = os.path.join(self.data_path, "movies.csv")
        self.movies_df = pd.read_csv(movies_path)
        print(f"Loaded {len(self.movies_df)} movies")

        # Parse genres
        self.movies_df['genres_list'] = self.movies_df['genres'].apply(
            lambda x: x.split('|') if isinstance(x, str) else []
        )

        # Load ratings in chunks for large datasets
        ratings_path = os.path.join(self.data_path, "ratings.csv")

        # First, get the file size to decide whether to use chunked reading
        file_size = os.path.getsize(ratings_path)
        use_chunks = file_size > 50 * 1024 * 1024  # Use chunks for files > 50MB

        if use_chunks:
            # Estimate total chunks for progress bar
            total_rows = sum(1 for _ in open(ratings_path)) - 1  # minus header
            total_chunks = (total_rows // chunksize) + 1

            print(f"Loading ratings in chunks (file size: {file_size / (1024*1024):.1f}MB)...")
            chunks = []
            for chunk in tqdm(pd.read_csv(ratings_path, chunksize=chunksize),
                            total=total_chunks,
                            desc="Loading chunks",
                            unit="chunk"):
                chunks.append(chunk)
                # Free memory between chunks
                gc.collect()
            self.ratings_df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
        else:
            self.ratings_df = pd.read_csv(ratings_path)

        print(f"Loaded {len(self.ratings_df)} ratings")

        # Statistics
        n_users = self.ratings_df['userId'].nunique()
        n_movies = self.ratings_df['movieId'].nunique()
        avg_rating = self.ratings_df['rating'].mean()

        print(f"\nDataset Statistics:")
        print(f"  Users: {n_users}")
        print(f"  Movies rated: {n_movies}")
        print(f"  Total ratings: {len(self.ratings_df)}")
        print(f"  Average rating: {avg_rating:.2f}")

        return self.movies_df, self.ratings_df

    def split_dataset(self, test_ratio: float = 0.2):
        """Hold out the latest ratings globally, keeping timestamp ties together."""
        if not 0 < test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")
        ratings = self.ratings_df.sort_values(
            ['timestamp', 'userId', 'movieId'], kind='stable'
        ).reset_index(drop=True)
        split_index = int(len(ratings) * (1 - test_ratio))
        if not 0 < split_index < len(ratings):
            raise ValueError("Not enough ratings for a chronological split")
        cutoff = int(ratings.iloc[split_index]['timestamp'])
        self.ratings_df = ratings[ratings['timestamp'] < cutoff].copy()
        self.test_ratings_df = ratings[ratings['timestamp'] >= cutoff].copy()
        if self.ratings_df.empty:
            raise ValueError("Timestamp ties leave no training ratings")
        self.split_metadata = {
            'protocol': 'global_temporal_warm_start',
            'requested_test_ratio': test_ratio,
            'cutoff_timestamp': cutoff,
            'raw_ratings': len(ratings),
            'raw_train_ratings': len(self.ratings_df),
            'raw_test_ratings': len(self.test_ratings_df),
        }
        return self.ratings_df, self.test_ratings_df

    def filter_dataset(self):
        """Select the user/item catalogue using training interactions only."""
        if self.test_ratings_df is None:
            raise ValueError("Call split_dataset before filtering or extracting features")
        print("\n" + "="*60)
        print("FILTERING DATASET")
        print("="*60)

        # Count interactions per user and item
        user_counts = self.ratings_df['userId'].value_counts()
        item_counts = self.ratings_df['movieId'].value_counts()

        # Get active users and popular items
        active_users = user_counts[user_counts >= self.min_interactions]
        popular_items = item_counts[item_counts >= self.min_interactions]

        # Apply max limits if specified (not None and not -1)
        if self.max_users and self.max_users != -1:
            active_users = active_users.head(self.max_users).index
        else:
            active_users = active_users.index

        if self.max_items and self.max_items != -1:
            popular_items = popular_items.head(self.max_items).index
        else:
            popular_items = popular_items.index

        # Filter ratings (use isin which is more memory efficient)
        filtered_df = self.ratings_df[
            self.ratings_df['userId'].isin(active_users) &
            self.ratings_df['movieId'].isin(popular_items)
        ].copy()

        print(f"After filtering:")
        print(f"  Users: {len(active_users)}")
        print(f"  Movies: {len(popular_items)}")
        print(f"  Ratings: {len(filtered_df)}")

        # Update ratings and movies
        self.ratings_df = filtered_df
        if filtered_df.empty:
            raise ValueError("No training ratings remain after filtering")
        self.movies_df = self.movies_df[
            self.movies_df['movieId'].isin(filtered_df['movieId'].unique())
        ].copy()
        known_user = self.test_ratings_df['userId'].isin(filtered_df['userId'].unique())
        known_item = self.test_ratings_df['movieId'].isin(filtered_df['movieId'].unique())
        self.split_metadata.update({
            'train_ratings': len(filtered_df),
            'train_users': int(filtered_df['userId'].nunique()),
            'train_items': int(filtered_df['movieId'].nunique()),
            'test_ratings_excluded_unknown_user': int((~known_user).sum()),
            'test_ratings_excluded_unknown_item_for_known_user': int((known_user & ~known_item).sum()),
        })
        self.test_ratings_df = self.test_ratings_df[known_user & known_item].copy()
        self.split_metadata['test_ratings_in_catalogue'] = len(self.test_ratings_df)
        for name, frame in [('train', self.ratings_df), ('test', self.test_ratings_df)]:
            payload = frame[['userId', 'movieId', 'rating', 'timestamp']].to_csv(index=False)
            self.split_metadata[f'{name}_sha256'] = hashlib.sha256(payload.encode()).hexdigest()

        # Free memory
        del filtered_df, user_counts, item_counts
        gc.collect()

        return self.ratings_df

    def extract_features(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Extract user and item features from ratings data - memory optimized."""
        print("\n" + "="*60)
        print("EXTRACTING FEATURES")
        print("="*60)

        user_features = defaultdict(dict)
        item_features = defaultdict(dict)

        # Build movie genre map for fast lookup (avoids expensive merge)
        movie_genre_map = {}
        for _, row in self.movies_df.iterrows():
            movie_genre_map[int(row['movieId'])] = row['genres_list']

        # User features - optimized to avoid per-user filtering
        user_stats = df.groupby('userId')['rating'].agg(['mean', 'count', 'std'])
        user_stats['std'] = user_stats['std'].fillna(0)

        # Count high ratings per user efficiently using vectorized operations
        high_ratings_mask = df['rating'] >= self.rating_threshold
        high_rating_counts = df[high_ratings_mask].groupby('userId').size()

        # Calculate genre preferences per user without merge
        user_genre_counts = defaultdict(lambda: defaultdict(int))
        for _, row in tqdm(df.iterrows(), total=len(df),
                          desc="Processing genres"):
            user_id = row['userId']
            movie_id = int(row['movieId'])
            genres = movie_genre_map.get(movie_id, [])
            for genre in genres:
                user_genre_counts[user_id][genre] += 1

        for user_id in tqdm(df['userId'].unique(), desc="Extracting user features"):
            uid = str(int(user_id))
            stats = user_stats.loc[user_id]

            # Rating behavior (categorical) - use pre-computed counts
            high_count = high_rating_counts.get(user_id, 0)
            high_ratio = high_count / stats['count']

            user_features[uid]['num_ratings'] = int(stats['count'])

            # Discretize rating behavior into categories
            if high_ratio >= 0.7:
                user_features[uid]['rating_pattern'] = 'selective_positive'
            elif high_ratio >= 0.4:
                user_features[uid]['rating_pattern'] = 'mixed'
            else:
                user_features[uid]['rating_pattern'] = 'critical'

            # Discretize average rating
            avg_rating = stats['mean']
            if avg_rating >= 4.0:
                user_features[uid]['rating_level'] = 'high'
            elif avg_rating >= 3.0:
                user_features[uid]['rating_level'] = 'medium'
            else:
                user_features[uid]['rating_level'] = 'low'

            # Activity level
            if stats['count'] >= 50:
                user_features[uid]['activity_level'] = 'high'
            elif stats['count'] >= 20:
                user_features[uid]['activity_level'] = 'medium'
            else:
                user_features[uid]['activity_level'] = 'low'

            # Top genre preference - use pre-computed counts
            if user_id in user_genre_counts:
                genres = user_genre_counts[user_id]
                if genres:
                    top_genre = max(genres.items(), key=lambda x: x[1])[0]
                    user_features[uid]['top_genre'] = top_genre

        # Free memory
        del user_genre_counts, movie_genre_map, high_rating_counts
        gc.collect()

        # Item features - pre-compute movie stats efficiently
        movie_stats = df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['movieId', 'avg_rating', 'rating_count']

        for _, movie in tqdm(self.movies_df.iterrows(),
                           total=len(self.movies_df),
                           desc="Extracting item features"):
            mid = str(int(movie['movieId']))

            # Extract year from title
            title = movie['title']
            year = None
            if '(' in title and ')' in title:
                try:
                    year_str = title[title.rfind('(')+1:title.rfind(')')]
                    year = int(year_str)
                except:
                    pass

            item_features[mid]['title'] = title

            if year:
                item_features[mid]['year'] = year
                item_features[mid]['decade'] = (year // 10) * 10

            # Genres
            genres = movie['genres_list']
            if genres:
                item_features[mid]['genre'] = genres[0]
                item_features[mid]['num_genres'] = len(genres)
            else:
                item_features[mid]['genre'] = 'Unknown'
                item_features[mid]['num_genres'] = 0

            # Rating stats for this movie - use pre-computed stats
            movie_stat = movie_stats[movie_stats['movieId'] == int(movie['movieId'])]
            if len(movie_stat) > 0:
                avg_rating = movie_stat['avg_rating'].values[0]
                rating_count = movie_stat['rating_count'].values[0]

                # Discretize average rating
                if avg_rating >= 4.0:
                    item_features[mid]['rating_level'] = 'high'
                elif avg_rating >= 3.0:
                    item_features[mid]['rating_level'] = 'medium'
                else:
                    item_features[mid]['rating_level'] = 'low'

                # Popularity tier
                if rating_count >= 50:
                    item_features[mid]['popularity'] = 'high'
                elif rating_count >= 20:
                    item_features[mid]['popularity'] = 'medium'
                else:
                    item_features[mid]['popularity'] = 'low'

        # Free memory
        del movie_stats
        gc.collect()

        print(f"Extracted features for {len(user_features)} users")
        print(f"Extracted features for {len(item_features)} movies")

        return dict(user_features), dict(item_features)

    def create_lfio_objects(self, user_features: Dict, item_features: Dict):
        """Create LFIO User, Item, and Interaction objects - memory optimized."""
        print("\n" + "="*60)
        print("CREATING LFIO OBJECTS")
        print("="*60)

        self.users = {}
        self.items = {}
        self.interactions = []

        # Create users
        for user_id, features in user_features.items():
            user = User(user_id=user_id, features=features)
            self.users[user_id] = user

        # Create items
        for item_id, features in item_features.items():
            item = Item(item_id=item_id, features=features)
            self.items[item_id] = item

        # Create interactions - use list comprehension for better memory efficiency
        positive_count = 0
        for _, row in tqdm(self.ratings_df.iterrows(),
                         total=len(self.ratings_df),
                         desc="Creating interactions"):
            user_id = str(int(row['userId']))
            item_id = str(int(row['movieId']))
            rating = float(row['rating'])

            interaction = Interaction(
                user_id=user_id,
                item_id=item_id,
                rating=rating,
                clicked=rating >= self.rating_threshold,
                timestamp=float(row['timestamp'])
            )

            self.interactions.append(interaction)

            if interaction.is_positive(threshold=self.rating_threshold):
                positive_count += 1

        positive_rate = positive_count / len(self.interactions)
        print(f"Created {len(self.users)} users")
        print(f"Created {len(self.items)} items")
        print(f"Created {len(self.interactions)} interactions")
        print(f"Positive interaction rate (rating >= {self.rating_threshold}): {positive_rate:.2%}")

        # Free memory - we don't need the original dataframes anymore
        del self.ratings_df, self.movies_df
        self.ratings_df = None
        self.movies_df = None
        gc.collect()

        return self.users, self.items, self.interactions

    def run_experiment(self,
                      min_support: float = 0.02,
                      min_confidence: float = 0.6,
                      min_lift: float = 1.2,
                      max_rules: int = 50):
        """Run the main LFIO experiment."""
        print("\n" + "="*60)
        print("RUNNING LFIO RECOMMENDER EXPERIMENT")
        print("="*60)

        # Initialize LFIO recommender
        recsys = LFIORecSys(
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_rules=max_rules,
            max_condition_size=3,
            enable_generalization=True,
            feedback_type=FeedbackType.EXPLICIT,
            rating_threshold=self.rating_threshold
        )

        # Load data
        recsys.load_data(list(self.users.values()),
                        list(self.items.values()),
                        self.interactions)

        # Learn rules
        recsys.fit()

        return recsys

    def evaluate_rules(self, recsys: LFIORecSys, top_k: int = 10):
        """Evaluate the fitted training model on unseen, warm-start test positives.

        Rank the full training catalogue for each eligible user. Empty or short
        recommendation lists count as misses; no model is refitted here.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self.test_ratings_df is None:
            raise ValueError("Call split_dataset before evaluation")
        positive_counts = defaultdict(int)
        for interaction in self.interactions:
            if interaction.is_positive(self.rating_threshold):
                positive_counts[interaction.item_id] += 1
        popularity_order = sorted(self.items, key=lambda iid: (-positive_counts[iid], iid))

        relevant_by_user = defaultdict(set)
        excluded_seen = 0
        for row in self.test_ratings_df.itertuples(index=False):
            uid, iid = str(int(row.userId)), str(int(row.movieId))
            if row.rating < self.rating_threshold:
                continue
            if iid in self.users[uid].history:
                excluded_seen += 1
                continue
            relevant_by_user[uid].add(iid)

        self.evaluation_rows = []
        recommended_items = {'lfio': set(), 'popularity': set()}
        for uid in tqdm(sorted(relevant_by_user), desc="Evaluating users"):
            relevant = relevant_by_user[uid]
            seen = set(self.users[uid].history)
            popular = [iid for iid in popularity_order if iid not in seen][:top_k]
            baseline = [Recommendation(uid, iid, float(positive_counts[iid]), 'Training popularity', [])
                        for iid in popular]
            for method, recs in [('lfio', recsys.recommend(uid, top_k=top_k)),
                                 ('popularity', baseline)]:
                rec_items = {rec.item_id for rec in recs}
                recommended_items[method].update(rec_items)
                self.evaluation_rows.append({
                    'user_id': uid,
                    'method': method,
                    'num_relevant': len(relevant),
                    'num_recommendations': len(recs),
                    'hits': len(rec_items & relevant),
                    'precision': RecEvaluator.precision_at_k(recs, relevant, top_k),
                    'recall': RecEvaluator.recall_at_k(recs, relevant, top_k),
                    'ndcg': float(RecEvaluator.ndcg_at_k(recs, dict.fromkeys(relevant, 1.0), top_k)),
                })

        def summarize(method):
            rows = [r for r in self.evaluation_rows if r['method'] == method]
            # No eligible users means undefined metrics, not a measured zero.
            means = {key: float(np.mean([r[key] for r in rows])) if rows else None
                     for key in ('precision', 'recall', 'ndcg')}
            p, r = means['precision'], means['recall']
            return {
                **means,
                'f1': (2 * p * r / (p + r) if p + r else 0.0) if rows else None,
                'hit_rate': float(np.mean([r['hits'] > 0 for r in rows])) if rows else None,
                'user_coverage': float(np.mean([r['num_recommendations'] > 0 for r in rows])) if rows else None,
                'catalogue_coverage': len(recommended_items[method]) / len(self.items) if rows else None,
                'num_users': len(rows),
            }

        metrics = {
            **summarize('lfio'),
            'popularity_baseline': summarize('popularity'),
            'top_k': top_k,
            'num_rules': len(recsys.rules),
            'split': {
                **self.split_metadata,
                'test_users_in_catalogue': int(self.test_ratings_df['userId'].nunique()),
                'test_users_without_unseen_positive': int(self.test_ratings_df['userId'].nunique()) - len(relevant_by_user),
                'test_positive_ratings_excluded_seen': excluded_seen,
                'test_unseen_positive_pairs': sum(map(len, relevant_by_user.values())),
            },
        }
        print("\nTemporal holdout results (warm-start, full training catalogue):")
        print(json.dumps(metrics, indent=2))
        return metrics

    def analyze_rules(self, recsys: LFIORecSys):
        """Analyze and summarize learned rules."""
        print("\n" + "="*60)
        print("RULE ANALYSIS")
        print("="*60)

        if not recsys.rules:
            print("No rules met the configured training thresholds.")
            return

        # Rule type distribution
        rule_types = defaultdict(int)
        for rule in recsys.rules:
            rule_types[rule.rule_type.value] += 1

        print(f"\nRule Type Distribution:")
        for rule_type, count in rule_types.items():
            print(f"  {rule_type}: {count}")

        # Condition analysis
        user_condition_features = defaultdict(int)
        item_condition_features = defaultdict(int)

        for rule in recsys.rules:
            for feat, _ in rule.user_conditions:
                user_condition_features[feat] += 1
            for feat, _ in rule.item_conditions:
                item_condition_features[feat] += 1

        print(f"\nMost Common User Condition Features:")
        for feat, count in sorted(user_condition_features.items(),
                                  key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feat}: {count}")

        print(f"\nMost Common Item Condition Features:")
        for feat, count in sorted(item_condition_features.items(),
                                  key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {feat}: {count}")

        # Quality stats
        confidences = [r.confidence for r in recsys.rules]
        lifts = [r.lift for r in recsys.rules]
        supports = [r.support for r in recsys.rules]

        print(f"\nRule Quality Statistics:")
        print(f"  Confidence - Mean: {np.mean(confidences):.3f}, Std: {np.std(confidences):.3f}")
        print(f"  Lift - Mean: {np.mean(lifts):.3f}, Std: {np.std(lifts):.3f}")
        print(f"  Support - Mean: {np.mean(supports):.3f}, Std: {np.std(supports):.3f}")

    def show_example_recommendations(self, recsys: LFIORecSys, num_users: int = 5):
        """Show example recommendations with explanations."""
        print("\n" + "="*60)
        print("EXAMPLE RECOMMENDATIONS")
        print("="*60)

        # Get users with sufficient interactions
        user_interaction_counts = defaultdict(int)
        for interaction in self.interactions:
            if interaction.is_positive(self.rating_threshold):
                user_interaction_counts[interaction.user_id] += 1

        # Get top users
        top_users = sorted(user_interaction_counts.items(),
                          key=lambda x: x[1], reverse=True)[:num_users]

        for user_id, count in top_users:
            user = self.users.get(user_id)
            if not user:
                continue

            print(f"\n--- User {user_id} ({count} positive ratings) ---")
            print(f"Features: {user.features}")

            recommendations = recsys.recommend(user_id, top_k=3)

            if not recommendations:
                print("  No recommendations generated")
                continue

            for i, rec in enumerate(recommendations, 1):
                item = self.items.get(rec.item_id)
                title = item.features.get('title', 'Unknown') if item else 'Unknown'
                print(f"  {i}. {title}")
                print(f"     {rec.explanation}")
                print(f"     Score: {rec.score:.3f}")

    def save_results(self, recsys: LFIORecSys, metrics: Dict, output_dir: str = "results"):
        """Save experiment results."""
        os.makedirs(output_dir, exist_ok=True)

        # Save rules
        rules_path = os.path.join(output_dir, "movielens_rules.json")
        recsys.save_rules(rules_path)

        # Save metrics
        metrics_path = os.path.join(output_dir, "movielens_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, allow_nan=False)

        pd.DataFrame(self.evaluation_rows).to_csv(
            os.path.join(output_dir, 'movielens_per_user.csv'), index=False
        )

        print(f"\nResults saved to {output_dir}/")

    def run_full_experiment(self, min_support=0.02, min_confidence=0.6,
                            min_lift=1.2, max_rules=50, test_ratio=0.2,
                            top_k=10, output_dir='results'):
        """Run the complete experiment pipeline."""
        print("\n" + "#"*60)
        print("# MOVIELENS LFIO RECOMMENDER EXPERIMENT")
        print("#"*60)

        start_time = time.perf_counter()

        # Load data
        self.load_data()

        # Split before selecting the catalogue or constructing rating features.
        self.split_dataset(test_ratio=test_ratio)

        # Filter dataset
        self.filter_dataset()

        # Extract features
        user_features, item_features = self.extract_features(self.ratings_df)

        # Create LFIO objects
        self.create_lfio_objects(user_features, item_features)

        # Run LFIO experiment
        fit_start = time.perf_counter()
        recsys = self.run_experiment(
            min_support=min_support,
            min_confidence=min_confidence,
            min_lift=min_lift,
            max_rules=max_rules
        )
        fit_seconds = time.perf_counter() - fit_start

        # Print learned rules
        recsys.print_rules(max_rules=20)

        # Analyze rules
        self.analyze_rules(recsys)

        # Evaluate rules
        metrics = self.evaluate_rules(recsys, top_k=top_k)
        metrics['config'] = {
            'data_path': self.data_path,
            'max_users': self.max_users, 'max_items': self.max_items,
            'min_interactions': self.min_interactions,
            'rating_threshold': self.rating_threshold,
            'min_support': min_support, 'min_confidence': min_confidence,
            'min_lift': min_lift, 'max_rules': max_rules,
            'test_ratio': test_ratio, 'top_k': top_k,
            'feature_protocol': 'original_categorical_training_only',
        }
        metrics['fit_seconds'] = fit_seconds
        metrics['elapsed_seconds'] = time.perf_counter() - start_time

        # Show example recommendations
        self.show_example_recommendations(recsys, num_users=5)

        # Save results
        self.save_results(recsys, metrics, output_dir=output_dir)

        print("\n" + "#"*60)
        print("# EXPERIMENT COMPLETE")
        print("#"*60)

        return recsys, metrics


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='MovieLens LFIO Experiment')
    parser.add_argument('--data-path', type=str, default='ml-latest-small',
                       help='Path to MovieLens dataset')
    parser.add_argument('--min-support', type=float, default=0.02,
                       help='Minimum support for rules')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                       help='Minimum confidence for rules')
    parser.add_argument('--min-lift', type=float, default=1.2,
                       help='Minimum lift for rules')
    parser.add_argument('--max-rules', type=int, default=50,
                       help='Maximum number of rules to learn')
    parser.add_argument('--max-users', type=int, default=-1,
                       help='Maximum users for experiment (-1 for all users)')
    parser.add_argument('--max-items', type=int, default=-1,
                       help='Maximum items for experiment (-1 for all items)')
    parser.add_argument('--test-ratio', type=float, default=0.2,
                       help='Latest fraction of ratings held out globally')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--output-dir', default='results/movielens')

    args = parser.parse_args()

    # Run experiment
    experiment = MovieLensExperiment(data_path=args.data_path)
    experiment.max_users = args.max_users
    experiment.max_items = args.max_items

    experiment.run_full_experiment(
        min_support=args.min_support, min_confidence=args.min_confidence,
        min_lift=args.min_lift, max_rules=args.max_rules,
        test_ratio=args.test_ratio, top_k=args.top_k, output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
