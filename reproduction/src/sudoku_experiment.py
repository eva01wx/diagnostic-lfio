"""
Sudoku Experiment for XAI Rule Extraction

This module implements a complete experiment pipeline for Sudoku validation:
1. Generate Sudoku datasets (valid/invalid grids)
2. Train black box neural network
3. Extract rules using Decision Tree, ILP, and LFIO
4. Compare and visualize results
"""

import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import random
from pathlib import Path
import torch

from src.sudoku import (
    SudokuDataset, SudokuGrid, Sudoku4x4,
    SudokuFeatureExtractor, create_sudoku_dataset
)
from src.black_box import create_black_box, BlackBoxTrainer
from src.rule_extractor import RuleExplainer
from src.ilp_learner import BooleanILPLearner
from src.lfio_learner import LFIOBoolean, LFIOBlackBox
from src.lfio_extended import LFIOBooleanWithNegation


class SudokuExperiment:
    """
    Complete experiment pipeline for Sudoku validation rule extraction.
    """

    def __init__(self, size: int = 4, n_samples: int = 500,
                 random_state: int = 42, constraint_features: str = "primitive"):
        """
        Initialize Sudoku experiment.

        Args:
            size: Sudoku grid size (4 for 4×4, 9 for 9×9)
            n_samples: Total number of samples to generate
            random_state: Random seed for reproducibility
        """
        self.size = size
        self.n_samples = n_samples
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
        torch.manual_seed(random_state)
        if constraint_features not in {"primitive", "engineered"}:
            raise ValueError("Unknown constraint feature protocol")
        self.constraint_features = constraint_features

        self.dataset = SudokuDataset(size=size, n_samples=n_samples)

        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_val = None
        self.y_train = None
        self.y_test = None
        self.feature_names = []

        self.black_box = None
        self.trainer = None

        self.results = {}

    def generate_data(self, balanced: bool = True, test_size: float = 0.2):
        """
        Generate Sudoku dataset.

        Args:
            balanced: Balance valid/invalid examples
            test_size: Proportion of data for testing
        """
        print(f"\n{'='*60}")
        print(f"Sudoku Experiment: {self.size}×{self.size} Grid")
        print(f"{'='*60}")

        X, y, feature_names = self.dataset.generate(balanced=balanced)
        if self.constraint_features == "primitive":
            keep = [feature_names.index(name) for name in ("no_row_duplicates", "no_col_duplicates", "no_box_duplicates")]
            X = X[:, keep]
            feature_names = [feature_names[i] for i in keep]
        self.feature_names = feature_names

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y
        )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.25,
            random_state=self.random_state, stratify=self.y_train
        )
        self.results['protocol'] = {'train_samples': len(self.X_train),
                                   'validation_samples': len(self.X_val),
                                   'test_samples': len(self.X_test),
                                   'constraint_features': self.constraint_features,
                                   'seed': self.random_state,
                                   'early_stopping_uses_validation': True}

        print(f"\nDataset split:")
        print(f"  Training: {len(self.X_train)} samples")
        print(f"  Testing: {len(self.X_test)} samples")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Class balance - Train: {np.mean(self.y_train):.2%}")
        print(f"  Class balance - Test: {np.mean(self.y_test):.2%}")

        return X, y, feature_names

    def train_black_box(self, hidden_dims: List[int] = None, epochs: int = 50):
        """
        Train black box neural network.

        Args:
            hidden_dims: Hidden layer dimensions
            epochs: Training epochs
        """
        print(f"\n{'='*60}")
        print("Training Black Box Model")
        print(f"{'='*60}")

        if hidden_dims is None:
            hidden_dims = [32, 16] if self.size == 4 else [64, 32, 16]

        self.black_box = create_black_box(
            input_dim=len(self.feature_names),
            hidden_dims=hidden_dims
        )
        self.trainer = BlackBoxTrainer(self.black_box)

        self.trainer.train(
            self.X_train, self.y_train,
            X_val=self.X_val, y_val=self.y_val,
            epochs=epochs
        )

        train_metrics = self.trainer.evaluate(self.X_train, self.y_train)
        test_metrics = self.trainer.evaluate(self.X_test, self.y_test)

        print(f"\nBlack Box Performance:")
        print(f"  Train Accuracy: {train_metrics['accuracy']:.3f}")
        print(f"  Test Accuracy: {test_metrics['accuracy']:.3f}")

        self.results['black_box'] = {
            'train_accuracy': train_metrics['accuracy'],
            'test_accuracy': test_metrics['accuracy']
        }

        return test_metrics['accuracy']

    def train_random_forest(self, n_estimators: int = 100) -> RandomForestClassifier:
        """
        Train Random Forest as a comparison baseline.

        Args:
            n_estimators: Number of trees in forest

        Returns:
            Trained Random Forest classifier
        """
        print(f"\n{'='*60}")
        print("Training Random Forest Baseline")
        print(f"{'='*60}")

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)

        train_acc = rf.score(self.X_train, self.y_train)
        test_acc = rf.score(self.X_test, self.y_test)

        print(f"\nRandom Forest Performance:")
        print(f"  Train Accuracy: {train_acc:.3f}")
        print(f"  Test Accuracy: {test_acc:.3f}")

        # Feature importance
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f"\nTop 5 Important Features:")
        for _, row in importances.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")

        self.results['random_forest'] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'feature_importances': importances.to_dict()
        }

        return rf

    def extract_decision_tree_rules(self) -> List:
        """
        Extract rules using decision tree surrogate.

        Returns:
            List of extracted rules
        """
        print(f"\n{'='*60}")
        print("Decision Tree Rule Extraction")
        print(f"{'='*60}")

        explainer = RuleExplainer(method="decision_tree")
        rules = explainer.extract_from_blackbox(
            self.X_train, self.black_box, self.feature_names
        )

        explainer.print_rules(max_rules=10)

        rule_preds = explainer.extractor.tree.predict(self.X_test)
        bb_preds = self.black_box.predict(self.X_test)

        rule_accuracy = np.mean(rule_preds == self.y_test)
        fidelity = np.mean(rule_preds == bb_preds)

        print(f"\nDecision Tree Performance:")
        print(f"  Test Accuracy: {rule_accuracy:.3f}")
        print(f"  Fidelity to Black Box: {fidelity:.3f}")

        self.results['decision_tree'] = {
            'num_rules': len(rules),
            'accuracy': rule_accuracy,
            'fidelity': fidelity
        }

        return rules

    def extract_ilp_rules(self, method: str = "boolean",
                          use_negation: bool = False) -> List:
        """
        Extract rules using ILP.

        Args:
            method: ILP method ('boolean', 'lfio', 'lfio_black_box')
            use_negation: Allow negation in rules

        Returns:
            List of learned rules
        """
        print(f"\n{'='*60}")
        print(f"ILP Rule Extraction ({method})")
        print(f"{'='*60}")

        bb_train_preds = self.black_box.predict(self.X_train)
        bb_test_preds = self.black_box.predict(self.X_test)

        if method == "boolean":
            ilp = BooleanILPLearner(
                max_rule_length=4,
                min_coverage=0.05,
                min_precision=0.85
            )
            ilp.learn_boolean_rules(self.X_train, bb_train_preds)

            print("Learned Boolean Rules:")
            for i, rule in enumerate(ilp.rules[:10], 1):
                # ILPRule has 'description' attribute
                print(f"{i}. {rule.description}")
                print(f"   Coverage: {rule.coverage:.2%}, Precision: {rule.precision:.2%}")

            ilp_preds = ilp.predict(self.X_test)
            ilp_accuracy = np.mean(ilp_preds == self.y_test)
            ilp_fidelity = np.mean(ilp_preds == bb_test_preds)

            print(f"\nILP Performance:")
            print(f"  Test Accuracy: {ilp_accuracy:.3f}")
            print(f"  Fidelity to Black Box: {ilp_fidelity:.3f}")

            self.results['ilp_boolean'] = {
                'num_rules': len(ilp.rules),
                'accuracy': ilp_accuracy,
                'fidelity': ilp_fidelity
            }

            return ilp.rules

        elif method == "lfio":
            lfio = LFIOBoolean(
                min_coverage=max(5, len(self.X_train) // 50),
                min_precision=0.85,
                max_literals=5
            )
            lfio.fit(self.X_train, bb_train_preds, self.feature_names)
            lfio.print_rules()

            lfio_preds = lfio.predict(self.X_test)
            lfio_accuracy = np.mean(lfio_preds == self.y_test)
            lfio_fidelity = np.mean(lfio_preds == bb_test_preds)

            print(f"\nLFIO Performance:")
            print(f"  Test Accuracy: {lfio_accuracy:.3f}")
            print(f"  Fidelity to Black Box: {lfio_fidelity:.3f}")

            self.results['lfio'] = {
                'num_rules': len(lfio.rules),
                'accuracy': lfio_accuracy,
                'fidelity': lfio_fidelity
            }

            return lfio.rules

        elif method == "lfio_negation":
            lfio = LFIOBooleanWithNegation(
                min_coverage=max(5, len(self.X_train) // 50),
                min_precision=0.85,
                max_literals=5,
                allow_negation=True
            )
            lfio.fit(self.X_train, bb_train_preds, self.feature_names)
            lfio.print_rules()

            lfio_preds = lfio.predict(self.X_test)
            lfio_accuracy = np.mean(lfio_preds == self.y_test)
            lfio_fidelity = np.mean(lfio_preds == bb_test_preds)

            print(f"\nLFIO with Negation Performance:")
            print(f"  Test Accuracy: {lfio_accuracy:.3f}")
            print(f"  Fidelity to Black Box: {lfio_fidelity:.3f}")

            self.results['lfio_negation'] = {
                'num_rules': len(lfio.rules),
                'accuracy': lfio_accuracy,
                'fidelity': lfio_fidelity
            }

            return lfio.rules

        elif method == "lfio_black_box":
            lfio_bb = LFIOBlackBox(
                black_box_model=self.black_box,
                min_coverage=0.05,
                min_fidelity=0.85,
                max_rules=20
            )
            lfio_bb.fit(self.X_train, self.feature_names)
            lfio_bb.print_rules()

            lfio_preds = lfio_bb.predict(self.X_test)
            lfio_accuracy = np.mean(lfio_preds == self.y_test)
            lfio_fidelity = np.mean(lfio_preds == bb_test_preds)

            print(f"\nLFIO Black Box Performance:")
            print(f"  Test Accuracy: {lfio_accuracy:.3f}")
            print(f"  Fidelity to Black Box: {lfio_fidelity:.3f}")

            self.results['lfio_black_box'] = {
                'num_rules': len(lfio_bb.rules),
                'accuracy': lfio_accuracy,
                'fidelity': lfio_fidelity
            }

            return lfio_bb.rules

        return []

    def compare_all_methods(self) -> pd.DataFrame:
        """
        Compare all rule extraction methods.

        Returns:
            DataFrame with comparison results
        """
        print(f"\n{'='*60}")
        print("Method Comparison")
        print(f"{'='*60}")

        # Get black box predictions
        bb_test_preds = self.black_box.predict(self.X_test)
        bb_train_preds = self.black_box.predict(self.X_train)

        comparison = []

        # Decision Tree
        dt_explainer = RuleExplainer(method="decision_tree")
        dt_rules = dt_explainer.extract_from_blackbox(
            self.X_train, self.black_box, self.feature_names
        )
        dt_preds = dt_explainer.extractor.tree.predict(self.X_test)
        comparison.append({
            'method': 'Decision Tree',
            'num_rules': len(dt_rules),
            'accuracy': np.mean(dt_preds == self.y_test),
            'fidelity': np.mean(dt_preds == bb_test_preds)
        })

        # ILP Boolean
        ilp = BooleanILPLearner(max_rule_length=4, min_coverage=0.05, min_precision=0.85)
        ilp.learn_boolean_rules(self.X_train, bb_train_preds)
        ilp_preds = ilp.predict(self.X_test)
        comparison.append({
            'method': 'ILP Boolean',
            'num_rules': len(ilp.rules),
            'accuracy': np.mean(ilp_preds == self.y_test),
            'fidelity': np.mean(ilp_preds == bb_test_preds)
        })

        # LFIO
        lfio = LFIOBoolean(
            min_coverage=max(5, len(self.X_train) // 50),
            min_precision=0.85
        )
        lfio.fit(self.X_train, bb_train_preds, self.feature_names)
        lfio_preds = lfio.predict(self.X_test)
        comparison.append({
            'method': 'LFIO',
            'num_rules': len(lfio.rules),
            'accuracy': np.mean(lfio_preds == self.y_test),
            'fidelity': np.mean(lfio_preds == bb_test_preds)
        })

        # LFIO with Negation
        lfio_neg = LFIOBooleanWithNegation(
            min_coverage=max(5, len(self.X_train) // 50),
            min_precision=0.85,
            allow_negation=True
        )
        lfio_neg.fit(self.X_train, bb_train_preds, self.feature_names)
        lfio_neg_preds = lfio_neg.predict(self.X_test)
        comparison.append({
            'method': 'LFIO + Negation',
            'num_rules': len(lfio_neg.rules),
            'accuracy': np.mean(lfio_neg_preds == self.y_test),
            'fidelity': np.mean(lfio_neg_preds == bb_test_preds)
        })

        from src.lfio_research import ValidatedLFIO
        validated = ValidatedLFIO().fit(self.X_train, bb_train_preds, self.X_val,
                                        self.black_box.predict(self.X_val), self.feature_names)
        prediction = validated.predict(self.X_test)
        validated_metrics = {'num_rules': len(validated.rules),
                             'accuracy': float(np.mean(prediction == self.y_test)),
                             'fidelity': float(np.mean(prediction == bb_test_preds))}
        comparison.append(dict(method='Validated LFIO', **validated_metrics))
        self.results['validated_lfio'] = dict(validated_metrics, program=validated.export())
        df = pd.DataFrame(comparison)
        print("\n", df.to_string(index=False))

        self.results['comparison'] = df.to_dict()

        return df

    def visualize_results(self, save_path: str = None):
        """
        Create visualization of results.

        Args:
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Method comparison
        if 'comparison' in self.results:
            comp_data = self.results['comparison']
            methods = list(comp_data['method'].values())
            accuracies = list(comp_data['accuracy'].values())
            fidelities = list(comp_data['fidelity'].values())

            x = np.arange(len(methods))
            width = 0.35

            ax = axes[0, 0]
            bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
            bars2 = ax.bar(x + width/2, fidelities, width, label='Fidelity', alpha=0.8)
            ax.set_xlabel('Method')
            ax.set_ylabel('Score')
            ax.set_title('Method Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=15, ha='right')
            ax.legend()
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)

        # 2. Feature statistics
        ax = axes[0, 1]
        pos_mask = self.y_test == 1
        neg_mask = self.y_test == 0

        pos_mean = self.X_test[pos_mask].mean(axis=0) if np.any(pos_mask) else np.zeros(len(self.feature_names))
        neg_mean = self.X_test[neg_mask].mean(axis=0) if np.any(neg_mask) else np.zeros(len(self.feature_names))
        diffs = np.abs(pos_mean - neg_mean)
        sorted_idx = np.argsort(diffs)[-10:]

        y_pos = np.arange(len(sorted_idx))
        ax.barh(y_pos, diffs[sorted_idx], alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_names[i] for i in sorted_idx])
        ax.set_xlabel('Absolute Difference')
        ax.set_title('Most Discriminative Features')
        ax.grid(axis='x', alpha=0.3)

        # 3. Confusion Matrix (Black Box)
        ax = axes[1, 0]
        bb_preds = self.black_box.predict(self.X_test)
        cm = confusion_matrix(self.y_test, bb_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Black Box Confusion Matrix')

        # 4. Rule complexity vs accuracy
        ax = axes[1, 1]
        if 'comparison' in self.results:
            comp_data = self.results['comparison']
            n_rules = list(comp_data['num_rules'].values())
            accs = list(comp_data['accuracy'].values())

            scatter = ax.scatter(n_rules, accs, s=100, alpha=0.7, c=range(len(methods)),
                               cmap='viridis')
            for i, m in enumerate(methods):
                ax.annotate(m, (n_rules[i], accs[i]), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)

            ax.set_xlabel('Number of Rules')
            ax.set_ylabel('Accuracy')
            ax.set_title('Rule Complexity vs Accuracy')
            ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to {save_path}")
        else:
            path = f"results/sudoku_{self.size}x{self.size}_results.png"
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to {path}")

        plt.close()

    def print_summary(self):
        """Print experiment summary."""
        print(f"\n{'='*60}")
        print(f"SUMMARY: {self.size}×{self.size} Sudoku Experiment")
        print(f"{'='*60}")

        print(f"\nDataset: {self.n_samples} samples, {len(self.feature_names)} features")

        if 'black_box' in self.results:
            bb = self.results['black_box']
            print(f"\nBlack Box: {bb['test_accuracy']:.3f} test accuracy")

        if 'random_forest' in self.results:
            rf = self.results['random_forest']
            print(f"Random Forest: {rf['test_accuracy']:.3f} test accuracy")

        print("\nRule Extraction Methods:")
        for key, value in self.results.items():
            if key in ['black_box', 'random_forest', 'comparison']:
                continue
            if 'accuracy' in value:
                print(f"  {key}: {value['accuracy']:.3f} accuracy, {value.get('num_rules', 0)} rules")

    def save_results(self, filepath: str):
        """Save results to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filepath}")


def run_sudoku_experiment(size: int = 4, n_samples: int = 500,
                         visualize: bool = True, epochs: int = 50,
                         seed: int = 42, constraint_features: str = "primitive",
                         output_dir: str = "results") -> Dict:
    """
    Run complete Sudoku experiment.

    Args:
        size: Grid size (4 or 9)
        n_samples: Number of samples
        visualize: Create visualizations

    Returns:
        Results dictionary
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    exp = SudokuExperiment(size=size, n_samples=n_samples, random_state=seed,
                           constraint_features=constraint_features)

    # Generate data
    exp.generate_data(balanced=True)

    # Train models
    exp.train_black_box(epochs=epochs)
    exp.train_random_forest()

    # Extract rules
    exp.extract_decision_tree_rules()
    exp.extract_ilp_rules(method="boolean")
    exp.extract_ilp_rules(method="lfio")
    exp.extract_ilp_rules(method="lfio_negation")

    # Compare
    exp.compare_all_methods()

    # Visualize
    if visualize:
        exp.visualize_results(save_path=str(Path(output_dir) / f"sudoku_{size}x{size}_results.png"))

    # Summary
    exp.print_summary()

    # Save results
    exp.save_results(str(Path(output_dir) / f"sudoku_{size}x{size}_results.json"))

    return exp.results


def run_sudoku_4x4_demo():
    """Run demonstration with 4×4 Sudoku (faster)."""
    print("\n" + "="*80)
    print("SUDOKU 4×4 RULE EXTRACTION EXPERIMENT")
    print("="*80)

    return run_sudoku_experiment(size=4, n_samples=500, visualize=True)


def run_sudoku_9x9_demo():
    """Run demonstration with 9×9 Sudoku."""
    print("\n" + "="*80)
    print("SUDOKU 9×9 RULE EXTRACTION EXPERIMENT")
    print("="*80)

    return run_sudoku_experiment(size=9, n_samples=300, visualize=True)


def interactive_sudoku_demo():
    """Interactive Sudoku demonstration."""
    print("="*80)
    print("SUDOKU RULE EXTRACTION EXPERIMENT")
    print("="*80)

    print("\nChoose Sudoku size:")
    print("1. 4×4 Sudoku (faster, for testing)")
    print("2. 9×9 Standard Sudoku")

    choice = input("\nEnter choice (1-2): ").strip()

    size = 4 if choice == '1' else 9
    n_samples = 500 if size == 4 else 300

    print(f"\nRunning {size}×{size} Sudoku experiment with {n_samples} samples...")

    exp = SudokuExperiment(size=size, n_samples=n_samples)

    # Generate data
    exp.generate_data(balanced=True)

    # Train models
    exp.train_black_box(epochs=50)
    exp.train_random_forest()

    # Extract rules with different methods
    print("\nChoose rule extraction methods (comma separated):")
    print("1. Decision Tree")
    print("2. ILP Boolean")
    print("3. LFIO")
    print("4. LFIO with Negation")
    print("5. All methods")

    method_choice = input("\nEnter choice (1-5): ").strip()

    method_map = {
        '1': 'dt',
        '2': 'boolean',
        '3': 'lfio',
        '4': 'negation',
        '5': 'all'
    }

    if method_choice == '1' or method_choice == '5':
        exp.extract_decision_tree_rules()
    if method_choice == '2' or method_choice == '5':
        exp.extract_ilp_rules(method="boolean")
    if method_choice == '3' or method_choice == '5':
        exp.extract_ilp_rules(method="lfio")
    if method_choice == '4' or method_choice == '5':
        exp.extract_ilp_rules(method="lfio_negation")

    if method_choice == '5':
        exp.compare_all_methods()

    visualize = input("\nCreate visualizations? (y/n): ").strip().lower() == 'y'
    if visualize:
        exp.visualize_results()

    exp.print_summary()
    exp.save_results(f"results/sudoku_{size}x{size}_results.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Sudoku Rule Extraction Experiment')
    parser.add_argument('--size', type=int, default=4, choices=[4, 9],
                        help='Sudoku grid size (4 or 9)')
    parser.add_argument('--samples', type=int, default=500,
                        help='Number of samples to generate')
    parser.add_argument('--no-viz', action='store_true',
                        help='Disable visualization')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs for neural network')

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--constraint-features", choices=["primitive", "engineered"], default="primitive")
    parser.add_argument("--output-dir", default="results")

    args = parser.parse_args()

    if args.interactive:
        interactive_sudoku_demo()
    else:
        run_sudoku_experiment(
            size=args.size,
            n_samples=args.samples,
            visualize=not args.no_viz, epochs=args.epochs, seed=args.seed,
            constraint_features=args.constraint_features, output_dir=args.output_dir
        )
