"""Black box model implementation using neural networks."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import pickle
import os
import copy


class BlackBoxModel(nn.Module):
    """Neural network as a black box model."""

    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16],
                 output_dim: int = 2, dropout: float = 0.1):
        super(BlackBoxModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

    def forward(self, x):
        return self.network(x)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            if len(x.shape) == 1:
                x_tensor = x_tensor.unsqueeze(0)
            logits = self.forward(x_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Get predictions."""
        probs = self.predict_proba(x)
        return np.argmax(probs, axis=1)

    def get_hidden_representation(self, x: np.ndarray, layer_idx: int = -1) -> np.ndarray:
        """Get hidden layer representations for analysis."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            if len(x.shape) == 1:
                x_tensor = x_tensor.unsqueeze(0)

            # Forward pass through layers up to the specified one
            x_current = x_tensor
            layer_count = 0
            target_layer = len(self.hidden_dims) if layer_idx == -1 else layer_idx

            for layer in self.network:
                x_current = layer(x_current)
                if isinstance(layer, nn.Linear):
                    if layer_count == target_layer:
                        return x_current.cpu().numpy()
                    layer_count += 1

        return x_current.cpu().numpy()


class BlackBoxTrainer:
    """Train and manage black box models."""

    def __init__(self, model: BlackBoxModel, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32, lr: float = 0.001,
              early_stopping: int = 10) -> dict:
        """
        Train the black box model.

        Returns:
            Dictionary with training history
        """
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)

        if X_val is not None:
            X_val_t = torch.FloatTensor(X_val).to(self.device)
            y_val_t = torch.LongTensor(y_val).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        best_state = None
        patience = early_stopping

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0

            # Mini-batch training
            indices = torch.randperm(len(X_train_t))
            for i in range(0, len(X_train_t), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_x = X_train_t[batch_indices]
                batch_y = y_train_t[batch_indices]

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += len(batch_x)

            train_loss = epoch_loss / total
            train_acc = correct / total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation phase
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(X_val_t)
                    val_loss = criterion(outputs, y_val_t).item()
                    _, predicted = torch.max(outputs, 1)
                    val_acc = (predicted == y_val_t).float().mean().item()

                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = early_stopping
                    # state_dict().copy() shares tensor storage with the model.
                    best_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience -= 1
                    if patience == 0:
                        break

            if (epoch + 1) % 20 == 0:
                val_msg = f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}" if X_val is not None else ""
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}{val_msg}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate the model on test data."""
        y_pred = self.model.predict(X_test)

        accuracy = np.mean(y_pred == y_test)

        # Per-class metrics
        metrics = {"accuracy": accuracy}

        for class_label in np.unique(y_test):
            mask = y_test == class_label
            if mask.sum() > 0:
                class_acc = np.mean(y_pred[mask] == y_test[mask])
                metrics[f"class_{class_label}_acc"] = class_acc

        return metrics

    def save_model(self, path: str):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.model.input_dim,
            "hidden_dims": self.model.hidden_dims,
            "output_dim": self.model.output_dim
        }, path)

    def load_model(self, path: str):
        """Load a model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def get_all_predictions(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get both predictions and probabilities."""
        probs = self.model.predict_proba(X)
        preds = np.argmax(probs, axis=1)
        return preds, probs


def create_black_box(input_dim: int, hidden_dims: list = None,
                     device: str = "cpu") -> BlackBoxModel:
    """Factory function to create a black box model."""
    if hidden_dims is None:
        hidden_dims = [64, 32, 16]

    model = BlackBoxModel(input_dim, hidden_dims)
    return model.to(device)


if __name__ == "__main__":
    # Test the black box model
    from data_generator import SyntheticDataGenerator

    print("=== Testing Black Box Model ===")

    # Generate synthetic data
    generator = SyntheticDataGenerator(n_samples=1000, n_features=5)
    X, y, rule_desc = generator.generate_boolean_dataset(rule_type="complex")
    print(f"Generated data with rule: {rule_desc}")

    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Create and train model
    model = create_black_box(input_dim=5, hidden_dims=[32, 16])
    trainer = BlackBoxTrainer(model)

    print("\nTraining model...")
    history = trainer.train(X_train, y_train, epochs=50)

    # Evaluate
    print("\nEvaluating model...")
    metrics = trainer.evaluate(X_test, y_test)
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
