"""
Training loop for neuronal cell-type classifiers.

Supports:
  - Stratified k-fold cross-validation
  - Early stopping on validation loss
  - Class-weighted loss (handles imbalanced subtypes)
  - Metric tracking (macro-F1, accuracy, per-class F1)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import Optional
import time


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_names: Optional[list[str]] = None,
) -> dict:
    """
    Evaluate model on a DataLoader.

    Returns a dict with loss, accuracy, macro-F1, and per-class F1.
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "report": classification_report(
            all_labels, all_preds,
            target_names=label_names,
            zero_division=0,
        ),
    }


def cross_validate(
    model_class,
    model_kwargs: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 15,
    label_names: Optional[list[str]] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Stratified k-fold cross-validation for PyTorch models.

    Parameters
    ----------
    model_class : nn.Module subclass
        The model class to instantiate for each fold.
    model_kwargs : dict
        Keyword arguments passed to model_class().
    X : np.ndarray, shape (n_cells, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_cells,)
        Integer class labels.
    n_splits : int
        Number of CV folds.
    n_epochs : int
        Maximum epochs per fold.
    patience : int
        Early stopping patience (epochs without val loss improvement).
    label_names : list of str, optional
        Class names for the classification report.

    Returns
    -------
    results : dict
        Per-fold and aggregated metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    # Class weights for imbalanced datasets
    class_counts = np.bincount(y)
    class_weights = torch.tensor(
        1.0 / class_counts, dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")

        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.long)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val = torch.tensor(y[val_idx], dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(X_val, y_val),
            batch_size=batch_size, shuffle=False,
        )

        model = model_class(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=7, factor=0.5
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        for epoch in range(n_epochs):
            t0 = time.time()
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device, label_names)
            scheduler.step(val_metrics["loss"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1:3d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_metrics['loss']:.4f} | "
                    f"val_F1={val_metrics['macro_f1']:.3f} | "
                    f"{time.time()-t0:.1f}s"
                )

            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Reload best weights and evaluate
        if best_state is not None:
            model.load_state_dict(best_state)
        final_metrics = evaluate(model, val_loader, criterion, device, label_names)
        fold_results.append(final_metrics)
        print(f"  Best val macro-F1: {final_metrics['macro_f1']:.4f}")
        print(final_metrics["report"])

    # Aggregate across folds
    mean_f1 = np.mean([r["macro_f1"] for r in fold_results])
    std_f1 = np.std([r["macro_f1"] for r in fold_results])
    mean_acc = np.mean([r["accuracy"] for r in fold_results])

    print(f"\n=== CV Summary ===")
    print(f"Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"Accuracy: {mean_acc:.4f}")

    return {
        "fold_results": fold_results,
        "mean_macro_f1": float(mean_f1),
        "std_macro_f1": float(std_f1),
        "mean_accuracy": float(mean_acc),
    }
