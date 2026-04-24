"""
PyTorch model definitions for neuronal cell-type classification.

Two architectures:
  - MLP: operates on hand-crafted electrophysiological features
  - CNN1D: operates directly on raw spike waveforms (no feature engineering)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class NeuronalMLP(nn.Module):
    """
    Multilayer Perceptron for electrophysiological feature classification.

    Takes a vector of hand-crafted features (ISI statistics, spike shape,
    passive membrane properties) and classifies the neuron into one of
    K cell-type classes.

    Architecture
    ------------
    Input → BatchNorm
      → Linear(n_features → 256) → BatchNorm → ReLU → Dropout(0.3)
      → Linear(256 → 64) → BatchNorm → ReLU → Dropout(0.2)
      → Linear(64 → n_classes) → (Softmax at inference)

    Parameters
    ----------
    n_features : int
        Number of input features (after NaN imputation).
    n_classes : int
        Number of output classes (2 for Exc/Inh, up to 6 for subtypes).
    dropout_1 : float
        Dropout rate after first hidden layer.
    dropout_2 : float
        Dropout rate after second hidden layer.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        dropout_1: float = 0.3,
        dropout_2: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_norm = nn.BatchNorm1d(n_features)

        self.hidden1 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_1),
        )

        self.hidden2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_2),
        )

        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_features)

        Returns
        -------
        logits : torch.Tensor, shape (batch, n_classes)
        """
        x = self.input_norm(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        return self.classifier(x)


class SpikeCNN1D(nn.Module):
    """
    1D Convolutional Network operating on raw spike waveforms.

    Learns discriminative features directly from spike shape without
    manual feature engineering. Serves as an ablation test: does the CNN
    recover the same shape features (half-width, AHP) that electrophysiologists
    use manually?

    Input is a batch of spike waveforms, each of length `waveform_length`
    samples, averaged across all detected spikes per cell.

    Parameters
    ----------
    waveform_length : int
        Number of samples per spike waveform (e.g. 60 samples at 20 kHz = 3 ms).
    n_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        waveform_length: int = 60,
        n_classes: int = 2,
    ) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, waveform_length)
            Mean spike waveforms, one per cell.

        Returns
        -------
        logits : torch.Tensor, shape (batch, n_classes)
        """
        x = x.unsqueeze(1)  # (batch, 1, waveform_length) for Conv1d
        x = self.conv_block(x)
        return self.classifier(x)
