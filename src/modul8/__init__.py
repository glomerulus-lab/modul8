"""modul8: three-factor Hebbian learning for standard PyTorch layers."""

from .feedback import DirectFeedback, relu_derivative, sigmoid_derivative, tanh_derivative
from .local import LocalHead, LocalLoss, similarity_matrix
from .three_factor import ThreeFactor

__all__ = [
    "DirectFeedback",
    "LocalHead",
    "LocalLoss",
    "ThreeFactor",
    "relu_derivative",
    "sigmoid_derivative",
    "similarity_matrix",
    "tanh_derivative",
]
