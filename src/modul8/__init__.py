"""modul8: three-factor Hebbian learning for standard PyTorch layers."""

from .feedback import DirectFeedback, relu_derivative, sigmoid_derivative, tanh_derivative
from .local import LinearFA, LocalHead, LocalLoss, similarity_matrix
from .three_factor import ThreeFactor

__all__ = [
    "DirectFeedback",
    "LinearFA",
    "LocalHead",
    "LocalLoss",
    "ThreeFactor",
    "relu_derivative",
    "sigmoid_derivative",
    "similarity_matrix",
    "tanh_derivative",
]
