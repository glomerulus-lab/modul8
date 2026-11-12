"""modul8: three-factor Hebbian learning for standard PyTorch layers."""

from .feedback import DirectFeedback, relu_derivative, sigmoid_derivative, tanh_derivative
from .three_factor import ThreeFactor

__all__ = [
    "DirectFeedback",
    "ThreeFactor",
    "relu_derivative",
    "sigmoid_derivative",
    "tanh_derivative",
]
