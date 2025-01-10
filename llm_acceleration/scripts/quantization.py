import torch
from typing import Tuple
from torch import Tensor, CharTensor


def absmax_quantization(x: Tensor) -> Tuple[float, CharTensor]:
    """
    Выполняет квантование тензора `x` с использованием метода максимального значения по модулю (AbsMax).
    """
    s = 127 / x.abs().max().item()

    x_q = (x * s).round().to(torch.int8)

    return s, x_q


def absmax_dequantization(s: float, x_q: Tensor) -> Tensor:
    """
    Выполняет деквантование тензора `x_q`, полученного методом AbsMax.

    Args:
        s (float): Масштабный коэффициент, использованный для квантования.
        x_q (Tensor): Квантованный тензор типа `int8`.

    Returns:
        Tensor: Восстановленный (деквантованный) тензор с типом `float`.
    """
    return x_q.to(torch.float32) / s


def zeropoint_quantization(x: Tensor) -> Tuple[float, int, CharTensor]:
    """
    Выполняет квантование тензора `x` с использованием метода нулевой точки (Zero-Point Quantization).

    Args:
        x (Tensor): Входной тензор для квантования.

    Returns:
        Tuple[float, int, CharTensor]: Кортеж, содержащий масштабный коэффициент `s`, значение нулевой точки `z`,
            и квантованный тензор `x_q` типа `int8`.
            - `s` (float): Масштабный коэффициент, использованный для квантования.
            - `z` (int): Смещение (нулевая точка), использованное для квантования.
            - `x_q` (CharTensor): Квантованный тензор с значениями типа `int8`.
    """
    x_min = x.min().item()
    x_max = x.max().item()

    s = 255 / (x_max - x_min)
    z = round(-s * x_min - 128)
    x_q = (s * x + z).round().clamp(-128, 127).to(torch.int8)

    return s, int(z), x_q


def zeropoint_dequantization(s: float, z: int, x_q: Tensor) -> Tensor:
    """
    Выполняет деквантование тензора `x_q`, полученного методом Zero-Point Quantization.

    Args:
        s (float): Масштабный коэффициент, использованный для квантования.
        z (int): Смещение (нулевая точка), использованное для квантования.
        x_q (Tensor): Квантованный тензор типа `int8`.

    Returns:
        Tensor: Восстановленный (деквантованный) тензор с типом `float`.
    """
    return (x_q.to(torch.float32) - z) / s
