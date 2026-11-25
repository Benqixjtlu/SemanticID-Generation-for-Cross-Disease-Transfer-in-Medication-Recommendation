"""Build the four-drug example with a fixed, one-dimensional quantizer."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from random import Random


TEACHING_VALUES = {"A": 74, "B": 78, "C": 116, "D": 222}
TEACHING_CODEBOOKS = ({2: 80, 3: 220}, {7: 0, 9: 30}, {4: 0, 1: 10})


@dataclass(frozen=True)
class QuantizationStep:
    """One representative value and the residual it leaves behind."""

    level: int
    code: int
    representative: float
    residual_before: float
    residual_after: float


def quantize(
    value: float, codebooks: Sequence[Mapping[int, float]]
) -> tuple[QuantizationStep, ...]:
    """Select the closest representative at each level by absolute distance."""
    if not isfinite(value):
        raise ValueError("The input value must be finite.")
    if not codebooks:
        raise ValueError("At least one codebook is required.")

    steps = []
    residual = value
    for level, codebook in enumerate(codebooks, start=1):
        if not codebook:
            raise ValueError(f"Codebook {level} is empty.")
        if any(type(code) is not int for code in codebook):
            raise ValueError(f"Codebook {level} requires integer code labels.")
        if any(not isfinite(representative) for representative in codebook.values()):
            raise ValueError(f"Codebook {level} requires finite representatives.")

        # Include the code itself so equal distances select the smaller code.
        code = min(codebook, key=lambda candidate: (
            abs(residual - codebook[candidate]), candidate
        ))
        representative = codebook[code]
        next_residual = residual - representative
        if not isfinite(next_residual):
            raise ValueError(f"Residual at level {level} is outside the finite range.")
        steps.append(QuantizationStep(
            level, code, representative, residual, next_residual
        ))
        residual = next_residual
    return tuple(steps)


def build_ids(
    values: Mapping[str, float] = TEACHING_VALUES,
    codebooks: Sequence[Mapping[int, float]] = TEACHING_CODEBOOKS,
) -> dict[str, tuple[int, ...]]:
    """Append a collision suffix assigned in alphabetical drug order."""
    if not values or any(not isinstance(drug, str) or not drug.strip() for drug in values):
        raise ValueError("The catalogue requires at least one nonempty drug name.")

    used_suffixes: dict[tuple[int, ...], int] = {}
    paths = {}
    for drug in sorted(values):
        codes = tuple(step.code for step in quantize(values[drug], codebooks))
        suffix = used_suffixes.get(codes, 0)
        paths[drug] = (*codes, suffix)
        used_suffixes[codes] = suffix + 1
    return paths


def permute_ids(
    paths: Mapping[str, tuple[int, ...]], *, seed: int
) -> dict[str, tuple[int, ...]]:
    """Reassign whole paths to drugs using a reproducible random shuffle."""
    drugs = sorted(paths)
    reassigned_paths = [paths[drug] for drug in drugs]
    Random(seed).shuffle(reassigned_paths)
    return dict(zip(drugs, reassigned_paths))
