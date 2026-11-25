"""按就诊计算集合分数，按患者成组估计四组实验的差值区间。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from random import Random
from statistics import fmean

from .data import Visit


TRANSFER_ARMS = ("rq_target", "rq_transfer", "perm_target", "perm_transfer")


@dataclass(frozen=True)
class Scores:
    jaccard: float
    f1: float

    @property
    def overall(self) -> float:
        return (self.jaccard + self.f1) / 2


def score_visit(gold: Collection[str], predicted: Collection[str]) -> Scores:
    """对同一次就诊按药物集合打分，约定两个空集视为完全匹配。"""
    if isinstance(gold, str) or isinstance(predicted, str):
        raise ValueError("Pass a collection of drug names for each visit.")
    actual, prediction = set(gold), set(predicted)
    if not actual and not prediction:
        return Scores(jaccard=1.0, f1=1.0)

    hits = len(actual & prediction)
    return Scores(
        jaccard=hits / len(actual | prediction),
        f1=2 * hits / (len(actual) + len(prediction)),
    )


@dataclass(frozen=True)
class Evaluation:
    averages: Scores
    n_visits: int
    n_patients: int
    n_missing_gold: int
    n_both_empty: int


def _labeled_visits(visits: Iterable[Visit]) -> tuple[list[tuple[Visit, frozenset[str]]], int]:
    labeled = []
    seen_visit_ids = set()
    missing_count = 0
    for visit in visits:
        if visit.visit_id in seen_visit_ids:
            raise ValueError(f"Duplicate visit ID {visit.visit_id!r}")
        seen_visit_ids.add(visit.visit_id)
        if visit.discharge_medications is None:
            missing_count += 1
        else:
            labeled.append((visit, visit.discharge_medications))
    if not labeled:
        raise ValueError("Evaluation requires at least one labeled visit.")
    return labeled, missing_count


def _prediction_for(
    predictions: Mapping[str, Collection[str]], visit_id: str
) -> Collection[str]:
    if visit_id not in predictions:
        raise ValueError(f"Missing prediction for labeled visit {visit_id!r}")
    return predictions[visit_id]


def evaluate(
    visits: Iterable[Visit], predictions: Mapping[str, Collection[str]]
) -> Evaluation:
    """先按就诊计分，再计算平均分。标签缺失的记录计数但不评分。"""
    labeled, missing_count = _labeled_visits(visits)
    scores = []
    both_empty_count = 0
    for visit, gold in labeled:
        predicted = _prediction_for(predictions, visit.visit_id)
        scores.append(score_visit(gold, predicted))
        if not gold and not predicted:
            both_empty_count += 1

    return Evaluation(
        averages=Scores(
            jaccard=fmean(score.jaccard for score in scores),
            f1=fmean(score.f1 for score in scores),
        ),
        n_visits=len(labeled),
        n_patients=len({visit.patient_id for visit, _ in labeled}),
        n_missing_gold=missing_count,
        n_both_empty=both_empty_count,
    )


@dataclass(frozen=True)
class TransferEffects:
    transfer: float
    direct: float


def transfer_effects(scores: Mapping[str, float]) -> TransferEffects:
    """计算源训练增益差和最终分差，四个输入均为平均 Jaccard。"""
    if set(scores) != set(TRANSFER_ARMS):
        raise ValueError(f"Expected exactly these experimental arms: {', '.join(TRANSFER_ARMS)}")
    rq_gain = scores["rq_transfer"] - scores["rq_target"]
    perm_gain = scores["perm_transfer"] - scores["perm_target"]
    return TransferEffects(
        transfer=rq_gain - perm_gain,
        direct=scores["rq_transfer"] - scores["perm_transfer"],
    )


@dataclass(frozen=True)
class BootstrapResult:
    point: TransferEffects
    transfer_ci: tuple[float, float]
    direct_ci: tuple[float, float]
    n_visits: int
    n_patients: int
    replicates: int


def _effects_for_patient_sample(
    patient_sample: Sequence[str],
    visit_count: Mapping[str, int],
    jaccard_totals: Mapping[str, Mapping[str, float]],
) -> TransferEffects:
    total_visits = sum(visit_count[patient] for patient in patient_sample)
    averages = {
        arm: sum(jaccard_totals[patient][arm] for patient in patient_sample) / total_visits
        for arm in TRANSFER_ARMS
    }
    return transfer_effects(averages)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = ceil(position)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def _interval(values: Iterable[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return _percentile(ordered, 0.025), _percentile(ordered, 0.975)


def paired_bootstrap(
    visits: Iterable[Visit],
    predictions_by_arm: Mapping[str, Mapping[str, Collection[str]]],
    *,
    replicates: int = 2000,
    seed: int = 42,
) -> BootstrapResult:
    """用一套四组预测重抽患者，每轮保留抽中患者的全部有标签就诊。"""
    if replicates < 1:
        raise ValueError("At least one bootstrap replicate is required.")
    if set(predictions_by_arm) != set(TRANSFER_ARMS):
        raise ValueError(f"Expected exactly these experimental arms: {', '.join(TRANSFER_ARMS)}")

    labeled, _ = _labeled_visits(visits)
    visit_count = Counter(visit.patient_id for visit, _ in labeled)
    patients = sorted(visit_count)
    jaccard_totals = {patient: dict.fromkeys(TRANSFER_ARMS, 0.0) for patient in patients}
    for visit, gold in labeled:
        for arm in TRANSFER_ARMS:
            try:
                predicted = _prediction_for(predictions_by_arm[arm], visit.visit_id)
            except ValueError as error:
                raise ValueError(f"{arm}: {error}") from error
            jaccard_totals[visit.patient_id][arm] += score_visit(gold, predicted).jaccard

    # 患者负责划分和重抽样，平均分的分母仍是抽中患者对应的就诊数。
    point = _effects_for_patient_sample(patients, visit_count, jaccard_totals)
    rng = Random(seed)
    resampled_effects = []
    for _ in range(replicates):
        patient_sample = rng.choices(patients, k=len(patients))
        effects = _effects_for_patient_sample(patient_sample, visit_count, jaccard_totals)
        resampled_effects.append(effects)

    return BootstrapResult(
        point=point,
        transfer_ci=_interval(effect.transfer for effect in resampled_effects),
        direct_ci=_interval(effect.direct for effect in resampled_effects),
        n_visits=len(labeled),
        n_patients=len(patients),
        replicates=replicates,
    )
