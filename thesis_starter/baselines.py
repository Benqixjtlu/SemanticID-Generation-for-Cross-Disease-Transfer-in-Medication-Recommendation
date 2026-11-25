"""两个只依赖教学数据表的出院药名基线。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass

from .data import Visit


def _rank_drugs(counts: Counter[str]) -> tuple[str, ...]:
    return tuple(sorted(counts, key=lambda drug: (-counts[drug], drug)))


@dataclass(frozen=True)
class FrequencyBaseline:
    overall_ranking: tuple[str, ...]
    diagnosis_rankings: Mapping[str, tuple[str, ...]]

    @classmethod
    def fit(cls, visits: Iterable[Visit]) -> FrequencyBaseline:
        """只从调用者提供的训练就诊统计标签事件频数。"""
        overall_counts: Counter[str] = Counter()
        diagnosis_counts: dict[str, Counter[str]] = {}
        for visit in visits:
            if visit.discharge_medications is None:
                continue
            group_counts = diagnosis_counts.setdefault(visit.primary_diagnosis, Counter())
            overall_counts.update(visit.discharge_medications)
            group_counts.update(visit.discharge_medications)
        if not diagnosis_counts:
            raise ValueError("Frequency training requires at least one labeled visit.")

        return cls(
            overall_ranking=_rank_drugs(overall_counts),
            diagnosis_rankings={
                diagnosis: _rank_drugs(counts)
                for diagnosis, counts in diagnosis_counts.items()
            },
        )

    def predict(
        self, visit: Visit, *, top_n: int = 3, by_diagnosis: bool = False
    ) -> frozenset[str]:
        """可按主诊断取常见药，遇到训练时未见的诊断则回退到总体频率。"""
        if top_n < 0:
            raise ValueError("top_n must be nonnegative.")
        ranking = self.overall_ranking
        if by_diagnosis:
            ranking = self.diagnosis_rankings.get(visit.primary_diagnosis, ranking)
        return frozenset(ranking[:top_n])


def mention_copy(visit: Visit, catalog: Collection[str]) -> frozenset[str]:
    """复制预测输入中已标记的药名线索，只保留目录内的药。"""
    return visit.medication_mentions.intersection(catalog)
