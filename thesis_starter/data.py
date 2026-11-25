"""合成就诊、教学 JSON 和按患者划分的跨队列实验。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from random import Random


@dataclass(frozen=True)
class Visit:
    """一次住院。None 表示未知的出院标签，空集合表示已确认无药。"""

    patient_id: str
    visit_id: str
    primary_diagnosis: str
    text: str
    discharge_medications: frozenset[str] | None
    secondary_diagnoses: tuple[str, ...] = ()
    history_diagnoses: tuple[str, ...] = ()
    medication_mentions: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CohortSplit:
    """四个互斥的患者集合，每组只储存符合条件的有标签就诊。"""

    source_train: tuple[Visit, ...]
    target_test: tuple[Visit, ...]
    target_pool: tuple[Visit, ...]
    other_test: tuple[Visit, ...]

    def support(self, k: int) -> tuple[Visit, ...]:
        """按池内患者的固定次序取前 k 人，保留这 k 人的全部目标就诊。"""
        patient_order = tuple(dict.fromkeys(visit.patient_id for visit in self.target_pool))
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= len(patient_order):
            raise ValueError(f"k 必须是 0 到 {len(patient_order)} 之间的患者人数")
        selected = set(patient_order[:k])
        return tuple(visit for visit in self.target_pool if visit.patient_id in selected)


def make_split(
    visits: Iterable[Visit],
    *,
    target_diagnosis: str,
    source_diagnoses: Collection[str],
    seed: int = 42,
    test_fraction: float = 0.25,
    min_target_test: int = 48,
    strict: bool = False,
) -> CohortSplit:
    """用所有就诊判定目标患者，再用有标签就诊构造训练和测试集合。

    非目标患者按其首条（以就诊 ID 排序）符合条件的源组有标签就诊分层。
    只有一位患者的层归入训练。严格规则仅过滤源训练，保留已划好的留出集。
    """
    if isinstance(source_diagnoses, str) or not source_diagnoses:
        raise ValueError("source_diagnoses 需要至少一个源组诊断")
    if not target_diagnosis or target_diagnosis in source_diagnoses:
        raise ValueError("目标主诊断必须有值且与源组主诊断不同")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction 必须在 0 和 1 之间")
    valid_minimum = type(min_target_test) is int and min_target_test > 0
    if not valid_minimum:
        raise ValueError("min_target_test 必须是正整数")

    by_patient: dict[str, list[Visit]] = defaultdict(list)
    visit_ids: set[str] = set()
    for visit in sorted(visits, key=lambda visit: visit.visit_id):
        if visit.visit_id in visit_ids:
            raise ValueError(f"发现重复就诊 ID：{visit.visit_id}")
        visit_ids.add(visit.visit_id)
        by_patient[visit.patient_id].append(visit)

    target_primary_patients = {
        patient_id
        for patient_id, records in by_patient.items()
        if any(visit.primary_diagnosis == target_diagnosis for visit in records)
    }
    target_records = {
        patient_id: _labeled_visits(by_patient[patient_id], {target_diagnosis})
        for patient_id in target_primary_patients
    }
    target_order = sorted(patient_id for patient_id, records in target_records.items() if records)
    Random(seed).shuffle(target_order)
    target_test_count = max(min_target_test, math.ceil(test_fraction * len(target_order)))
    if target_test_count > len(target_order):
        raise ValueError(
            f"目标组只有 {len(target_order)} 名有标签患者，无法留出 {target_test_count} 人"
        )

    # 根据源组主诊断分层，每名患者只进入一个层。
    source_records: dict[str, tuple[Visit, ...]] = {}
    source_strata: dict[str, list[str]] = defaultdict(list)
    source_names = set(source_diagnoses)
    for patient_id in sorted(by_patient.keys() - target_primary_patients):
        records = _labeled_visits(by_patient[patient_id], source_names)
        if records:
            source_records[patient_id] = records
            source_strata[records[0].primary_diagnosis].append(patient_id)

    source_train_ids: list[str] = []
    other_test_ids: list[str] = []
    for diagnosis, patient_ids in sorted(source_strata.items()):
        Random(f"{seed}:source:{diagnosis}").shuffle(patient_ids)
        test_count = min(len(patient_ids) - 1, math.ceil(test_fraction * len(patient_ids)))
        other_test_ids.extend(patient_ids[:test_count])
        source_train_ids.extend(patient_ids[test_count:])

    if strict:
        source_train_ids = [
            patient_id
            for patient_id in source_train_ids
            if not any(
                target_diagnosis in visit.secondary_diagnoses
                or target_diagnosis in visit.history_diagnoses
                for visit in by_patient[patient_id]
            )
        ]

    def records_for(
        records: dict[str, tuple[Visit, ...]], patient_ids: Sequence[str]
    ) -> tuple[Visit, ...]:
        return tuple(visit for patient_id in patient_ids for visit in records[patient_id])

    return CohortSplit(
        source_train=records_for(source_records, source_train_ids),
        target_test=records_for(target_records, target_order[:target_test_count]),
        target_pool=records_for(target_records, target_order[target_test_count:]),
        other_test=records_for(source_records, other_test_ids),
    )


def _labeled_visits(
    records: Iterable[Visit], diagnoses: Collection[str]
) -> tuple[Visit, ...]:
    return tuple(
        visit for visit in records
        if visit.primary_diagnosis in diagnoses and visit.discharge_medications is not None
    )


def read_visits(path: str | Path) -> tuple[Visit, ...]:
    """读取 Visit 字段命名的教学 JSON 数组。原始赛题数据需要单独适配。"""
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise ValueError("教学 JSON 顶层必须是数组")

    records: list[Visit] = []
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"第 {row_number} 条记录必须是对象")
        fields = ("patient_id", "visit_id", "primary_diagnosis", "text")
        for field in fields:
            value = row.get(field)
            if not isinstance(value, str) or (field != "text" and not value.strip()):
                raise ValueError(f"第 {row_number} 条记录的 {field} 字段格式不正确")

        labels = row.get("discharge_medications")
        records.append(Visit(
            patient_id=row["patient_id"],
            visit_id=row["visit_id"],
            primary_diagnosis=row["primary_diagnosis"],
            text=row["text"],
            discharge_medications=(
                None if labels is None else frozenset(
                    _string_array(labels, "discharge_medications", row_number)
                )
            ),
            secondary_diagnoses=_string_array(
                row.get("secondary_diagnoses", []), "secondary_diagnoses", row_number
            ),
            history_diagnoses=_string_array(
                row.get("history_diagnoses", []), "history_diagnoses", row_number
            ),
            medication_mentions=frozenset(_string_array(
                row.get("medication_mentions", []), "medication_mentions", row_number
            )),
        ))
    return tuple(records)


def _string_array(value: object, field: str, row_number: int) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"第 {row_number} 条记录的 {field} 字段必须是非空字符串组成的数组")
    return tuple(value)


def synthetic_visits() -> tuple[Visit, ...]:
    """生成虚构线索与 A–D 标签，包含重复住院、交叉主诊断和次诊断。"""
    source = [
        _synthetic_visit(f"S{index:03d}", f"VS{index:03d}", diagnosis, index)
        for index, diagnosis in enumerate(["源组一"] * 144 + ["源组二"] * 144, start=1)
    ]
    target = [
        _synthetic_visit(f"T{index:03d}", f"VT{index:03d}", "目标组", index)
        for index in range(1, 127)
    ]
    examples = (
        _synthetic_visit("P07", "V071", "源组一", 0),
        _synthetic_visit("P07", "V072", "目标组", 1),
        _synthetic_visit("P08", "V081", "源组二", 1, secondary=("目标组",)),
        _synthetic_visit("P12", "V121", "目标组", 4),
        _synthetic_visit("T001", "VT001-repeat", "目标组", 1),
    )
    return (*source, *target, *examples)


# 每行依次是虚构的病历线索、源组标签、目标组标签和虚构既往提及。
_PATTERNS = (
    ("线索甲 记录完整", ("A", "C"), ("B", "C"), ("A",)),
    ("线索甲 单项提示", ("A",), ("B",), ("A",)),
    ("线索乙 记录完整", ("C", "D"), ("C", "D"), ("D",)),
    ("线索乙 单项提示", ("D",), ("D",), ()),
    ("线索甲 观察复核", ("A", "C"), ("A", "C"), ("C",)),
    ("线索丙 单项提示", ("B",), ("B",), ("A",)),
    ("线索丙 记录完整", ("A", "B"), ("B", "C"), ("B",)),
    ("线索乙 观察复核", ("C",), ("C",), ("D",)),
)


def _synthetic_visit(
    patient_id: str,
    visit_id: str,
    diagnosis: str,
    pattern: int,
    *,
    secondary: tuple[str, ...] = (),
) -> Visit:
    text, source_labels, target_labels, mentions = _PATTERNS[pattern % len(_PATTERNS)]
    labels = target_labels if diagnosis == "目标组" else source_labels
    return Visit(
        patient_id=patient_id,
        visit_id=visit_id,
        primary_diagnosis=diagnosis,
        text=text,
        discharge_medications=frozenset(labels),
        secondary_diagnoses=secondary,
        medication_mentions=frozenset(mentions),
    )
