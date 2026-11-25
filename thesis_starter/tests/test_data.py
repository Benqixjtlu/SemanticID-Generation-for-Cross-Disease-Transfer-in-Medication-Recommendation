import itertools
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from thesis_starter.data import Visit, make_split, read_visits, synthetic_visits


def patient_ids(visits):
    return {visit.patient_id for visit in visits}


class CohortTests(unittest.TestCase):
    options = dict(target_diagnosis="目标组", source_diagnoses=("源组一", "源组二"))

    def test_synthetic_split_keeps_every_patient_in_one_group(self):
        visits = synthetic_visits()
        split = make_split(visits, **self.options)
        groups = (split.source_train, split.target_test, split.target_pool, split.other_test)
        for first, second in itertools.combinations(map(patient_ids, groups), 2):
            self.assertTrue(first.isdisjoint(second))
        self.assertEqual(len(patient_ids(split.target_test)), 48)
        self.assertEqual(len(patient_ids(split.target_pool)), 80)
        self.assertGreaterEqual(len(patient_ids(split.source_train)), 200)
        self.assertNotIn("P07", patient_ids(split.source_train) | patient_ids(split.other_test))
        self.assertIn("V072", {visit.visit_id for visit in split.target_test + split.target_pool})
        self.assertNotIn("V071", {visit.visit_id for group in groups for visit in group})
        self.assertEqual(split, make_split(reversed(visits), **self.options))

    def test_support_samples_patients_and_keeps_repeated_target_visits(self):
        gold = frozenset({"A"})
        visits = [
            Visit("P1", "V1", "目标组", "", gold),
            Visit("P2", "V2", "目标组", "", gold),
            Visit("P2", "V3", "目标组", "", gold),
            Visit("P3", "V4", "目标组", "", gold),
        ]
        split = make_split(visits, **self.options, min_target_test=1)
        self.assertEqual(split.support(0), ())
        self.assertEqual(len(patient_ids(split.support(1))), 1)
        self.assertTrue(set(split.support(1)) < set(split.support(2)))
        for group in (split.target_test, split.support(2)):
            both_visits = [visit for visit in group if visit.patient_id == "P2"]
            if both_visits:
                self.assertEqual({visit.visit_id for visit in both_visits}, {"V2", "V3"})
        with self.assertRaises(ValueError):
            split.support(3)

    def test_strict_filter_uses_all_visits_without_changing_holdouts(self):
        gold = frozenset({"A"})
        visits = (
            Visit("target", "V1", "目标组", "", gold),
            Visit("shared", "V2", "目标组", "", None),
            Visit("shared", "V3", "源组二", "", gold),
            Visit("history", "V4", "源组一", "", gold),
            Visit("history", "V5", "其他组", "", None, history_diagnoses=("目标组",)),
            Visit("secondary", "V6", "源组二", "", gold, secondary_diagnoses=("目标组",)),
        )
        ordinary = make_split(visits, **self.options, min_target_test=1)
        strict = make_split(visits, **self.options, min_target_test=1, strict=True)
        self.assertEqual(patient_ids(ordinary.source_train), {"history", "secondary"})
        self.assertEqual(strict.source_train, ())
        self.assertEqual(strict.target_test, ordinary.target_test)
        self.assertEqual(strict.target_pool, ordinary.target_pool)
        self.assertEqual(strict.other_test, ordinary.other_test)

    def test_duplicate_visit_ids_and_impossible_holdout_fail(self):
        duplicate = Visit("P1", "V1", "目标组", "", frozenset())
        with self.assertRaisesRegex(ValueError, "V1"):
            make_split([duplicate, duplicate], **self.options)
        with self.assertRaisesRegex(ValueError, "48"):
            make_split([duplicate], **self.options)


class TeachingJSONTests(unittest.TestCase):
    def test_missing_labels_stay_distinct_from_empty_prescriptions(self):
        base = {"patient_id": "P1", "primary_diagnosis": "目标组", "text": "虚构笔记"}
        rows = [
            {**base, "visit_id": "V1", "discharge_medications": []},
            {**base, "visit_id": "V2", "discharge_medications": None},
            {**base, "visit_id": "V3"},
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "teaching.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            visits = read_visits(path)
        self.assertEqual(visits[0].discharge_medications, frozenset())
        self.assertIsNone(visits[1].discharge_medications)
        self.assertIsNone(visits[2].discharge_medications)
        split = make_split(visits, **CohortTests.options, min_target_test=1)
        self.assertEqual(split.target_test, (visits[0],))
