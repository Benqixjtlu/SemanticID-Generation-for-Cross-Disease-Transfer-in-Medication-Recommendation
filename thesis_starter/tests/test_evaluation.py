"""固定导读的手算值及患者级和空标签评分约定。"""

import unittest

from thesis_starter.baselines import FrequencyBaseline, mention_copy
from thesis_starter.data import Visit
from thesis_starter.evaluation import (
    TRANSFER_ARMS,
    Scores,
    evaluate,
    paired_bootstrap,
    score_visit,
    transfer_effects,
)


def visit(patient_id, visit_id, labels, *, diagnosis="source", mentions=()):
    return Visit(
        patient_id=patient_id,
        visit_id=visit_id,
        primary_diagnosis=diagnosis,
        text="Synthetic teaching record",
        discharge_medications=None if labels is None else frozenset(labels),
        medication_mentions=frozenset(mentions),
    )


class EvaluationTests(unittest.TestCase):
    def test_worked_example_and_empty_sets(self):
        score = score_visit({"A", "C"}, {"B", "C", "D"})
        self.assertEqual(score, Scores(jaccard=0.25, f1=0.40))
        self.assertAlmostEqual(score.overall, 0.325)
        self.assertEqual(score_visit(set(), set()), Scores(1.0, 1.0))
        self.assertEqual(score_visit(set(), {"A"}), Scores(0.0, 0.0))
        self.assertEqual(score_visit({"A"}, set()), Scores(0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "collection of drug names"):
            score_visit({"A"}, "A")

    def test_missing_gold_is_excluded_but_labeled_visit_requires_prediction(self):
        visits = (
            visit("p1", "v1", {"A"}),
            visit("p1", "v2", frozenset()),
            visit("p2", "v3", None),
        )
        evaluation = evaluate(visits, {"v1": {"B"}, "v2": set()})
        self.assertEqual(evaluation.averages, Scores(0.5, 0.5))
        self.assertEqual(evaluation.n_visits, 2)
        self.assertEqual(evaluation.n_patients, 1)
        self.assertEqual(evaluation.n_missing_gold, 1)
        self.assertEqual(evaluation.n_both_empty, 1)
        with self.assertRaisesRegex(ValueError, "v2"):
            evaluate(visits, {"v1": {"B"}})
        with self.assertRaisesRegex(ValueError, "at least one labeled"):
            evaluate(visits[-1:], {})

    def test_tutorial_transfer_gains_and_final_scores_can_disagree(self):
        effects = transfer_effects(
            {"rq_target": 0.31, "rq_transfer": 0.43, "perm_target": 0.37, "perm_transfer": 0.45}
        )
        self.assertAlmostEqual(effects.transfer, 0.04)
        self.assertAlmostEqual(effects.direct, -0.02)

    def test_bootstrap_resamples_whole_patients_and_keeps_visit_weight(self):
        visits = (
            visit("p1", "v1", {"A"}),
            visit("p1", "v2", {"A"}),
            visit("p1", "v3", {"A"}),
            visit("p2", "v4", {"A"}),
        )
        misses = {item.visit_id: set() for item in visits}
        partial_hits = {"v1": {"A"}, "v2": {"A"}, "v3": {"A"}, "v4": set()}
        arms = dict.fromkeys(TRANSFER_ARMS, misses)
        arms["rq_transfer"] = partial_hits
        result = paired_bootstrap(visits, arms, replicates=100, seed=3)
        self.assertEqual(result.n_visits, 4)
        self.assertEqual(result.n_patients, 2)
        self.assertEqual(result.point.transfer, 0.75)
        self.assertEqual(result.point.direct, 0.75)
        self.assertEqual(result.transfer_ci, (0.0, 1.0))
        self.assertEqual(result, paired_bootstrap(visits, arms, replicates=100, seed=3))

        # 两套编码逐条预测完全一样，每轮成对重抽的差值应为 0。
        arms["perm_transfer"] = partial_hits
        paired_result = paired_bootstrap(visits, arms, replicates=100, seed=3)
        self.assertEqual(paired_result.transfer_ci, (0.0, 0.0))
        self.assertEqual(paired_result.direct_ci, (0.0, 0.0))

        arms["perm_target"] = {}
        with self.assertRaisesRegex(ValueError, "perm_target: .*v1"):
            paired_bootstrap(visits, arms, replicates=1)


class BaselineTests(unittest.TestCase):
    def test_frequency_and_mention_copy_use_different_allowed_inputs(self):
        training = (
            visit("p1", "s1", {"A"}, diagnosis="x"),
            visit("p2", "s2", {"B"}, diagnosis="y"),
            visit("p3", "s3", {"B"}, diagnosis="y"),
        )
        intake = visit("p4", "t1", None, diagnosis="x", mentions=("A", "out-of-catalog"))
        model = FrequencyBaseline.fit(training)
        self.assertEqual(model.predict(intake, top_n=1), frozenset({"B"}))
        self.assertEqual(model.predict(intake, top_n=1, by_diagnosis=True), frozenset({"A"}))
        self.assertEqual(mention_copy(intake, {"A", "B"}), frozenset({"A"}))
