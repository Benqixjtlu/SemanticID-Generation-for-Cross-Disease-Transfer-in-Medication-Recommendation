from unittest import TestCase

from thesis_starter.data import Visit
from thesis_starter.decoding import CatalogDecoder, DecodeState, STOP
from thesis_starter.ids import build_ids, permute_ids
from thesis_starter.model import PrefixCountModel


class PrefixCountModelTests(TestCase):
    def test_invalid_label_does_not_partially_update_the_model(self) -> None:
        model = PrefixCountModel(CatalogDecoder(build_ids()))
        before = model.token_probabilities("one", DecodeState())
        visits = [
            Visit("P1", "V1", "source", "one", frozenset({"A"})),
            Visit("P2", "V2", "source", "one", frozenset({"unknown"})),
        ]
        with self.assertRaises(ValueError):
            model.learn(visits)
        self.assertEqual(model.token_probabilities("one", DecodeState()), before)

    def test_teacher_forcing_learns_legal_next_tokens(self) -> None:
        decoder = CatalogDecoder(build_ids())
        model = PrefixCountModel(decoder)
        labels = frozenset({"A", "C"})
        model.learn([Visit("P1", "V1", "source", "hint", labels)])

        state = DecodeState(prefix=(2,))
        probabilities = model.token_probabilities("hint", state)
        self.assertEqual(set(probabilities), {7, 9})
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(probabilities[7], probabilities[9])

    def test_predicts_fitted_set_under_both_mappings(self) -> None:
        ids = build_ids()
        gold = frozenset({"A", "C"})
        visits = [
            Visit(f"P{i}", f"V{i}", "source", "one", gold)
            for i in range(12)
        ]
        visits += [
            Visit("P12", "V12", "source", "two", frozenset()),
            Visit("P13", "V13", "source", "one", None),
        ]
        for mapping in (ids, permute_ids(ids, seed=42)):
            with self.subTest(mapping=mapping):
                decoder = CatalogDecoder(mapping)
                model = PrefixCountModel(decoder)
                model.learn(visits)
                self.assertEqual(model.predict("one"), ("A", "C"))
                self.assertEqual(model.predict("two"), ())
                self.assertEqual(decoder.serialize(model.predict("two")), (STOP,))
