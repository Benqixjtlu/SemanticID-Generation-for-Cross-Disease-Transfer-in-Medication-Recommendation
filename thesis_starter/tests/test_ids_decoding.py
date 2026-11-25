"""Check the worked example and drug-path safety invariants."""

from itertools import combinations
import unittest

from thesis_starter.decoding import CatalogDecoder, DecodeState, STOP
from thesis_starter.ids import TEACHING_CODEBOOKS, build_ids, permute_ids, quantize


class QuantizationTests(unittest.TestCase):
    def test_residual_trace_and_collision_suffix(self) -> None:
        steps = quantize(116, TEACHING_CODEBOOKS)
        self.assertEqual([step.code for step in steps], [2, 9, 1])
        self.assertEqual([step.residual_before for step in steps], [116, 36, 6])
        self.assertEqual([step.residual_after for step in steps], [36, 6, -4])
        self.assertEqual(build_ids(), {
            "A": (2, 7, 4, 0),
            "B": (2, 7, 4, 1),
            "C": (2, 9, 1, 0),
            "D": (3, 7, 4, 0),
        })
        self.assertEqual(build_ids({"B": 78, "A": 74}), {
            "A": (2, 7, 4, 0),
            "B": (2, 7, 4, 1),
        })

    def test_permutation_preserves_catalogue_and_full_paths(self) -> None:
        paths = build_ids()
        shuffled = permute_ids(paths, seed=42)
        self.assertEqual(set(shuffled), set(paths))
        self.assertEqual(set(shuffled.values()), set(paths.values()))
        self.assertEqual(shuffled, permute_ids(
            dict(reversed(tuple(paths.items()))), seed=42
        ))


class DecodingTests(unittest.TestCase):
    def test_every_subset_round_trips_for_both_mappings(self) -> None:
        paths = build_ids()
        ranking = ("D", "B", "A", "C")
        for mapping in (paths, permute_ids(paths, seed=42)):
            decoder = CatalogDecoder(mapping, ranking=ranking)
            for size in range(len(ranking) + 1):
                for subset in combinations(ranking, size):
                    with self.subTest(mapping=mapping, subset=subset):
                        tokens = decoder.serialize(tuple(reversed(subset)))
                        self.assertEqual(decoder.decode(tokens), subset)

    def test_rejects_illegal_stop_path_order_and_extra_tokens(self) -> None:
        paths = build_ids()
        decoder = CatalogDecoder(paths)
        prefix = decoder.advance(DecodeState(), 2)
        self.assertEqual(decoder.allowed(prefix), (7, 9))
        illegal_sequences = (
            (2, STOP),
            (2, 3, STOP),
            (*paths["A"], *paths["A"], STOP),
            (*paths["B"], *paths["A"], STOP),
            (*paths["A"],),
            (STOP, STOP),
        )
        for tokens in illegal_sequences:
            with self.subTest(tokens=tokens):
                with self.assertRaises(ValueError):
                    decoder.decode(tokens)

    def test_rank_prunes_only_branches_without_an_eligible_drug(self) -> None:
        paths = build_ids()
        decoder = CatalogDecoder(paths)
        state = DecodeState()
        for token in paths["B"]:
            state = decoder.advance(state, token)
        self.assertEqual(decoder.allowed(state), (2, 3, STOP))
        self.assertEqual(decoder.allowed(decoder.advance(state, 2)), (9,))
        with self.assertRaises(ValueError):
            decoder.serialize({"X"})

    def test_catalogue_rejects_duplicate_paths_and_invalid_ranking(self) -> None:
        with self.assertRaises(ValueError):
            CatalogDecoder({"A": (1,), "B": (1,)})
        with self.assertRaises(ValueError):
            CatalogDecoder(build_ids(), ranking=("A", "A", "B", "C"))
