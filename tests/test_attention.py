import math
import unittest

import examples.attention as attention_module
from examples.attention import (
    causal_mask,
    multi_head_attention,
    rotate_pair,
    scaled_dot_product_attention,
)


class AttentionTest(unittest.TestCase):
    def assertVectorClose(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertTrue(math.isfinite(actual_value))
            self.assertAlmostEqual(actual_value, expected_value, delta=1e-9)

    def assertMatrixClose(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_row, expected_row in zip(actual, expected):
            self.assertVectorClose(actual_row, expected_row)

    @staticmethod
    def fixture():
        values = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        return values, [row[:] for row in values], [row[:] for row in values]

    def test_scaled_attention_computes_scores_normalized_weights_and_output(self):
        q, k, v = self.fixture()
        result = scaled_dot_product_attention(q, k, v, 1 / math.sqrt(2))

        self.assertMatrixClose(
            result["scores"],
            [
                [math.sqrt(0.5), 0, math.sqrt(0.5)],
                [0, math.sqrt(0.5), math.sqrt(0.5)],
                [math.sqrt(0.5), math.sqrt(0.5), math.sqrt(2)],
            ],
        )
        self.assertMatrixClose(
            result["weights"],
            [
                [0.4011120927, 0.1977758146, 0.4011120927],
                [0.1977758146, 0.4011120927, 0.4011120927],
                [0.2482550783, 0.2482550783, 0.5034898435],
            ],
        )
        for row in result["weights"]:
            self.assertAlmostEqual(sum(row), 1, delta=1e-9)
        self.assertMatrixClose(
            result["output"],
            [
                [0.8022241854, 0.5988879073],
                [0.5988879073, 0.8022241854],
                [0.7517449217, 0.7517449217],
            ],
        )
        expected = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        self.assertEqual(q, expected)
        self.assertEqual(k, expected)
        self.assertEqual(v, expected)

    def test_scaled_attention_applies_scale_before_softmax_and_uses_values(self):
        scores = [[2.0, 1.0, 3.0], [1.0, 1.0, 1.0], [3.0, 1.0, 2.0]]
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        result = scaled_dot_product_attention(
            scores, identity, identity, 1 / math.sqrt(2)
        )

        self.assertVectorClose(
            result["weights"][0], [0.2839954097, 0.1400292450, 0.5759753452]
        )
        self.assertAlmostEqual(result["output"][0][0], 0.2839954097, delta=1e-9)
        self.assertAlmostEqual(result["output"][0][2], 0.5759753452, delta=1e-9)
        for row in result["weights"]:
            self.assertAlmostEqual(sum(row), 1, delta=1e-9)

    def test_scaled_attention_distinguishes_keys_from_values_and_dimensions(self):
        q = [[1, 0]]
        k = [[1, 0], [0, 1]]
        v = [[10], [20]]
        result = scaled_dot_product_attention(q, k, v, 1)

        self.assertEqual(result["scores"], [[1, 0]])
        self.assertMatrixClose(
            result["weights"], [[0.7310585786, 0.2689414214]]
        )
        self.assertMatrixClose(result["output"], [[12.6894142137]])

    def test_multi_head_attention_uses_keys_for_scores_and_values_for_output(self):
        q = [[1, 0]]
        k = [[1, 0], [0, 1]]
        v = [[10, 10], [20, 20]]
        result = multi_head_attention(q, k, v, 2)

        self.assertEqual(result["heads"][0]["scores"], [[1, 0]])
        self.assertEqual(result["heads"][1]["scores"], [[0, 0]])
        self.assertMatrixClose(
            result["heads"][0]["weights"], [[0.7310585786, 0.2689414214]]
        )
        self.assertMatrixClose(result["heads"][1]["weights"], [[0.5, 0.5]])
        self.assertMatrixClose(result["heads"][0]["output"], [[12.6894142137]])
        self.assertMatrixClose(result["heads"][1]["output"], [[15]])
        self.assertMatrixClose(result["output"], [[12.6894142137, 15]])

    def test_causal_mask_blocks_only_strict_future_positions_without_mutation(self):
        scores = [[2.0, 1.0, 3.0], [1.0, 1.0, 1.0], [3.0, 1.0, 2.0]]
        masked = causal_mask(scores)

        self.assertEqual(masked[0][1], -math.inf)
        self.assertEqual(masked[0][2], -math.inf)
        self.assertEqual(masked[1][2], -math.inf)
        self.assertEqual(masked[0][0], 2)
        self.assertEqual(masked[1][:2], [1, 1])
        self.assertEqual(masked[2], [3, 1, 2])
        self.assertEqual(scores, [[2, 1, 3], [1, 1, 1], [3, 1, 2]])
        self.assertIsNot(masked, scores)
        for actual_row, original_row in zip(masked, scores):
            self.assertIsNot(actual_row, original_row)

    def test_masked_softmax_composes_with_causal_mask_and_zeroes_future_weights(self):
        scores = [[2.0, 1.0, 3.0], [1.0, 1.0, 1.0], [3.0, 1.0, 2.0]]
        weights = [
            attention_module.masked_softmax(row) for row in causal_mask(scores)
        ]

        self.assertMatrixClose(
            weights,
            [
                [1, 0, 0],
                [0.5, 0.5, 0],
                [0.6652409558, 0.0900305735, 0.2447284711],
            ],
        )
        self.assertEqual(weights[0][1], 0)
        self.assertEqual(weights[0][2], 0)
        self.assertEqual(weights[1][2], 0)
        for row in weights:
            self.assertAlmostEqual(sum(row), 1, delta=1e-9)
        self.assertVectorClose(
            attention_module.masked_softmax([1000, 1001, -math.inf]),
            [0.2689414214, 0.7310585786, 0],
        )

    def test_masked_softmax_rejects_invalid_values_and_fully_masked_rows(self):
        invalid_rows = [
            [],
            [-math.inf, -math.inf],
            [math.nan, -math.inf],
            [math.inf, -math.inf],
        ]
        for row in invalid_rows:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    attention_module.masked_softmax(row)

    def test_multi_head_attention_computes_independent_heads_and_concatenates(self):
        q, k, v = self.fixture()
        result = multi_head_attention(q, k, v, 2)

        self.assertEqual(len(result["heads"]), 2)
        self.assertMatrixClose(
            result["heads"][0]["output"],
            [[0.8446375965], [0.6666666667], [0.8446375965]],
        )
        self.assertMatrixClose(
            result["heads"][1]["output"],
            [[0.6666666667], [0.8446375965], [0.8446375965]],
        )
        self.assertMatrixClose(
            result["output"],
            [
                [0.8446375965, 0.6666666667],
                [0.6666666667, 0.8446375965],
                [0.8446375965, 0.8446375965],
            ],
        )
        for head in result["heads"]:
            for row in head["weights"]:
                self.assertAlmostEqual(sum(row), 1, delta=1e-9)

    def test_single_head_output_matches_scaled_dot_product_attention(self):
        q, k, v = self.fixture()
        single = scaled_dot_product_attention(q, k, v, 1 / math.sqrt(2))
        result = multi_head_attention(q, k, v, 1)

        self.assertEqual(result["output"], single["output"])
        self.assertEqual(result["heads"], [single])

    def test_rope_rotates_adjacent_pairs_in_positive_direction_without_mutation(self):
        vector = [1.0, 0.0, 1.0, 0.0]
        quarter_turn = rotate_pair(vector, math.pi / 2)
        self.assertVectorClose(quarter_turn, [0, 1, 0, 1])
        self.assertVectorClose(rotate_pair([1, 0], 0), [1, 0])
        self.assertVectorClose(rotate_pair([1, 0], math.pi), [-1, 0])
        self.assertVectorClose(rotate_pair([1, 2, 3, 4], math.pi / 2), [-2, 1, -4, 3])
        self.assertEqual(vector, [1, 0, 1, 0])
        self.assertIsNot(quarter_turn, vector)

    def test_attention_validates_matrices_dimensions_scales_and_head_counts(self):
        operations = [
            lambda: scaled_dot_product_attention([], [[1]], [[1]], 1),
            lambda: scaled_dot_product_attention([[1]], [], [[1]], 1),
            lambda: scaled_dot_product_attention([[1]], [[1]], [], 1),
            lambda: scaled_dot_product_attention([[]], [[1]], [[1]], 1),
            lambda: scaled_dot_product_attention([[1, 2]], [[1]], [[1]], 1),
            lambda: scaled_dot_product_attention([[1]], [[1], [2]], [[1]], 1),
            lambda: scaled_dot_product_attention([[1, 2], [3]], [[1, 2]], [[1], [2]], 1),
            lambda: multi_head_attention([[1, 0]], [[1, 0]], [[1, 0]], 0),
            lambda: multi_head_attention([[1, 0]], [[1, 0]], [[1, 0]], -1),
            lambda: multi_head_attention([[1, 0]], [[1, 0]], [[1, 0]], 1.5),
            lambda: multi_head_attention([[1, 0]], [[1, 0]], [[1, 0]], math.nan),
            lambda: multi_head_attention([[1, 0]], [[1, 0]], [[1, 0, 0]], 2),
        ]
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    operation()

        for scale in (0, -1, math.nan, math.inf, -math.inf):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    scaled_dot_product_attention([[1]], [[1]], [[1]], scale)
        for value in (math.nan, math.inf, -math.inf):
            non_finite_operations = [
                lambda: scaled_dot_product_attention([[value]], [[1]], [[1]], 1),
                lambda: scaled_dot_product_attention([[1]], [[value]], [[1]], 1),
                lambda: scaled_dot_product_attention([[1]], [[1]], [[value]], 1),
            ]
            for operation in non_finite_operations:
                with self.subTest(value=value, operation=operation):
                    with self.assertRaises(ValueError):
                        operation()

    def test_causal_mask_and_rope_reject_malformed_or_non_finite_inputs(self):
        invalid_masks = [
            [],
            [[1, 2]],
            [[1], [2]],
            [[1, 2], [3]],
            [[1, math.nan], [1, 2]],
            [[1, math.inf], [1, 2]],
        ]
        for scores in invalid_masks:
            with self.subTest(scores=scores):
                with self.assertRaises(ValueError):
                    causal_mask(scores)

        with self.assertRaises(ValueError):
            rotate_pair([], 0)
        with self.assertRaises(ValueError):
            rotate_pair([1, 2, 3], 0)
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    rotate_pair([value, 0], 0)
                with self.assertRaises(ValueError):
                    rotate_pair([1, 0], value)


if __name__ == "__main__":
    unittest.main()
