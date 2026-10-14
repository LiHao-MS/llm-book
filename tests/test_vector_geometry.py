import math
import unittest

from examples.vector_geometry import add, cosine, dot, norm, projection, transform


class VectorGeometryTest(unittest.TestCase):
    def assertClose(self, actual, expected):
        self.assertTrue(math.isfinite(actual))
        self.assertAlmostEqual(actual, expected, delta=1e-9)

    def test_linalg_computes_the_fixed_vector_and_matrix_examples(self):
        self.assertEqual(add([1, 2], [3, 4]), [4, 6])
        self.assertEqual(dot([1, 2], [3, 4]), 11)
        self.assertEqual(norm([3, 4]), 5)
        self.assertClose(cosine([1, 0], [1, 1]), math.sqrt(0.5))
        self.assertEqual(projection([2, 2], [1, 0]), [2, 0])
        self.assertEqual(projection([2, 2], [2, 0]), [2, 0])
        self.assertEqual(transform([[0, -1], [1, 0]], [1, 0]), [0, 1])
        self.assertEqual(transform([[2, 1], [0, 3]], [1, 1]), [3, 3])

    def test_norm_cosine_and_projection_avoid_extreme_scale_errors(self):
        for value in (1e200, 1e-200):
            operations = [
                ("norm", lambda: norm([value]), value),
                ("cosine", lambda: cosine([value], [value]), 1),
                ("projection", lambda: projection([value], [value]), [value]),
            ]
            for name, operation, expected in operations:
                with self.subTest(value=value, operation=name):
                    self.assertEqual(operation(), expected)

    def test_projection_rejects_a_genuinely_unrepresentable_result(self):
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        with self.assertRaises(ValueError):
            projection([maximum, maximum], [1, 0.1])

    def test_cosine_and_projection_reject_zero_vectors_in_denominators(self):
        with self.assertRaises(ValueError):
            cosine([0, 0], [1, 0])
        with self.assertRaises(ValueError):
            cosine([1, 0], [0, 0])
        with self.assertRaises(ValueError):
            projection([1, 0], [0, 0])

    def test_operations_reject_empty_vectors_and_mismatched_dimensions(self):
        empty_operations = [
            lambda: add([], []),
            lambda: dot([], []),
            lambda: norm([]),
            lambda: cosine([], []),
            lambda: projection([], []),
            lambda: transform([[1]], []),
            lambda: transform([[1, 2], [3, 4]], []),
        ]
        for operation in empty_operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    operation()

        mismatched_operations = [
            lambda: add([1], [1, 2]),
            lambda: dot([1], [1, 2]),
            lambda: cosine([1], [1, 2]),
            lambda: projection([1], [1, 2]),
            lambda: transform([[1, 2], [3, 4]], [1, 2, 3]),
        ]
        for operation in mismatched_operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    operation()

    def test_operations_reject_non_finite_vector_and_matrix_values(self):
        for value in (math.nan, math.inf, -math.inf):
            operations = [
                lambda: add([value], [1]),
                lambda: add([1], [value]),
                lambda: dot([value], [1]),
                lambda: dot([1], [value]),
                lambda: norm([value]),
                lambda: cosine([value], [1]),
                lambda: cosine([1], [value]),
                lambda: projection([value], [1]),
                lambda: projection([1], [value]),
                lambda: transform([[value]], [1]),
                lambda: transform([[1]], [value]),
            ]
            for operation in operations:
                with self.subTest(value=value, operation=operation):
                    with self.assertRaises(ValueError):
                        operation()

    def test_transform_rejects_non_rectangular_matrices(self):
        with self.assertRaises(ValueError):
            transform([], [1])
        with self.assertRaises(ValueError):
            transform([[1, 2], [3]], [1, 2])
        with self.assertRaises(ValueError):
            transform([[1], [2, 3]], [1])


if __name__ == "__main__":
    unittest.main()
