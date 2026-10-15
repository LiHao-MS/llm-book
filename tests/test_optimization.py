import math
import unittest

from examples.optimization import (
    adam_step,
    adamw_step,
    clip_global_norm,
    cosine_learning_rate,
    momentum_step,
    quadratic,
    quadratic_gradient,
    sgd_step,
)


class OptimizationTest(unittest.TestCase):
    def assertVectorClose(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertTrue(math.isfinite(actual_value))
            self.assertAlmostEqual(actual_value, expected_value, delta=1e-8)

    @staticmethod
    def adam_state():
        return {
            "m": [0, 0],
            "v": [0, 0],
            "t": 0,
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
        }

    def test_quadratic_computes_objective_and_gradient_without_mutation(self):
        point = [3, 1]
        center = [1, -1]
        self.assertEqual(quadratic(point, center), 4)
        self.assertEqual(quadratic_gradient(point, center), [2, 2])
        self.assertEqual(point, [3, 1])
        self.assertEqual(center, [1, -1])

    def test_sgd_and_momentum_compute_fixed_updates_without_mutation(self):
        point = [1, 1]
        gradient = [2, -1]
        self.assertEqual(sgd_step(point, gradient, 0.1), [0.8, 1.1])

        state = {"velocity": [1, 1], "beta": 0.5}
        result = momentum_step(point, gradient, 0.1, state)
        self.assertEqual(result["point"], [0.75, 1.05])
        self.assertEqual(result["velocity"], [2.5, -0.5])
        self.assertEqual(result["beta"], 0.5)
        self.assertIsNot(result["point"], point)
        self.assertIsNot(result["velocity"], state["velocity"])
        self.assertEqual(point, [1, 1])
        self.assertEqual(gradient, [2, -1])
        self.assertEqual(state, {"velocity": [1, 1], "beta": 0.5})

    def test_adam_updates_moments_and_time_with_bias_correction(self):
        point = [1, 1]
        gradient = [2, -1]
        state = self.adam_state()
        result = adam_step(point, gradient, 0.1, state)

        self.assertVectorClose(result["point"], [0.9, 1.1])
        self.assertVectorClose(result["m"], [0.2, -0.1])
        self.assertVectorClose(result["v"], [0.004, 0.001])
        self.assertEqual(result["t"], 1)
        self.assertEqual(result["beta1"], state["beta1"])
        self.assertEqual(result["beta2"], state["beta2"])
        self.assertEqual(result["epsilon"], state["epsilon"])
        self.assertIsNot(result["point"], point)
        self.assertIsNot(result["m"], state["m"])
        self.assertIsNot(result["v"], state["v"])
        self.assertEqual(point, [1, 1])
        self.assertEqual(gradient, [2, -1])
        self.assertEqual(state, self.adam_state())

    def test_adam_handles_large_finite_gradients_without_overflow(self):
        point = [1, 1]
        gradient = [1.5e154, 0]
        state = self.adam_state()
        result = adam_step(point, gradient, 0.1, state)

        self.assertVectorClose(result["point"], [0.9, 1])
        self.assertVectorClose([result["v"][0] / 2.25e305], [1])
        self.assertEqual(point, [1, 1])
        self.assertEqual(gradient, [1.5e154, 0])
        self.assertEqual(state, self.adam_state())

    def test_adam_uses_prior_moments_and_sqrt_vhat_plus_epsilon(self):
        point = [1, -1]
        gradient = [0, 0]
        state = {
            "m": [0.00015, 0.0003],
            "v": [1.5e-8, 6e-8],
            "t": 1,
            "beta1": 0.5,
            "beta2": 0.5,
            "epsilon": 1e-8,
        }
        result = adam_step(point, gradient, 0.1, state)

        self.assertVectorClose(
            result["point"], [0.9000099990001, -1.0999950002499875]
        )
        self.assertEqual([round(value, 1) for value in result["point"]], [0.9, -1.1])
        self.assertEqual(result["m"], [0.000075, 0.00015])
        self.assertEqual(result["v"], [7.5e-9, 3e-8])
        self.assertEqual(result["t"], 2)
        self.assertEqual(point, [1, -1])
        self.assertEqual(gradient, [0, 0])
        self.assertEqual(
            state,
            {
                "m": [0.00015, 0.0003],
                "v": [1.5e-8, 6e-8],
                "t": 1,
                "beta1": 0.5,
                "beta2": 0.5,
                "epsilon": 1e-8,
            },
        )

    def test_adamw_applies_decoupled_decay_after_adam(self):
        point = [1, 1]
        gradient = [2, -1]
        state = self.adam_state()
        result = adamw_step(point, gradient, 0.1, state, 0.01)

        self.assertVectorClose(result["point"], [0.891, 1.089])
        self.assertVectorClose(result["m"], [0.2, -0.1])
        self.assertVectorClose(result["v"], [0.004, 0.001])
        self.assertEqual(result["t"], 1)
        self.assertEqual(point, [1, 1])
        self.assertEqual(gradient, [2, -1])
        self.assertEqual(state, self.adam_state())

    def test_momentum_rejects_incomplete_state_with_required_field_error(self):
        with self.assertRaisesRegex(ValueError, r"^state\.beta is required$"):
            momentum_step([1, 1], [1, 0], 0.1, {"velocity": [0, 0]})

    def test_adam_rejects_incomplete_state_with_required_field_error(self):
        state = {
            "m": [0, 0],
            "v": [0, 0],
            "t": 0,
            "beta1": 0.9,
            "epsilon": 1e-8,
        }
        with self.assertRaisesRegex(ValueError, r"^state\.beta2 is required$"):
            adam_step([1, 1], [1, 0], 0.1, state)

    def test_zero_gradient_and_zero_learning_rate_are_legal(self):
        point = [1, 1]
        zero = [0, 0]
        self.assertEqual(sgd_step(point, zero, 0.1), point)
        self.assertEqual(sgd_step(point, [2, -1], 0), point)

        momentum = momentum_step(point, zero, 0.1, {"velocity": zero, "beta": 0.5})
        self.assertEqual(momentum["point"], point)
        self.assertEqual(momentum["velocity"], zero)

        adam = adam_step(point, zero, 0.1, self.adam_state())
        self.assertEqual(adam["point"], point)
        self.assertEqual(adam["m"], zero)
        self.assertEqual(adam["v"], zero)
        self.assertEqual(adam_step(point, [2, -1], 0, self.adam_state())["point"], point)
        self.assertEqual(
            adamw_step(point, [2, -1], 0, self.adam_state(), 0)["point"], point
        )

    def test_clip_global_norm_scales_only_when_needed(self):
        gradient = [3, 4]
        clipped = clip_global_norm(gradient, 1)
        self.assertEqual(clipped, [0.6, 0.8])
        self.assertIsNot(clipped, gradient)
        self.assertEqual(gradient, [3, 4])
        self.assertEqual(clip_global_norm(gradient, 5), gradient)
        self.assertEqual(clip_global_norm(gradient, 0), [0, 0])
        self.assertEqual(clip_global_norm([0, 0], 1), [0, 0])
        self.assertEqual(clip_global_norm([0, 0], 0), [0, 0])

    def test_clip_global_norm_remains_finite_at_extreme_scales(self):
        self.assertVectorClose(clip_global_norm([3e200, 4e200], 1), [0.6, 0.8])
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        self.assertVectorClose(
            clip_global_norm([maximum, maximum], 1),
            [math.sqrt(0.5), math.sqrt(0.5)],
        )

    def test_cosine_learning_rate_follows_endpoints_and_midpoint(self):
        self.assertAlmostEqual(cosine_learning_rate(0, 10, 0, 1), 1)
        self.assertAlmostEqual(cosine_learning_rate(5, 10, 0, 1), 0.5)
        self.assertAlmostEqual(cosine_learning_rate(10, 10, 0, 1), 0)
        self.assertAlmostEqual(cosine_learning_rate(0, 10, 0.2, 1), 1)
        self.assertAlmostEqual(cosine_learning_rate(10, 10, 0.2, 1), 0.2)

    def test_vector_apis_reject_invalid_dimensions_and_non_finite_values(self):
        operations = [
            lambda: quadratic([], []),
            lambda: quadratic([1], [0]),
            lambda: quadratic([1, 2, 3], [0, 0, 0]),
            lambda: quadratic_gradient([1, 2], [0]),
            lambda: sgd_step([1], [1], 0.1),
            lambda: sgd_step([1, 2], [1], 0.1),
            lambda: momentum_step([1, 2], [1, 2], 0.1, {"velocity": [0], "beta": 0.5}),
            lambda: adam_step([1, 2], [1, 2], 0.1, {**self.adam_state(), "m": [0]}),
            lambda: adam_step([1, 2], [1, 2], 0.1, {**self.adam_state(), "v": [0]}),
            lambda: clip_global_norm([], 1),
        ]
        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    operation()

        for value in (math.nan, math.inf, -math.inf):
            non_finite_operations = [
                lambda: quadratic([value, 0], [0, 0]),
                lambda: quadratic([1, 0], [0, value]),
                lambda: quadratic_gradient([value, 0], [0, 0]),
                lambda: sgd_step([1, 1], [value, 0], 0.1),
                lambda: sgd_step([1, 1], [1, 0], value),
                lambda: momentum_step(
                    [1, 1], [1, 0], 0.1, {"velocity": [value, 0], "beta": 0.5}
                ),
                lambda: adam_step(
                    [1, 1], [1, 0], 0.1, {**self.adam_state(), "m": [value, 0]}
                ),
                lambda: clip_global_norm([value, 0], 1),
                lambda: cosine_learning_rate(value, 10, 0, 1),
                lambda: cosine_learning_rate(0, value, 0, 1),
                lambda: cosine_learning_rate(0, 10, value, 1),
                lambda: cosine_learning_rate(0, 10, 0, value),
            ]
            for operation in non_finite_operations:
                with self.subTest(value=value, operation=operation):
                    with self.assertRaises(ValueError):
                        operation()

    def test_hyperparameters_and_schedule_bounds_are_validated(self):
        for learning_rate in (-0.1, math.nan, math.inf, -math.inf):
            with self.subTest(learning_rate=learning_rate):
                with self.assertRaises(ValueError):
                    sgd_step([1, 1], [1, 1], learning_rate)
                with self.assertRaises(ValueError):
                    momentum_step(
                        [1, 1], [1, 1], learning_rate, {"velocity": [0, 0], "beta": 0.5}
                    )
                with self.assertRaises(ValueError):
                    adam_step([1, 1], [1, 1], learning_rate, self.adam_state())

        for beta in (-0.1, 1, 1.1, math.nan, math.inf, -math.inf):
            with self.subTest(beta=beta):
                with self.assertRaises(ValueError):
                    momentum_step([1, 1], [1, 1], 0.1, {"velocity": [0, 0], "beta": beta})
                with self.assertRaises(ValueError):
                    adam_step([1, 1], [1, 1], 0.1, {**self.adam_state(), "beta1": beta})
                with self.assertRaises(ValueError):
                    adam_step([1, 1], [1, 1], 0.1, {**self.adam_state(), "beta2": beta})

        for epsilon in (0, -1, math.nan, math.inf, -math.inf):
            with self.subTest(epsilon=epsilon):
                with self.assertRaises(ValueError):
                    adam_step(
                        [1, 1], [1, 1], 0.1, {**self.adam_state(), "epsilon": epsilon}
                    )
        for t in (-1, 0.5, math.nan, math.inf, -math.inf):
            with self.subTest(t=t):
                with self.assertRaises(ValueError):
                    adam_step([1, 1], [1, 1], 0.1, {**self.adam_state(), "t": t})
        for weight_decay in (-0.1, math.nan, math.inf, -math.inf):
            with self.subTest(weight_decay=weight_decay):
                with self.assertRaises(ValueError):
                    adamw_step([1, 1], [1, 1], 0.1, self.adam_state(), weight_decay)
        for threshold in (-0.1, math.nan, math.inf, -math.inf):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    clip_global_norm([1, 1], threshold)

        for total_steps in (0, -1, math.nan, math.inf, -math.inf):
            with self.subTest(total_steps=total_steps):
                with self.assertRaises(ValueError):
                    cosine_learning_rate(0, total_steps, 0.2, 1)
        invalid_schedules = [
            (-1, 10, 0, 1),
            (11, 10, 0, 1),
            (0, 10, -0.1, 1),
            (0, 10, 0, -1),
            (0, 10, 1.1, 1),
        ]
        for arguments in invalid_schedules:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    cosine_learning_rate(*arguments)


if __name__ == "__main__":
    unittest.main()
