import math
import unittest

from examples.softmax_cross_entropy import cross_entropy, perplexity, softmax


class ProbabilityTest(unittest.TestCase):
    def assertClose(self, actual, expected):
        self.assertTrue(math.isfinite(actual))
        self.assertAlmostEqual(actual, expected, delta=1e-9)

    def test_softmax_computes_normalized_probabilities_stably(self):
        probabilities = softmax([1, 2, 3])
        self.assertClose(probabilities[2], 0.6652409558)
        self.assertClose(sum(probabilities), 1)
        self.assertClose(softmax([1000, 1001])[0], 0.2689414214)

    def test_softmax_allows_negative_logits(self):
        self.assertClose(softmax([-1, -2, -3])[0], 0.6652409558)

    def test_softmax_rejects_empty_and_non_finite_logits(self):
        for logits in ([], [0, math.nan], [0, math.inf], [0, -math.inf]):
            with self.subTest(logits=logits):
                with self.assertRaises(ValueError):
                    softmax(logits)

    def test_cross_entropy_and_perplexity_use_natural_logarithms(self):
        self.assertClose(cross_entropy(0.5), math.log(2))
        self.assertClose(cross_entropy(1), 0)
        self.assertClose(perplexity(0.5), 2)
        self.assertClose(perplexity(0.25), 4)
        self.assertClose(perplexity(1), 1)
        with self.assertRaises(ValueError):
            cross_entropy(0)

    def test_cross_entropy_and_perplexity_reject_invalid_probabilities(self):
        for function in (cross_entropy, perplexity):
            for probability in (0, -0.1, 1.1, math.nan, math.inf, -math.inf):
                with self.subTest(function=function.__name__, probability=probability):
                    with self.assertRaises(ValueError):
                        function(probability)


if __name__ == "__main__":
    unittest.main()
