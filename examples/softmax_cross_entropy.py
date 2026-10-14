import math
from collections.abc import Sequence


def softmax(logits: Sequence[float]) -> list[float]:
    values = list(logits)
    if not values:
        raise ValueError("logits must be non-empty")
    if not all(math.isfinite(logit) for logit in values):
        raise ValueError("logits must contain only finite numbers")

    maximum = max(values)
    exponentials = [math.exp(logit - maximum) for logit in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def cross_entropy(probability: float) -> float:
    if not math.isfinite(probability) or probability <= 0 or probability > 1:
        raise ValueError("probability must be finite and in (0, 1]")
    return -math.log(probability)


def perplexity(probability: float) -> float:
    return math.exp(cross_entropy(probability))


def main() -> None:
    logits = [1.0, 2.0, 3.0]
    probabilities = softmax(logits)
    target_probability = probabilities[-1]
    print(f"softmax({logits}) = {probabilities}")
    print(f"cross_entropy(target) = {cross_entropy(target_probability)}")
    print(f"perplexity(target) = {perplexity(target_probability)}")


if __name__ == "__main__":
    main()
