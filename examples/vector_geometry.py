import math
from collections.abc import Sequence


def _validate_vector(vector: Sequence[float], name: str) -> list[float]:
    values = list(vector)
    if not values:
        raise ValueError(f"{name} must be non-empty")
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite numbers")
    return values


def _validate_pair(a: Sequence[float], b: Sequence[float]) -> tuple[list[float], list[float]]:
    left = _validate_vector(a, "a")
    right = _validate_vector(b, "b")
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    return left, right


def _validate_matrix(
    matrix: Sequence[Sequence[float]], vector_length: int
) -> list[list[float]]:
    rows = [list(row) for row in matrix]
    if not rows:
        raise ValueError("matrix must be non-empty")

    column_count = len(rows[0])
    for row in rows:
        _validate_vector(row, "matrix row")
        if len(row) != column_count:
            raise ValueError("matrix must be rectangular")
    if column_count != vector_length:
        raise ValueError("matrix columns must match the vector dimension")
    return rows


def _is_zero_vector(vector: Sequence[float]) -> bool:
    for value in vector:
        if value != 0:
            return False
    return True


def add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    left, right = _validate_pair(a, b)
    result = []
    for index in range(len(left)):
        result.append(left[index] + right[index])
    return result


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    left, right = _validate_pair(a, b)
    result = 0
    for index in range(len(left)):
        result += left[index] * right[index]
    return result


def _normalize_by_max(vector: Sequence[float]) -> tuple[float, list[float]]:
    scale = max(abs(value) for value in vector)
    if scale == 0:
        return scale, [0.0 for _ in vector]
    return scale, [value / scale for value in vector]


def norm(a: Sequence[float]) -> float:
    values = _validate_vector(a, "a")
    result = 0.0
    for value in values:
        result = math.hypot(result, value)
    return result


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    left, right = _validate_pair(a, b)
    if _is_zero_vector(left) or _is_zero_vector(right):
        raise ValueError("cosine is undefined for zero vectors")
    _, normalized_left = _normalize_by_max(left)
    _, normalized_right = _normalize_by_max(right)
    return dot(normalized_left, normalized_right) / (
        norm(normalized_left) * norm(normalized_right)
    )


def projection(a: Sequence[float], b: Sequence[float]) -> list[float]:
    left, right = _validate_pair(a, b)
    if _is_zero_vector(right):
        raise ValueError("projection direction must be non-zero")

    left_scale, normalized_left = _normalize_by_max(left)
    _, normalized_right = _normalize_by_max(right)
    scale = dot(normalized_left, normalized_right) / dot(
        normalized_right, normalized_right
    )
    result = []
    for value in normalized_right:
        projected = scale * value * left_scale
        if not math.isfinite(projected):
            raise ValueError("projection result is not representable")
        result.append(projected)
    return result


def transform(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    values = _validate_vector(vector, "vector")
    rows = _validate_matrix(matrix, len(values))

    result = []
    for row in rows:
        value = 0
        for index in range(len(values)):
            value += row[index] * values[index]
        result.append(value)
    return result


def main() -> None:
    projected = projection([2, 2], [1, 0])
    transformed = transform([[0, -1], [1, 0]], [1, 0])
    print(f"projection([2, 2], [1, 0]) = {projected}")
    print(f"transform([[0, -1], [1, 0]], [1, 0]) = {transformed}")


if __name__ == "__main__":
    main()
