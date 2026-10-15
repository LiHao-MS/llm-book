import math
from collections.abc import Sequence


def _validate_vector(vector: Sequence[float], name: str) -> list[float]:
    # 统一检查向量：非空，并且每个分量都是有限数值。
    values = list(vector)
    if not values:
        raise ValueError(f"{name} must be non-empty")
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite numbers")
    return values


def _validate_pair(a: Sequence[float], b: Sequence[float]) -> tuple[list[float], list[float]]:
    # 逐元素运算前，先确认两个向量具有相同维度。
    left = _validate_vector(a, "a")
    right = _validate_vector(b, "b")
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    return left, right


def _validate_matrix(
    matrix: Sequence[Sequence[float]], vector_length: int
) -> list[list[float]]:
    # 矩阵按行主序存储；只有列数等于向量维度时才能相乘。
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
    # 余弦和投影都以方向向量的范数为分母，因此先排除零方向。
    for value in vector:
        if value != 0:
            return False
    return True


def add(a: Sequence[float], b: Sequence[float]) -> list[float]:
    # 向量加法：c_i = a_i + b_i。
    left, right = _validate_pair(a, b)
    result = []
    for index in range(len(left)):
        result.append(left[index] + right[index])
    return result


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    # 内积：a·b = Σ_i a_i b_i。
    left, right = _validate_pair(a, b)
    result = 0
    for index in range(len(left)):
        result += left[index] * right[index]
    return result


def _normalize_by_max(vector: Sequence[float]) -> tuple[float, list[float]]:
    # 用最大绝对分量归一化，避免极大或极小向量在后续乘积中溢出或下溢。
    scale = max(abs(value) for value in vector)
    if scale == 0:
        return scale, [0.0 for _ in vector]
    return scale, [value / scale for value in vector]


def norm(a: Sequence[float]) -> float:
    # 用 math.hypot 稳定计算欧几里得范数 ||a||。
    values = _validate_vector(a, "a")
    result = 0.0
    for value in values:
        result = math.hypot(result, value)
    return result


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    # 余弦相似度 = (a/||a||)·(b/||b||)，与两个向量的尺度无关。
    left, right = _validate_pair(a, b)
    if _is_zero_vector(left) or _is_zero_vector(right):
        raise ValueError("cosine is undefined for zero vectors")
    _, normalized_left = _normalize_by_max(left)
    _, normalized_right = _normalize_by_max(right)
    return dot(normalized_left, normalized_right) / (
        norm(normalized_left) * norm(normalized_right)
    )


def projection(a: Sequence[float], b: Sequence[float]) -> list[float]:
    # 投影到 b 上：proj_b(a) = ((a·b)/(b·b)) b。先归一化以避免尺度造成的数值问题。
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
    # 矩阵—向量乘法：结果第 i 个分量 = 矩阵第 i 行与向量的内积。
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
    # 两个最小可核对示例：投影到 x 轴，以及把 (1, 0) 旋转 90°。
    projected = projection([2, 2], [1, 0])
    transformed = transform([[0, -1], [1, 0]], [1, 0])
    print(f"projection([2, 2], [1, 0]) = {projected}")
    print(f"transform([[0, -1], [1, 0]], [1, 0]) = {transformed}")


if __name__ == "__main__":
    main()
