import math
from collections.abc import Sequence

from examples.softmax_cross_entropy import softmax


def _validate_vector(vector: Sequence[float], name: str) -> list[float]:
    try:
        values = list(vector)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence") from error
    if not values:
        raise ValueError(f"{name} must be non-empty")
    for value in values:
        try:
            finite = math.isfinite(value)
        except TypeError as error:
            raise ValueError(f"{name} must contain only finite numbers") from error
        if not finite:
            raise ValueError(f"{name} must contain only finite numbers")
    return values


def _validate_matrix(
    matrix: Sequence[Sequence[float]], name: str
) -> list[list[float]]:
    try:
        raw_rows = list(matrix)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of rows") from error
    if not raw_rows:
        raise ValueError(f"{name} must be a non-empty matrix")

    rows = [_validate_vector(row, f"{name} row") for row in raw_rows]
    column_count = len(rows[0])
    if any(len(row) != column_count for row in rows):
        raise ValueError(f"{name} must be rectangular")
    return rows


def _validate_attention_inputs(q, k, v):
    queries = _validate_matrix(q, "q")
    keys = _validate_matrix(k, "k")
    values = _validate_matrix(v, "v")
    if len(queries[0]) != len(keys[0]):
        raise ValueError("q and k must have the same feature dimension")
    if len(keys) != len(values):
        raise ValueError("k and v must have the same sequence length")
    return queries, keys, values


def _validate_scale(scale: float) -> float:
    try:
        finite = math.isfinite(scale)
    except TypeError as error:
        raise ValueError("scale must be finite and positive") from error
    if not finite or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return scale


def scaled_dot_product_attention(
    q: Sequence[Sequence[float]],
    k: Sequence[Sequence[float]],
    v: Sequence[Sequence[float]],
    scale: float,
) -> dict:
    queries, keys, values = _validate_attention_inputs(q, k, v)
    scale = _validate_scale(scale)

    scores = []
    for query in queries:
        score_row = []
        for key in keys:
            score = 0.0
            for index in range(len(query)):
                score += query[index] * key[index]
            score *= scale
            if not math.isfinite(score):
                raise ValueError("attention scores must be finite")
            score_row.append(score)
        scores.append(score_row)

    weights = [softmax(row) for row in scores]
    output = []
    for weight_row in weights:
        output_row = [0.0 for _ in range(len(values[0]))]
        for value_index, value_row in enumerate(values):
            for dimension in range(len(output_row)):
                output_row[dimension] += weight_row[value_index] * value_row[dimension]
        if not all(math.isfinite(value) for value in output_row):
            raise ValueError("attention output must be finite")
        output.append(output_row)

    return {"scores": scores, "weights": weights, "output": output}


def causal_mask(scores: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = _validate_matrix(scores, "scores")
    if len(rows) != len(rows[0]):
        raise ValueError("scores must be a square matrix")
    return [
        [
            -math.inf if column_index > row_index else value
            for column_index, value in enumerate(row)
        ]
        for row_index, row in enumerate(rows)
    ]


def masked_softmax(logits: Sequence[float]) -> list[float]:
    values = list(logits)
    finite_logits = [value for value in values if value != -math.inf]
    finite_weights = iter(softmax(finite_logits))
    return [
        0.0 if value == -math.inf else next(finite_weights) for value in values
    ]


def multi_head_attention(
    q: Sequence[Sequence[float]],
    k: Sequence[Sequence[float]],
    v: Sequence[Sequence[float]],
    head_count: int,
) -> dict:
    queries, keys, values = _validate_attention_inputs(q, k, v)
    if (
        isinstance(head_count, bool)
        or not isinstance(head_count, int)
        or head_count <= 0
    ):
        raise ValueError("head_count must be a positive integer")

    query_dimension = len(queries[0])
    value_dimension = len(values[0])
    if query_dimension % head_count != 0 or value_dimension % head_count != 0:
        raise ValueError("head_count must divide q, k, and v feature dimensions")

    head_dimension = query_dimension // head_count
    value_head_dimension = value_dimension // head_count
    scale = 1 / math.sqrt(head_dimension)
    heads = []
    for head in range(head_count):
        query_start = head * head_dimension
        value_start = head * value_head_dimension
        head_queries = [
            row[query_start : query_start + head_dimension] for row in queries
        ]
        head_keys = [row[query_start : query_start + head_dimension] for row in keys]
        head_values = [
            row[value_start : value_start + value_head_dimension] for row in values
        ]
        heads.append(
            scaled_dot_product_attention(head_queries, head_keys, head_values, scale)
        )

    output = [
        [
            value
            for head in heads
            for value in head["output"][row_index]
        ]
        for row_index in range(len(queries))
    ]
    return {"heads": heads, "output": output}


def rotate_pair(vector: Sequence[float], angle: float) -> list[float]:
    values = _validate_vector(vector, "vector")
    if len(values) % 2 != 0:
        raise ValueError("vector must have an even dimension")
    try:
        finite_angle = math.isfinite(angle)
    except TypeError as error:
        raise ValueError("angle must be finite") from error
    if not finite_angle:
        raise ValueError("angle must be finite")

    cosine = math.cos(angle)
    sine = math.sin(angle)
    result = []
    for index in range(0, len(values), 2):
        first = values[index]
        second = values[index + 1]
        rotated_first = first * cosine - second * sine
        rotated_second = first * sine + second * cosine
        if not math.isfinite(rotated_first) or not math.isfinite(rotated_second):
            raise ValueError("rotated vector must be finite")
        result.extend([rotated_first, rotated_second])
    return result
