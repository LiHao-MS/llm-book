import math
from collections.abc import Sequence


DIMENSION = 2


def _validate_vector(
    vector: Sequence[float], name: str, dimension: int | None = None
) -> list[float]:
    try:
        values = list(vector)
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence") from error
    if not values or (dimension is not None and len(values) != dimension):
        expected = "non-empty" if dimension is None else f"of length {dimension}"
        raise ValueError(f"{name} must be {expected}")
    for value in values:
        try:
            finite = math.isfinite(value)
        except TypeError as error:
            raise ValueError(f"{name} must contain only finite numbers") from error
        if not finite:
            raise ValueError(f"{name} must contain only finite numbers")
    return values


def _validate_pair(
    a: Sequence[float], b: Sequence[float], dimension: int = DIMENSION
) -> tuple[list[float], list[float]]:
    return _validate_vector(a, "a", dimension), _validate_vector(b, "b", dimension)


def _validate_finite(value: float, name: str) -> float:
    try:
        finite = math.isfinite(value)
    except TypeError as error:
        raise ValueError(f"{name} must be finite") from error
    if not finite:
        raise ValueError(f"{name} must be finite")
    return value


def _validate_non_negative(value: float, name: str) -> float:
    _validate_finite(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _validate_positive(value: float, name: str) -> float:
    _validate_finite(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_beta(value: float, name: str) -> float:
    _validate_finite(value, name)
    if value < 0 or value >= 1:
        raise ValueError(f"{name} must be in [0, 1)")
    return value


def _validate_time(value: int) -> int:
    _validate_finite(value, "t")
    if value < 0 or not float(value).is_integer():
        raise ValueError("t must be a non-negative integer")
    return int(value)


def _validate_result(result: Sequence[float]) -> list[float]:
    return _validate_vector(result, "result")


def quadratic(point: Sequence[float], center: Sequence[float]) -> float:
    point_values, center_values = _validate_pair(point, center)
    x = point_values[0] - center_values[0]
    y = point_values[1] - center_values[1]
    return _validate_finite(0.5 * (x * x + y * y), "quadratic result")


def quadratic_gradient(point: Sequence[float], center: Sequence[float]) -> list[float]:
    point_values, center_values = _validate_pair(point, center)
    return _validate_result(
        [point_values[0] - center_values[0], point_values[1] - center_values[1]]
    )


def sgd_step(
    point: Sequence[float], gradient: Sequence[float], learning_rate: float
) -> list[float]:
    point_values, gradient_values = _validate_pair(point, gradient)
    _validate_non_negative(learning_rate, "learning_rate")
    return _validate_result(
        [
            value - learning_rate * gradient_values[index]
            for index, value in enumerate(point_values)
        ]
    )


def _required_state_value(state: dict, name: str):
    try:
        return state[name]
    except KeyError as error:
        raise ValueError(f"state.{name} is required") from error


def momentum_step(
    point: Sequence[float],
    gradient: Sequence[float],
    learning_rate: float,
    state: dict,
) -> dict:
    point_values, gradient_values = _validate_pair(point, gradient)
    _validate_non_negative(learning_rate, "learning_rate")
    if not isinstance(state, dict):
        raise ValueError("state must be a dict")
    previous_velocity = _validate_vector(
        _required_state_value(state, "velocity"), "velocity", DIMENSION
    )
    beta = _validate_beta(_required_state_value(state, "beta"), "beta")

    velocity = _validate_result(
        [
            beta * value + gradient_values[index]
            for index, value in enumerate(previous_velocity)
        ]
    )
    next_point = _validate_result(
        [
            value - learning_rate * velocity[index]
            for index, value in enumerate(point_values)
        ]
    )
    return {"point": next_point, "velocity": velocity, "beta": beta}


def adam_step(
    point: Sequence[float],
    gradient: Sequence[float],
    learning_rate: float,
    state: dict,
) -> dict:
    point_values, gradient_values = _validate_pair(point, gradient)
    _validate_non_negative(learning_rate, "learning_rate")
    if not isinstance(state, dict):
        raise ValueError("state must be a dict")
    previous_m = _validate_vector(_required_state_value(state, "m"), "m", DIMENSION)
    previous_v = _validate_vector(_required_state_value(state, "v"), "v", DIMENSION)
    beta1 = _validate_beta(_required_state_value(state, "beta1"), "beta1")
    beta2 = _validate_beta(_required_state_value(state, "beta2"), "beta2")
    epsilon = _validate_positive(_required_state_value(state, "epsilon"), "epsilon")
    previous_time = _validate_time(_required_state_value(state, "t"))

    m = _validate_result(
        [
            beta1 * value + (1 - beta1) * gradient_values[index]
            for index, value in enumerate(previous_m)
        ]
    )
    v = _validate_result(
        [
            beta2 * value
            + (1 - beta2) * gradient_values[index] * gradient_values[index]
            for index, value in enumerate(previous_v)
        ]
    )
    t = previous_time + 1
    correction1 = 1 - beta1**t
    correction2 = 1 - beta2**t
    next_point = _validate_result(
        [
            value
            - learning_rate
            * (m[index] / correction1)
            / (math.sqrt(v[index]) / math.sqrt(correction2) + epsilon)
            for index, value in enumerate(point_values)
        ]
    )
    return {
        "point": next_point,
        "m": m,
        "v": v,
        "t": t,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
    }


def adamw_step(
    point: Sequence[float],
    gradient: Sequence[float],
    learning_rate: float,
    state: dict,
    weight_decay: float,
) -> dict:
    _validate_non_negative(weight_decay, "weight_decay")
    next_state = adam_step(point, gradient, learning_rate, state)
    decayed_point = _validate_result(
        [value * (1 - weight_decay) for value in next_state["point"]]
    )
    return {**next_state, "point": decayed_point}


def clip_global_norm(gradient: Sequence[float], threshold: float) -> list[float]:
    values = _validate_vector(gradient, "gradient")
    _validate_non_negative(threshold, "threshold")

    scale = max(abs(value) for value in values)
    if scale == 0:
        return [0 for _ in values]

    normalized_norm = 0.0
    for value in values:
        normalized_norm = math.hypot(normalized_norm, value / scale)
    factor = min(1, (threshold / scale) / normalized_norm)
    if factor == 1:
        return values.copy()

    # 先归一化再缩放，避免范数或其倒数在极端尺度下溢出。
    return _validate_result(
        [(value / scale / normalized_norm) * threshold for value in values]
    )


def cosine_learning_rate(
    step: float, total_steps: float, min_rate: float, max_rate: float
) -> float:
    _validate_finite(step, "step")
    _validate_positive(total_steps, "total_steps")
    _validate_non_negative(min_rate, "min_rate")
    _validate_non_negative(max_rate, "max_rate")
    if step < 0 or step > total_steps:
        raise ValueError("step must be in [0, total_steps]")
    if min_rate > max_rate:
        raise ValueError("min_rate must not exceed max_rate")

    return min_rate + 0.5 * (max_rate - min_rate) * (
        1 + math.cos(math.pi * step / total_steps)
    )
