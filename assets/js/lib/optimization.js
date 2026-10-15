const DIMENSION = 2;

function validateVector(vector, name, dimension = null) {
  if (
    !Array.isArray(vector)
    || vector.length === 0
    || (dimension !== null && vector.length !== dimension)
  ) {
    const expected = dimension === null ? 'a non-empty array' : `an array of length ${dimension}`;
    throw new RangeError(`${name} must be ${expected}`);
  }
  for (const value of vector) {
    if (!Number.isFinite(value)) {
      throw new RangeError(`${name} must contain only finite numbers`);
    }
  }
  return vector;
}

function validatePair(a, b, dimension = DIMENSION) {
  validateVector(a, 'a', dimension);
  validateVector(b, 'b', dimension);
}

function validateFinite(value, name) {
  if (!Number.isFinite(value)) {
    throw new RangeError(`${name} must be finite`);
  }
  return value;
}

function validateNonNegative(value, name) {
  validateFinite(value, name);
  if (value < 0) {
    throw new RangeError(`${name} must be non-negative`);
  }
  return value;
}

function validatePositive(value, name) {
  validateFinite(value, name);
  if (value <= 0) {
    throw new RangeError(`${name} must be positive`);
  }
  return value;
}

function validateBeta(value, name) {
  validateFinite(value, name);
  if (value < 0 || value >= 1) {
    throw new RangeError(`${name} must be in [0, 1)`);
  }
  return value;
}

function validateTime(value) {
  if (!Number.isSafeInteger(value) || value < 0 || value >= Number.MAX_SAFE_INTEGER) {
    throw new RangeError('t must be a non-negative safe integer below MAX_SAFE_INTEGER');
  }
  return value;
}

function validateResult(result) {
  return validateVector(result, 'result');
}

export function quadratic(point, center) {
  validatePair(point, center);
  const x = point[0] - center[0];
  const y = point[1] - center[1];
  return validateFinite(0.5 * (x * x + y * y), 'quadratic result');
}

export function quadraticGradient(point, center) {
  validatePair(point, center);
  return validateResult([point[0] - center[0], point[1] - center[1]]);
}

export function sgdStep(point, gradient, learningRate) {
  validatePair(point, gradient);
  validateNonNegative(learningRate, 'learningRate');
  return validateResult(point.map((value, index) => value - learningRate * gradient[index]));
}

function requiredStateField(state, name) {
  if (!Object.hasOwn(state, name) || state[name] === undefined) {
    throw new RangeError(`state.${name} is required`);
  }
  return state[name];
}

export function momentumStep(point, gradient, learningRate, state) {
  validatePair(point, gradient);
  validateNonNegative(learningRate, 'learningRate');
  if (state === null || typeof state !== 'object') {
    throw new RangeError('state must be an object');
  }
  const previousVelocity = validateVector(
    requiredStateField(state, 'velocity'),
    'velocity',
    DIMENSION,
  );
  const beta = validateBeta(requiredStateField(state, 'beta'), 'beta');

  const velocity = validateResult(previousVelocity.map(
    (value, index) => beta * value + gradient[index],
  ));
  const nextPoint = validateResult(point.map(
    (value, index) => value - learningRate * velocity[index],
  ));
  return { point: nextPoint, velocity, beta };
}

export function adamStep(point, gradient, learningRate, state) {
  validatePair(point, gradient);
  validateNonNegative(learningRate, 'learningRate');
  if (state === null || typeof state !== 'object') {
    throw new RangeError('state must be an object');
  }
  const previousM = validateVector(requiredStateField(state, 'm'), 'm', DIMENSION);
  const previousV = validateVector(requiredStateField(state, 'v'), 'v', DIMENSION);
  const beta1 = validateBeta(requiredStateField(state, 'beta1'), 'beta1');
  const beta2 = validateBeta(requiredStateField(state, 'beta2'), 'beta2');
  const epsilon = validatePositive(requiredStateField(state, 'epsilon'), 'epsilon');
  const previousTime = validateTime(requiredStateField(state, 't'));

  const m = validateResult(previousM.map(
    (value, index) => beta1 * value + (1 - beta1) * gradient[index],
  ));
  const v = validateResult(previousV.map(
    (value, index) => beta2 * value + (1 - beta2) * gradient[index] * gradient[index],
  ));
  const t = previousTime + 1;
  const correction1 = 1 - beta1 ** t;
  const correction2 = 1 - beta2 ** t;
  const nextPoint = validateResult(point.map((value, index) => {
    const correctedMean = m[index] / correction1;
    const correctedStandardDeviation = Math.sqrt(v[index]) / Math.sqrt(correction2);
    return value - learningRate * correctedMean / (correctedStandardDeviation + epsilon);
  }));

  return { point: nextPoint, m, v, t, beta1, beta2, epsilon };
}

export function adamwStep(point, gradient, learningRate, state, weightDecay) {
  validateNonNegative(weightDecay, 'weightDecay');
  const next = adamStep(point, gradient, learningRate, state);
  const decayedPoint = validateResult(next.point.map((value) => value * (1 - weightDecay)));
  return { ...next, point: decayedPoint };
}

export function clipGlobalNorm(gradient, threshold) {
  validateVector(gradient, 'gradient');
  validateNonNegative(threshold, 'threshold');

  let scale = 0;
  for (const value of gradient) {
    scale = Math.max(scale, Math.abs(value));
  }
  if (scale === 0) {
    return gradient.map(() => 0);
  }

  let normalizedNorm = 0;
  for (const value of gradient) {
    normalizedNorm = Math.hypot(normalizedNorm, value / scale);
  }
  const factor = Math.min(1, (threshold / scale) / normalizedNorm);
  if (factor === 1) {
    return gradient.slice();
  }

  // Scale through the normalized vector so neither the norm nor its reciprocal overflows.
  return validateResult(gradient.map(
    (value) => (value / scale / normalizedNorm) * threshold,
  ));
}

export function cosineLearningRate(step, totalSteps, minRate, maxRate) {
  validateFinite(step, 'step');
  validatePositive(totalSteps, 'totalSteps');
  validateNonNegative(minRate, 'minRate');
  validateNonNegative(maxRate, 'maxRate');
  if (step < 0 || step > totalSteps) {
    throw new RangeError('step must be in [0, totalSteps]');
  }
  if (minRate > maxRate) {
    throw new RangeError('minRate must not exceed maxRate');
  }

  return minRate + 0.5 * (maxRate - minRate) * (1 + Math.cos(Math.PI * step / totalSteps));
}
