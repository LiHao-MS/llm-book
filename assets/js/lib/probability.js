function validateProbabilities(probabilities) {
  if (!Array.isArray(probabilities) || probabilities.length === 0) {
    throw new RangeError('probabilities must be a non-empty array');
  }

  let sum = 0;
  for (const probability of probabilities) {
    if (!Number.isFinite(probability) || probability < 0) {
      throw new RangeError('probabilities must contain finite non-negative numbers');
    }
    sum += probability;
  }
  if (!Number.isFinite(sum) || sum <= 0) {
    throw new RangeError('probabilities must have a finite positive sum');
  }
  return sum;
}

function validateProbability(probability) {
  if (!Number.isFinite(probability) || probability <= 0 || probability > 1) {
    throw new RangeError('probability must be finite and in (0, 1]');
  }
}

export function softmax(logits) {
  if (!Array.isArray(logits) || logits.length === 0) {
    throw new RangeError('logits must be a non-empty array');
  }

  let maximum = -Infinity;
  for (const logit of logits) {
    if (!Number.isFinite(logit)) {
      throw new RangeError('logits must contain only finite numbers');
    }
    maximum = Math.max(maximum, logit);
  }

  const exponentials = logits.map((logit) => Math.exp(logit - maximum));
  const sum = exponentials.reduce((total, value) => total + value, 0);
  return exponentials.map((value) => value / sum);
}

export function crossEntropy(probability) {
  validateProbability(probability);
  return -Math.log(probability);
}

export function perplexity(probability) {
  return Math.exp(crossEntropy(probability));
}

export function temperatureProbabilities(probabilities, temperature) {
  validateProbabilities(probabilities);
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError('temperature must be finite and positive');
  }

  const transformed = probabilities.map((probability) => probability ** (1 / temperature));
  const sum = transformed.reduce((total, value) => total + value, 0);
  if (!Number.isFinite(sum) || sum <= 0) {
    throw new RangeError('transformed probabilities must have a finite positive sum');
  }
  return transformed.map((probability) => probability / sum);
}

const UNIT_ROUNDOFF = Number.EPSILON / 2;

function addCompensated(accumulator, value) {
  const next = accumulator.sum + value;
  accumulator.correction += Math.abs(accumulator.sum) >= Math.abs(value)
    ? (accumulator.sum - next) + value
    : (value - next) + accumulator.sum;
  accumulator.sum = next;
}

function compensatedSum(values) {
  const accumulator = { sum: 0, correction: 0 };
  for (const value of values) {
    addCompensated(accumulator, value);
  }
  return accumulator.sum + accumulator.correction;
}

function reachesThreshold(ratio, threshold, inputLength) {
  const operationCount = 2 * inputLength + 1;
  const roundingOperations = operationCount * UNIT_ROUNDOFF;
  const gamma = roundingOperations / (1 - roundingOperations);
  const magnitude = Math.max(Math.abs(ratio), Math.abs(threshold));
  return ratio + gamma * magnitude >= threshold;
}

export function topP(probabilities, threshold) {
  validateProbabilities(probabilities);
  if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) {
    throw new RangeError('threshold must be finite and in (0, 1]');
  }

  const total = compensatedSum(probabilities);
  const sorted = probabilities
    .map((probability, index) => ({ probability, index }))
    .sort((left, right) => right.probability - left.probability || left.index - right.index);
  const selected = new Set();
  const cumulative = { sum: 0, correction: 0 };
  let keptSum = 0;

  for (const { probability, index } of sorted) {
    selected.add(index);
    addCompensated(cumulative, probability);
    keptSum = cumulative.sum + cumulative.correction;
    // γ(2n+1) conservatively covers two n-term sums and their division.
    if (reachesThreshold(keptSum / total, threshold, probabilities.length)) {
      break;
    }
  }

  return {
    probabilities: probabilities.map((probability, index) =>
      selected.has(index) ? probability / keptSum : 0,
    ),
    kept: probabilities.map((_, index) => selected.has(index)),
  };
}
