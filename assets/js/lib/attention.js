import { softmax } from './probability.js';

function validateVector(vector, name) {
  if (!Array.isArray(vector) || vector.length === 0) {
    throw new RangeError(`${name} must be a non-empty array`);
  }
  for (const value of vector) {
    if (!Number.isFinite(value)) {
      throw new RangeError(`${name} must contain only finite numbers`);
    }
  }
}

function validateMatrix(matrix, name) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    throw new RangeError(`${name} must be a non-empty matrix`);
  }

  let columnCount = -1;
  for (const row of matrix) {
    validateVector(row, `${name} row`);
    if (columnCount === -1) {
      columnCount = row.length;
    } else if (row.length !== columnCount) {
      throw new RangeError(`${name} must be rectangular`);
    }
  }
  return columnCount;
}

function validateAttentionInputs(q, k, v) {
  const queryDimension = validateMatrix(q, 'q');
  const keyDimension = validateMatrix(k, 'k');
  validateMatrix(v, 'v');
  if (queryDimension !== keyDimension) {
    throw new RangeError('q and k must have the same feature dimension');
  }
  if (k.length !== v.length) {
    throw new RangeError('k and v must have the same sequence length');
  }
}

function validateScale(scale) {
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new RangeError('scale must be finite and positive');
  }
}

export function scaledDotProductAttention(q, k, v, scale) {
  validateAttentionInputs(q, k, v);
  validateScale(scale);

  const scores = q.map((query) => k.map((key) => {
    let score = 0;
    for (let index = 0; index < query.length; index += 1) {
      score += query[index] * key[index];
    }
    score *= scale;
    if (!Number.isFinite(score)) {
      throw new RangeError('attention scores must be finite');
    }
    return score;
  }));
  const weights = scores.map((row) => softmax(row));
  const output = weights.map((row) => {
    const values = Array(v[0].length).fill(0);
    for (let valueIndex = 0; valueIndex < v.length; valueIndex += 1) {
      for (let dimension = 0; dimension < values.length; dimension += 1) {
        values[dimension] += row[valueIndex] * v[valueIndex][dimension];
      }
    }
    for (const value of values) {
      if (!Number.isFinite(value)) {
        throw new RangeError('attention output must be finite');
      }
    }
    return values;
  });

  return { scores, weights, output };
}

export function causalMask(scores) {
  const size = validateMatrix(scores, 'scores');
  if (size !== scores.length) {
    throw new RangeError('scores must be a square matrix');
  }
  return scores.map((row, rowIndex) => row.map(
    (value, columnIndex) => (columnIndex > rowIndex ? -Infinity : value),
  ));
}

export function maskedSoftmax(logits) {
  if (!Array.isArray(logits) || logits.length === 0) {
    throw new RangeError('logits must be a non-empty array');
  }

  const finiteLogits = logits.filter((logit) => logit !== -Infinity);
  const finiteWeights = softmax(finiteLogits);
  let finiteIndex = 0;
  return logits.map((logit) => {
    if (logit === -Infinity) {
      return 0;
    }
    const weight = finiteWeights[finiteIndex];
    finiteIndex += 1;
    return weight;
  });
}

export function multiHeadAttention(q, k, v, headCount) {
  validateAttentionInputs(q, k, v);
  if (!Number.isSafeInteger(headCount) || headCount <= 0) {
    throw new RangeError('headCount must be a positive integer');
  }

  const queryDimension = q[0].length;
  const valueDimension = v[0].length;
  if (queryDimension % headCount !== 0 || valueDimension % headCount !== 0) {
    throw new RangeError('headCount must divide q, k, and v feature dimensions');
  }

  const headDimension = queryDimension / headCount;
  const valueHeadDimension = valueDimension / headCount;
  const scale = 1 / Math.sqrt(headDimension);
  const heads = [];
  for (let head = 0; head < headCount; head += 1) {
    const queryStart = head * headDimension;
    const valueStart = head * valueHeadDimension;
    const headQueries = q.map((row) => row.slice(queryStart, queryStart + headDimension));
    const headKeys = k.map((row) => row.slice(queryStart, queryStart + headDimension));
    const headValues = v.map((row) => row.slice(valueStart, valueStart + valueHeadDimension));
    heads.push(scaledDotProductAttention(headQueries, headKeys, headValues, scale));
  }

  const output = q.map((_, rowIndex) => {
    const row = [];
    for (const head of heads) {
      row.push(...head.output[rowIndex]);
    }
    return row;
  });
  return { heads, output };
}

export function rotatePair(vector, angle) {
  validateVector(vector, 'vector');
  if (vector.length % 2 !== 0) {
    throw new RangeError('vector must have an even dimension');
  }
  if (!Number.isFinite(angle)) {
    throw new RangeError('angle must be finite');
  }

  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  const result = [];
  for (let index = 0; index < vector.length; index += 2) {
    const first = vector[index];
    const second = vector[index + 1];
    const rotatedFirst = first * cosine - second * sine;
    const rotatedSecond = first * sine + second * cosine;
    if (!Number.isFinite(rotatedFirst) || !Number.isFinite(rotatedSecond)) {
      throw new RangeError('rotated vector must be finite');
    }
    result.push(rotatedFirst, rotatedSecond);
  }
  return result;
}
