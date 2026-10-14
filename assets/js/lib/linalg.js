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

function validatePair(a, b) {
  validateVector(a, 'a');
  validateVector(b, 'b');
  if (a.length !== b.length) {
    throw new RangeError('vectors must have the same dimension');
  }
}

function validateMatrix(matrix, vectorLength) {
  if (!Array.isArray(matrix) || matrix.length === 0) {
    throw new RangeError('matrix must be a non-empty array');
  }

  let columnCount = -1;
  for (const row of matrix) {
    validateVector(row, 'matrix row');
    if (columnCount === -1) {
      columnCount = row.length;
    } else if (row.length !== columnCount) {
      throw new RangeError('matrix must be rectangular');
    }
  }
  if (columnCount !== vectorLength) {
    throw new RangeError('matrix columns must match the vector dimension');
  }
}

function isZeroVector(vector) {
  for (const value of vector) {
    if (value !== 0) {
      return false;
    }
  }
  return true;
}

export function add(a, b) {
  validatePair(a, b);
  const result = [];
  for (let index = 0; index < a.length; index += 1) {
    result.push(a[index] + b[index]);
  }
  return result;
}

export function dot(a, b) {
  validatePair(a, b);
  let result = 0;
  for (let index = 0; index < a.length; index += 1) {
    result += a[index] * b[index];
  }
  return result;
}

function normalizeByMax(vector) {
  let scale = 0;
  for (const value of vector) {
    scale = Math.max(scale, Math.abs(value));
  }
  if (scale === 0) {
    return { scale, values: vector.map(() => 0) };
  }
  return { scale, values: vector.map((value) => value / scale) };
}

export function norm(a) {
  validateVector(a, 'a');
  let result = 0;
  for (const value of a) {
    result = Math.hypot(result, value);
  }
  return result;
}

export function cosine(a, b) {
  validatePair(a, b);
  if (isZeroVector(a) || isZeroVector(b)) {
    throw new RangeError('cosine is undefined for zero vectors');
  }
  const { values: normalizedA } = normalizeByMax(a);
  const { values: normalizedB } = normalizeByMax(b);
  return dot(normalizedA, normalizedB) / (norm(normalizedA) * norm(normalizedB));
}

export function projection(a, b) {
  validatePair(a, b);
  if (isZeroVector(b)) {
    throw new RangeError('projection direction must be non-zero');
  }

  const { scale: aScale, values: normalizedA } = normalizeByMax(a);
  const { values: normalizedB } = normalizeByMax(b);
  const scale = dot(normalizedA, normalizedB) / dot(normalizedB, normalizedB);
  const result = [];
  for (const value of normalizedB) {
    const projected = scale * value * aScale;
    if (!Number.isFinite(projected)) {
      throw new RangeError('projection result is not representable');
    }
    result.push(projected);
  }
  return result;
}

export function transform(matrix, vector) {
  validateVector(vector, 'vector');
  validateMatrix(matrix, vector.length);

  const result = [];
  for (const row of matrix) {
    let value = 0;
    for (let index = 0; index < vector.length; index += 1) {
      value += row[index] * vector[index];
    }
    result.push(value);
  }
  return result;
}
