import test from 'node:test';
import assert from 'node:assert/strict';

import * as attention from '../../assets/js/lib/attention.js';

const {
  causalMask,
  maskedSoftmax,
  multiHeadAttention,
  rotatePair,
  scaledDotProductAttention,
} = attention;

const TOLERANCE = 1e-9;

function assertClose(actual, expected) {
  assert.ok(Number.isFinite(actual), `Expected a finite number, got ${actual}`);
  assert.ok(
    Math.abs(actual - expected) <= TOLERANCE,
    `Expected ${actual} to be within ${TOLERANCE} of ${expected}`,
  );
}

function assertVectorClose(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assertClose(value, expected[index]));
}

function assertMatrixClose(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((row, index) => assertVectorClose(row, expected[index]));
}

function fixture() {
  return {
    q: [[1, 0], [0, 1], [1, 1]],
    k: [[1, 0], [0, 1], [1, 1]],
    v: [[1, 0], [0, 1], [1, 1]],
    scale: 1 / Math.sqrt(2),
  };
}

test('scaled attention computes scores, normalized weights, and value output', () => {
  const { q, k, v, scale } = fixture();
  const result = scaledDotProductAttention(q, k, v, scale);

  assertMatrixClose(result.scores, [
    [Math.SQRT1_2, 0, Math.SQRT1_2],
    [0, Math.SQRT1_2, Math.SQRT1_2],
    [Math.SQRT1_2, Math.SQRT1_2, Math.sqrt(2)],
  ]);
  assertMatrixClose(result.weights, [
    [0.4011120927, 0.1977758146, 0.4011120927],
    [0.1977758146, 0.4011120927, 0.4011120927],
    [0.2482550783, 0.2482550783, 0.5034898435],
  ]);
  for (const row of result.weights) {
    assertClose(row.reduce((sum, value) => sum + value, 0), 1);
  }
  assertMatrixClose(result.output, [
    [0.8022241854, 0.5988879073],
    [0.5988879073, 0.8022241854],
    [0.7517449217, 0.7517449217],
  ]);
  assert.deepEqual(q, [[1, 0], [0, 1], [1, 1]]);
  assert.deepEqual(k, [[1, 0], [0, 1], [1, 1]]);
  assert.deepEqual(v, [[1, 0], [0, 1], [1, 1]]);
});

test('scaled attention applies scale before Softmax and uses value dimensions', () => {
  const scores = [[2, 1, 3], [1, 1, 1], [3, 1, 2]];
  const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const result = scaledDotProductAttention(scores, identity, identity, Math.SQRT1_2);

  assertVectorClose(result.weights[0], [0.2839954097, 0.1400292450, 0.5759753452]);
  assertClose(result.output[0][0], 0.2839954097);
  assertClose(result.output[0][2], 0.5759753452);
  for (const row of result.weights) {
    assertClose(row.reduce((sum, value) => sum + value, 0), 1);
  }
});

test('scaled attention distinguishes keys from values with independent dimensions', () => {
  const q = [[1, 0]];
  const k = [[1, 0], [0, 1]];
  const v = [[10], [20]];
  const result = scaledDotProductAttention(q, k, v, 1);

  assert.deepEqual(result.scores, [[1, 0]]);
  assertMatrixClose(result.weights, [[0.7310585786, 0.2689414214]]);
  assertMatrixClose(result.output, [[12.6894142137]]);
});

test('multi-head attention computes scores from keys and outputs from values', () => {
  const q = [[1, 0]];
  const k = [[1, 0], [0, 1]];
  const v = [[10, 10], [20, 20]];
  const result = multiHeadAttention(q, k, v, 2);

  assert.deepEqual(result.heads[0].scores, [[1, 0]]);
  assert.deepEqual(result.heads[1].scores, [[0, 0]]);
  assertMatrixClose(result.heads[0].weights, [[0.7310585786, 0.2689414214]]);
  assertMatrixClose(result.heads[1].weights, [[0.5, 0.5]]);
  assertMatrixClose(result.heads[0].output, [[12.6894142137]]);
  assertMatrixClose(result.heads[1].output, [[15]]);
  assertMatrixClose(result.output, [[12.6894142137, 15]]);
});

test('causal mask blocks only strict future positions without mutation', () => {
  const scores = [[2, 1, 3], [1, 1, 1], [3, 1, 2]];
  const masked = causalMask(scores);

  assert.equal(masked[0][1], -Infinity);
  assert.equal(masked[0][2], -Infinity);
  assert.equal(masked[1][2], -Infinity);
  assert.deepEqual(masked[0][0], 2);
  assert.deepEqual(masked[1].slice(0, 2), [1, 1]);
  assert.deepEqual(masked[2], [3, 1, 2]);
  assert.deepEqual(scores, [[2, 1, 3], [1, 1, 1], [3, 1, 2]]);
  assert.notEqual(masked, scores);
  masked.forEach((row, index) => assert.notEqual(row, scores[index]));
});

test('masked Softmax composes with causalMask and zeroes future weights', () => {
  const scores = [[2, 1, 3], [1, 1, 1], [3, 1, 2]];
  const weights = causalMask(scores).map((row) => maskedSoftmax(row));

  assertMatrixClose(weights, [
    [1, 0, 0],
    [0.5, 0.5, 0],
    [0.6652409558, 0.0900305735, 0.2447284711],
  ]);
  assert.equal(weights[0][1], 0);
  assert.equal(weights[0][2], 0);
  assert.equal(weights[1][2], 0);
  for (const row of weights) {
    assertClose(row.reduce((sum, value) => sum + value, 0), 1);
  }
  assertVectorClose(
    maskedSoftmax([1000, 1001, -Infinity]),
    [0.2689414214, 0.7310585786, 0],
  );
});

test('masked Softmax rejects invalid values and fully masked rows', () => {
  const invalidRows = [
    [],
    [-Infinity, -Infinity],
    [NaN, -Infinity],
    [Infinity, -Infinity],
  ];
  for (const row of invalidRows) {
    assert.throws(() => maskedSoftmax(row), RangeError);
  }
});

test('multi-head attention splits feature dimensions and concatenates head outputs', () => {
  const { q, k, v } = fixture();
  const result = multiHeadAttention(q, k, v, 2);

  assert.equal(result.heads.length, 2);
  assertMatrixClose(result.heads[0].output, [
    [0.8446375965],
    [0.6666666667],
    [0.8446375965],
  ]);
  assertMatrixClose(result.heads[1].output, [
    [0.6666666667],
    [0.8446375965],
    [0.8446375965],
  ]);
  assertMatrixClose(result.output, [
    [0.8446375965, 0.6666666667],
    [0.6666666667, 0.8446375965],
    [0.8446375965, 0.8446375965],
  ]);
  for (const head of result.heads) {
    for (const row of head.weights) {
      assertClose(row.reduce((sum, value) => sum + value, 0), 1);
    }
  }
});

test('single-head output matches scaled dot-product attention', () => {
  const { q, k, v, scale } = fixture();
  const single = scaledDotProductAttention(q, k, v, scale);
  const result = multiHeadAttention(q, k, v, 1);

  assert.deepEqual(result.output, single.output);
  assert.deepEqual(result.heads, [single]);
});

test('RoPE rotates each adjacent pair in the positive direction without mutation', () => {
  const vector = [1, 0, 1, 0];
  const quarterTurn = rotatePair(vector, Math.PI / 2);
  assertVectorClose(quarterTurn, [0, 1, 0, 1]);
  assertVectorClose(rotatePair([1, 0], 0), [1, 0]);
  assertVectorClose(rotatePair([1, 0], Math.PI), [-1, 0]);
  assertVectorClose(rotatePair([1, 2, 3, 4], Math.PI / 2), [-2, 1, -4, 3]);
  assert.deepEqual(vector, [1, 0, 1, 0]);
  assert.notEqual(quarterTurn, vector);
});

test('attention validates matrices, dimensions, scales, and head counts', () => {
  const operations = [
    () => scaledDotProductAttention([], [[1]], [[1]], 1),
    () => scaledDotProductAttention([[1]], [], [[1]], 1),
    () => scaledDotProductAttention([[1]], [[1]], [], 1),
    () => scaledDotProductAttention([[]], [[1]], [[1]], 1),
    () => scaledDotProductAttention([[1, 2]], [[1]], [[1]], 1),
    () => scaledDotProductAttention([[1]], [[1], [2]], [[1]], 1),
    () => scaledDotProductAttention([[1, 2], [3]], [[1, 2]], [[1], [2]], 1),
    () => multiHeadAttention([[1, 0]], [[1, 0]], [[1, 0]], 0),
    () => multiHeadAttention([[1, 0]], [[1, 0]], [[1, 0]], -1),
    () => multiHeadAttention([[1, 0]], [[1, 0]], [[1, 0]], 1.5),
    () => multiHeadAttention([[1, 0]], [[1, 0]], [[1, 0]], NaN),
    () => multiHeadAttention([[1, 0]], [[1, 0]], [[1, 0, 0]], 2),
  ];
  for (const operation of operations) {
    assert.throws(operation, RangeError);
  }

  for (const scale of [0, -1, NaN, Infinity, -Infinity]) {
    assert.throws(
      () => scaledDotProductAttention([[1]], [[1]], [[1]], scale),
      RangeError,
    );
  }
  for (const value of [NaN, Infinity, -Infinity]) {
    assert.throws(() => scaledDotProductAttention([[value]], [[1]], [[1]], 1), RangeError);
    assert.throws(() => scaledDotProductAttention([[1]], [[value]], [[1]], 1), RangeError);
    assert.throws(() => scaledDotProductAttention([[1]], [[1]], [[value]], 1), RangeError);
  }
});

test('causal mask and RoPE reject malformed or non-finite inputs', () => {
  const invalidMasks = [
    [],
    [[1, 2]],
    [[1], [2]],
    [[1, 2], [3]],
    [[1, NaN], [1, 2]],
    [[1, Infinity], [1, 2]],
  ];
  for (const scores of invalidMasks) {
    assert.throws(() => causalMask(scores), RangeError);
  }

  assert.throws(() => rotatePair([], 0), RangeError);
  assert.throws(() => rotatePair([1, 2, 3], 0), RangeError);
  for (const value of [NaN, Infinity, -Infinity]) {
    assert.throws(() => rotatePair([value, 0], 0), RangeError);
    assert.throws(() => rotatePair([1, 0], value), RangeError);
  }
});
