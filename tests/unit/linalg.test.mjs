import test from 'node:test';
import assert from 'node:assert/strict';

import {
  add,
  cosine,
  dot,
  norm,
  projection,
  transform,
} from '../../assets/js/lib/linalg.js';

const TOLERANCE = 1e-9;

function assertClose(actual, expected) {
  assert.ok(Number.isFinite(actual), `Expected a finite number, got ${actual}`);
  assert.ok(
    Math.abs(actual - expected) <= TOLERANCE,
    `Expected ${actual} to be within ${TOLERANCE} of ${expected}`,
  );
}

test('linalg computes the fixed vector and matrix examples', () => {
  assert.deepEqual(add([1, 2], [3, 4]), [4, 6]);
  assert.equal(dot([1, 2], [3, 4]), 11);
  assert.equal(norm([3, 4]), 5);
  assertClose(cosine([1, 0], [1, 1]), Math.SQRT1_2);
  assert.deepEqual(projection([2, 2], [1, 0]), [2, 0]);
  assert.deepEqual(projection([2, 2], [2, 0]), [2, 0]);
  assert.deepEqual(transform([[0, -1], [1, 0]], [1, 0]), [0, 1]);
  assert.deepEqual(transform([[2, 1], [0, 3]], [1, 1]), [3, 3]);
});

for (const value of [1e200, 1e-200]) {
  test(`norm remains exact at finite scale ${value}`, () => {
    assert.equal(norm([value]), value);
  });

  test(`cosine remains finite at finite scale ${value}`, () => {
    assert.equal(cosine([value], [value]), 1);
  });

  test(`projection remains finite at finite scale ${value}`, () => {
    assert.deepEqual(projection([value], [value]), [value]);
  });
}

test('projection rejects a genuinely unrepresentable result', () => {
  assert.throws(
    () => projection([Number.MAX_VALUE, Number.MAX_VALUE], [1, 0.1]),
    RangeError,
  );
});

test('cosine and projection reject zero vectors in their denominators', () => {
  assert.throws(() => cosine([0, 0], [1, 0]), RangeError);
  assert.throws(() => cosine([1, 0], [0, 0]), RangeError);
  assert.throws(() => projection([1, 0], [0, 0]), RangeError);
});

test('operations reject empty vectors and mismatched dimensions', () => {
  const emptyOperations = [
    () => add([], []),
    () => dot([], []),
    () => norm([]),
    () => cosine([], []),
    () => projection([], []),
    () => transform([[1]], []),
  ];
  for (const operation of emptyOperations) {
    assert.throws(operation, RangeError);
  }

  const mismatchedOperations = [
    () => add([1], [1, 2]),
    () => dot([1], [1, 2]),
    () => cosine([1], [1, 2]),
    () => projection([1], [1, 2]),
    () => transform([[1, 2], [3, 4]], [1, 2, 3]),
  ];
  for (const operation of mismatchedOperations) {
    assert.throws(operation, RangeError);
  }
});

test('operations reject non-finite vector and matrix values', () => {
  for (const value of [NaN, Infinity, -Infinity]) {
    const operations = [
      () => add([value], [1]),
      () => add([1], [value]),
      () => dot([value], [1]),
      () => dot([1], [value]),
      () => norm([value]),
      () => cosine([value], [1]),
      () => cosine([1], [value]),
      () => projection([value], [1]),
      () => projection([1], [value]),
      () => transform([[value]], [1]),
      () => transform([[1]], [value]),
    ];
    for (const operation of operations) {
      assert.throws(operation, RangeError);
    }
  }
});

test('transform rejects non-rectangular matrices', () => {
  assert.throws(() => transform([], [1]), RangeError);
  assert.throws(() => transform([[1, 2], [3]], [1, 2]), RangeError);
  assert.throws(() => transform([[1], [2, 3]], [1]), RangeError);
});
