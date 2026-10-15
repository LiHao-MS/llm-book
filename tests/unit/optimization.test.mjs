import test from 'node:test';
import assert from 'node:assert/strict';

import {
  adamStep,
  adamwStep,
  clipGlobalNorm,
  cosineLearningRate,
  momentumStep,
  quadratic,
  quadraticGradient,
  sgdStep,
} from '../../assets/js/lib/optimization.js';

const TOLERANCE = 1e-8;

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

function adamState() {
  return {
    m: [0, 0],
    v: [0, 0],
    t: 0,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
  };
}

test('quadratic computes the bowl objective and its gradient without mutation', () => {
  const point = [3, 1];
  const center = [1, -1];
  assert.equal(quadratic(point, center), 4);
  assert.deepEqual(quadraticGradient(point, center), [2, 2]);
  assert.deepEqual(point, [3, 1]);
  assert.deepEqual(center, [1, -1]);
});

test('SGD and Momentum compute the fixed updates and return fresh state', () => {
  const point = [1, 1];
  const gradient = [2, -1];
  assert.deepEqual(sgdStep(point, gradient, 0.1), [0.8, 1.1]);

  const state = { velocity: [1, 1], beta: 0.5 };
  const result = momentumStep(point, gradient, 0.1, state);
  assert.deepEqual(result.point, [0.75, 1.05]);
  assert.deepEqual(result.velocity, [2.5, -0.5]);
  assert.equal(result.beta, 0.5);
  assert.notEqual(result.point, point);
  assert.notEqual(result.velocity, state.velocity);
  assert.deepEqual(point, [1, 1]);
  assert.deepEqual(gradient, [2, -1]);
  assert.deepEqual(state, { velocity: [1, 1], beta: 0.5 });
});

test('Adam updates moments and time with bias correction', () => {
  const point = [1, 1];
  const gradient = [2, -1];
  const state = adamState();
  const result = adamStep(point, gradient, 0.1, state);

  assertVectorClose(result.point, [0.9, 1.1]);
  assertVectorClose(result.m, [0.2, -0.1]);
  assertVectorClose(result.v, [0.004, 0.001]);
  assert.equal(result.t, 1);
  assert.equal(result.beta1, state.beta1);
  assert.equal(result.beta2, state.beta2);
  assert.equal(result.epsilon, state.epsilon);
  assert.notEqual(result.point, point);
  assert.notEqual(result.m, state.m);
  assert.notEqual(result.v, state.v);
  assert.deepEqual(point, [1, 1]);
  assert.deepEqual(gradient, [2, -1]);
  assert.deepEqual(state, adamState());
});

test('Adam handles large finite gradients without overflow', () => {
  const point = [1, 1];
  const gradient = [1.5e154, 0];
  const state = adamState();
  const result = adamStep(point, gradient, 0.1, state);

  assertVectorClose(result.point, [0.9, 1]);
  assertClose(result.v[0] / 2.25e305, 1);
  assert.deepEqual(point, [1, 1]);
  assert.deepEqual(gradient, [1.5e154, 0]);
  assert.deepEqual(state, adamState());
});

test('Adam uses prior moments and sqrt(vhat) plus epsilon on the second step', () => {
  const point = [1, -1];
  const gradient = [0, 0];
  const state = {
    m: [0.00015, 0.0003],
    v: [1.5e-8, 6e-8],
    t: 1,
    beta1: 0.5,
    beta2: 0.5,
    epsilon: 1e-8,
  };
  const result = adamStep(point, gradient, 0.1, state);

  assertVectorClose(result.point, [0.9000099990001, -1.0999950002499875]);
  assert.deepEqual(result.point.map((value) => Number(value.toFixed(1))), [0.9, -1.1]);
  assert.deepEqual(result.m, [0.000075, 0.00015]);
  assert.deepEqual(result.v, [7.5e-9, 3e-8]);
  assert.equal(result.t, 2);
  assert.deepEqual(point, [1, -1]);
  assert.deepEqual(gradient, [0, 0]);
  assert.deepEqual(state, {
    m: [0.00015, 0.0003],
    v: [1.5e-8, 6e-8],
    t: 1,
    beta1: 0.5,
    beta2: 0.5,
    epsilon: 1e-8,
  });
});

test('AdamW applies decoupled decay after the Adam update', () => {
  const point = [1, 1];
  const gradient = [2, -1];
  const state = adamState();
  const result = adamwStep(point, gradient, 0.1, state, 0.01);

  assertVectorClose(result.point, [0.891, 1.089]);
  assertVectorClose(result.m, [0.2, -0.1]);
  assertVectorClose(result.v, [0.004, 0.001]);
  assert.equal(result.t, 1);
  assert.deepEqual(point, [1, 1]);
  assert.deepEqual(gradient, [2, -1]);
  assert.deepEqual(state, adamState());
});

test('Momentum rejects incomplete state with a required-field RangeError', () => {
  assert.throws(
    () => momentumStep([1, 1], [1, 0], 0.1, { velocity: [0, 0] }),
    { name: 'RangeError', message: 'state.beta is required' },
  );
});

test('Adam rejects incomplete state with a required-field RangeError', () => {
  const state = {
    m: [0, 0],
    v: [0, 0],
    t: 0,
    beta1: 0.9,
    epsilon: 1e-8,
  };
  assert.throws(
    () => adamStep([1, 1], [1, 0], 0.1, state),
    { name: 'RangeError', message: 'state.beta2 is required' },
  );
});

test('zero gradients and zero learning rates are legal', () => {
  const point = [1, 1];
  const zero = [0, 0];
  assert.deepEqual(sgdStep(point, zero, 0.1), point);
  assert.deepEqual(sgdStep(point, [2, -1], 0), point);

  const momentum = momentumStep(point, zero, 0.1, { velocity: zero, beta: 0.5 });
  assert.deepEqual(momentum.point, point);
  assert.deepEqual(momentum.velocity, zero);

  const adam = adamStep(point, zero, 0.1, adamState());
  assert.deepEqual(adam.point, point);
  assert.deepEqual(adam.m, zero);
  assert.deepEqual(adam.v, zero);

  assert.deepEqual(adamStep(point, [2, -1], 0, adamState()).point, point);
  assert.deepEqual(adamwStep(point, [2, -1], 0, adamState(), 0).point, point);
});

test('clipGlobalNorm scales only when needed and handles zero values', () => {
  const gradient = [3, 4];
  const clipped = clipGlobalNorm(gradient, 1);
  assert.deepEqual(clipped, [0.6, 0.8]);
  assert.notEqual(clipped, gradient);
  assert.deepEqual(gradient, [3, 4]);
  assert.deepEqual(clipGlobalNorm(gradient, 5), gradient);
  assert.deepEqual(clipGlobalNorm(gradient, 0), [0, 0]);
  assert.deepEqual(clipGlobalNorm([0, 0], 1), [0, 0]);
  assert.deepEqual(clipGlobalNorm([0, 0], 0), [0, 0]);
});

test('clipGlobalNorm remains finite at extreme finite scales', () => {
  assertVectorClose(clipGlobalNorm([3e200, 4e200], 1), [0.6, 0.8]);
  assertVectorClose(
    clipGlobalNorm([Number.MAX_VALUE, Number.MAX_VALUE], 1),
    [Math.SQRT1_2, Math.SQRT1_2],
  );
});

test('cosineLearningRate follows the cosine endpoints and midpoint', () => {
  assertClose(cosineLearningRate(0, 10, 0, 1), 1);
  assertClose(cosineLearningRate(5, 10, 0, 1), 0.5);
  assertClose(cosineLearningRate(10, 10, 0, 1), 0);
  assertClose(cosineLearningRate(0, 10, 0.2, 1), 1);
  assertClose(cosineLearningRate(10, 10, 0.2, 1), 0.2);
});

test('vector APIs reject invalid dimensions and non-finite values', () => {
  const operations = [
    () => quadratic([], []),
    () => quadratic([1], [0]),
    () => quadratic([1, 2, 3], [0, 0, 0]),
    () => quadraticGradient([1, 2], [0]),
    () => sgdStep([1], [1], 0.1),
    () => sgdStep([1, 2], [1], 0.1),
    () => momentumStep([1, 2], [1, 2], 0.1, { velocity: [0], beta: 0.5 }),
    () => adamStep([1, 2], [1, 2], 0.1, { ...adamState(), m: [0] }),
    () => adamStep([1, 2], [1, 2], 0.1, { ...adamState(), v: [0] }),
    () => clipGlobalNorm([], 1),
  ];
  for (const operation of operations) {
    assert.throws(operation, RangeError);
  }

  for (const value of [NaN, Infinity, -Infinity]) {
    const nonFiniteOperations = [
      () => quadratic([value, 0], [0, 0]),
      () => quadratic([1, 0], [0, value]),
      () => quadraticGradient([value, 0], [0, 0]),
      () => sgdStep([1, 1], [value, 0], 0.1),
      () => sgdStep([1, 1], [1, 0], value),
      () => momentumStep([1, 1], [1, 0], 0.1, { velocity: [value, 0], beta: 0.5 }),
      () => adamStep([1, 1], [1, 0], 0.1, { ...adamState(), m: [value, 0] }),
      () => clipGlobalNorm([value, 0], 1),
      () => cosineLearningRate(value, 10, 0, 1),
      () => cosineLearningRate(0, value, 0, 1),
      () => cosineLearningRate(0, 10, value, 1),
      () => cosineLearningRate(0, 10, 0, value),
    ];
    for (const operation of nonFiniteOperations) {
      assert.throws(operation, RangeError);
    }
  }
});

test('optimizer hyperparameters and schedule bounds are validated', () => {
  for (const learningRate of [-0.1, NaN, Infinity, -Infinity]) {
    assert.throws(() => sgdStep([1, 1], [1, 1], learningRate), RangeError);
    assert.throws(
      () => momentumStep([1, 1], [1, 1], learningRate, { velocity: [0, 0], beta: 0.5 }),
      RangeError,
    );
    assert.throws(() => adamStep([1, 1], [1, 1], learningRate, adamState()), RangeError);
  }

  for (const beta of [-0.1, 1, 1.1, NaN, Infinity, -Infinity]) {
    assert.throws(
      () => momentumStep([1, 1], [1, 1], 0.1, { velocity: [0, 0], beta }),
      RangeError,
    );
    assert.throws(() => adamStep([1, 1], [1, 1], 0.1, { ...adamState(), beta1: beta }), RangeError);
    assert.throws(() => adamStep([1, 1], [1, 1], 0.1, { ...adamState(), beta2: beta }), RangeError);
  }

  for (const epsilon of [0, -1, NaN, Infinity, -Infinity]) {
    assert.throws(
      () => adamStep([1, 1], [1, 1], 0.1, { ...adamState(), epsilon }),
      RangeError,
    );
  }
  for (const t of [-1, 0.5, NaN, Infinity, -Infinity]) {
    assert.throws(() => adamStep([1, 1], [1, 1], 0.1, { ...adamState(), t }), RangeError);
  }
  for (const weightDecay of [-0.1, NaN, Infinity, -Infinity]) {
    assert.throws(() => adamwStep([1, 1], [1, 1], 0.1, adamState(), weightDecay), RangeError);
  }
  for (const threshold of [-0.1, NaN, Infinity, -Infinity]) {
    assert.throws(() => clipGlobalNorm([1, 1], threshold), RangeError);
  }

  for (const totalSteps of [0, -1, NaN, Infinity, -Infinity]) {
    assert.throws(() => cosineLearningRate(0, totalSteps, 0.2, 1), RangeError);
  }
  assert.throws(() => cosineLearningRate(-1, 10, 0, 1), RangeError);
  assert.throws(() => cosineLearningRate(11, 10, 0, 1), RangeError);
  assert.throws(() => cosineLearningRate(0, 10, -0.1, 1), RangeError);
  assert.throws(() => cosineLearningRate(0, 10, 0, -1), RangeError);
  assert.throws(() => cosineLearningRate(0, 10, 1.1, 1), RangeError);
});
