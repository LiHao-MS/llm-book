import test from 'node:test';
import assert from 'node:assert/strict';

import {
  speculativeAcceptance,
  speculativeResidual,
} from '../../assets/js/components/speculative-decoding.js';

const TOLERANCE = 1e-12;

function assertClose(actual, expected) {
  assert.ok(Number.isFinite(actual), `Expected a finite number, got ${actual}`);
  assert.ok(
    Math.abs(actual - expected) <= TOLERANCE,
    `Expected ${actual} to be within ${TOLERANCE} of ${expected}`,
  );
}

function assertDistributionClose(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assertClose(value, expected[index]));
}

test('speculativeAcceptance reproduces the worked distribution example', () => {
  const result = speculativeAcceptance([0.5, 0.3, 0.2], [0.4, 0.4, 0.2]);

  assertDistributionClose(result.perToken, [1, 0.75, 1]);
  assertClose(result.total, 0.9);
});

test('speculativeResidual normalizes positive rejected target mass', () => {
  const residual = speculativeResidual([0.5, 0.3, 0.2], [0.4, 0.4, 0.2]);

  assertDistributionClose(residual, [1, 0, 0]);
  assertClose(residual.reduce((sum, value) => sum + value, 0), 1);
});

test('identical distributions have total acceptance one and no residual', () => {
  const distribution = [0.5, 0.3, 0.2];
  const acceptance = speculativeAcceptance(distribution, distribution);

  assertDistributionClose(acceptance.perToken, [1, 1, 1]);
  assertClose(acceptance.total, 1);
  assert.equal(speculativeResidual(distribution, distribution), null);
});

test('a token unreachable from the draft has null acceptance and zero total contribution', () => {
  const acceptance = speculativeAcceptance([0.4, 0.6], [1, 0]);

  assertClose(acceptance.perToken[0], 0.4);
  assert.equal(acceptance.perToken[1], null);
  assertClose(acceptance.total, 0.4);
  assertDistributionClose(speculativeResidual([0.4, 0.6], [1, 0]), [0, 1]);
});

test('zero draft probability stays null even when target probability is also zero', () => {
  const acceptance = speculativeAcceptance([0.5, 0.5, 0], [0.5, 0, 0.5]);

  assertDistributionClose([acceptance.perToken[0], acceptance.perToken[2]], [1, 0]);
  assert.equal(acceptance.perToken[1], null);
  assertClose(acceptance.total, 0.5);
  assertDistributionClose(speculativeResidual([0.5, 0.5, 0], [0.5, 0, 0.5]), [0, 1, 0]);
});

test('acceptance and residual reject mismatched or empty distributions', () => {
  for (const fn of [speculativeAcceptance, speculativeResidual]) {
    assert.throws(() => fn([0.5, 0.5], [1]), RangeError);
    assert.throws(() => fn([], []), RangeError);
    assert.throws(() => fn('not-an-array', [1]), RangeError);
  }
});

test('normalization validation tolerates roundoff but rejects a real excess mass', () => {
  const roundoffDistribution = Array(10).fill(0.1);
  assertClose(speculativeAcceptance(roundoffDistribution, roundoffDistribution).total, 1);
  assert.equal(speculativeResidual(roundoffDistribution, roundoffDistribution), null);

  for (const fn of [speculativeAcceptance, speculativeResidual]) {
    assert.throws(() => fn([0.5, 0.5000000001], [0.5, 0.5]), RangeError);
  }
});

test('acceptance and residual reject invalid or non-normalized probabilities', () => {
  const invalidPairs = [
    [[0.6, 0.6], [0.5, 0.5]],
    [[0.5, 0.5], [0.4, 0.4]],
    [[-0.1, 1.1], [0.5, 0.5]],
    [[1.1, -0.1], [0.5, 0.5]],
    [[NaN, 1], [0.5, 0.5]],
    [[Infinity, 0], [0.5, 0.5]],
    [[0.5, 0.5], [NaN, 1]],
    [[0.5, 0.5], [Infinity, 0]],
  ];

  for (const fn of [speculativeAcceptance, speculativeResidual]) {
    for (const [p, q] of invalidPairs) {
      assert.throws(() => fn(p, q), RangeError, `${fn.name} accepted ${p} / ${q}`);
    }
  }
});
