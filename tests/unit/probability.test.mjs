import test from 'node:test';
import assert from 'node:assert/strict';

import {
  crossEntropy,
  perplexity,
  softmax,
  temperatureProbabilities,
  topP,
} from '../../assets/js/lib/probability.js';

const TOLERANCE = 1e-9;

function assertClose(actual, expected) {
  assert.ok(Number.isFinite(actual), `Expected a finite number, got ${actual}`);
  assert.ok(
    Math.abs(actual - expected) <= TOLERANCE,
    `Expected ${actual} to be within ${TOLERANCE} of ${expected}`,
  );
}

test('softmax computes normalized probabilities stably', () => {
  const probabilities = softmax([1, 2, 3]);
  assertClose(probabilities[2], 0.6652409558);
  assertClose(probabilities.reduce((sum, value) => sum + value, 0), 1);
  assertClose(softmax([1000, 1001])[0], 0.2689414214);
  assertClose(softmax([-1, -2, -3])[0], 0.6652409558);
});

test('softmax rejects empty and non-finite logits', () => {
  for (const logits of [[], [0, NaN], [0, Infinity], [0, -Infinity]]) {
    assert.throws(() => softmax(logits), RangeError);
  }
});

test('crossEntropy and perplexity use natural logarithms', () => {
  assertClose(crossEntropy(0.5), Math.LN2);
  assertClose(crossEntropy(1), 0);
  assertClose(perplexity(0.5), 2);
  assertClose(perplexity(0.25), 4);
  assertClose(perplexity(1), 1);
  assert.throws(() => crossEntropy(0), RangeError);
});

test('crossEntropy and perplexity reject invalid probabilities', () => {
  for (const fn of [crossEntropy, perplexity]) {
    for (const probability of [0, -0.1, 1.1, NaN, Infinity, -Infinity]) {
      assert.throws(() => fn(probability), RangeError);
    }
  }
});

test('temperatureProbabilities normalizes and flattens at high temperature', () => {
  const original = temperatureProbabilities([0.8, 0.2], 1);
  const heated = temperatureProbabilities([0.8, 0.2], 10);

  assertClose(heated.reduce((sum, value) => sum + value, 0), 1);
  assert.ok(heated[0] < original[0]);
  assert.ok(heated[1] > original[1]);
  assert.ok(heated[0] - heated[1] < original[0] - original[1]);
  assert.throws(() => temperatureProbabilities([0.5, 0.5], 0), RangeError);
});

test('temperatureProbabilities rejects invalid inputs', () => {
  for (const temperature of [0, -1, NaN, Infinity, -Infinity]) {
    assert.throws(() => temperatureProbabilities([0.5, 0.5], temperature), RangeError);
  }
  for (const probabilities of [[], [0, 0], [-0.1, 1.1], [NaN, 1], [Infinity, 0]]) {
    assert.throws(() => temperatureProbabilities(probabilities, 1), RangeError);
  }
});

test('topP keeps the smallest reaching set in original token order', () => {
  const input = [0.2, 0.5, 0.3];
  const result = topP(input, 0.7);

  assert.deepEqual(result.kept, [false, true, true]);
  assertClose(result.probabilities[0], 0);
  assertClose(result.probabilities[1], 0.625);
  assertClose(result.probabilities[2], 0.375);
  assertClose(result.probabilities.reduce((sum, value) => sum + value, 0), 1);
  assert.deepEqual(input, [0.2, 0.5, 0.3]);

  const allKept = topP(input, 1);
  assert.deepEqual(allKept.kept, [true, true, true]);
  assertClose(allKept.probabilities.reduce((sum, value) => sum + value, 0), 1);
});

test('topP stops at an exact cumulative threshold in original order', () => {
  const cases = [
    {
      input: [0.1, 0.2, 0.7],
      kept: [false, true, true],
      probabilities: [0, 2 / 9, 7 / 9],
    },
    {
      input: [0.7, 0.1, 0.2],
      kept: [true, false, true],
      probabilities: [7 / 9, 0, 2 / 9],
    },
    {
      input: [0.2, 0.7, 0.1],
      kept: [true, true, false],
      probabilities: [2 / 9, 7 / 9, 0],
    },
  ];

  for (const { input, kept, probabilities } of cases) {
    const result = topP(input, 0.9);
    assert.deepEqual(result.kept, kept, `input order: ${input}`);
    assert.equal(result.kept.filter(Boolean).length, 2);
    assert.ok(0.7 < 0.9, 'the largest token alone cannot reach the threshold');
    result.probabilities.forEach((probability, index) => {
      assertClose(probability, probabilities[index]);
    });
    assertClose(result.probabilities.reduce((sum, value) => sum + value, 0), 1);
  }
});

test('topP at threshold one excludes trailing zero-probability tokens', () => {
  const result = topP([0.1, 0.2, 0.7, 0], 1);

  assert.deepEqual(result.kept, [true, true, true, false]);
  assert.deepEqual(result.probabilities.slice(0, 3), [0.1, 0.2, 0.7]);
  assert.equal(result.probabilities[3], 0);
  assertClose(result.probabilities.reduce((sum, value) => sum + value, 0), 1);
});

test('topP rejects invalid inputs', () => {
  for (const threshold of [0, -0.1, 1.1, NaN, Infinity, -Infinity]) {
    assert.throws(() => topP([0.5, 0.5], threshold), RangeError);
  }
  for (const probabilities of [[], [0, 0], [-0.1, 1.1], [NaN, 1], [Infinity, 0]]) {
    assert.throws(() => topP(probabilities, 0.7), RangeError);
  }
});
