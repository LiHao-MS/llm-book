import test from 'node:test';
import assert from 'node:assert/strict';

import { ppoClippedObjective } from '../../assets/js/components/ppo-clipping.js';

test('PPO clipping uses the clipped term for a large ratio with positive advantage', () => {
  assert.equal(ppoClippedObjective(1.4, 1, 0.2), 1.2);
});

test('PPO clipping uses the clipped term for a small ratio with negative advantage', () => {
  assert.equal(ppoClippedObjective(0.6, -1, 0.2), -0.8);
});

test('PPO clipping handles a zero ratio for both advantage signs', () => {
  assert.equal(ppoClippedObjective(0, 1, 0.2), 0);
  assert.equal(ppoClippedObjective(0, -1, 0.2), -0.8);
});

test('PPO clipping leaves unclipped ratios and zero advantage exact', () => {
  assert.equal(ppoClippedObjective(1.1, 2, 0.2), 2.2);
  assert.equal(ppoClippedObjective(0.9, -2, 0.2), -1.8);
  assert.equal(ppoClippedObjective(1.4, -2, 0.2), -2.8);
  assert.equal(ppoClippedObjective(0.6, 2, 0.2), 1.2);
  assert.equal(ppoClippedObjective(1.4, 0, 0.2), 0);
});

test('PPO clipping accepts a zero epsilon as the degenerate clip at ratio one', () => {
  assert.equal(ppoClippedObjective(1.4, 1, 0), 1);
  assert.equal(ppoClippedObjective(0.6, -1, 0), -1);
});

test('PPO clipping rejects negative or non-finite epsilon', () => {
  for (const epsilon of [-0.2, NaN, Infinity, -Infinity]) {
    assert.throws(() => ppoClippedObjective(1, 1, epsilon), RangeError);
  }
});

test('PPO clipping rejects negative ratios and non-finite numeric inputs', () => {
  assert.throws(() => ppoClippedObjective(-0.1, 1, 0.2), RangeError);
  for (const value of [NaN, Infinity, -Infinity]) {
    assert.throws(() => ppoClippedObjective(value, 1, 0.2), RangeError);
    assert.throws(() => ppoClippedObjective(1, value, 0.2), RangeError);
  }
});
