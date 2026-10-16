import test from 'node:test';
import assert from 'node:assert/strict';

import { estimateStaticStates } from '../../assets/js/components/distributed-training.js';

const toyModel = {
  parameters: 1_000_000_000,
  layers: 12,
};

function configuration(overrides = {}) {
  return {
    ...toyModel,
    dataParallel: 1,
    tensorParallel: 1,
    pipelineParallel: 1,
    zeroStage: 0,
    ...overrides,
  };
}

test('data parallel replication keeps all 16 decimal GB of static states on each device', () => {
  assert.deepEqual(
    estimateStaticStates(configuration({ dataParallel: 8 })),
    {
      weightBytes: 2_000_000_000,
      gradientBytes: 2_000_000_000,
      optimizerBytes: 12_000_000_000,
      totalBytes: 16_000_000_000,
    },
  );
});

test('TP, PP, and ZeRO-3 divide the toy state account to 2 decimal GB per device', () => {
  assert.deepEqual(
    estimateStaticStates(configuration({
      dataParallel: 2,
      tensorParallel: 2,
      pipelineParallel: 2,
      zeroStage: 3,
    })),
    {
      weightBytes: 250_000_000,
      gradientBytes: 250_000_000,
      optimizerBytes: 1_500_000_000,
      totalBytes: 2_000_000_000,
    },
  );
});

test('ZeRO stages shard optimizer, gradients, and weights cumulatively', () => {
  const expected = [
    { weightBytes: 2e9, gradientBytes: 2e9, optimizerBytes: 12e9, totalBytes: 16e9 },
    { weightBytes: 2e9, gradientBytes: 2e9, optimizerBytes: 3e9, totalBytes: 7e9 },
    { weightBytes: 2e9, gradientBytes: 0.5e9, optimizerBytes: 3e9, totalBytes: 5.5e9 },
    { weightBytes: 0.5e9, gradientBytes: 0.5e9, optimizerBytes: 3e9, totalBytes: 4e9 },
  ];

  for (const zeroStage of [0, 1, 2, 3]) {
    assert.deepEqual(
      estimateStaticStates(configuration({ dataParallel: 4, zeroStage })),
      expected[zeroStage],
    );
  }
});

test('non-uniform pipeline and parameter partitions are rejected', () => {
  assert.throws(
    () => estimateStaticStates(configuration({ pipelineParallel: 5 })),
    RangeError,
  );
  assert.throws(
    () => estimateStaticStates(configuration({ parameters: 12, tensorParallel: 5 })),
    RangeError,
  );
  assert.throws(
    () => estimateStaticStates(configuration({ parameters: 12, dataParallel: 5, zeroStage: 1 })),
    RangeError,
  );
});

test('invalid model sizes and parallel degrees cannot produce NaN or negative states', () => {
  for (const field of ['parameters', 'layers']) {
    for (const value of [0, -1, 1.5, NaN, Infinity, -Infinity]) {
      assert.throws(() => estimateStaticStates(configuration({ [field]: value })), RangeError);
    }
  }

  for (const field of ['dataParallel', 'tensorParallel', 'pipelineParallel']) {
    for (const value of [0, -2, 1.5, NaN, Infinity, -Infinity]) {
      assert.throws(() => estimateStaticStates(configuration({ [field]: value })), RangeError);
    }
  }

  for (const zeroStage of [-1, 4, 1.5, NaN, Infinity, -Infinity]) {
    assert.throws(() => estimateStaticStates(configuration({ zeroStage })), RangeError);
  }
});

test('large exact accounts divide parameter shares before multiplying bytes', () => {
  assert.deepEqual(
    estimateStaticStates(configuration({
      parameters: 9_007_199_254_740_986,
      tensorParallel: 26,
    })),
    {
      weightBytes: 692_861_481_133_922,
      gradientBytes: 692_861_481_133_922,
      optimizerBytes: 4_157_168_886_803_532,
      totalBytes: 5_542_891_849_071_376,
    },
  );
});

test('unsafe state components and totals are rejected', () => {
  assert.throws(
    () => estimateStaticStates(configuration({ parameters: Number.MAX_SAFE_INTEGER })),
    RangeError,
  );
  assert.throws(
    () => estimateStaticStates(configuration({ parameters: 562_949_953_421_312 })),
    RangeError,
  );
});
