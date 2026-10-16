import test from 'node:test';
import assert from 'node:assert/strict';

import { computePatchLayout } from '../../assets/js/components/multimodal-patch.js';

test('224 by 224 images produce the fixed visual and attention accounts', () => {
  const cases = [
    {
      patchSize: 14,
      expected: {
        patchRows: 16,
        patchColumns: 16,
        visualTokens: 256,
        sequenceLength: 276,
        attentionScores: 76176,
      },
    },
    {
      patchSize: 16,
      expected: {
        patchRows: 14,
        patchColumns: 14,
        visualTokens: 196,
        sequenceLength: 216,
        attentionScores: 46656,
      },
    },
    {
      patchSize: 32,
      expected: {
        patchRows: 7,
        patchColumns: 7,
        visualTokens: 49,
        sequenceLength: 69,
        attentionScores: 4761,
      },
    },
  ];

  for (const { patchSize, expected } of cases) {
    assert.deepEqual(
      computePatchLayout({ width: 224, height: 224, patchSize, textTokens: 20 }),
      expected,
      `patchSize=${patchSize}`,
    );
  }
});

test('partial patches are padded on the right and bottom and counted with ceil', () => {
  assert.deepEqual(
    computePatchLayout({ width: 225, height: 100, patchSize: 32, textTokens: 20 }),
    {
      patchRows: 4,
      patchColumns: 8,
      visualTokens: 32,
      sequenceLength: 52,
      attentionScores: 2704,
    },
  );
});

test('zero text tokens are valid and contribute only to the sequence account', () => {
  assert.deepEqual(
    computePatchLayout({ width: 32, height: 64, patchSize: 32, textTokens: 0 }),
    {
      patchRows: 2,
      patchColumns: 1,
      visualTokens: 2,
      sequenceLength: 2,
      attentionScores: 4,
    },
  );
});

test('non-positive or non-integer image dimensions and patch sizes are rejected', () => {
  for (const field of ['width', 'height', 'patchSize']) {
    for (const value of [0, -1, 1.5, NaN, Infinity, -Infinity]) {
      assert.throws(
        () => computePatchLayout({
          width: 224,
          height: 224,
          patchSize: 16,
          textTokens: 20,
          [field]: value,
        }),
        RangeError,
        `${field}=${value}`,
      );
    }
  }
});

test('text token counts must be non-negative safe integers', () => {
  for (const textTokens of [-1, 1.5, NaN, Infinity, -Infinity]) {
    assert.throws(
      () => computePatchLayout({ width: 224, height: 224, patchSize: 16, textTokens }),
      RangeError,
      `textTokens=${textTokens}`,
    );
  }
});

test('accounts that exceed the exact safe-integer range are rejected', () => {
  assert.throws(
    () => computePatchLayout({
      width: Number.MAX_SAFE_INTEGER,
      height: Number.MAX_SAFE_INTEGER,
      patchSize: 1,
      textTokens: 20,
    }),
    RangeError,
  );
  assert.throws(
    () => computePatchLayout({
      width: Number.MAX_SAFE_INTEGER,
      height: 1,
      patchSize: 1,
      textTokens: 20,
    }),
    RangeError,
  );
  assert.throws(
    () => computePatchLayout({ width: 100_000_000, height: 1, patchSize: 1, textTokens: 0 }),
    RangeError,
  );
});
