import test from 'node:test';
import assert from 'node:assert/strict';

import {
  rankDocuments,
  selectContext,
} from '../../assets/js/components/retrieval-rag.js';

const TOLERANCE = 1e-12;

function assertClose(actual, expected) {
  assert.ok(Number.isFinite(actual), `Expected a finite score, got ${actual}`);
  assert.ok(
    Math.abs(actual - expected) <= TOLERANCE,
    `Expected ${actual} to be within ${TOLERANCE} of ${expected}`,
  );
}

function retrievalDocuments() {
  return [
    {
      id: 'long-off-axis',
      vector: [10, 1],
      text: '长且偏离查询方向的文档',
      tokens: 8,
    },
    {
      id: 'short-aligned',
      vector: [1, 0],
      text: '短且与查询方向对齐的文档',
      tokens: 3,
    },
  ];
}

test('dot product favors the long off-axis document while cosine favors the aligned one', () => {
  const query = [1, 0];
  const documents = retrievalDocuments();

  const byDot = rankDocuments(query, documents, 'dot');
  assert.deepEqual(byDot.map((document) => document.id), ['long-off-axis', 'short-aligned']);
  assert.deepEqual(byDot.map((document) => document.score), [10, 1]);
  assert.deepEqual(byDot.map((document) => document.rank), [1, 2]);

  const byCosine = rankDocuments(query, documents, 'cosine');
  assert.deepEqual(byCosine.map((document) => document.id), ['short-aligned', 'long-off-axis']);
  assertClose(byCosine[0].score, 1);
  assertClose(byCosine[1].score, 10 / Math.sqrt(101));
  assert.deepEqual(byCosine.map((document) => document.rank), [1, 2]);
});

test('ranking preserves raw scores and returns new objects without mutating caller documents', () => {
  const documents = retrievalDocuments().map((document) => Object.freeze({
    ...document,
    vector: Object.freeze([...document.vector]),
  }));
  const originalOrder = documents.map((document) => document.id);
  const frozenDocuments = Object.freeze(documents);

  const ranked = rankDocuments([1, 0], frozenDocuments, 'dot');

  assert.notEqual(ranked, frozenDocuments);
  assert.notEqual(ranked[0], documents[0]);
  assert.equal(ranked[0].score, 10);
  assert.equal(ranked[0].vector, documents[0].vector);
  assert.deepEqual(documents.map((document) => document.id), originalOrder);
  assert.deepEqual(documents.map((document) => document.score), [undefined, undefined]);
  assert.deepEqual(documents.map((document) => document.rank), [undefined, undefined]);
});

test('equal scores use input order as a stable tie-break and ranks are one-based positions', () => {
  const documents = ['first', 'second', 'third'].map((id, index) => ({
    id,
    vector: [index + 1, 0],
    text: id,
    tokens: index + 1,
  }));

  const ranked = rankDocuments([1, 0], documents, 'cosine');

  assert.deepEqual(ranked.map((document) => document.id), ['first', 'second', 'third']);
  assert.deepEqual(ranked.map((document) => document.score), [1, 1, 1]);
  assert.deepEqual(ranked.map((document) => document.rank), [1, 2, 3]);
});

test('ranking accepts an empty document collection but rejects an empty query', () => {
  assert.deepEqual(rankDocuments([1, 0], [], 'dot'), []);
  assert.throws(() => rankDocuments([], retrievalDocuments(), 'dot'), RangeError);
});

test('ranking rejects non-finite values, invalid metrics, and undefined cosine scores', () => {
  for (const value of [NaN, Infinity, -Infinity]) {
    assert.throws(() => rankDocuments([value, 0], retrievalDocuments(), 'dot'), RangeError);
    assert.throws(() => rankDocuments([1, 0], [{
      id: 'invalid', vector: [value, 0], text: 'invalid', tokens: 1,
    }], 'dot'), RangeError);
  }
  assert.throws(() => rankDocuments([1, 0], retrievalDocuments(), 'probability'), RangeError);
  assert.throws(() => rankDocuments([0, 0], retrievalDocuments(), 'cosine'), RangeError);
  assert.throws(() => rankDocuments([1, 0], [{
    id: 'zero', vector: [0, 0], text: 'zero', tokens: 1,
  }], 'cosine'), RangeError);
});

test('context selection keeps complete documents in ranked order and skips an oversized document', () => {
  const ranked = [
    { id: 'a', vector: [1, 0], text: 'AAAAA', tokens: 5, score: 4, rank: 1 },
    { id: 'b', vector: [0, 1], text: 'BBBBBBB', tokens: 7, score: 3, rank: 2 },
    { id: 'c', vector: [1, 1], text: 'CCC', tokens: 3, score: 2, rank: 3 },
  ];
  const frozen = Object.freeze(ranked.map((document) => Object.freeze(document)));

  const selected = selectContext(frozen, 8);

  assert.notEqual(selected, frozen);
  assert.deepEqual(selected.map((document) => document.id), ['a', 'c']);
  assert.equal(selected.reduce((total, document) => total + document.tokens, 0), 8);
  assert.deepEqual(frozen.map((document) => document.id), ['a', 'b', 'c']);
});

test('context selection admits a document that exactly consumes the remaining budget', () => {
  const ranked = [
    { id: 'a', vector: [1, 0], text: 'AAAAA', tokens: 5, score: 4, rank: 1 },
    { id: 'b', vector: [0, 1], text: 'BBBBBBB', tokens: 7, score: 3, rank: 2 },
    { id: 'c', vector: [1, 1], text: 'CCC', tokens: 3, score: 2, rank: 3 },
  ];

  assert.deepEqual(selectContext(ranked, 12).map((document) => document.id), ['a', 'b']);
  assert.deepEqual(selectContext(ranked, 15).map((document) => document.id), ['a', 'b', 'c']);
  assert.deepEqual(selectContext(ranked, 0), []);
  assert.deepEqual(selectContext([], 10), []);
});

test('context selection rejects non-finite or fractional budgets and token counts', () => {
  const ranked = [
    { id: 'a', vector: [1, 0], text: 'AAAAA', tokens: 5, score: 4, rank: 1 },
  ];

  for (const budget of [NaN, Infinity, -Infinity, -1, 1.5]) {
    assert.throws(() => selectContext(ranked, budget), RangeError);
  }
  for (const tokens of [NaN, Infinity, -Infinity, -1, 1.5]) {
    assert.throws(() => selectContext([{ ...ranked[0], tokens }], 10), RangeError);
  }
});
