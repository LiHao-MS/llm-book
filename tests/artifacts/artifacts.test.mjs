import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import {
  existsSync,
  readFileSync,
  readdirSync,
  statSync,
} from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const projectRoot = fileURLToPath(new URL('../../', import.meta.url));
const bookDir = join(projectRoot, '_book');
const htmlPath = join(bookDir, 'index.html');
const chapterHtmlPath = join(bookDir, 'softmax-cross-entropy.html');
const pdfPath = join(bookDir, 'llm-book.pdf');

function directorySize(directory) {
  return readdirSync(directory, { withFileTypes: true }).reduce((total, entry) => {
    const entryPath = join(directory, entry.name);
    if (entry.isDirectory()) {
      return total + directorySize(entryPath);
    }
    return entry.isFile() ? total + statSync(entryPath).size : total;
  }, 0);
}

test('HTML home page contains the book title', () => {
  assert.ok(existsSync(htmlPath), `${htmlPath} does not exist`);
  assert.match(readFileSync(htmlPath, 'utf8'), /理解大语言模型/);
});

function readChapterHtml() {
  assert.ok(existsSync(chapterHtmlPath), `${chapterHtmlPath} does not exist`);
  return readFileSync(chapterHtmlPath, 'utf8');
}

test('Softmax chapter HTML is published', () => {
  readChapterHtml();
});

test('Softmax chapter contains required learning content', () => {
  const html = readChapterHtml();
  for (const text of ['从 Softmax 到交叉熵', 'Teacher Forcing', '常见误解', '练习']) {
    assert.ok(html.includes(text), `Chapter is missing required text: ${text}`);
  }
});

test('Softmax chapter references all static figures', () => {
  const html = readChapterHtml();
  for (const figure of [
    'assets/figures/next-token-pipeline.svg',
    'assets/figures/cross-entropy-curve.svg',
    'assets/figures/nucleus-sampling.svg',
  ]) {
    assert.ok(html.includes(figure), `Chapter does not reference ${figure}`);
  }
});

test('Softmax chapter exposes stable section and figure anchors', () => {
  const html = readChapterHtml();
  for (const anchor of [
    'sec-softmax',
    'sec-cross-entropy',
    'sec-decoding',
    'fig-next-token',
    'fig-cross-entropy',
    'fig-top-p',
  ]) {
    assert.ok(html.includes(`id="${anchor}"`), `Chapter is missing #${anchor}`);
  }
});

test('Softmax chapter cites the three required sources', () => {
  const html = readChapterHtml();
  const citationKeys = new Set(
    [...html.matchAll(/data-cites="([^"]*)"/g)].flatMap((match) =>
      match[1].split(/\s+/),
    ),
  );
  for (const key of [
    'bengio2003neural',
    'goodfellow2016deep',
    'holtzman2019curious',
  ]) {
    assert.ok(citationKeys.has(key), `Chapter does not cite ${key}`);
  }
});

test('Softmax chapter has no unresolved citation markers', () => {
  assert.doesNotMatch(readChapterHtml(), /\[\?\]/);
});

test('Typst PDF is present and non-trivial', () => {
  assert.ok(existsSync(pdfPath), `${pdfPath} does not exist`);
  assert.ok(
    statSync(pdfPath).size > 100 * 1024,
    'Expected the PDF to be larger than 100 KB',
  );
  assert.equal(readFileSync(pdfPath).subarray(0, 5).toString('ascii'), '%PDF-');
});

test('published site stays below the GitHub Pages 1 GB limit', () => {
  assert.ok(existsSync(bookDir), `${bookDir} does not exist`);
  const size = directorySize(bookDir);
  assert.ok(size < 1024 ** 3, `Site size ${size} bytes exceeds the 1 GB limit`);
});

test('emitted HTML has no third-party runtime dependencies', () => {
  const runtimePattern = /<(?:script|link|img|iframe|source|embed|object|video|audio|track|input)\b[^>]*\b(?:src|href|data|poster)\s*=\s*["'](?:https?:)?\/\//gi;
  const cssPattern = /(?:@import\s+(?:url\()?|url\()\s*["']?(?:https?:)?\/\//gi;
  for (const path of [htmlPath, chapterHtmlPath]) {
    const html = readFileSync(path, 'utf8');
    assert.deepEqual([...html.matchAll(runtimePattern)].map((match) => match[0]), []);
    assert.deepEqual([...html.matchAll(cssPattern)].map((match) => match[0]), []);
  }
});

test('Top-p figure uses one shared scale for original and normalized bars', () => {
  const svg = readFileSync(join(projectRoot, 'assets/figures/nucleus-sampling.svg'), 'utf8');
  const bars = [...svg.matchAll(
    /<rect x="(120|760)" y="(?:175|245|315)" width="([0-9.]+)"/g,
  )];
  assert.equal(bars.length, 6);

  for (const [x, probabilities] of [
    ['120', [0.45, 0.30, 0.15]],
    ['760', [0.5, 1 / 3, 1 / 6]],
  ]) {
    const widths = bars
      .filter((match) => match[1] === x)
      .map((match) => Number(match[2]));
    probabilities.forEach((probability, index) => {
      assert.ok(
        Math.abs(widths[index] - probability * 500) < 0.01,
        `${x} bar ${index + 1} does not use the shared 500-unit scale`,
      );
    });
  }
  assert.match(svg, /两组条形共用同一概率标尺/);
});

test('Typst PDF uses Chinese chapter labels and keeps all static figures', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });

  assert.match(text, /章节\s+1\.\s+理解大语言模型/);
  assert.match(text, /章节\s+2\.\s+从 Softmax 到交叉熵/);
  assert.doesNotMatch(text, /Chapter\s+[12]\./);
  for (const figureText of [
    '从上下文到下一 Token 分布',
    '交叉熵：L(p) = −ln p',
    '两组条形共用同一概率标尺',
  ]) {
    assert.ok(text.includes(figureText), `PDF is missing figure text: ${figureText}`);
  }
});
