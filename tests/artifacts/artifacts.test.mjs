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
const geometryHtmlPath = join(bookDir, 'embedding-geometry.html');
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

function readGeometryHtml() {
  assert.ok(existsSync(geometryHtmlPath), `${geometryHtmlPath} does not exist`);
  return readFileSync(geometryHtmlPath, 'utf8');
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

test('Embedding geometry chapter HTML is published', () => {
  readGeometryHtml();
});

test('Embedding geometry chapter contains required learning content', () => {
  const html = readGeometryHtml();
  for (const text of [
    '向量、矩阵与 Embedding 几何',
    '内积',
    '投影',
    'SVD',
    '常见误解',
    '练习',
  ]) {
    assert.ok(html.includes(text), `Chapter is missing required text: ${text}`);
  }
});

test('Embedding geometry chapter references all static figures', () => {
  const html = readGeometryHtml();
  for (const figure of [
    'assets/figures/vector-projection.svg',
    'assets/figures/matrix-transform.svg',
    'assets/figures/embedding-space.svg',
  ]) {
    assert.ok(html.includes(figure), `Chapter does not reference ${figure}`);
  }
});

test('Embedding geometry chapter exposes stable section and figure anchors', () => {
  const html = readGeometryHtml();
  for (const anchor of [
    'sec-dot-product',
    'sec-matrix-transform',
    'sec-svd',
    'sec-embeddings',
    'fig-vector-projection',
    'fig-matrix-transform',
    'fig-embedding-space',
  ]) {
    assert.ok(html.includes(`id="${anchor}"`), `Chapter is missing #${anchor}`);
  }
});

test('Embedding geometry chapter cites the three required sources', () => {
  const html = readGeometryHtml();
  const citationKeys = new Set(
    [...html.matchAll(/data-cites="([^"]*)"/g)].flatMap((match) =>
      match[1].split(/\s+/),
    ),
  );
  for (const key of [
    'axler2015linear',
    'mikolov2013efficient',
    'vaswani2017attention',
  ]) {
    assert.ok(citationKeys.has(key), `Chapter does not cite ${key}`);
  }
});

test('Embedding geometry chapter has no unresolved citation markers', () => {
  assert.doesNotMatch(readGeometryHtml(), /\[\?\]/);
});

test('Embedding geometry chapter mounts both interactions and loads the shared module once', () => {
  const html = readGeometryHtml();
  for (const tagName of ['llm-vector-projector', 'llm-matrix-transformer']) {
    assert.equal(
      [...html.matchAll(new RegExp(`<${tagName}\\b`, 'g'))].length,
      1,
      `Expected exactly one <${tagName}> mount`,
    );
    assert.doesNotMatch(html, new RegExp(`<p>\\s*<${tagName}\\b`));
  }
  assert.equal(
    [...html.matchAll(/<script type="module">import '\.\/assets\/js\/book\.js';<\/script>/g)].length,
    1,
    'Expected exactly one shared book.js module import',
  );
});

test('Typst PDF contains geometry formulas, figures, exercises, and resolved references', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  for (const requiredText of [
    '向量、矩阵与 Embedding 几何',
    '投影',
    'SVD',
    '向量投影：正交分解',
    '矩阵变换：单位方块的像',
    '低秩子空间与 Embedding 几何',
    '基本运算',
    '手算投影',
    '截断 SVD',
    '检查 Q/K/V 形状',
  ]) {
    assert.ok(text.includes(requiredText), `PDF is missing text: ${requiredText}`);
  }
  for (const formula of [
    /𝑎\s*⋅\s*𝑏\s*=\s*𝑎𝘛\s*𝑏/,
    /proj𝑏\s*\(𝑎\)\s*=/,
    /𝐴\s*=\s*𝑈\s*Σ𝑉\s*𝘛/,
    /Attention\s*\(𝑄,\s*𝐾,\s*𝑉\s*\)/,
    /𝑄𝐾\s*𝘛/,
    /=\s*softmax/,
    /√𝑑𝑘/,
  ]) {
    assert.match(text, formula);
  }
  assert.match(text, /3\.9\s+练习/);
  assert.doesNotMatch(text, /\[\?\]|@(?:fig|sec)-[a-z0-9-]+/);
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
  for (const path of [htmlPath, chapterHtmlPath, geometryHtmlPath]) {
    assert.ok(existsSync(path), `${path} does not exist`);
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

function readMatrixGridLines() {
  const svg = readFileSync(join(projectRoot, 'assets/figures/matrix-transform.svg'), 'utf8');
  const groups = [...svg.matchAll(
    /<g stroke="#DDE3E7" stroke-width="1\.2">([\s\S]*?)<\/g>/g,
  )];
  assert.equal(groups.length, 2);

  return groups.map((group) =>
    [...group[1].matchAll(
      /<line x1="([0-9.]+)" y1="([0-9.]+)" x2="([0-9.]+)" y2="([0-9.]+)"\/>/g,
    )].map((match) => match.slice(1).map(Number)),
  );
}

function transformOriginalGridPoint([x, y]) {
  const mathematicalX = (x - 100) / 70;
  const mathematicalY = (500 - y) / 70;
  return [
    680 + 70 * (mathematicalX + mathematicalY),
    500 - 70 * mathematicalY,
  ];
}

test('Matrix figure maps every original grid endpoint with A', () => {
  const [original, transformed] = readMatrixGridLines();
  assert.equal(original.length, 10);

  const expected = original.map(([x1, y1, x2, y2]) => [
    ...transformOriginalGridPoint([x1, y1]),
    ...transformOriginalGridPoint([x2, y2]),
  ]);
  assert.deepEqual(transformed, expected);
});

test('Matrix figure keeps transformed vertical grid lines collinear with Ae2', () => {
  const svg = readFileSync(join(projectRoot, 'assets/figures/matrix-transform.svg'), 'utf8');
  const [original, transformed] = readMatrixGridLines();
  const ae2Match = svg.match(
    /<line x1="680" y1="500" x2="([0-9.]+)" y2="([0-9.]+)" stroke="#B55224"/,
  );
  assert.ok(ae2Match, 'Transformed Ae2 edge is missing');
  const ae2 = [Number(ae2Match[1]) - 680, Number(ae2Match[2]) - 500];
  assert.deepEqual(ae2, [70, -70]);

  const verticalIndexes = original.flatMap((line, index) =>
    line[0] === line[2] ? [index] : [],
  );
  assert.equal(verticalIndexes.length, 5);
  for (const index of verticalIndexes) {
    const [x1, y1, x2, y2] = transformed[index];
    const direction = [x2 - x1, y2 - y1];
    assert.equal(
      direction[0] * ae2[1] - direction[1] * ae2[0],
      0,
      `Grid line ${index} is not collinear with Ae2`,
    );
  }
});

test('Typst PDF uses Chinese chapter labels and keeps all static figures', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });

  assert.match(text, /章节\s+1\.\s+理解大语言模型/);
  assert.match(text, /章节\s+2\.\s+从 Softmax 到交叉熵/);
  assert.match(text, /章节\s+3\.\s+向量、矩阵与 Embedding 几何/);
  assert.doesNotMatch(text, /Chapter\s+[123]\./);
  for (const figureText of [
    '从上下文到下一 Token 分布',
    '交叉熵：L(p) = −ln p',
    '两组条形共用同一概率标尺',
  ]) {
    assert.ok(text.includes(figureText), `PDF is missing figure text: ${figureText}`);
  }
});
