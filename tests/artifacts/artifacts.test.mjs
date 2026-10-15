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
const optimizationHtmlPath = join(bookDir, 'optimization.html');
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

function readOptimizationHtml() {
  assert.ok(existsSync(optimizationHtmlPath), `${optimizationHtmlPath} does not exist`);
  return readFileSync(optimizationHtmlPath, 'utf8');
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

test('Optimization chapter HTML is published with required learning content', () => {
  const html = readOptimizationHtml();
  for (const text of [
    '优化：从经验风险到 AdamW',
    '经验风险',
    'Momentum',
    'AdamW',
    '梯度裁剪',
    '权重衰减',
  ]) {
    assert.ok(html.includes(text), `Chapter is missing required text: ${text}`);
  }
});

test('Optimization chapter references all static figures', () => {
  const html = readOptimizationHtml();
  for (const figure of [
    'assets/figures/loss-surface.svg',
    'assets/figures/optimizer-trajectory.svg',
    'assets/figures/lr-schedule.svg',
  ]) {
    assert.ok(html.includes(figure), `Chapter does not reference ${figure}`);
  }
});

test('Optimization chapter exposes stable section and figure anchors', () => {
  const html = readOptimizationHtml();
  for (const anchor of [
    'sec-empirical-risk',
    'sec-sgd',
    'sec-momentum',
    'sec-adam',
    'sec-schedule',
    'sec-adamw',
    'fig-loss-surface',
    'fig-optimizer-trajectory',
    'fig-lr-schedule',
  ]) {
    assert.ok(html.includes(`id="${anchor}"`), `Chapter is missing #${anchor}`);
  }
});

test('Optimization chapter cites the three required sources', () => {
  const html = readOptimizationHtml();
  const citationKeys = new Set(
    [...html.matchAll(/data-cites="([^"]*)"/g)].flatMap((match) =>
      match[1].split(/\s+/),
    ),
  );
  for (const key of [
    'kingma2014adam',
    'loshchilov2017decoupled',
    'goodfellow2016deep',
  ]) {
    assert.ok(citationKeys.has(key), `Chapter does not cite ${key}`);
  }
});

test('Optimization chapter has no unresolved citation markers', () => {
  assert.doesNotMatch(readOptimizationHtml(), /\[\?\]/);
});

test('Typst PDF contains optimization chapter and key optimization terms', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  for (const requiredText of [
    '优化：从经验风险到 AdamW',
    'Momentum',
    'AdamW',
    '梯度裁剪',
    'decoupled weight decay',
  ]) {
    assert.ok(text.includes(requiredText), `PDF is missing text: ${requiredText}`);
  }
});

test('Typst PDF contains optimization formulas, figures, exercises, and resolved references', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  for (const requiredText of [
    '优化：从经验风险到 AdamW',
    '二维二次损失的等高线、负梯度方向与学习率缩放后的实际更新',
    'Warmup、cosine 学习率与全局范数裁剪',
    '同一二维二次损失上 SGD、Momentum、Adam 和 AdamW 的十步优化轨迹',
    '计算经验风险',
    '手算 SGD',
    '展开 Momentum',
    '核对 Adam 偏差校正',
    '代入学习率 schedule',
    '执行全局裁剪',
    '区分 L2 与 AdamW',
    '检查混合精度尺度',
    '设计数值检查',
  ]) {
    assert.ok(text.includes(requiredText), `PDF is missing text: ${requiredText}`);
  }
  for (const formula of [
    /ℓ𝑖\s*\(𝜃\)\s*=\s*ℓ\(𝑓𝜃/,
    /𝜃𝑡\+1\s*=\s*𝜃𝑡\s*−\s*𝜂𝑡\s*𝑔𝑡/,
    /𝑣𝑡\s*=\s*𝛽𝑣𝑡−1\s*\+\s*𝑔𝑡/,
    /𝑚𝑡\s*=\s*𝛽1\s*𝑚𝑡−1\s*\+\s*\(1 − 𝛽1 \)𝑔𝑡/,
    /𝜂\(55\)\s*=\s*0\.55/,
    /𝑔̃\s*=\s*𝑐𝑔/,
    /𝜃\+\s*=\s*\(1 − 0\.01\)𝑎\s*=\s*\(0\.891, 1\.089\)/,
  ]) {
    assert.match(text, formula);
  }
  assert.match(text, /4\.12\s+练习/);
  assert.doesNotMatch(text, /\[\?\]|@(?:fig|sec)-[a-z0-9-]+/);
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
  for (const path of [htmlPath, chapterHtmlPath, geometryHtmlPath, optimizationHtmlPath]) {
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

function relativeLuminance(hexColor) {
  assert.match(hexColor, /^#[0-9A-Fa-f]{6}$/);
  const channels = hexColor
    .match(/[0-9A-Fa-f]{2}/g)
    .map((channel) => Number.parseInt(channel, 16) / 255)
    .map((channel) => channel <= 0.04045
      ? channel / 12.92
      : ((channel + 0.055) / 1.055) ** 2.4);
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function assertMinimumContrast(foreground, background, minimum, context) {
  const [lighter, darker] = [relativeLuminance(foreground), relativeLuminance(background)]
    .sort((a, b) => b - a);
  const ratio = (lighter + 0.05) / (darker + 0.05);
  assert.ok(
    ratio >= minimum,
    `${context} contrast ${ratio.toFixed(2)}:1 is below ${minimum}:1 (${foreground} on ${background})`,
  );
}

function svgElements(svg, tagName) {
  return [...svg.matchAll(new RegExp(`<${tagName}\\b[^>]*>`, 'g'))].map((match) =>
    Object.fromEntries(
      [...match[0].matchAll(/([a-zA-Z_:][\w:.-]*)="([^"]*)"/g)]
        .map((attribute) => attribute.slice(1)),
    ),
  );
}

test('optimization figure graphics and labels meet static contrast thresholds', () => {
  const trajectory = readFileSync(
    join(projectRoot, 'assets/figures/optimizer-trajectory.svg'),
    'utf8',
  );
  const trajectoryGroups = svgElements(trajectory, 'g');
  const trajectoryContours = trajectoryGroups.find((attributes) =>
    attributes['clip-path'] === 'url(#trajectory-clip)' && attributes.stroke,
  );
  assert.ok(trajectoryContours, 'Trajectory contour group is missing');
  assertMinimumContrast(trajectoryContours.stroke, '#F8FAFB', 3, 'Static contours');

  const contourLabel = svgElements(trajectory, 'text').find((attributes) =>
    attributes.x === '500' && attributes.y === '535',
  );
  assert.ok(contourLabel, 'Trajectory contour label is missing');
  assertMinimumContrast(contourLabel.fill, '#F8FAFB', 4.5, 'Contour label');

  const paths = svgElements(trajectory, 'path').filter((attributes) =>
    /url\(#(?:sgd|momentum|adam|adamw)-arrow\)/.test(attributes['marker-end'] ?? ''),
  );
  assert.equal(paths.length, 4);
  for (const path of paths) {
    assertMinimumContrast(path.stroke, '#F8FAFB', 3, `Trajectory ${path['marker-end']}`);
  }

  const lossSurface = readFileSync(join(projectRoot, 'assets/figures/loss-surface.svg'), 'utf8');
  const lossContours = svgElements(lossSurface, 'g').find((attributes) =>
    attributes['clip-path'] === 'url(#plot-clip)' && attributes.stroke,
  );
  assert.ok(lossContours, 'Loss-surface contour group is missing');
  assertMinimumContrast(lossContours.stroke, '#F8FAFB', 3, 'Loss-surface contours');

  const scss = readFileSync(join(projectRoot, 'styles/site.scss'), 'utf8');
  for (const selector of ['optimizer-contour', 'optimizer-full-trajectory']) {
    const declaration = scss.match(
      new RegExp(`\\.${selector}\\s*\\{[^}]*?stroke:\\s*(#[0-9A-Fa-f]{6})`),
    );
    assert.ok(declaration, `Missing interactive stroke for .${selector}`);
    assertMinimumContrast(declaration[1], '#FFFFFF', 3, `Interactive .${selector}`);
  }
});

test('static optimizer paths preserve geometry and mirror distinct dash patterns in the legend', () => {
  const svg = readFileSync(join(projectRoot, 'assets/figures/optimizer-trajectory.svg'), 'utf8');
  const paths = svgElements(svg, 'path');
  const legendLines = svgElements(svg, 'line');
  const patterns = [];

  for (const algorithm of ['sgd', 'momentum', 'adam', 'adamw']) {
    const path = paths.find((attributes) => attributes['data-algorithm'] === algorithm);
    const legend = legendLines.find((attributes) => attributes['data-algorithm'] === algorithm);
    assert.ok(path, `Missing ${algorithm} trajectory`);
    assert.ok(legend, `Missing ${algorithm} legend swatch`);
    const pathPattern = path['stroke-dasharray'] ?? 'solid';
    const legendPattern = legend['stroke-dasharray'] ?? 'solid';
    assert.equal(legendPattern, pathPattern, `${algorithm} legend pattern differs from its path`);
    assertMinimumContrast(legend.stroke, '#FBF8F3', 3, `${algorithm} legend swatch`);
    patterns.push(pathPattern);
  }
  assert.equal(new Set(patterns).size, 4, 'Every optimizer needs a distinct line pattern');

  const sgd = paths.find((attributes) => attributes['data-algorithm'] === 'sgd');
  assert.ok(
    sgd.d.startsWith('M 570.00 260.00 L 465.00 330.00'),
    `SGD first segment changed: ${sgd.d}`,
  );
});

test('matrix figure preserves the complete unit-square polygon set', () => {
  const svg = readFileSync(join(projectRoot, 'assets/figures/matrix-transform.svg'), 'utf8');
  const polygons = svgElements(svg, 'polygon').map((attributes) =>
    attributes.points.trim().split(/\s+/).map((point) => point.split(',').map(Number)),
  );
  assert.deepEqual(polygons, [
    [[100, 500], [170, 500], [170, 430], [100, 430]],
    [[680, 500], [750, 500], [820, 430], [750, 430]],
  ]);
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
  assert.match(text, /章节\s+4\.\s+优化：从经验风险到 AdamW/);
  assert.doesNotMatch(text, /Chapter\s+[1234]\./);
  for (const figureText of [
    '从上下文到下一 Token 分布',
    '交叉熵：L(p) = −ln p',
    '两组条形共用同一概率标尺',
    '二维二次损失的等高线、负梯度方向与学习率缩放后的实际更新',
    'Warmup、cosine 学习率与全局范数裁剪',
    '同一二维二次损失上 SGD、Momentum、Adam 和 AdamW 的十步优化轨迹',
  ]) {
    assert.ok(text.includes(figureText), `PDF is missing figure text: ${figureText}`);
  }
});
