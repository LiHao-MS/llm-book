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
const attentionHtmlPath = join(bookDir, 'attention.html');
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

test('HTML home page describes the current published scope', () => {
  const html = readFileSync(htmlPath, 'utf8');
  assert.match(html, /当前版本已发布 11 个章节页面/);
  assert.doesNotMatch(html, /纵向样板|不表示所有章节均已发布/);
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

function readAttentionHtml() {
  assert.ok(existsSync(attentionHtmlPath), `${attentionHtmlPath} does not exist`);
  return readFileSync(attentionHtmlPath, 'utf8');
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

test('Attention chapter HTML is published with required learning content', () => {
  const html = readAttentionHtml();
  for (const text of [
    '序列建模与自注意力机制',
    'RNN',
    '缩放点积注意力',
    '多头注意力',
    '因果掩码',
    'RoPE',
    '常见误解',
    '练习',
  ]) {
    assert.ok(html.includes(text), `Chapter is missing required text: ${text}`);
  }
});

test('Attention chapter references all static figures', () => {
  const html = readAttentionHtml();
  for (const figure of [
    'assets/figures/attention-scores.svg',
    'assets/figures/causal-mask.svg',
    'assets/figures/multi-head.svg',
    'assets/figures/rope-rotation.svg',
  ]) {
    assert.ok(html.includes(figure), `Chapter does not reference ${figure}`);
  }
});

test('Attention chapter exposes stable section and figure anchors', () => {
  const html = readAttentionHtml();
  for (const anchor of [
    'sec-rnn-limits',
    'sec-attention',
    'sec-scaled-attention',
    'sec-multi-head',
    'sec-causal-mask',
    'sec-rope',
    'fig-attention-scores',
    'fig-causal-mask',
    'fig-multi-head',
    'fig-rope-rotation',
  ]) {
    assert.ok(html.includes(`id="${anchor}"`), `Chapter is missing #${anchor}`);
  }
});

test('Attention chapter cites the four required sources', () => {
  const html = readAttentionHtml();
  const citationKeys = new Set(
    [...html.matchAll(/data-cites="([^"]*)"/g)].flatMap((match) =>
      match[1].split(/\s+/),
    ),
  );
  for (const key of [
    'bahdanau2014neural',
    'luong2015effective',
    'vaswani2017attention',
    'su2021roformer',
  ]) {
    assert.ok(citationKeys.has(key), `Chapter does not cite ${key}`);
  }
});

test('Attention chapter has no unresolved citation markers', () => {
  assert.doesNotMatch(readAttentionHtml(), /\[\?\]/);
});

test('Typst PDF contains attention chapter and key attention terms', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  for (const requiredText of [
    '序列建模与自注意力机制',
    'Softmax',
    'RoPE',
    '因果掩码',
  ]) {
    assert.ok(text.includes(requiredText), `PDF is missing text: ${requiredText}`);
  }
});

test('Typst PDF contains attention formulas, figures, exercises, and resolved references', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  const chapterStart = text.indexOf('5. 序列建模与自注意力机制');
  assert.ok(chapterStart >= 0, 'PDF is missing the attention chapter body');
  const chapterText = text.slice(chapterStart);

  for (const requiredText of [
    '第一个查询的缩放分数经 Softmax 变为行归一化权重，再对 Value 加权得到输出',
    '两个一维头从同一输入的不同坐标产生不同注意力矩阵，再按坐标拼接输出',
    '因果掩码把严格上三角分数设为负无穷，Softmax 后未来位置权重为零',
    'RoPE 将二维向量逆时针旋转并保持范数，Query 与 Key 的相对夹角由位置差决定',
    '分析递归路径',
    '手算值加权输出',
    '推导缩放尺度',
    '检查多头形状',
    '执行因果掩码',
    '区分两种掩码',
    '核对 RoPE 方向与范数',
    '推导相对位置',
    '反驳固定标签解释',
  ]) {
    assert.ok(chapterText.includes(requiredText), `PDF is missing text: ${requiredText}`);
  }
  for (const formula of [
    /ℎ𝑡\s*=\s*𝑓𝜃\s*\(𝑥𝑡\s*,\s*ℎ𝑡−1\s*\)/,
    /𝑜\s*=\s*∑\s*𝛼𝑗\s*𝑣𝑗/,
    /𝑄𝐾\s*𝘛/,
    /√𝑑𝑘/,
    /𝐴\s*=\s*softmaxrow\s*\(𝑆\)/,
    /𝑂\s*=\s*𝐴𝑉/,
    /MHA\s*\(𝑋\)\s*=\s*Concat𝐻/,
    /𝐴\s*=\s*softmaxrow\s*\(𝑆\s*\+\s*𝑀\s*\)/,
    /⟨𝑅\(𝑚𝜃\)𝑞,\s*𝑅\(𝑛𝜃\)𝑘⟩\s*=\s*𝑞\s*𝘛\s*𝑅\(𝑚𝜃\)𝘛\s*𝑅\(𝑛𝜃\)𝑘/,
    /=\s*𝑞\s*𝘛\s*𝑅\(\(𝑛\s*−\s*𝑚\)𝜃\)𝑘/,
  ]) {
    assert.match(chapterText, formula);
  }
  assert.match(chapterText, /5\.11\s+练习/);
  assert.doesNotMatch(chapterText, /\[\?\]|@(?:fig|sec)-[a-z0-9-]+/);
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
  for (const path of [
    htmlPath,
    chapterHtmlPath,
    geometryHtmlPath,
    optimizationHtmlPath,
    attentionHtmlPath,
    join(bookDir, 'transformer-architecture.html'),
    join(bookDir, 'pretraining-scaling.html'),
    join(bookDir, 'inference-systems.html'),
    join(bookDir, 'rl-foundations.html'),
    join(bookDir, 'rlhf-alignment.html'),
    join(bookDir, 'frontier-rl.html'),
  ]) {
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

test('causal mask figure separates the score-panel title from column labels', () => {
  const svg = readFileSync(join(projectRoot, 'assets/figures/causal-mask.svg'), 'utf8');
  const title = svg.match(/<text x="195" y="([0-9.]+)"[^>]*>原始分数 S<\/text>/);
  assert.ok(title, 'Score-panel title is missing');
  const labels = [...svg.matchAll(
    /<text x="(?:135|215|295)" y="([0-9.]+)">k[₁₂₃]<\/text>/g,
  )];
  assert.equal(labels.length, 3);
  for (const label of labels) {
    assert.ok(
      Number(label[1]) - Number(title[1]) >= 30,
      'Score-panel title leaves less than 30 SVG units before the column labels',
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
  assert.match(text, /章节\s+5\.\s+序列建模与自注意力机制/);
  assert.match(text, /章节\s+6\.\s+Transformer 架构与预训练/);
  assert.match(text, /章节\s+7\.\s+预训练与 Scaling Laws/);
  assert.match(text, /章节\s+8\.\s+推理与系统工程/);
  assert.match(text, /章节\s+9\.\s+强化学习基础：从 MDP 到 PPO/);
  assert.match(text, /章节\s+10\.\s+后训练、偏好优化与 RLHF/);
  assert.match(text, /章节\s+11\.\s+前沿 RL、Agentic RL 与开放问题/);
  assert.doesNotMatch(text, /Chapter\s+(?:[1-9]|10|11)\./);
  for (const figureText of [
    '从上下文到下一 Token 分布',
    '交叉熵：L(p) = −ln p',
    '两组条形共用同一概率标尺',
    '二维二次损失的等高线、负梯度方向与学习率缩放后的实际更新',
    'Warmup、cosine 学习率与全局范数裁剪',
    '同一二维二次损失上 SGD、Momentum、Adam 和 AdamW 的十步优化轨迹',
    '第一个查询的缩放分数经 Softmax 变为行归一化权重，再对 Value 加权得到输出',
    '两个一维头从同一输入的不同坐标产生不同注意力矩阵，再按坐标拼接输出',
    '因果掩码把严格上三角分数设为负无穷，Softmax 后未来位置权重为零',
    'RoPE 将二维向量逆时针旋转并保持范数，Query 与 Key 的相对夹角由位置差决定',
    'Pre-LN Transformer Block 数值图',
    '教学 Scaling Laws：系数仅用于演示，并非论文拟合值',
    '推理系统的显存层次、KV cache、连续批处理与成本换算',
    '两状态 MDP：价值迭代与贪心策略',
    '后训练流程与 RLHF／DPO 玩具目标对比',
    '四种回答在结果奖励、含漏洞的文本标记奖励和重算检查奖励下的总奖励与组内优势',
  ]) {
    assert.ok(text.includes(figureText), `PDF is missing figure text: ${figureText}`);
  }
});

function readChapterFile(filename) {
  return readFileSync(join(bookDir, filename), 'utf8');
}

function assertChapterIncludes(filename, anchors, texts) {
  const html = readChapterFile(filename);
  for (const anchor of anchors) {
    assert.ok(html.includes(`id="${anchor}"`), `${filename} is missing #${anchor}`);
  }
  for (const text of texts) {
    assert.ok(html.includes(text), `${filename} is missing required text: ${text}`);
  }
}

test('Transformer architecture chapter is complete', () => {
  assertChapterIncludes('transformer-architecture.html', [
    'sec-transformer-block',
    'sec-layernorm',
    'sec-ffn',
    'sec-position',
    'sec-decoder-only',
    'fig-transformer-block',
  ], ['Transformer', 'LayerNorm', 'FFN', '位置编码', '练习']);
});

test('Pretraining and scaling chapter is complete', () => {
  assertChapterIncludes('pretraining-scaling.html', [
    'sec-pretraining-objectives',
    'sec-data-quality',
    'sec-tokenization-pipeline',
    'sec-scaling-laws',
    'sec-moe',
    'sec-cost',
    'fig-pretraining-scaling',
  ], ['Scaling Laws', 'Kaplan', 'Chinchilla', 'MoE', '练习']);
});

test('Inference systems chapter is complete', () => {
  assertChapterIncludes('inference-systems.html', [
    'sec-inference-pipeline',
    'sec-gpu-memory',
    'sec-kv-cache',
    'sec-quantization',
    'sec-distillation',
    'sec-speculative-decoding',
    'sec-batching',
    'sec-inference-cost',
    'sec-serving',
    'fig-inference-systems',
  ], ['KV cache', '量化', '投机解码', '连续批处理', '练习']);
});

test('RL foundations chapter is complete', () => {
  assertChapterIncludes('rl-foundations.html', [
    'sec-mdp',
    'sec-return-value',
    'sec-bellman',
    'sec-monte-carlo',
    'sec-td',
    'sec-policy-gradient',
    'sec-actor-critic',
    'sec-ppo',
    'fig-rl-foundations',
  ], ['MDP', 'Bellman', '策略梯度', 'Actor-Critic', 'PPO', '练习']);
});

test('RL foundations keeps discount weights in policy-gradient updates', () => {
  const html = readChapterFile('rl-foundations.html');
  const policyStart = html.indexOf('id="sec-policy-gradient"');
  const actorStart = html.indexOf('id="sec-actor-critic"');
  const ppoStart = html.indexOf('id="sec-ppo"');
  assert.ok(policyStart >= 0 && actorStart > policyStart && ppoStart > actorStart);
  const policy = html.slice(policyStart, actorStart);
  const actor = html.slice(actorStart, ppoStart);
  assert.match(policy, /\\gamma\^t/);
  assert.match(actor, /\\gamma\^t/);
  assert.match(readChapterFile('frontier-rl.html'), /\\gamma\^t\s*A_t/);
});

test('RLHF alignment chapter is complete', () => {
  assertChapterIncludes('rlhf-alignment.html', [
    'sec-sft',
    'sec-preference-data',
    'sec-reward-model',
    'sec-rlhf-objective',
    'sec-ppo-kl',
    'sec-dpo',
    'sec-process-reward',
    'sec-reward-hacking',
    'sec-evaluation',
    'sec-alignment-risk',
    'fig-rlhf-objective',
  ], ['RLHF', 'DPO', '奖励模型', 'Reward hacking', '练习']);
});

test('Frontier RL chapter is complete', () => {
  assertChapterIncludes('frontier-rl.html', [
    'sec-rlvr',
    'sec-process-reward',
    'sec-reward-hacking',
    'sec-agentic-rl',
    'sec-environment',
    'sec-long-horizon',
    'sec-agent-evaluation',
    'sec-interpretability',
    'sec-open-problems',
    'fig-frontier-rl',
  ], ['RLVR', 'Agentic RL', '可验证奖励', '开放问题', '练习']);
});

test('advanced chapters reference their existing static figures', () => {
  for (const [filename, figure] of [
    ['transformer-architecture.html', 'transformer-block.svg'],
    ['pretraining-scaling.html', 'pretraining-scaling.svg'],
    ['inference-systems.html', 'inference-systems.svg'],
    ['rl-foundations.html', 'rl-foundations.svg'],
    ['rlhf-alignment.html', 'rlhf-alignment.svg'],
    ['frontier-rl.html', 'frontier-rl.svg'],
  ]) {
    assert.ok(
      readChapterFile(filename).includes(`assets/figures/${figure}`),
      `${filename} does not reference ${figure}`,
    );
    assert.ok(
      existsSync(join(projectRoot, 'assets/figures', figure)),
      `Static figure is missing: ${figure}`,
    );
  }
});

test('HTML navigation stays relative and each page has bounded payload', () => {
  for (const filename of [
    'index.html',
    'softmax-cross-entropy.html',
    'embedding-geometry.html',
    'optimization.html',
    'attention.html',
    'transformer-architecture.html',
    'pretraining-scaling.html',
    'inference-systems.html',
    'rl-foundations.html',
    'rlhf-alignment.html',
    'frontier-rl.html',
  ]) {
    const path = join(bookDir, filename);
    assert.ok(existsSync(path), `${path} does not exist`);
    assert.ok(
      statSync(path).size < 32 * 1024 ** 2,
      `${filename} exceeds the 32 MiB per-page payload budget`,
    );
    assert.doesNotMatch(
      readFileSync(path, 'utf8'),
      /data:text\/html/,
      `${filename} embeds a complete chapter in a navigation link`,
    );
  }
});

test('Typst PDF includes every new chapter title', () => {
  const text = execFileSync('pdftotext', ['-layout', join(bookDir, 'llm-book.pdf'), '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  for (const title of [
    '章节 6. Transformer 架构与预训练',
    '章节 7. 预训练与 Scaling Laws',
    '章节 8. 推理与系统工程',
    '章节 9. 强化学习基础：从 MDP 到 PPO',
    '章节 10. 后训练、偏好优化与 RLHF',
    '章节 11. 前沿 RL、Agentic RL 与开放问题',
  ]) {
    assert.ok(text.includes(title), `PDF is missing chapter title: ${title}`);
  }
  assert.doesNotMatch(text, /Chapter\s+[6-9]|Chapter\s+10|Chapter\s+11/);
});

test('Typst PDF keeps advanced exercises and cited source names', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  for (const marker of [
    /6\.11\s+练习/,
    /7\.13\s+练习/,
    /8\.14\s+练习/,
    /9\.13\s+练习/,
    /10\.15\s+练习/,
    /11\.13\s+练习/,
  ]) {
    assert.match(text, marker);
  }
  const compactText = text.replace(/\s+/g, '');
  for (const source of [
    'Kaplan 等（2020）',
    'Chinchilla',
    'Muennighoff 等（2023）',
    'Shazeer 等（2017）',
    'Fedus 等（2021）',
    'Ouyang 等（2022）',
    'Christiano 等（2017）',
    'Schulman 等（2017）',
    'Rafailov 等（2023）',
    'GRPO',
    'HumanEval',
    'ReAct',
  ]) {
    assert.ok(
      compactText.includes(source.replace(/\s+/g, '')),
      `PDF is missing cited source: ${source}`,
    );
  }
});
