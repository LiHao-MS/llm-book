import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const chapterPath = '/llm-book/multimodal-models.html';
const localOrigin = 'http://127.0.0.1:4173';
const projectRoot = fileURLToPath(new URL('../../', import.meta.url));
const pdfPath = join(projectRoot, '_book', 'llm-book.pdf');

async function openChapter(page) {
  const pageErrors = [];
  const localResources = [];
  page.on('pageerror', (error) => pageErrors.push(error));
  page.on('response', (response) => {
    const url = new URL(response.url());
    if (url.origin === localOrigin && /\.(?:m?js|svg)$/.test(url.pathname)) {
      localResources.push(response);
    }
  });

  const response = await page.goto(chapterPath, { waitUntil: 'networkidle' });
  return { localResources, pageErrors, response };
}

function expectCleanResources({ localResources, pageErrors }) {
  expect(pageErrors).toEqual([]);
  expect(localResources.length).toBeGreaterThan(0);
  for (const response of localResources) {
    expect(response.status(), `${response.url()} should load successfully`).toBe(200);
  }
}

function axeSummary(violations) {
  return violations.map((violation) => ({
    id: violation.id,
    impact: violation.impact,
    help: violation.help,
    nodes: violation.nodes.map((node) => ({
      target: node.target,
      html: node.html,
      failureSummary: node.failureSummary,
    })),
  }));
}

function ppmPixels(buffer) {
  const header = buffer.subarray(0, 128).toString('ascii');
  const match = /^P6\s+(\d+)\s+(\d+)\s+255\s/.exec(header);
  expect(match, 'pdftoppm should emit a binary PPM image').not.toBeNull();
  const pixels = buffer.subarray(match[0].length);
  expect(pixels.length).toBe(Number(match[1]) * Number(match[2]) * 3);
  return pixels;
}

function countSimilarPixels(pixels, expected, tolerance = 24) {
  let count = 0;
  for (let index = 0; index < pixels.length; index += 3) {
    if (expected.every((channel, offset) =>
      Math.abs(pixels[index + offset] - channel) <= tolerance
    )) {
      count += 1;
    }
  }
  return count;
}

const accounts = [
  { patch: '14', rows: 16, columns: 16, visual: 256, sequence: 276, scores: 76176, formattedScores: '76,176' },
  { patch: '16', rows: 14, columns: 14, visual: 196, sequence: 216, scores: 46656, formattedScores: '46,656' },
  { patch: '32', rows: 7, columns: 7, visual: 49, sequence: 69, scores: 4761, formattedScores: '4,761' },
];

async function expectAccount(page, account) {
  await page.getByTestId('multimodal-patch-size').selectOption(account.patch);
  await expect(page.getByTestId('multimodal-patch-grid')).toHaveText(`${account.rows} × ${account.columns}`);
  await expect(page.getByTestId('multimodal-visual-tokens')).toHaveText(String(account.visual));
  await expect(page.getByTestId('multimodal-visual-tokens')).toHaveAttribute('data-value', String(account.visual));
  await expect(page.getByTestId('multimodal-text-tokens')).toHaveText('20');
  await expect(page.getByTestId('multimodal-text-tokens')).toHaveAttribute('data-value', '20');
  await expect(page.getByTestId('multimodal-sequence-length')).toHaveText(String(account.sequence));
  await expect(page.getByTestId('multimodal-sequence-length')).toHaveAttribute('data-value', String(account.sequence));
  await expect(page.getByTestId('multimodal-attention-scores')).toHaveText(account.formattedScores);
  await expect(page.getByTestId('multimodal-attention-scores')).toHaveAttribute('data-value', String(account.scores));
  await expect(page.getByTestId('multimodal-account-equation')).toHaveText(
    `${account.visual} + 20 = ${account.sequence}；${account.sequence}² = ${account.formattedScores}`,
  );

  const cells = page.getByTestId('multimodal-patch-cell');
  await expect(cells).toHaveCount(account.visual);
  await expect(cells.last()).toHaveAttribute('data-row', String(account.rows));
  await expect(cells.last()).toHaveAttribute('data-column', String(account.columns));
  await expect(page.getByTestId('multimodal-grid-caption')).toHaveText(
    `${account.rows} 行 × ${account.columns} 列 = ${account.visual} 个视觉 Token`,
  );

  const chart = page.getByTestId('multimodal-patch-chart');
  await expect(chart).toHaveAttribute(
    'data-signature',
    `${account.patch}|${account.rows}|${account.columns}|${account.visual}|20|${account.sequence}|${account.scores}`,
  );
  await expect(chart).toHaveAccessibleDescription(
    new RegExp(`patch 边长 ${account.patch}.*${account.rows} 行 × ${account.columns} 列.*${account.visual} 个视觉 Token.*序列长度 ${account.sequence}.*${account.formattedScores} 个.*不对应某一真实模型`),
  );
  await expect(page.getByTestId('multimodal-attention-dimensions')).toHaveText(`${account.sequence} × ${account.sequence}`);
  await expect(page.getByTestId('multimodal-attention-total')).toHaveText(`= ${account.formattedScores} 个分数`);

  const visualWidth = Number(await page.getByTestId('multimodal-sequence-visual').getAttribute('width'));
  expect(visualWidth).toBeCloseTo(420 * account.visual / account.sequence, 8);
  const textX = Number(await page.getByTestId('multimodal-sequence-text').getAttribute('x'));
  const textWidth = Number(await page.getByTestId('multimodal-sequence-text').getAttribute('width'));
  expect(textX + textWidth).toBeCloseTo(890, 8);

  const selectedRows = page.locator('[data-testid="multimodal-account-row"][data-current="true"]');
  await expect(selectedRows).toHaveCount(1);
  await expect(selectedRows).toHaveAttribute('data-patch-size', account.patch);
}

test('patch control synchronizes all three grids, token accounts, quadratic scores, and descriptions', async ({ page }) => {
  const monitor = await openChapter(page);

  await expect(page.getByRole('heading', { name: '多模态模型与统一表示', level: 1 })).toBeVisible();
  await expect(page.getByTestId('multimodal-patch-size')).toHaveValue('14');
  await expect(page.getByTestId('multimodal-patch-chart')).toHaveAccessibleName('Patch 网格、拼接序列与玩具二次注意力账');
  for (const account of accounts) {
    await expectAccount(page, account);
  }
  const teachingNote = page.locator('llm-multimodal-patch p').filter({ hasText: '教学序列账：' });
  await expect(teachingNote).toContainText('不运行真实多模态模型');
  expectCleanResources(monitor);
});

test('native select, dynamic text data, and reduced-motion rendering are accessible', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const monitor = await openChapter(page);
  const select = page.getByTestId('multimodal-patch-size');

  await expect(page.getByRole('combobox', { name: 'Patch 边长 P（像素）' })).toHaveAttribute('data-testid', 'multimodal-patch-size');
  const id = await select.getAttribute('id');
  expect(id).toBeTruthy();
  await expect(page.locator(`label[for="${id}"]`)).toHaveCount(1);
  await select.focus();
  const outline = await select.evaluate((element) => {
    const style = getComputedStyle(element);
    return { style: style.outlineStyle, width: Number.parseFloat(style.outlineWidth) };
  });
  expect(outline.style).not.toBe('none');
  expect(outline.width).toBeGreaterThan(0);
  await select.press('ArrowDown');
  await expect(select).toHaveValue('16');
  await expect(page.getByTestId('multimodal-visual-tokens')).toHaveText('196');
  await select.press('ArrowUp');
  await expect(select).toHaveValue('14');

  const textEquivalent = page.getByText('三组 patch 序列账的文字数据', { exact: true });
  await textEquivalent.click();
  const table = page.getByTestId('multimodal-text-equivalent').locator('table');
  await expect(table).toBeVisible();
  for (const text of [
    '256 + 20 = 276',
    '276² = 76,176',
    '196 + 20 = 216',
    '216² = 46,656',
    '49 + 20 = 69',
    '69² = 4,761',
  ]) {
    await expect(table).toContainText(text);
  }

  const duplicateIds = await page.locator('llm-multimodal-patch').evaluate((root) => {
    const ids = [root.id, ...[...root.querySelectorAll('[id]')].map((node) => node.id)];
    return ids.filter((item, index) => item && ids.indexOf(item) !== index);
  });
  expect(duplicateIds).toEqual([]);

  const motion = await page.locator('llm-multimodal-patch').evaluate((root) => {
    const elements = [root, ...root.querySelectorAll('*')];
    return elements.flatMap((element) => {
      const style = getComputedStyle(element);
      const animationDurations = style.animationDuration.split(',').map(Number.parseFloat);
      const transitionDurations = style.transitionDuration.split(',').map(Number.parseFloat);
      const hasMotion = (style.animationName !== 'none' && animationDurations.some((duration) => duration > 0))
        || transitionDurations.some((duration) => duration > 0);
      return [
        ...element.getAnimations().map((animation) => animation.playState),
        ...(hasMotion ? [style.animationName || style.transitionProperty] : []),
      ];
    });
  });
  expect(motion).toEqual([]);
  expectCleanResources(monitor);
});

test('chapter uses local resources and covers required topics, primary sources, and six exercises', async ({ page }) => {
  const offOriginRequests = [];
  await page.route('**/*', async (route) => {
    const url = new URL(route.request().url());
    if (['http:', 'https:'].includes(url.protocol) && url.origin !== localOrigin) {
      offOriginRequests.push(url.href);
      await route.abort('blockedbyclient');
      return;
    }
    await route.continue();
  });
  const monitor = await openChapter(page);

  expect(monitor.response.status()).toBe(200);
  expect(offOriginRequests, `发现跨源运行时请求：${offOriginRequests.join(', ')}`).toEqual([]);
  await expect(page.locator('.interactive-mount > llm-multimodal-patch')).toHaveCount(1);
  await expect(page.locator('p > llm-multimodal-patch')).toHaveCount(0);
  await expect(page.locator('llm-multimodal-patch#fig-multimodal-models')).toBeVisible();
  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toContain('/llm-book/assets/js/components/multimodal-patch.js');

  const formulas = page.locator('#quarto-document-content mjx-container[jax="SVG"]');
  await expect(formulas.first()).toBeVisible();
  expect(await formulas.count()).toBeGreaterThan(10);
  const invalidFormulas = await formulas.evaluateAll((nodes) =>
    nodes.flatMap((node, index) => {
      const svg = node.querySelector('svg');
      if (!svg) {
        return [{ index, text: node.textContent, reason: 'missing SVG' }];
      }
      const bounds = svg.getBoundingClientRect();
      return bounds.width > 0 && bounds.height > 0
        ? []
        : [{ index, text: node.textContent, width: bounds.width, height: bounds.height }];
    }),
  );
  expect(invalidFormulas, JSON.stringify(invalidFormulas)).toEqual([]);

  const content = page.locator('#quarto-document-content');
  for (const text of [
    '图像 patch',
    '音频帧',
    '模态编码器',
    '对比对齐',
    'CLIP',
    'Q-Former',
    'early fusion',
    'cross-attention',
    '视觉指令微调',
    '统一表示的边界',
    '幻觉',
    '模态偏置',
    '多模态评测',
    '常见误解',
    '总结',
  ]) {
    await expect(content).toContainText(text);
  }
  await expect(content).not.toContainText('[?]');
  await expect(page.locator('#sec-exercises > ol > li')).toHaveCount(6);

  const sources = page.locator('#sec-primary-sources');
  for (const [name, key, href] of [
    ['An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', 'dosovitskiy2020image', 'https://arxiv.org/abs/2010.11929'],
    ['wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations', 'baevski2020wav2vec', 'https://arxiv.org/abs/2006.11477'],
    ['Learning Transferable Visual Models From Natural Language Supervision', 'radford2021clip', 'https://arxiv.org/abs/2103.00020'],
    ['Flamingo: a Visual Language Model for Few-Shot Learning', 'alayrac2022flamingo', 'https://arxiv.org/abs/2204.14198'],
    ['BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models', 'li2023blip2', 'https://arxiv.org/abs/2301.12597'],
    ['Visual Instruction Tuning', 'liu2023visualinstruction', 'https://arxiv.org/abs/2304.08485'],
    ['Evaluating Object Hallucination in Large Vision-Language Models', 'li2023pope', 'https://arxiv.org/abs/2305.10355'],
  ]) {
    const source = sources.locator('li').filter({ hasText: name });
    await expect(source).toHaveCount(1);
    await expect(source.locator(`.citation[data-cites~="${key}"]`)).toHaveCount(1);
    const reference = page.locator(`#ref-${key}`);
    await expect(reference).toHaveCount(1);
    await expect(reference.getByRole('link', { name: href, exact: true })).toHaveAttribute('href', href);
  }
  expectCleanResources(monitor);
});

test('多模态模型章通过 WCAG 2.1 AA axe 扫描', async ({ page }) => {
  await openChapter(page);
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();

  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);
});

test('chapter has no page, control, text-table, widget, or display-formula clipping at responsive widths', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openChapter(page);
  await page.getByText('三组 patch 序列账的文字数据', { exact: true }).click();

  for (const width of [390, 600, 768, 1280]) {
    await page.setViewportSize({ width, height: 844 });
    const dimensions = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
      formulaOverflow: [...document.querySelectorAll('mjx-container[display="true"]')]
        .map((node, index) => ({
          index,
          text: node.textContent,
          clientWidth: node.clientWidth,
          scrollWidth: node.scrollWidth,
        }))
        .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
      controlOverflow: [...document.querySelectorAll('llm-multimodal-patch .interactive-control')]
        .map((element, index) => ({
          index,
          clientWidth: element.clientWidth,
          scrollWidth: element.scrollWidth,
        }))
        .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
      widgetOverflow: [...document.querySelectorAll('llm-multimodal-patch .interactive-widget')]
        .map((element, index) => ({
          index,
          clientWidth: element.clientWidth,
          scrollWidth: element.scrollWidth,
        }))
        .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
      textTableOverflow: [...document.querySelectorAll('llm-multimodal-patch .multimodal-text-equivalent table')]
        .map((element, index) => ({
          index,
          clientWidth: element.clientWidth,
          scrollWidth: element.scrollWidth,
        }))
        .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
    }));
    expect(dimensions.scrollWidth, `页面在 ${width}px 视口不应横向溢出`).toBeLessThanOrEqual(dimensions.clientWidth);
    expect(dimensions.formulaOverflow, `${width}px 的公式溢出：${JSON.stringify(dimensions.formulaOverflow)}`).toEqual([]);
    expect(dimensions.controlOverflow, `${width}px 的控件溢出：${JSON.stringify(dimensions.controlOverflow)}`).toEqual([]);
    expect(dimensions.widgetOverflow, `${width}px 的组件溢出：${JSON.stringify(dimensions.widgetOverflow)}`).toEqual([]);
    expect(dimensions.textTableOverflow, `${width}px 的文字表溢出：${JSON.stringify(dimensions.textTableOverflow)}`).toEqual([]);
  }
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows the multimodal pipeline figure and complete static patch table', async ({ page }) => {
    const monitor = await openChapter(page);
    const image = page.locator('noscript img[src="assets/figures/multimodal-models.svg"]');
    await expect(image).toBeVisible();
    await expect(image).toHaveAttribute('alt', /多模态静态教学图/);
    await expect(image).toHaveAttribute('alt', /256 个视觉 Token/);
    await expect(image).toHaveAttribute('alt', /L²=46656/);
    await expect(image).toHaveAttribute('alt', /不是完整 FLOPs 或 benchmark/);
    await expect(page.locator('#fig-multimodal-models')).toBeVisible();

    const fallback = page.locator('noscript #tbl-multimodal-patch-static');
    const table = fallback.locator('table');
    await expect(table).toHaveCount(1);
    await expect(fallback).toContainText('三组 patch 的静态序列账');
    for (const text of [
      '256 + 20 = 276',
      '276² = 76,176',
      '196 + 20 = 216',
      '216² = 46,656',
      '49 + 20 = 69',
      '69² = 4,761',
    ]) {
      await expect(table).toContainText(text);
    }
    await expect(page.getByRole('combobox')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});

test('Typst PDF retains the multimodal pipeline figure and complete static patch account', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  const compact = text.replace(/\s+/g, '');
  for (const requiredText of [
    '多模态模型与统一表示',
    '三组patch的静态序列账',
    '256+20=276',
    '276²=76,176',
    '196+20=216',
    '216²=46,656',
    '49+20=69',
    '69²=4,761',
    '不对应某一真实模型结构',
  ]) {
    expect(compact).toContain(requiredText);
  }

  const pages = text.split('\f');
  const figurePageIndex = pages.findIndex((page) => {
    const compactPage = page.replace(/\s+/g, '');
    return [
      '1.模态编码',
      '2.投影或融合',
      '3.共享语言模型',
      'Patch教学账',
    ].every((label) => compactPage.includes(label));
  });
  expect(figurePageIndex, 'PDF is missing the multimodal pipeline figure text').toBeGreaterThanOrEqual(0);

  const temporaryDirectory = mkdtempSync(join(tmpdir(), 'multimodal-models-pdf-'));
  const outputPrefix = join(temporaryDirectory, 'figure-page');
  try {
    execFileSync('pdftoppm', [
      '-f', String(figurePageIndex + 1),
      '-l', String(figurePageIndex + 1),
      '-singlefile',
      '-r', '144',
      pdfPath,
      outputPrefix,
    ]);
    const pixels = ppmPixels(readFileSync(`${outputPrefix}.ppm`));
    const panelPixels = countSimilarPixels(pixels, [0x8A, 0x9A, 0xA5]);
    const gridPixels = countSimilarPixels(pixels, [0xBB, 0xD5, 0xE2]);
    expect(panelPixels, 'Rendered PDF page lacks the multimodal panel graphics').toBeGreaterThanOrEqual(100);
    expect(gridPixels, 'Rendered PDF page lacks the static patch grid').toBeGreaterThanOrEqual(100);
  } finally {
    rmSync(temporaryDirectory, { recursive: true, force: true });
  }
});
