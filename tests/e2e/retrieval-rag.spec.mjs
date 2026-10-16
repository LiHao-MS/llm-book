import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const chapterPath = '/llm-book/retrieval-rag.html';
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

async function expectRanking(page, expected) {
  const rows = page.getByTestId('retrieval-ranking-body').locator('tr');
  expect(await rows.evaluateAll((nodes) => nodes.map((node) => node.dataset.documentId))).toEqual(expected);
  const bars = page.getByTestId('retrieval-ranking-bars').locator(':scope > g');
  expect(await bars.evaluateAll((nodes) => nodes.map((node) => node.dataset.documentId))).toEqual(expected);
}

async function expectContext(page, expected) {
  const items = page.getByTestId('retrieval-context').locator('li[data-document-id]');
  expect(await items.evaluateAll((nodes) => nodes.map((node) => node.dataset.documentId))).toEqual(expected);
}

test('dot and cosine reverse the geometric toy ranking while top-k and whole-document budget stay distinct', async ({ page }) => {
  const monitor = await openChapter(page);

  await expect(page.getByRole('heading', { name: '检索、RAG 与上下文管理', level: 1 })).toBeVisible();
  await expect(page.getByTestId('retrieval-metric')).toHaveValue('dot');
  await expect(page.getByTestId('retrieval-top-k')).toHaveValue('3');
  await expect(page.getByTestId('retrieval-token-budget')).toHaveValue('12');
  await expectRanking(page, ['D1', 'D5', 'D3', 'D2', 'D4']);
  await expectContext(page, ['D1', 'D3']);
  await expect(page.getByTestId('retrieval-used-tokens')).toHaveText('12 / 12 Token');
  await expect(page.getByTestId('retrieval-used-tokens')).toHaveAttribute('data-value', '12');
  await expect(page.getByTestId('retrieval-row-D1').getByTestId('retrieval-score')).toHaveText('10');
  await expect(page.getByTestId('retrieval-row-D1').getByTestId('retrieval-score'))
    .toHaveAttribute('data-value', '10');
  await expect(page.getByTestId('retrieval-row-D5')).toContainText('预算不足');
  await expect(page.getByTestId('retrieval-row-D3')).toContainText('进入上下文');
  await expect(page.getByTestId('retrieval-point-D1')).toHaveAttribute('data-state', 'selected');
  await expect(page.getByTestId('retrieval-point-D5')).toHaveAttribute('data-state', 'budget-skipped');
  await expect(page.getByTestId('retrieval-point-D2')).toHaveAttribute('data-state', 'outside-top-k');

  const chart = page.getByTestId('retrieval-chart');
  await expect(chart).toHaveAccessibleName('二维几何与原始检索分数排名');
  await expect(chart).toHaveAccessibleDescription(/几何玩具.*不运行真实语义模型.*点积.*Top-k=3.*预算 12.*D1.*D5.*D3.*D1、D3.*12 Token/);
  await expect(chart).toHaveAttribute(
    'data-signature',
    'dot|3|12|D1,D5,D3,D2,D4|D1,D3|12',
  );

  await page.getByTestId('retrieval-metric').selectOption('cosine');
  await expectRanking(page, ['D2', 'D1', 'D5', 'D3', 'D4']);
  await expectContext(page, ['D2', 'D1']);
  await expect(page.getByTestId('retrieval-used-tokens')).toHaveText('11 / 12 Token');
  await expect(page.getByTestId('retrieval-row-D2').getByTestId('retrieval-score')).toHaveText('1.000000');
  await expect(page.getByTestId('retrieval-row-D1').getByTestId('retrieval-score')).toHaveText('0.995037');
  await expect(chart).toHaveAccessibleDescription(/余弦.*D2、D1.*11 Token/);

  await page.getByTestId('retrieval-token-budget').selectOption('4');
  await expectContext(page, ['D2']);
  await expect(page.getByTestId('retrieval-used-tokens')).toHaveText('3 / 4 Token');
  await expect(page.getByTestId('retrieval-row-D1')).toContainText('预算不足');

  await page.getByTestId('retrieval-token-budget').selectOption('12');
  await page.getByTestId('retrieval-top-k').selectOption('1');
  await expectContext(page, ['D2']);
  await expect(page.getByTestId('retrieval-row-D1')).toContainText('Top-k 外');
  await expect(chart).toHaveAccessibleDescription(/Top-k=1/);
  expectCleanResources(monitor);
});

test('native controls, dynamic SVG description, text data, and reduced motion are accessible', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const monitor = await openChapter(page);
  const controls = [
    { testid: 'retrieval-metric', name: '相似度度量' },
    { testid: 'retrieval-top-k', name: 'Top-k 文档数' },
    { testid: 'retrieval-token-budget', name: '上下文 Token 预算' },
  ];

  for (const { testid, name } of controls) {
    const locator = page.getByTestId(testid);
    await expect(page.getByRole('combobox', { name })).toHaveAttribute('data-testid', testid);
    const id = await locator.getAttribute('id');
    expect(id).toBeTruthy();
    await expect(page.locator(`label[for="${id}"]`)).toHaveCount(1);
    await locator.focus();
    const outline = await locator.evaluate((element) => {
      const style = getComputedStyle(element);
      return { style: style.outlineStyle, width: Number.parseFloat(style.outlineWidth) };
    });
    expect(outline.style).not.toBe('none');
    expect(outline.width).toBeGreaterThan(0);

    const initial = await locator.inputValue();
    const optionCount = await locator.evaluate((element) => element.options.length);
    const selectedIndex = await locator.evaluate((element) => element.selectedIndex);
    const forward = selectedIndex < optionCount - 1;
    await locator.press(forward ? 'ArrowDown' : 'ArrowUp');
    expect(await locator.inputValue()).not.toBe(initial);
    await locator.press(forward ? 'ArrowUp' : 'ArrowDown');
    await expect(locator).toHaveValue(initial);
  }

  await expect(page.getByText('这是固定二维几何玩具，不运行真实语义模型。', { exact: true })).toBeVisible();
  const textEquivalent = page.getByText('当前排名与预算的文字数据', { exact: true });
  await textEquivalent.click();
  const table = page.getByTestId('retrieval-text-equivalent').locator('table');
  await expect(table).toBeVisible();
  await expect(table).toContainText('原始分数');
  await expect(table).toContainText('预算不足，跳过完整文档');

  const duplicateIds = await page.locator('llm-retrieval-rag').evaluate((root) => {
    const ids = [root.id, ...[...root.querySelectorAll('[id]')].map((node) => node.id)];
    return ids.filter((id, index) => id && ids.indexOf(id) !== index);
  });
  expect(duplicateIds).toEqual([]);

  const motion = await page.locator('llm-retrieval-rag').evaluate((root) => {
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

test('chapter uses only local runtime resources and covers required topics, primary sources, and six exercises', async ({ page }) => {
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
  await expect(page.locator('.interactive-mount > llm-retrieval-rag')).toHaveCount(1);
  await expect(page.locator('p > llm-retrieval-rag')).toHaveCount(0);
  await expect(page.locator('llm-retrieval-rag#fig-retrieval-rag')).toBeVisible();
  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toContain('/llm-book/assets/js/components/retrieval-rag.js');

  const content = page.locator('#quarto-document-content');
  for (const text of [
    'chunking',
    '向量检索',
    'ANN',
    'metadata filter',
    'Top-k',
    '重排',
    '上下文组装',
    'groundedness',
    'RAG 评测',
    '扩大上下文不等于有效利用上下文',
    '常见误解',
    '总结',
  ]) {
    await expect(content).toContainText(text);
  }
  await expect(page.locator('#sec-exercises > ol > li')).toHaveCount(6);
  const filterExercise = page.locator('#sec-exercises > ol > li').nth(2);
  await expect(filterExercise).toContainText('只保留 Top-5');
  await expect(filterExercise).toContainText('保留 20 条候选');
  await expect(filterExercise).toContainText('候选池耗尽或没有更多合格文档');
  for (const [name, key, href] of [
    ['Retrieval-Augmented Generation', 'lewis2020rag', 'https://arxiv.org/abs/2005.11401'],
    ['Dense Passage Retrieval', 'karpukhin2020dpr', 'https://arxiv.org/abs/2004.04906'],
    ['HNSW', 'malkov2020hnsw', 'https://arxiv.org/abs/1603.09320'],
    ['FAISS', 'johnson2021faiss', 'https://arxiv.org/abs/1702.08734'],
    ['ColBERT', 'khattab2020colbert', 'https://arxiv.org/abs/2004.12832'],
    ['Lost in the Middle', 'liu2024lost', 'https://arxiv.org/abs/2307.03172'],
    ['ARES', 'saadfalcon2024ares', 'https://arxiv.org/abs/2311.09476'],
  ]) {
    await expect(content).toContainText(name);
    const citation = page.locator(`.citation[data-cites~="${key}"]`);
    await expect(citation).toHaveCount(1);
    await expect(citation.getByRole('link')).toHaveAttribute('href', `${localOrigin}${chapterPath}#ref-${key}`);
    const reference = page.locator(`#ref-${key}`);
    await expect(reference).toHaveCount(1);
    await expect(reference.getByRole('link', { name: href, exact: true })).toHaveAttribute('href', href);
  }
  await expect(content).not.toContainText('[?]');
  expectCleanResources(monitor);
});

test('检索、RAG 与上下文管理章通过 WCAG 2.1 AA axe 扫描', async ({ page }) => {
  await openChapter(page);
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();

  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);
});

test('chapter has no page, control, text-table, widget, or display-formula clipping at responsive widths', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openChapter(page);
  await page.getByText('当前排名与预算的文字数据', { exact: true }).click();

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
      controlOverflow: [...document.querySelectorAll('llm-retrieval-rag .interactive-control')]
        .map((element, index) => ({
          index,
          clientWidth: element.clientWidth,
          scrollWidth: element.scrollWidth,
        }))
        .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
      widgetOverflow: [...document.querySelectorAll('llm-retrieval-rag .interactive-widget')]
        .map((element, index) => ({
          index,
          clientWidth: element.clientWidth,
          scrollWidth: element.scrollWidth,
        }))
        .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
      textTableOverflow: [...document.querySelectorAll('llm-retrieval-rag .retrieval-text-equivalent table')]
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

  test('shows the pipeline figure and complete static score and budget table', async ({ page }) => {
    const monitor = await openChapter(page);
    const image = page.locator('noscript img[src="assets/figures/retrieval-rag.svg"]');
    await expect(image).toBeVisible();
    await expect(image).toHaveAttribute('alt', /检索到重排、Token 预算、生成与引用的静态流程图/);
    await expect(image).toHaveAttribute('alt', /D1、D3/);
    await expect(image).toHaveAttribute('alt', /原始分数不是概率/);
    await expect(page.locator('#fig-retrieval-rag')).toBeVisible();

    const fallback = page.locator('noscript #tbl-retrieval-rag-static');
    const table = fallback.locator('table');
    await expect(table).toHaveCount(1);
    await expect(fallback).toContainText('静态排名与预算结果');
    await expect(table).toContainText('10 / 1');
    await expect(table).toContainText('0.995037 / 2');
    await expect(table).toContainText('1.000000 / 1');
    await expect(table).toContainText('点积 Top-3、预算 12：D1、D3，共 12 Token');
    await expect(table).toContainText('余弦 Top-3、预算 12：D2、D1，共 11 Token');
    await expect(page.getByRole('combobox')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});

test('Typst PDF retains the RAG pipeline figure and static geometric fallback', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  const compact = text.replace(/\s+/g, '');
  for (const requiredText of [
    '检索、RAG与上下文管理',
    '静态排名与预算结果',
    '原始分数不是概率',
    '扩大上下文不等于有效利用上下文',
    '10/1',
    '0.995037/2',
    '1/4',
    '1.000000/1',
    '2/3',
    '0.707107/4',
    '0/5',
    '0.000000/5',
    '3/2',
    '0.986394/3',
    '点积Top-3、预算12：D1、D3，共12Token',
    '余弦Top-3、预算12：D2、D1，共11Token',
  ]) {
    expect(compact).toContain(requiredText);
  }

  const pages = text.split('\f');
  const figurePageIndex = pages.findIndex((page) => {
    const compactPage = page.replace(/\s+/g, '');
    return [
      'RAG：候选相关不等于上下文可用',
      '1.检索候选',
      '2.可选重排',
      '3.Token预算',
      '4.生成与引用',
    ].every((label) => compactPage.includes(label));
  });
  expect(figurePageIndex, 'PDF is missing the RAG pipeline figure text').toBeGreaterThanOrEqual(0);

  const temporaryDirectory = mkdtempSync(join(tmpdir(), 'retrieval-rag-pdf-'));
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
    const accentPixels = countSimilarPixels(pixels, [0x8C, 0x42, 0x1F]);
    expect(
      panelPixels,
      'Rendered PDF page lacks the RAG pipeline panel graphics',
    ).toBeGreaterThanOrEqual(100);
    expect(
      accentPixels,
      'Rendered PDF page lacks the RAG worked-example accent graphics',
    ).toBeGreaterThanOrEqual(20);
  } finally {
    rmSync(temporaryDirectory, { recursive: true, force: true });
  }
});
