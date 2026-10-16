import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const chapterPath = '/llm-book/distributed-training.html';
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

  await page.goto(chapterPath, { waitUntil: 'networkidle' });
  return { localResources, pageErrors };
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

async function expectAccount(page, expected) {
  for (const [testid, text, bytes] of [
    ['distributed-weight-bytes', expected.weight, expected.weightBytes],
    ['distributed-gradient-bytes', expected.gradient, expected.gradientBytes],
    ['distributed-optimizer-bytes', expected.optimizer, expected.optimizerBytes],
    ['distributed-total-bytes', expected.total, expected.totalBytes],
  ]) {
    const output = page.getByTestId(testid);
    await expect(output).toHaveText(text);
    await expect(output).toHaveAttribute('data-value', String(bytes));
  }
}

test('state account and communication events reproduce both worked configurations', async ({ page }) => {
  const monitor = await openChapter(page);

  await expect(page.getByRole('heading', { name: '分布式训练与显存优化', level: 1 })).toBeVisible();
  await expect(page.getByTestId('distributed-zero-stage')).toHaveValue('0');
  await expect(page.getByTestId('distributed-data-parallel')).toHaveValue('8');
  await expect(page.getByTestId('distributed-tensor-parallel')).toHaveValue('1');
  await expect(page.getByTestId('distributed-pipeline-parallel')).toHaveValue('1');
  await expectAccount(page, {
    weight: '2.000 GB', weightBytes: 2e9,
    gradient: '2.000 GB', gradientBytes: 2e9,
    optimizer: '12.000 GB', optimizerBytes: 12e9,
    total: '16.000 GB', totalBytes: 16e9,
  });
  await expect(page.getByTestId('distributed-stage-layers')).toHaveText('12 / 1 = 12 层');
  await expect(page.getByTestId('distributed-communication')).toContainText('DP=8：梯度 AllReduce');
  await expect(page.getByTestId('distributed-chart')).toHaveAttribute('data-signature', '8|1|1|0');
  await expect(page.getByTestId('distributed-chart')).toHaveAccessibleDescription(/DP=8.*ZeRO-0.*16\.000 GB.*不表示真实吞吐/);

  await page.getByTestId('distributed-zero-stage').selectOption('3');
  await page.getByTestId('distributed-data-parallel').selectOption('2');
  await page.getByTestId('distributed-tensor-parallel').selectOption('2');
  await page.getByTestId('distributed-pipeline-parallel').selectOption('2');
  await expectAccount(page, {
    weight: '0.250 GB', weightBytes: 0.25e9,
    gradient: '0.250 GB', gradientBytes: 0.25e9,
    optimizer: '1.500 GB', optimizerBytes: 1.5e9,
    total: '2.000 GB', totalBytes: 2e9,
  });
  await expect(page.getByTestId('distributed-stage-layers')).toHaveText('12 / 2 = 6 层');
  const communication = page.getByTestId('distributed-communication');
  await expect(communication).toContainText('DP=2：梯度 ReduceScatter');
  await expect(communication).toContainText('AllGather 权重');
  await expect(communication).toContainText('TP=2');
  await expect(communication).toContainText('PP=2：相邻阶段 Send/Recv');
  await expect(page.getByTestId('distributed-chart')).toHaveAttribute('data-signature', '2|2|2|3');
  await expect(page.getByTestId('distributed-chart')).toHaveAccessibleDescription(/DP=2.*TP=2.*PP=2.*ZeRO-3.*2\.000 GB/);

  const barWidth = await page.getByTestId('distributed-weight-bar').evaluate((weightBar) => {
    const gradientBar = weightBar.ownerSVGElement.querySelector('[data-testid="distributed-gradient-bar"]');
    const optimizerBar = weightBar.ownerSVGElement.querySelector('[data-testid="distributed-optimizer-bar"]');
    return [weightBar, gradientBar, optimizerBar]
      .reduce((sum, bar) => sum + Number(bar.getAttribute('width')), 0);
  });
  expect(barWidth).toBeCloseTo(77.5, 8);
  expectCleanResources(monitor);
});

test('native selects, dynamic SVG description, text equivalent, and reduced motion are accessible', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const monitor = await openChapter(page);
  const controls = [
    { testid: 'distributed-zero-stage', name: '状态切分策略' },
    { testid: 'distributed-data-parallel', name: '数据并行度 DP' },
    { testid: 'distributed-tensor-parallel', name: '张量并行度 TP' },
    { testid: 'distributed-pipeline-parallel', name: '流水线并行度 PP' },
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

  const chart = page.getByTestId('distributed-chart');
  await expect(chart).toHaveAttribute('role', 'img');
  await expect(chart).toHaveAccessibleName('每设备静态状态账目与四阶段流水线概念时间线');
  await page.getByTestId('distributed-zero-stage').selectOption('2');
  await page.getByTestId('distributed-data-parallel').selectOption('4');
  await expect(chart).toHaveAccessibleDescription(/DP=4.*ZeRO-2.*5\.500 GB.*ReduceScatter/);
  await expect(chart).toContainText('4 阶段 × 4 microbatch');

  const schedule = page.getByText('流水线时间线的文字等价表');
  await schedule.click();
  const scheduleTable = page.locator('.distributed-schedule-text table');
  await expect(scheduleTable).toBeVisible();
  await expect(scheduleTable).toContainText('阶段 1（L1–L3）');
  await expect(scheduleTable).toContainText('t11–t14：B4→B1');
  await expect(scheduleTable).toContainText('阶段 4（L10–L12）');

  const duplicateIds = await page.locator('llm-distributed-training').evaluate((root) => {
    const ids = [root.id, ...[...root.querySelectorAll('[id]')].map((node) => node.id)];
    return ids.filter((id, index) => id && ids.indexOf(id) !== index);
  });
  expect(duplicateIds).toEqual([]);

  const motion = await page.locator('llm-distributed-training').evaluate((root) => {
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

test('chapter loads only local resources and covers required topics, sources, and six exercises', async ({ page }) => {
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

  expect(offOriginRequests, `发现跨源运行时请求：${offOriginRequests.join(', ')}`).toEqual([]);
  await expect(page.locator('.interactive-mount > llm-distributed-training')).toHaveCount(1);
  await expect(page.locator('p > llm-distributed-training')).toHaveCount(0);
  await expect(page.locator('llm-distributed-training#fig-distributed-training')).toBeVisible();
  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toContain('/llm-book/assets/js/components/distributed-training.js');

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
    '梯度累积',
    '数据并行',
    '张量并行',
    '流水线并行',
    'ZeRO 与 FSDP',
    '激活重计算',
    '三维并行',
    '常见误解',
    '总结',
  ]) {
    await expect(content).toContainText(text);
  }
  await expect(page.locator('#sec-exercises ol > li')).toHaveCount(6);
  for (const [name, key, href] of [
    ['Micikevicius 等（2018）', 'micikevicius2018mixed', 'https://arxiv.org/abs/1710.03740'],
    ['Megatron-LM', 'shoeybi2019megatron', 'https://arxiv.org/abs/1909.08053'],
    ['GPipe', 'huang2019gpipe', 'https://arxiv.org/abs/1811.06965'],
    ['ZeRO', 'rajbhandari2020zero', 'https://arxiv.org/abs/1910.02054'],
    ['Zhao 等（2023）', 'zhao2023pytorch', 'https://arxiv.org/abs/2304.11277'],
    ['Chen 等（2016）', 'chen2016training', 'https://arxiv.org/abs/1604.06174'],
    ['Narayanan 等（2021）', 'narayanan2021efficient', 'https://arxiv.org/abs/2104.04473'],
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

test('分布式训练章通过 WCAG 2.1 AA axe 扫描', async ({ page }) => {
  await openChapter(page);
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();

  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);
});

test('分布式训练章在移动端无页面、控件、时间线或长公式横向裁切', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openChapter(page);

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
      controlOverflow: [...document.querySelectorAll('llm-distributed-training .interactive-control')]
        .map((element, index) => ({
          index,
          clientWidth: element.clientWidth,
          scrollWidth: element.scrollWidth,
        }))
        .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
      widgetOverflow: [...document.querySelectorAll('llm-distributed-training .interactive-widget')]
        .map((element, index) => ({
          index,
          clientWidth: element.clientWidth,
          scrollWidth: element.scrollWidth,
        }))
        .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
    }));
    expect(dimensions.scrollWidth, `页面在 ${width}px 视口不应横向溢出`).toBeLessThanOrEqual(dimensions.clientWidth);
    expect(dimensions.formulaOverflow, `${width}px 的公式溢出：${JSON.stringify(dimensions.formulaOverflow)}`).toEqual([]);
    expect(dimensions.controlOverflow, `${width}px 的控件溢出`).toEqual([]);
    expect(dimensions.widgetOverflow, `${width}px 的组件溢出`).toEqual([]);
  }
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows the static parallel flows, pipeline timeline, and complete state table', async ({ page }) => {
    const monitor = await openChapter(page);
    const image = page.locator('noscript img[src="assets/figures/distributed-training.svg"]');
    await expect(image).toBeVisible();
    await expect(image).toHaveAttribute('alt', /分布式训练静态教学图/);
    await expect(image).toHaveAttribute('alt', /16 GB\/设备/);
    await expect(image).toHaveAttribute('alt', /2 GB\/设备/);
    await expect(image).toHaveAttribute('alt', /不表示真实吞吐/);
    await expect(page.locator('#fig-distributed-training')).toBeVisible();
    const table = page.locator('noscript table');
    await expect(table).toHaveCount(1);
    await expect(table).toContainText('DP=8，TP=PP=1，ZeRO-0');
    await expect(table).toContainText('16 GB/设备');
    await expect(table).toContainText('DP=TP=PP=2，ZeRO-3');
    await expect(table).toContainText('2 GB/设备');
    await expect(page.getByRole('combobox')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});

test('Typst PDF retains the distributed-training figure and static fallback account', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  const compact = text.replace(/\s+/g, '');
  for (const requiredText of [
    '分布式训练与显存优化',
    'GPipe式概念时间线',
    '静态通信等价说明',
    '16GB/设备',
    '2GB/设备',
    '不模拟真实吞吐',
  ]) {
    expect(compact).toContain(requiredText);
  }
});
