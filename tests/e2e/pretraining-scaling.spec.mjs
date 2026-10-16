import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const chapterPath = '/llm-book/pretraining-scaling.html';
const localOrigin = 'http://127.0.0.1:4173';

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

test('scaling controls reproduce the toy law, allocation, and cost calculations', async ({ page }) => {
  const monitor = await openChapter(page);
  const compute = page.getByTestId('scaling-compute');
  const ratio = page.getByTestId('scaling-ratio');
  const ratioChart = page.getByTestId('scaling-ratio-chart');
  const computeChart = page.getByTestId('scaling-compute-chart');

  await expect(compute).toHaveValue('480');
  await expect(ratio).toHaveValue('20');
  await expect(page.getByTestId('scaling-compute-value')).toHaveText('480');
  await expect(page.getByTestId('scaling-ratio-value')).toHaveText('20');
  await expect(page.getByTestId('scaling-params')).toHaveText('2.000 M');
  await expect(page.getByTestId('scaling-tokens')).toHaveText('40.000 M');
  await expect(page.getByTestId('scaling-compute-check')).toHaveText('480.000 TFLOP');
  await expect(page.getByTestId('scaling-loss')).toHaveText('4.328427');
  await expect(page.getByTestId('scaling-optimal-loss')).toHaveText('4.328427');
  await expect(page.getByTestId('scaling-optimal-config')).toHaveText('2.000 M / 40.000 M');
  await expect(page.getByTestId('scaling-training-time')).toHaveText('1.333333 h');
  await expect(page.getByTestId('scaling-inference')).toHaveText('4.000000 MFLOP/Token');
  await expect(page.getByTestId('scaling-memory')).toHaveText('4.000000 MB');

  const initialRatioSignature = await ratioChart.getAttribute('data-signature');
  const initialComputeSignature = await computeChart.getAttribute('data-signature');
  const initialRatioPath = await page.getByTestId('scaling-ratio-path').getAttribute('d');
  await ratio.fill('10');
  await expect(page.getByTestId('scaling-params')).toHaveText('2.828 M');
  await expect(page.getByTestId('scaling-tokens')).toHaveText('28.284 M');
  await expect(page.getByTestId('scaling-compute-check')).toHaveText('480.000 TFLOP');
  await expect(page.getByTestId('scaling-loss')).toHaveText('4.371000');
  await expect(page.getByTestId('scaling-optimal-loss')).toHaveText('4.328427');
  await expect(page.getByTestId('scaling-training-time')).toHaveText('1.333333 h');
  await expect(page.getByTestId('scaling-inference')).toHaveText('5.656854 MFLOP/Token');
  await expect(page.getByTestId('scaling-memory')).toHaveText('5.656854 MB');
  expect(await ratioChart.getAttribute('data-signature')).not.toBe(initialRatioSignature);
  expect(await computeChart.getAttribute('data-signature')).toBe(initialComputeSignature);

  await ratio.fill('40');
  await expect(page.getByTestId('scaling-params')).toHaveText('1.414 M');
  await expect(page.getByTestId('scaling-tokens')).toHaveText('56.569 M');
  await expect(page.getByTestId('scaling-loss')).toHaveText('4.371000');
  await ratio.fill('20');
  await compute.fill('60');
  await expect(page.getByTestId('scaling-params')).toHaveText('0.707 M');
  await expect(page.getByTestId('scaling-tokens')).toHaveText('14.142 M');
  await expect(page.getByTestId('scaling-compute-check')).toHaveText('60.000 TFLOP');
  await expect(page.getByTestId('scaling-loss')).toHaveText('6.256828');
  await expect(page.getByTestId('scaling-optimal-loss')).toHaveText('6.256828');
  await expect(page.getByTestId('scaling-optimal-config')).toHaveText('0.707 M / 14.142 M');
  await expect(page.getByTestId('scaling-training-time')).toHaveText('0.166667 h');
  await expect(page.getByTestId('scaling-inference')).toHaveText('1.414214 MFLOP/Token');
  await expect(page.getByTestId('scaling-memory')).toHaveText('1.414214 MB');
  expect(await page.getByTestId('scaling-ratio-path').getAttribute('d')).not.toBe(initialRatioPath);
  expect(await computeChart.getAttribute('data-signature')).not.toBe(initialComputeSignature);
  expectCleanResources(monitor);
});

test('native controls, dynamic SVG descriptions, and reduced-motion rendering are accessible', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const monitor = await openChapter(page);
  const controls = [
    { testid: 'scaling-compute', name: '训练算力 c（TFLOP）' },
    { testid: 'scaling-ratio', name: 'Token/参数比 r' },
  ];

  for (const { testid, name } of controls) {
    const locator = page.getByTestId(testid);
    await expect(page.getByRole('slider', { name })).toHaveAttribute('data-testid', testid);
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
    await locator.press('ArrowRight');
    expect(await locator.inputValue()).not.toBe(initial);
    await locator.press('ArrowLeft');
    await expect(locator).toHaveValue(initial);
  }

  const ratioChart = page.getByTestId('scaling-ratio-chart');
  const computeChart = page.getByTestId('scaling-compute-chart');
  await expect(ratioChart).toHaveAttribute('role', 'img');
  await expect(ratioChart).toHaveAccessibleName('固定算力下 Token/参数比与玩具损失');
  await expect(computeChart).toHaveAccessibleName('最优配置的玩具损失随训练算力变化');
  await page.getByTestId('scaling-ratio').fill('10');
  await expect(ratioChart).toHaveAccessibleDescription(/c=480.*r=10.*4\.371000.*最优损失 4\.328427/);
  await page.getByTestId('scaling-compute').fill('60');
  await expect(computeChart).toHaveAccessibleDescription(/c=60.*n=0\.707107.*d=14\.142136.*6\.256828/);

  const duplicateIds = await page.locator('llm-scaling-laws').evaluate((root) => {
    const ids = [root.id, ...[...root.querySelectorAll('[id]')].map((node) => node.id)];
    return ids.filter((id, index) => id && ids.indexOf(id) !== index);
  });
  expect(duplicateIds).toEqual([]);

  const motion = await page.locator('llm-scaling-laws').evaluate((root) => {
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

test('chapter mounts its local component and renders MathJax SVG without cross-origin requests', async ({ page }) => {
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
  await expect(page.locator('.interactive-mount > llm-scaling-laws')).toHaveCount(1);
  await expect(page.locator('p > llm-scaling-laws')).toHaveCount(0);
  await expect(page.locator('llm-scaling-laws#fig-pretraining-scaling')).toBeVisible();
  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toContain('/llm-book/assets/js/components/scaling-laws.js');

  const formulas = page.locator('#quarto-document-content mjx-container[jax="SVG"]');
  await expect(formulas.first()).toBeVisible();
  expect(await formulas.count()).toBeGreaterThan(20);
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
  expect(invalidFormulas, `未完整渲染的公式：${JSON.stringify(invalidFormulas)}`).toEqual([]);
  expectCleanResources(monitor);
});

test('预训练章通过 WCAG 2.1 AA axe 扫描', async ({ page }) => {
  await openChapter(page);
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();

  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);
});

test('预训练章在移动端无页面、控件或长公式横向裁切', async ({ page }) => {
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
      controlOverflow: [...document.querySelectorAll('llm-scaling-laws .interactive-control')]
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
  }
});

test('chapter covers the required topics with exactly six exercises and resolved source links', async ({ page }) => {
  await openChapter(page);
  const content = page.locator('#quarto-document-content');
  for (const text of [
    '预训练目标',
    '数据质量与混合',
    'Token 化与数据管线',
    'Scaling Laws',
    '数据受限时的 Scaling',
    'Mixture of Experts',
    '从训练算力到推理成本',
    '常见误解',
  ]) {
    await expect(content).toContainText(text);
  }
  await expect(page.locator('#sec-exercises ol > li')).toHaveCount(6);
  for (const [name, href] of [
    ['Kaplan 等（2020）', 'https://arxiv.org/abs/2001.08361'],
    ['Hoffmann 等（2022）的 Chinchilla 工作', 'https://arxiv.org/abs/2203.15556'],
    ['Muennighoff 等（2023）', 'https://arxiv.org/abs/2305.16264'],
    ['Shazeer 等（2017）', 'https://arxiv.org/abs/1701.06538'],
    ['Fedus 等（2021）', 'https://arxiv.org/abs/2101.03961'],
  ]) {
    await expect(page.getByRole('link', { name })).toHaveAttribute('href', href);
  }
  await expect(content).not.toContainText('[?]');
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows the static scaling figure with an anchor and detailed alternative', async ({ page }) => {
    const monitor = await openChapter(page);
    const image = page.locator('noscript img[src="assets/figures/pretraining-scaling.svg"]');
    await expect(image).toBeVisible();
    await expect(image).toHaveAttribute('alt', /教学 Scaling 图/);
    await expect(image).toHaveAttribute('alt', /而非论文拟合值|不是论文拟合值|并非论文拟合值/);
    await expect(page.locator('#fig-pretraining-scaling')).toBeVisible();
    await expect(page.getByTestId('scaling-compute')).toHaveCount(0);
    await expect(page.getByTestId('scaling-ratio')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});
