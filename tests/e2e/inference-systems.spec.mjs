import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const chapterPath = '/llm-book/inference-systems.html';
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

function expectClose(actual, expected, tolerance = 1e-9) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
}

async function metricValue(page, testid) {
  const output = page.getByTestId(testid);
  await expect(output).toHaveAttribute('data-value', /.+/);
  return Number(await output.getAttribute('data-value'));
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

test('KV cache calculator reproduces the worked example and scales linearly', async ({ page }) => {
  const monitor = await openChapter(page);

  await expect(page.getByRole('heading', { name: '推理与系统工程', level: 1 })).toBeVisible();
  await expect(page.getByTestId('kv-layers')).toHaveValue('2');
  await expect(page.getByTestId('kv-heads')).toHaveValue('2');
  await expect(page.getByTestId('kv-head-dim')).toHaveValue('64');
  await expect(page.getByTestId('kv-sequence-length')).toHaveValue('512');
  await expect(page.getByTestId('kv-batch-size')).toHaveValue('4');
  await expect(page.getByTestId('kv-bytes-per-element')).toHaveValue('2');
  await expect(page.getByTestId('kv-budget-mib')).toHaveValue('16');

  expectClose(await metricValue(page, 'kv-element-count'), 1048576);
  expectClose(await metricValue(page, 'kv-total-bytes'), 2097152);
  expectClose(await metricValue(page, 'kv-total-mib'), 2);
  expectClose(await metricValue(page, 'kv-per-request-mib'), 0.5);
  expectClose(await metricValue(page, 'kv-max-batch'), 32);
  await expect(page.getByTestId('kv-summary')).toContainText('2BLSH_kvd_hb');

  await page.getByTestId('kv-sequence-length').fill('1024');
  expectClose(await metricValue(page, 'kv-total-bytes'), 4194304);
  expectClose(await metricValue(page, 'kv-total-mib'), 4);
  expectClose(await metricValue(page, 'kv-per-request-mib'), 1);
  expectClose(await metricValue(page, 'kv-max-batch'), 16);

  await page.getByTestId('kv-batch-size').fill('2');
  expectClose(await metricValue(page, 'kv-total-mib'), 2);
  expectClose(await metricValue(page, 'kv-per-request-mib'), 1);
  expectClose(await metricValue(page, 'kv-max-batch'), 16);

  const figure = page.locator('img[src="assets/figures/inference-systems.svg"]');
  await expect(figure).toBeVisible();
  await expect(figure).toHaveAttribute('alt', /四面板推理系统教学图/);
  expectCleanResources(monitor);
});

test('service cost calculator keeps throughput, request rate, and cost units consistent', async ({ page }) => {
  const monitor = await openChapter(page);

  expectClose(await metricValue(page, 'cost-effective-throughput'), 20);
  expectClose(await metricValue(page, 'cost-hourly-total'), 8);
  expectClose(await metricValue(page, 'cost-per-million'), 8 * 1e6 / (3600 * 20));
  expectClose(await metricValue(page, 'cost-per-request'), 8 * 250 / (3600 * 20));
  expectClose(await metricValue(page, 'cost-requests-per-hour'), 288);
  await expect(page.getByTestId('cost-per-million')).toContainText('成本单位');
  await expect(page.getByTestId('cost-summary')).toContainText('利用率设为 100%');

  await page.getByTestId('cost-aggregate-throughput').fill('80');
  expectClose(await metricValue(page, 'cost-effective-throughput'), 40);
  expectClose(await metricValue(page, 'cost-per-million'), 8 * 1e6 / (3600 * 40));
  expectClose(await metricValue(page, 'cost-per-request'), 8 * 250 / (3600 * 40));
  expectClose(await metricValue(page, 'cost-requests-per-hour'), 576);

  await page.getByTestId('cost-hourly-price').fill('0');
  await expect(page.getByTestId('cost-hourly-price')).toHaveAttribute('aria-invalid', 'false');
  expectClose(await metricValue(page, 'cost-per-million'), 0);
  expectClose(await metricValue(page, 'cost-per-request'), 0);
  expectClose(await metricValue(page, 'cost-requests-per-hour'), 576);
  expectCleanResources(monitor);
});

test('invalid calculator input has accessible errors and clears stale metrics', async ({ page }) => {
  const monitor = await openChapter(page);
  const kvSequence = page.getByTestId('kv-sequence-length');
  const kvError = page.getByTestId('kv-sequence-length-error');

  await kvSequence.fill('0');
  await expect(kvSequence).toHaveAttribute('aria-invalid', 'true');
  await expect(kvError).toBeVisible();
  await expect(kvError).toHaveText('请输入 1 到 10000000 之间的整数。');
  await expect(kvSequence).toHaveAccessibleDescription(/Token.*请输入 1 到 10000000 之间的整数。/);
  await expect(page.getByTestId('kv-cache-form')).toHaveAttribute('data-valid', 'false');
  for (const testid of [
    'kv-element-count',
    'kv-total-bytes',
    'kv-total-mib',
    'kv-per-request-mib',
    'kv-max-batch',
  ]) {
    await expect(page.getByTestId(testid)).toHaveText('');
    await expect(page.getByTestId(testid)).not.toHaveAttribute('data-value', /.+/);
  }

  await kvSequence.fill('512');
  await expect(kvSequence).toHaveAttribute('aria-invalid', 'false');
  await expect(kvError).toBeHidden();
  expectClose(await metricValue(page, 'kv-total-mib'), 2);

  await page.getByTestId('kv-layers').fill('1.5');
  await expect(page.getByTestId('kv-layers-error')).toBeVisible();
  await expect(page.getByTestId('kv-cache-form')).toHaveAttribute('data-valid', 'false');
  await page.getByTestId('kv-layers').fill('2');
  expectClose(await metricValue(page, 'kv-total-mib'), 2);

  const utilization = page.getByTestId('cost-utilization');
  await utilization.fill('0');
  await expect(utilization).toHaveAttribute('aria-invalid', 'true');
  await expect(page.getByTestId('cost-utilization-error')).toHaveText('请输入大于 0 且不超过 100 的数值。');
  await expect(page.getByTestId('inference-cost-form')).toHaveAttribute('data-valid', 'false');
  for (const testid of [
    'cost-effective-throughput',
    'cost-hourly-total',
    'cost-per-million',
    'cost-per-request',
    'cost-requests-per-hour',
  ]) {
    await expect(page.getByTestId(testid)).toHaveText('');
  }

  await utilization.fill('50');
  expectClose(await metricValue(page, 'cost-effective-throughput'), 20);
  expectCleanResources(monitor);
});

test('all native inputs are labeled, keyboard operable, unique, and free of animation', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const monitor = await openChapter(page);
  const inputs = page.getByRole('spinbutton');
  await expect(inputs).toHaveCount(12);

  for (let index = 0; index < 12; index += 1) {
    const input = inputs.nth(index);
    const id = await input.getAttribute('id');
    expect(id).toBeTruthy();
    await expect(page.locator(`label[for="${id}"]`)).toHaveCount(1);
    await expect(input).toHaveAttribute('required', '');
    await expect(input).toHaveAttribute('aria-invalid', 'false');

    await input.focus();
    const outline = await input.evaluate((element) => {
      const style = getComputedStyle(element);
      return { style: style.outlineStyle, width: Number.parseFloat(style.outlineWidth) };
    });
    expect(outline.style).not.toBe('none');
    expect(outline.width).toBeGreaterThan(0);

    const initial = await input.inputValue();
    await input.press('ArrowUp');
    expect(await input.inputValue()).not.toBe(initial);
    await input.fill(initial);
    await expect(input).toHaveValue(initial);
  }

  const duplicateIds = await page.locator('llm-kv-cache-calculator, llm-inference-cost-calculator')
    .evaluateAll((roots) => {
      const ids = roots.flatMap((root) => [
        root.id,
        ...[...root.querySelectorAll('[id]')].map((node) => node.id),
      ]);
      return ids.filter((id, index) => id && ids.indexOf(id) !== index);
    });
  expect(duplicateIds).toEqual([]);

  const motion = await page.locator('llm-kv-cache-calculator, llm-inference-cost-calculator')
    .evaluateAll((roots) => {
      const elements = roots.flatMap((root) => [root, ...root.querySelectorAll('*')]);
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

test('chapter passes WCAG 2.1 AA and has no mobile page, control, or formula clipping', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const monitor = await openChapter(page);
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);

  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    controlOverflow: [...document.querySelectorAll('.inference-calculator-form .interactive-control')]
      .map((node, index) => ({ index, clientWidth: node.clientWidth, scrollWidth: node.scrollWidth }))
      .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
    formulaOverflow: [...document.querySelectorAll('mjx-container[display="true"]')]
      .map((node, index) => ({
        index,
        text: node.textContent,
        clientWidth: node.clientWidth,
        scrollWidth: node.scrollWidth,
      }))
      .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
  }));
  expect(dimensions.scrollWidth, JSON.stringify(dimensions)).toBeLessThanOrEqual(dimensions.clientWidth);
  expect(dimensions.controlOverflow, JSON.stringify(dimensions.controlOverflow)).toEqual([]);
  expect(dimensions.formulaOverflow, JSON.stringify(dimensions.formulaOverflow)).toEqual([]);
  expectCleanResources(monitor);
});

test('formulas and all runtime resources remain local with cross-origin requests blocked', async ({ page }) => {
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
  expect(invalidFormulas, JSON.stringify(invalidFormulas)).toEqual([]);

  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toContain('/llm-book/assets/js/components/inference-calculator.js');
  expect(paths).toContain('/llm-book/assets/figures/inference-systems.svg');
  expectCleanResources(monitor);
});

test.describe('inference chapter without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows accessible SVG and complete static calculator results', async ({ page }) => {
    await openChapter(page);
    const figure = page.locator('img[src="assets/figures/inference-systems.svg"]');
    await expect(figure).toBeVisible();
    await expect(figure).toHaveAttribute('alt', /面板 B.*2 MiB/);

    const tables = page.locator('noscript table');
    await expect(tables).toHaveCount(2);
    await expect(tables.nth(0)).toContainText('2,097,152 字节 = 2 MiB');
    await expect(tables.nth(0)).toContainText('32');
    await expect(tables.nth(1)).toContainText('111.111111 成本单位');
    await expect(tables.nth(1)).toContainText('288 请求/小时');
    await expect(page.getByRole('spinbutton')).toHaveCount(0);
    await expect(page.getByTestId('kv-layers')).toHaveCount(0);
    await expect(page.getByTestId('cost-gpu-count')).toHaveCount(0);
  });
});
