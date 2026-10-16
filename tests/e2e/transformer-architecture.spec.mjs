import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const chapterPath = '/llm-book/transformer-architecture.html';
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

test('publishes the complete chapter with six exercises and local MathJax formulas', async ({ page }) => {
  const monitor = await openChapter(page);
  expect(monitor.response.status()).toBe(200);
  await expect(page.getByRole('heading', { level: 1, name: 'Transformer 架构与预训练' })).toBeVisible();

  for (const anchor of [
    'sec-transformer-block',
    'sec-layernorm',
    'sec-ffn',
    'sec-position',
    'sec-training-objective',
    'sec-decoder-only',
    'sec-pretrain-sft',
    'sec-exercises',
  ]) {
    await expect(page.locator(`#${anchor}`)).toHaveCount(1);
  }
  for (const term of ['残差连接', 'LayerNorm', '前馈网络', '位置编码', 'decoder-only', '指令微调']) {
    await expect(page.locator('#quarto-document-content')).toContainText(term);
  }
  await expect(page.locator('section#sec-exercises > ol > li')).toHaveCount(6);
  const formulas = page.locator('#quarto-document-content mjx-container[jax="SVG"]');
  await expect(formulas.first()).toBeVisible();
  expect(await formulas.count()).toBeGreaterThan(20);
  await expect(page.locator('#quarto-document-content')).not.toContainText('[?]');
  expectCleanResources(monitor);
});

test('controls recompute all four Pre/Post-LN and residual states', async ({ page }) => {
  const monitor = await openChapter(page);
  const normPosition = page.getByTestId('transformer-norm-position');
  const residual = page.getByTestId('transformer-residual');
  const normalizedInput = page.getByTestId('transformer-normalized-input');
  const sublayerOutput = page.getByTestId('transformer-sublayer-output');
  const merged = page.getByTestId('transformer-merged-output');
  const output = page.getByTestId('transformer-block-output');
  const rule = page.getByTestId('transformer-block-rule');
  const chart = page.getByTestId('transformer-block-chart');
  const residualPath = page.getByTestId('transformer-residual-path');
  const residualLabel = page.getByTestId('transformer-residual-label');

  await expect(normPosition).toHaveValue('pre');
  await expect(residual).toBeChecked();
  await expect(normalizedInput).toHaveText('(0, -1.2247448714, 1.2247448714)');
  await expect(sublayerOutput).toHaveText('(0, 0, 1.2247448714)');
  await expect(merged).toHaveText('(1, -2, 5.2247448714)');
  await expect(output).toHaveText('(1, -2, 5.2247448714)');
  await expect(rule).toHaveText('y = x + F(LN(x))');
  await expect(residualPath).toHaveAttribute('visibility', 'visible');
  await expect(residualLabel).toHaveAttribute('visibility', 'visible');
  await expect(chart).toHaveAccessibleDescription(/Pre-LN.*启用残差连接.*5\.2247448714/);
  const initialSignature = await chart.getAttribute('data-signature');

  await residual.uncheck();
  await expect(merged).toHaveText('(0, 0, 1.2247448714)');
  await expect(output).toHaveText('(0, 0, 1.2247448714)');
  await expect(rule).toHaveText('y = F(LN(x))');
  await expect(residualPath).toHaveAttribute('visibility', 'hidden');
  await expect(residualLabel).toHaveAttribute('visibility', 'hidden');

  await normPosition.selectOption('post');
  await expect(normalizedInput).toHaveText('(1, -2, 4)');
  await expect(sublayerOutput).toHaveText('(1, 0, 4)');
  await expect(merged).toHaveText('(1, 0, 4)');
  await expect(output).toHaveText('(-0.3922322703, -0.9805806757, 1.372812946)');
  await expect(rule).toHaveText('y = LN(F(x))');
  await expect(chart).toHaveAccessibleDescription(/Post-LN.*关闭残差连接.*1\.372812946/);

  await residual.check();
  await expect(merged).toHaveText('(2, -2, 8)');
  await expect(output).toHaveText('(-0.1622214211, -1.1355499479, 1.297771369)');
  await expect(rule).toHaveText('y = LN(x + F(x))');
  await expect(residualPath).toHaveAttribute('visibility', 'visible');
  await expect(residualLabel).toHaveAttribute('visibility', 'visible');
  expect(await chart.getAttribute('data-signature')).not.toBe(initialSignature);

  await normPosition.selectOption('pre');
  await expect(output).toHaveText('(1, -2, 5.2247448714)');
  expectCleanResources(monitor);
});

test('native controls, dynamic SVG, module mounting, and mobile layout stay accessible', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const monitor = await openChapter(page);
  const normPosition = page.getByTestId('transformer-norm-position');
  const residual = page.getByTestId('transformer-residual');

  await expect(page.getByRole('combobox', { name: '归一化位置' })).toHaveAttribute(
    'data-testid',
    'transformer-norm-position',
  );
  await expect(page.getByRole('checkbox', { name: '启用残差连接' })).toHaveAttribute(
    'data-testid',
    'transformer-residual',
  );
  for (const control of [normPosition, residual]) {
    const id = await control.getAttribute('id');
    expect(id).toBeTruthy();
    await expect(page.locator(`label[for="${id}"]`)).toHaveCount(1);
    await control.focus();
    const outline = await control.evaluate((element) => {
      const style = getComputedStyle(element);
      return { style: style.outlineStyle, width: Number.parseFloat(style.outlineWidth) };
    });
    expect(outline.style).not.toBe('none');
    expect(outline.width).toBeGreaterThan(0);
  }

  await normPosition.press('p');
  await expect(normPosition).toHaveValue('post');
  await normPosition.press('p');
  await expect(normPosition).toHaveValue('pre');
  await residual.press('Space');
  await expect(residual).not.toBeChecked();
  await residual.press('Space');
  await expect(residual).toBeChecked();

  const chart = page.getByTestId('transformer-block-chart');
  await expect(chart).toHaveAttribute('role', 'img');
  await expect(chart).toHaveAccessibleName('Transformer Block 的归一化与残差数据流');
  await expect(page.locator('.interactive-mount > llm-transformer-block')).toHaveCount(1);
  await expect(page.locator('p > llm-transformer-block')).toHaveCount(0);
  await expect(page.locator('llm-transformer-block#fig-transformer-block')).toBeVisible();

  const duplicateIds = await page.locator('llm-transformer-block').evaluateAll((roots) => {
    const ids = roots.flatMap((root) => [
      root.id,
      ...[...root.querySelectorAll('[id]')].map((node) => node.id),
    ]);
    return ids.filter((id, index) => id && ids.indexOf(id) !== index);
  });
  expect(duplicateIds).toEqual([]);

  const motion = await page.locator('llm-transformer-block').evaluateAll((roots) => {
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

  for (const width of [390, 600, 768, 1280]) {
    await page.setViewportSize({ width, height: 844 });
    const dimensions = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
      formulaOverflow: [...document.querySelectorAll('mjx-container[display="true"]')]
        .map((node, index) => ({ index, clientWidth: node.clientWidth, scrollWidth: node.scrollWidth }))
        .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
    }));
    expect(dimensions.scrollWidth, `页面在 ${width}px 视口不应横向溢出`).toBeLessThanOrEqual(
      dimensions.clientWidth,
    );
    expect(dimensions.formulaOverflow, `公式在 ${width}px 视口不应被裁切`).toEqual([]);
  }

  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toContain('/llm-book/assets/js/components/transformer-block.js');
  expectCleanResources(monitor);
});

test('chapter passes WCAG 2.1 AA axe scanning', async ({ page }) => {
  await openChapter(page);
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);
});

test('formulas remain local and complete when every cross-origin request is rejected', async ({ page }) => {
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

  await openChapter(page);
  expect(offOriginRequests, `发现跨源运行时请求：${offOriginRequests.join(', ')}`).toEqual([]);
  const formulas = page.locator('#quarto-document-content mjx-container[jax="SVG"]');
  await expect(formulas.first()).toBeVisible();
  expect(await formulas.count()).toBeGreaterThan(20);
  const invalidFormulas = await formulas.evaluateAll((nodes) =>
    nodes.flatMap((node, index) => {
      const svg = node.querySelector('svg');
      if (!svg) {
        return [{ index, reason: 'missing SVG' }];
      }
      const bounds = svg.getBoundingClientRect();
      return bounds.width > 0 && bounds.height > 0
        ? []
        : [{ index, width: bounds.width, height: bounds.height }];
    }),
  );
  expect(invalidFormulas).toEqual([]);
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows the static Transformer Block figure with anchor and alternative', async ({ page }) => {
    const monitor = await openChapter(page);
    const image = page.locator('noscript img[src="assets/figures/transformer-block.svg"]');
    await expect(image).toBeVisible();
    await expect(image).toHaveAttribute('alt', /Pre-LN Transformer Block 数值图/);
    await expect(page.locator('#fig-transformer-block')).toBeVisible();
    await expect(page.getByTestId('transformer-norm-position')).toHaveCount(0);
    await expect(page.getByTestId('transformer-residual')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});
