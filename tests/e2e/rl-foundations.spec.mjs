import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const chapterPath = '/llm-book/rl-foundations.html';
const figurePath = '/llm-book/assets/figures/rl-foundations.svg';
const componentPath = '/llm-book/assets/js/components/rl-foundations.js';
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

test('value iteration applies synchronous Bellman updates and extracts greedy actions', async ({ page }) => {
  const monitor = await openChapter(page);
  const next = page.getByTestId('rl-next');
  const iteration = page.getByTestId('rl-iteration');
  const valueA = page.getByTestId('rl-value-a');
  const valueB = page.getByTestId('rl-value-b');
  const policyA = page.getByTestId('rl-policy-a');
  const policyB = page.getByTestId('rl-policy-b');

  await expect(iteration).toHaveText('0');
  await expect(valueA).toHaveText('0');
  await expect(valueB).toHaveText('0');
  await expect(policyA).toHaveText('a₀ 与 a₁ 并列');
  await expect(policyB).toHaveText('a₁');

  await next.click();
  await expect(iteration).toHaveText('1');
  await expect(valueA).toHaveText('0');
  await expect(valueB).toHaveText('1');
  await expect(policyA).toHaveText('a₁');
  await expect(policyB).toHaveText('a₁');
  await expect(page.getByTestId('rl-bar-a')).toHaveAttribute('width', '0');
  await expect(page.getByTestId('rl-bar-b')).toHaveAttribute('width', '200');

  await next.click();
  await expect(iteration).toHaveText('2');
  await expect(valueA).toHaveText('0.5');
  await expect(valueB).toHaveText('1.5');
  await expect(page.getByTestId('rl-bar-a')).toHaveAttribute('width', '100');
  await expect(page.getByTestId('rl-bar-b')).toHaveAttribute('width', '300');

  await next.click();
  await expect(valueA).toHaveText('0.75');
  await expect(valueB).toHaveText('1.75');
  await expect(page.getByTestId('rl-value-chart')).toHaveAccessibleDescription(
    /第 3 次迭代.*γ = 0\.5.*V\(A\) = 0\.75.*V\(B\) = 1\.75.*a₁/,
  );
  expectCleanResources(monitor);
});

test('changing discount resets the demo and reset restores its initial state', async ({ page }) => {
  const monitor = await openChapter(page);
  const gamma = page.getByTestId('rl-gamma');
  const next = page.getByTestId('rl-next');

  await gamma.fill('0.9');
  await expect(page.getByTestId('rl-gamma-value')).toHaveText('0.9');
  await next.click();
  await next.click();
  await expect(page.getByTestId('rl-value-a')).toHaveText('0.9');
  await expect(page.getByTestId('rl-value-b')).toHaveText('1.9');
  await expect(page.getByTestId('rl-bar-a')).toHaveAttribute('width', '36');
  await expect(page.getByTestId('rl-bar-b')).toHaveAttribute('width', '76');

  await page.getByTestId('rl-reset').click();
  await expect(page.getByTestId('rl-iteration')).toHaveText('0');
  await expect(page.getByTestId('rl-value-a')).toHaveText('0');
  await expect(page.getByTestId('rl-value-b')).toHaveText('0');
  await expect(page.getByTestId('rl-policy-a')).toHaveText('a₀ 与 a₁ 并列');
  await expect(gamma).toHaveValue('0.9');
  expectCleanResources(monitor);
});

test('native demo controls are labelled, keyboard-operable, and visibly focused', async ({ page }) => {
  const monitor = await openChapter(page);
  const gamma = page.getByRole('slider', { name: '折扣因子 γ' });
  await expect(gamma).toHaveAttribute('data-testid', 'rl-gamma');
  const id = await gamma.getAttribute('id');
  expect(id).toBeTruthy();
  await expect(page.locator(`label[for="${id}"]`)).toHaveCount(1);

  await gamma.focus();
  const outline = await gamma.evaluate((element) => {
    const style = getComputedStyle(element);
    return { style: style.outlineStyle, width: Number.parseFloat(style.outlineWidth) };
  });
  expect(outline.style).not.toBe('none');
  expect(outline.width).toBeGreaterThan(0);
  await gamma.press('ArrowRight');
  await expect(gamma).toHaveValue('0.55');
  await expect(page.getByTestId('rl-gamma-value')).toHaveText('0.55');
  await gamma.press('ArrowLeft');
  await expect(gamma).toHaveValue('0.5');

  await page.getByRole('button', { name: '价值迭代一步' }).press('Enter');
  await expect(page.getByTestId('rl-iteration')).toHaveText('1');
  await page.getByRole('button', { name: '重置价值' }).press('Space');
  await expect(page.getByTestId('rl-iteration')).toHaveText('0');

  const duplicateIds = await page.locator('llm-rl-foundations').evaluate((root) => {
    const ids = [root.id, ...[...root.querySelectorAll('[id]')].map((node) => node.id)];
    return ids.filter((value, index) => value && ids.indexOf(value) !== index);
  });
  expect(duplicateIds).toEqual([]);
  await expect(page.locator('.interactive-mount > llm-rl-foundations')).toHaveCount(1);
  await expect(page.locator('p > llm-rl-foundations')).toHaveCount(0);
  expectCleanResources(monitor);
});

test('interactive chart and local module stay coherent without motion', async ({ page }) => {
  const monitor = await openChapter(page);
  const chart = page.getByTestId('rl-value-chart');
  await expect(chart).toHaveAttribute('role', 'img');
  await expect(chart).toHaveAccessibleName('两状态 MDP 的价值与贪心策略');

  const motion = await page.locator('llm-rl-foundations').evaluate((root) => {
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

  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toContain(componentPath);
  expect(paths.filter((path) => path === componentPath)).toHaveLength(1);
  expectCleanResources(monitor);
});

test('强化学习章通过 WCAG 2.1 AA axe 扫描', async ({ page }) => {
  await openChapter(page);
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);
});

test('强化学习章在 390×844 视口无页面或长公式横向裁切', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openChapter(page);
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
  }));

  expect(dimensions.scrollWidth, JSON.stringify(dimensions)).toBeLessThanOrEqual(dimensions.clientWidth);
  expect(dimensions.formulaOverflow, JSON.stringify(dimensions.formulaOverflow)).toEqual([]);
});

test('核心公式在拒绝所有跨源请求时仍由本地 MathJax SVG 完整渲染', async ({ page }) => {
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
  expect(offOriginRequests, offOriginRequests.join(', ')).toEqual([]);
  const formulas = page.locator('mjx-container[jax="SVG"]');
  await expect(formulas.first()).toBeVisible();
  expect(await formulas.count()).toBeGreaterThan(35);
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
});

test('static SVG is a labelled image with a useful long description', async ({ request }) => {
  const response = await request.get(figurePath);
  expect(response.status()).toBe(200);
  const svg = await response.text();
  expect(svg).toContain('role="img"');
  expect(svg).toContain('aria-labelledby="title desc"');
  expect(svg).toMatch(/<title id="title">两状态 MDP/);
  expect(svg).toMatch(/<desc id="desc">.*价值迭代.*贪心策略.*<\/desc>/s);
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows the static value-policy figure and no interactive controls', async ({ page }) => {
    const monitor = await openChapter(page);
    const image = page.locator('noscript img[src="assets/figures/rl-foundations.svg"]');
    await expect(image).toBeVisible();
    await expect(image).toHaveAttribute('alt', /两状态 MDP 价值迭代图/);
    await expect(page.locator('#fig-rl-foundations')).toBeVisible();
    await expect(page.getByTestId('rl-gamma')).toHaveCount(0);
    await expect(page.getByTestId('rl-next')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});
