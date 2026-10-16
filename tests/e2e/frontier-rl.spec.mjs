import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const chapterPath = '/llm-book/frontier-rl.html';
const figurePath = '/llm-book/assets/figures/frontier-rl.svg';
const componentPath = '/llm-book/assets/js/components/frontier-rl.js';
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

test('RLVR demo compares outcome, process, and reward-hacking signals exactly', async ({ page }) => {
  const monitor = await openChapter(page);
  const verifier = page.getByTestId('frontier-verifier');
  const candidate = page.getByTestId('frontier-candidate');
  const outcome = page.getByTestId('frontier-outcome');
  const bonus = page.getByTestId('frontier-bonus');
  const reward = page.getByTestId('frontier-reward');
  const advantage = page.getByTestId('frontier-advantage');

  await expect(verifier).toHaveValue('outcome');
  await expect(candidate).toHaveValue('decomposition');
  await expect(outcome).toHaveText('1');
  await expect(bonus).toHaveText('0');
  await expect(reward).toHaveText('1');
  await expect(advantage).toHaveText('0.57735');

  await verifier.selectOption('marker');
  await expect(advantage).toHaveText('0.390567');
  await candidate.selectOption('marker-correct');
  await expect(outcome).toHaveText('1');
  await expect(bonus).toHaveText('0.2');
  await expect(reward).toHaveText('1.2');
  await expect(advantage).toHaveText('0.911322');
  await expect(page.getByTestId('frontier-response')).toContainText('VERIFIER_OK');
  await expect(page.getByTestId('frontier-diagnosis')).toContainText('最高组内优势');

  await candidate.selectOption('marker-wrong');
  await expect(outcome).toHaveText('0');
  await expect(bonus).toHaveText('0.2');
  await expect(reward).toHaveText('0.2');
  await expect(advantage).toHaveText('-1.692456');

  await verifier.selectOption('checked');
  await expect(reward).toHaveText('0');
  await expect(advantage).toHaveText('-1.705606');
  await candidate.selectOption('decomposition');
  await expect(bonus).toHaveText('0.2');
  await expect(reward).toHaveText('1.2');
  await expect(advantage).toHaveText('0.852803');

  const bars = page.locator('[data-testid^="frontier-bar-"]');
  await expect(bars).toHaveCount(4);
  expect(await bars.evaluateAll((nodes) =>
    nodes.map((node) => node.getAttribute('data-reward')),
  )).toEqual(['1.2', '1', '1', '0']);
  expect(await bars.evaluateAll((nodes) =>
    nodes.map((node) => node.getAttribute('data-advantage')),
  )).toEqual(['0.852803', '0.426401', '0.426401', '-1.705606']);
  expectCleanResources(monitor);
});

test('native controls, dynamic SVG alternative, mount, and local module stay coherent', async ({ page }) => {
  const monitor = await openChapter(page);
  const verifier = page.getByRole('combobox', { name: '奖励规则' });
  const candidate = page.getByRole('combobox', { name: '候选回答' });

  for (const locator of [verifier, candidate]) {
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
  }

  await candidate.focus();
  await candidate.press('b');
  await expect(candidate).toHaveValue('answer');

  const chart = page.getByTestId('frontier-chart');
  await expect(chart).toHaveAttribute('role', 'img');
  await expect(chart).toHaveAccessibleName('四种回答的 RLVR 奖励与组内优势');
  await verifier.selectOption('marker');
  await candidate.selectOption('marker-correct');
  await expect(chart).toHaveAccessibleDescription(
    /结果加文本标记.*C：正确回答加伪造标记，奖励 1\.2，优势 0\.911322/,
  );
  await expect(page.getByTestId('frontier-diagnosis')).toHaveAttribute('aria-live', 'polite');

  await expect(page.locator('.interactive-mount > llm-frontier-rl')).toHaveCount(1);
  await expect(page.locator('p > llm-frontier-rl')).toHaveCount(0);
  const duplicateIds = await page.locator('llm-frontier-rl').evaluate((root) => {
    const ids = [root.id, ...[...root.querySelectorAll('[id]')].map((node) => node.id)];
    return ids.filter((id, index) => id && ids.indexOf(id) !== index);
  });
  expect(duplicateIds).toEqual([]);

  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toContain(componentPath);
  expectCleanResources(monitor);
});

test('chapter covers required topics, local MathJax, and exactly six exercises', async ({ page }) => {
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

  for (const anchor of [
    'sec-rlvr',
    'sec-process-reward',
    'sec-reward-hacking',
    'sec-agentic-rl',
    'sec-environment',
    'sec-long-horizon',
    'sec-agent-evaluation',
    'sec-interpretability',
    'sec-open-problems',
    'sec-exercises',
  ]) {
    await expect(page.locator(`#${anchor}`)).toHaveCount(1);
  }
  await expect(page.locator('#sec-exercises > ol > li')).toHaveCount(6);
  expect(offOriginRequests, `发现跨源运行时请求：${offOriginRequests.join(', ')}`).toEqual([]);

  const formulas = page.locator('mjx-container[jax="SVG"]');
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

test('frontier RL chapter passes WCAG 2.1 AA axe scan', async ({ page }) => {
  await openChapter(page);
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();

  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);
});

test('chapter and long formulas do not overflow a 390×844 viewport', async ({ page }) => {
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

  expect(
    dimensions.scrollWidth,
    `页面宽度 ${JSON.stringify(dimensions)}`,
  ).toBeLessThanOrEqual(dimensions.clientWidth);
  expect(
    dimensions.formulaOverflow,
    `以下公式被横向裁切：${JSON.stringify(dimensions.formulaOverflow)}`,
  ).toEqual([]);
});

test('interactive component has no non-essential animation', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await openChapter(page);
  const motion = await page.locator('llm-frontier-rl').evaluate((root) => {
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
});

test('static SVG preserves all three reward tables and accessible text', async ({ request }) => {
  const response = await request.get(figurePath);
  expect(response.status()).toBe(200);
  const svg = await response.text();
  expect(svg).toContain('role="img"');
  expect(svg).toContain('<title id="title">');
  expect(svg).toContain('<desc id="desc">');
  for (const text of [
    'A = 0.577350',
    'A = 0.390567',
    'A = 0.911322（最高）',
    'R = 0.2（错误仍得分）',
    'A = 0.852803',
    'A = −1.705606',
    '颜色不是唯一编码',
  ]) {
    expect(svg).toContain(text);
  }
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows the complete static comparison instead of controls', async ({ page }) => {
    const monitor = await openChapter(page);
    const image = page.locator('noscript img[src="assets/figures/frontier-rl.svg"]');
    await expect(image).toBeVisible();
    await expect(image).toHaveAttribute('alt', /RLVR 奖励劫持对比图/);
    await expect(page.locator('#fig-frontier-rl')).toBeVisible();
    await expect(page.getByTestId('frontier-verifier')).toHaveCount(0);
    await expect(page.getByTestId('frontier-candidate')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});
