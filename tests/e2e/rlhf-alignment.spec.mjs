import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const chapterPath = '/llm-book/rlhf-alignment.html';
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

test('RLHF and DPO demo recomputes the exact two-response model', async ({ page }) => {
  const monitor = await openChapter(page);
  const alpha = page.getByTestId('rlhf-alpha');
  const beta = page.getByTestId('rlhf-beta');
  const chart = page.getByTestId('rlhf-chart');
  const objectivePath = page.getByTestId('rlhf-objective-path');

  await expect(alpha).toHaveValue('1');
  await expect(beta).toHaveValue('1');
  await expect(page.getByTestId('rlhf-preferred-probability')).toHaveText('0.880797');
  await expect(page.getByTestId('rlhf-expected-reward')).toHaveText('0.761594');
  await expect(page.getByTestId('rlhf-kl')).toHaveText('0.327813');
  await expect(page.getByTestId('rlhf-kl-penalty')).toHaveText('0.327813');
  await expect(page.getByTestId('rlhf-objective-value')).toHaveText('0.433781');
  await expect(page.getByTestId('rlhf-dpo-loss')).toHaveText('0.126928');
  await expect(page.getByTestId('rlhf-optimal-alpha')).toHaveText('1');
  await expect(page.getByTestId('rlhf-optimum-line')).toHaveAttribute('x1', '390');
  expect(await objectivePath.getAttribute('d')).toMatch(/^M 70\.000 /);

  const initialSignature = await chart.getAttribute('data-signature');
  const initialObjectivePath = await objectivePath.getAttribute('d');
  await alpha.fill('2');
  await beta.fill('2');
  await expect(page.getByTestId('rlhf-preferred-probability')).toHaveText('0.982014');
  await expect(page.getByTestId('rlhf-expected-reward')).toHaveText('0.964028');
  await expect(page.getByTestId('rlhf-kl')).toHaveText('0.603052');
  await expect(page.getByTestId('rlhf-kl-penalty')).toHaveText('1.206105');
  await expect(page.getByTestId('rlhf-objective-value')).toHaveText('-0.242077');
  await expect(page.getByTestId('rlhf-dpo-loss')).toHaveText('0.000335');
  await expect(page.getByTestId('rlhf-optimal-alpha')).toHaveText('0.5');
  await expect(page.getByTestId('rlhf-optimum-line')).toHaveAttribute('x1', '230');
  expect(await chart.getAttribute('data-signature')).not.toBe(initialSignature);
  expect(await objectivePath.getAttribute('d')).not.toBe(initialObjectivePath);

  await beta.fill('0.5');
  await expect(page.getByTestId('rlhf-objective-value')).toHaveText('0.662501');
  await expect(page.getByTestId('rlhf-dpo-loss')).toHaveText('0.126928');
  await expect(page.getByTestId('rlhf-optimal-alpha')).toHaveText('2');
  await expect(page.getByTestId('rlhf-optimum-line')).toHaveAttribute('x1', '710');

  await alpha.fill('0');
  await beta.fill('2');
  await expect(page.getByTestId('rlhf-preferred-probability')).toHaveText('0.5');
  await expect(page.getByTestId('rlhf-expected-reward')).toHaveText('0');
  await expect(page.getByTestId('rlhf-kl')).toHaveText('0');
  await expect(page.getByTestId('rlhf-objective-value')).toHaveText('0');
  await expect(page.getByTestId('rlhf-dpo-loss')).toHaveText('0.693147');

  await page.getByTestId('rlhf-reset').click();
  await expect(alpha).toHaveValue('1');
  await expect(beta).toHaveValue('1');
  await expect(page.getByTestId('rlhf-objective-value')).toHaveText('0.433781');
  expectCleanResources(monitor);
});

test('native controls, dynamic SVG description, mounts, and local module stay coherent', async ({ page }) => {
  const monitor = await openChapter(page);
  const controls = [
    { testid: 'rlhf-alpha', name: '策略倾斜强度 α' },
    { testid: 'rlhf-beta', name: 'KL 系数 β' },
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

  const chart = page.getByTestId('rlhf-chart');
  await expect(chart).toHaveAttribute('role', 'img');
  await expect(chart).toHaveAccessibleName('RLHF 目标与 DPO 偏好损失对比');
  await page.getByTestId('rlhf-alpha').fill('2');
  await page.getByTestId('rlhf-beta').fill('2');
  await expect(chart).toHaveAccessibleDescription(
    /α 为 2，β 为 2.*0\.982014.*0\.964028.*0\.603052.*1\.206105.*-0\.242077.*0\.000335.*理论最优 α 为 0\.5/,
  );

  await page.getByTestId('rlhf-alpha').fill('0.5');
  await page.getByTestId('rlhf-reset').press('Space');
  await expect(page.getByTestId('rlhf-alpha')).toHaveValue('1');
  await expect(page.getByTestId('rlhf-beta')).toHaveValue('1');

  const duplicateIds = await page.locator('llm-rlhf-alignment').evaluateAll((roots) => {
    const ids = roots.flatMap((root) => [
      root.id,
      ...[...root.querySelectorAll('[id]')].map((node) => node.id),
    ]);
    return ids.filter((id, index) => id && ids.indexOf(id) !== index);
  });
  expect(duplicateIds).toEqual([]);

  const motion = await page.locator('llm-rlhf-alignment').evaluateAll((roots) => {
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

  await expect(page.locator('.interactive-mount > llm-rlhf-alignment')).toHaveCount(1);
  await expect(page.locator('p > llm-rlhf-alignment')).toHaveCount(0);
  await expect(page.locator('llm-rlhf-alignment#fig-rlhf-objective')).toBeVisible();
  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toContain('/llm-book/assets/js/components/rlhf-alignment.js');
  expect(paths.filter((path) => path === '/llm-book/assets/js/components/rlhf-alignment.js')).toHaveLength(1);
  expectCleanResources(monitor);
});

test('chapter covers the alignment pipeline and exactly six exercises', async ({ page }) => {
  const monitor = await openChapter(page);
  const sections = [
    ['sec-sft', 'SFT'],
    ['sec-preference-data', '偏好数据'],
    ['sec-reward-model', 'Bradley–Terry'],
    ['sec-rlhf-objective', 'RLHF'],
    ['sec-ppo-kl', 'PPO'],
    ['sec-dpo', 'DPO'],
    ['sec-process-reward', '过程奖励'],
    ['sec-reward-hacking', 'Reward hacking'],
    ['sec-evaluation', '评估'],
    ['sec-alignment-risk', '对齐风险'],
  ];
  for (const [id, text] of sections) {
    await expect(page.locator(`#${id}`)).toContainText(text);
  }
  await expect(page.locator('#sec-exercises > ol > li')).toHaveCount(6);
  await expect(page.locator('body')).not.toContainText('[?]');
  expectCleanResources(monitor);
});

test('chapter passes WCAG 2.1 AA axe scan', async ({ page }) => {
  await openChapter(page);
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);
});

test('formulas and interactive layout do not overflow a narrow viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const monitor = await openChapter(page);
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
    controlOverflow: [...document.querySelectorAll('.rlhf-widget .interactive-control')]
      .map((node, index) => ({ index, clientWidth: node.clientWidth, scrollWidth: node.scrollWidth }))
      .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
  }));

  expect(dimensions.scrollWidth, JSON.stringify(dimensions)).toBeLessThanOrEqual(dimensions.clientWidth);
  expect(dimensions.formulaOverflow, JSON.stringify(dimensions.formulaOverflow)).toEqual([]);
  expect(dimensions.controlOverflow, JSON.stringify(dimensions.controlOverflow)).toEqual([]);
  expectCleanResources(monitor);
});

test('MathJax SVG formulas render without cross-origin runtime requests', async ({ page }) => {
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
  const formulas = page.locator('mjx-container[jax="SVG"]');
  await expect(formulas.first()).toBeVisible();
  expect(await formulas.count()).toBeGreaterThan(40);
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
  expectCleanResources(monitor);
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows the static RLHF figure with anchor and detailed alternative', async ({ page }) => {
    const monitor = await openChapter(page);
    const image = page.locator('noscript img[src="assets/figures/rlhf-alignment.svg"]');
    await expect(image).toBeVisible();
    await expect(image).toHaveAttribute('alt', /后训练与偏好优化教学图/);
    await expect(image).toHaveAttribute('alt', /0\.880797/);
    await expect(image).toHaveAttribute('alt', /0\.433781/);
    await expect(page.locator('#fig-rlhf-objective')).toBeVisible();
    await expect(page.getByTestId('rlhf-alpha')).toHaveCount(0);
    await expect(page.getByTestId('rlhf-beta')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});
