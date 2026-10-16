import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const chapterPath = '/llm-book/rl-foundations.html';
const figurePath = '/llm-book/assets/figures/ppo-clipping.svg';
const componentPath = '/llm-book/assets/js/components/ppo-clipping.js';
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

test('PPO curves, current points, and values track ratio, epsilon, and advantage sign', async ({ page }) => {
  const monitor = await openChapter(page);
  const ratio = page.getByTestId('ppo-ratio');
  const epsilon = page.getByTestId('ppo-epsilon');
  const advantage = page.getByTestId('ppo-advantage');
  const chart = page.getByTestId('ppo-chart');
  const objectivePath = page.getByTestId('ppo-objective-path');

  await expect(ratio).toHaveValue('1.4');
  await expect(epsilon).toHaveValue('0.2');
  await expect(advantage).toHaveValue('1');
  await expect(page.getByTestId('ppo-ratio-value')).toHaveText('1.4');
  await expect(page.getByTestId('ppo-epsilon-value')).toHaveText('0.2');
  await expect(page.getByTestId('ppo-unclipped-value')).toHaveText('1.4');
  await expect(page.getByTestId('ppo-clipped-ratio-value')).toHaveText('1.2');
  await expect(page.getByTestId('ppo-clipped-value')).toHaveText('1.2');
  await expect(page.getByTestId('ppo-objective-value')).toHaveText('1.2');
  await expect(page.getByTestId('ppo-unclipped-marker')).toHaveAttribute('cx', '518');
  await expect(page.getByTestId('ppo-unclipped-marker')).toHaveAttribute('cy', '76.25');
  await expect(page.getByTestId('ppo-clipped-marker')).toHaveAttribute('cx', '518');
  await expect(page.getByTestId('ppo-clipped-marker')).toHaveAttribute('cy', '90');
  await expect(page.getByTestId('ppo-objective-marker')).toHaveAttribute('cx', '518');
  await expect(page.getByTestId('ppo-objective-marker')).toHaveAttribute('cy', '90');
  await expect(chart).toHaveAccessibleDescription(
    /当前概率比 ρ 为 1\.4，ε 为 0\.2，优势 A 为 1。未截断项为 1\.4，截断比率为 1\.2，截断项为 1\.2，PPO 截断目标为 1\.2。/,
  );

  const initialObjectivePath = await objectivePath.getAttribute('d');
  await advantage.selectOption('-1');
  await ratio.fill('0.6');
  await expect(page.getByTestId('ppo-unclipped-value')).toHaveText('-0.6');
  await expect(page.getByTestId('ppo-clipped-ratio-value')).toHaveText('0.8');
  await expect(page.getByTestId('ppo-clipped-value')).toHaveText('-0.8');
  await expect(page.getByTestId('ppo-objective-value')).toHaveText('-0.8');
  await expect(page.getByTestId('ppo-unclipped-marker')).toHaveAttribute('cx', '262');
  await expect(page.getByTestId('ppo-unclipped-marker')).toHaveAttribute('cy', '213.75');
  await expect(page.getByTestId('ppo-clipped-marker')).toHaveAttribute('cy', '227.5');
  await expect(page.getByTestId('ppo-objective-marker')).toHaveAttribute('cy', '227.5');
  await expect(chart).toHaveAccessibleDescription(
    /当前概率比 ρ 为 0\.6，ε 为 0\.2，优势 A 为 -1。未截断项为 -0\.6，截断比率为 0\.8，截断项为 -0\.8，PPO 截断目标为 -0\.8。/,
  );
  expect(await objectivePath.getAttribute('d')).not.toBe(initialObjectivePath);

  const negativeObjectivePath = await objectivePath.getAttribute('d');
  await epsilon.fill('0.1');
  await expect(page.getByTestId('ppo-epsilon-value')).toHaveText('0.1');
  await expect(page.getByTestId('ppo-unclipped-value')).toHaveText('-0.6');
  await expect(page.getByTestId('ppo-clipped-ratio-value')).toHaveText('0.9');
  await expect(page.getByTestId('ppo-clipped-value')).toHaveText('-0.9');
  await expect(page.getByTestId('ppo-objective-value')).toHaveText('-0.9');
  await expect(page.getByTestId('ppo-objective-marker')).toHaveAttribute('cy', '234.375');
  expect(await objectivePath.getAttribute('d')).not.toBe(negativeObjectivePath);

  await page.getByTestId('ppo-reset').click();
  await expect(ratio).toHaveValue('1.4');
  await expect(epsilon).toHaveValue('0.2');
  await expect(advantage).toHaveValue('1');
  await expect(page.getByTestId('ppo-objective-value')).toHaveText('1.2');
  expectCleanResources(monitor);
});

test('native PPO controls are labelled, keyboard-operable, and visibly focused', async ({ page }) => {
  const monitor = await openChapter(page);
  const controls = [
    { testid: 'ppo-ratio', name: '概率比 ρ' },
    { testid: 'ppo-epsilon', name: '截断范围 ε' },
    { testid: 'ppo-advantage', name: '优势估计 A' },
  ];

  for (const { testid, name } of controls) {
    const locator = page.getByTestId(testid);
    await expect(page.getByRole(testid === 'ppo-advantage' ? 'combobox' : 'slider', { name }))
      .toHaveAttribute('data-testid', testid);
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

  const ratio = page.getByTestId('ppo-ratio');
  await ratio.focus();
  await ratio.press('ArrowRight');
  await expect(ratio).toHaveValue('1.5');
  await expect(page.getByTestId('ppo-ratio-value')).toHaveText('1.5');
  await ratio.press('ArrowLeft');
  await expect(ratio).toHaveValue('1.4');

  const advantage = page.getByTestId('ppo-advantage');
  await advantage.focus();
  await advantage.press('ArrowDown');
  await expect(advantage).toHaveValue('-1');
  await expect(page.getByTestId('ppo-objective-value')).toHaveText('-1.4');

  await page.getByTestId('ppo-reset').press('Space');
  await expect(advantage).toHaveValue('1');
  await expect(page.getByTestId('ppo-objective-value')).toHaveText('1.2');
  expectCleanResources(monitor);
});

test('PPO mount has unique anchors, an idempotent local module, and no reduced-motion animation', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const monitor = await openChapter(page);
  const chart = page.getByTestId('ppo-chart');
  await expect(chart).toHaveAttribute('role', 'img');
  await expect(chart).toHaveAccessibleName('PPO 未截断项、截断项与 min 目标');

  const duplicateIds = await page.locator('llm-ppo-clipping').evaluate((root) => {
    const ids = [root.id, ...[...root.querySelectorAll('[id]')].map((node) => node.id)];
    return ids.filter((value, index) => value && ids.indexOf(value) !== index);
  });
  expect(duplicateIds).toEqual([]);
  await expect(page.locator('#fig-ppo-clipping')).toHaveCount(1);
  await expect(page.locator('.interactive-mount > llm-ppo-clipping')).toHaveCount(1);
  await expect(page.locator('p > llm-ppo-clipping')).toHaveCount(0);

  const motion = await page.locator('llm-ppo-clipping').evaluate((root) => {
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

  const registration = await page.evaluate(async () => {
    const module = await import('/llm-book/assets/js/components/ppo-clipping.js');
    module.definePpoClippingElement();
    module.definePpoClippingElement();
    return customElements.get('llm-ppo-clipping') !== undefined;
  });
  expect(registration).toBe(true);
  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toContain(componentPath);
  expect(paths.filter((path) => path === componentPath)).toHaveLength(1);
  expectCleanResources(monitor);
});

test('PPO clipping chapter passes WCAG 2.1 AA axe scan', async ({ page }) => {
  await openChapter(page);
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);
});

test('PPO controls and chart do not overflow a 390px viewport', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const monitor = await openChapter(page);
  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    controlOverflow: [...document.querySelectorAll('.ppo-widget .interactive-control')]
      .map((node, index) => ({ index, clientWidth: node.clientWidth, scrollWidth: node.scrollWidth }))
      .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
    chartOverflow: [...document.querySelectorAll('.ppo-widget .interactive-chart')]
      .map((node, index) => ({ index, clientWidth: node.clientWidth, scrollWidth: node.scrollWidth }))
      .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
  }));

  expect(dimensions.scrollWidth, JSON.stringify(dimensions)).toBeLessThanOrEqual(dimensions.clientWidth);
  expect(dimensions.controlOverflow, JSON.stringify(dimensions.controlOverflow)).toEqual([]);
  expect(dimensions.chartOverflow, JSON.stringify(dimensions.chartOverflow)).toEqual([]);
  expectCleanResources(monitor);
});

test('static PPO SVG is a labelled image with both exact examples and no hard-constraint claim', async ({ request }) => {
  const response = await request.get(figurePath);
  expect(response.status()).toBe(200);
  const svg = await response.text();
  expect(svg).toContain('role="img"');
  expect(svg).toContain('aria-labelledby="title desc"');
  expect(svg).toMatch(/<title id="title">PPO 截断目标/);
  expect(svg).toMatch(/<desc id="desc">.*未截断项.*截断项.*min.*1\.2.*-0\.8.*不是.*硬约束.*<\/desc>/s);
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows the static PPO figure with its anchor and no interactive controls', async ({ page }) => {
    const monitor = await openChapter(page);
    const image = page.locator('noscript img[src="assets/figures/ppo-clipping.svg"]');
    await expect(image).toBeVisible();
    await expect(image).toHaveAttribute('alt', /PPO 截断目标曲线/);
    await expect(image).toHaveAttribute('alt', /1\.2/);
    await expect(image).toHaveAttribute('alt', /-0\.8/);
    await expect(page.locator('#fig-ppo-clipping')).toBeVisible();
    await expect(page.getByTestId('ppo-ratio')).toHaveCount(0);
    await expect(page.getByTestId('ppo-epsilon')).toHaveCount(0);
    await expect(page.getByTestId('ppo-advantage')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});
