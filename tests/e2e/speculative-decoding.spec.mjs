import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

const chapterPath = '/llm-book/inference-systems.html';
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

async function expectMetric(page, testid, expected) {
  const output = page.getByTestId(testid);
  await expect(output).toHaveAttribute('data-value', /.+/);
  const actual = Number(await output.getAttribute('data-value'));
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(1e-12);
  await expect(output).toHaveText(String(Number(expected.toFixed(6))));
}

async function expectClearedResults(page) {
  const outputs = page.locator('llm-speculative-decoding output[data-testid]');
  expect(await outputs.count()).toBeGreaterThan(0);
  for (let index = 0; index < await outputs.count(); index += 1) {
    await expect(outputs.nth(index)).toHaveText('');
    await expect(outputs.nth(index)).not.toHaveAttribute('data-value', /.+/);
  }
  await expect(page.getByTestId('speculative-residual-state')).toHaveText('');
  await expect(page.getByTestId('speculative-probability-bars')).toBeEmpty();
  await expect(page.getByTestId('speculative-chart')).not.toHaveAttribute('data-signature', /.+/);
}

test('worked example, unreachable draft token, and identical distributions update every result', async ({ page }) => {
  const monitor = await openChapter(page);

  await expect(page.locator('#sec-speculative-distribution-demo')).toHaveCount(1);
  await expect(page.locator('llm-speculative-decoding#fig-speculative-decoding')).toBeVisible();
  await expect(page.getByTestId('speculative-preset')).toHaveValue('worked');
  for (const [token, p, q, acceptance, residual] of [
    ['A', 0.5, 0.4, 1, 1],
    ['B', 0.3, 0.4, 0.75, 0],
    ['C', 0.2, 0.2, 1, 0],
  ]) {
    await expectMetric(page, `speculative-target-${token}`, p);
    await expectMetric(page, `speculative-draft-${token}`, q);
    await expectMetric(page, `speculative-acceptance-${token}`, acceptance);
    await expectMetric(page, `speculative-residual-${token}`, residual);
  }
  await expectMetric(page, 'speculative-total-acceptance', 0.9);
  await expect(page.getByTestId('speculative-residual-state')).toContainText('[1, 0, 0]');
  await expect(page.getByTestId('speculative-summary')).toContainText('保持目标分布');
  await expect(page.getByTestId('speculative-summary')).toContainText('减少严格串行');
  await expect(page.getByTestId('speculative-summary')).toContainText('不是测试时搜索');

  const chart = page.getByTestId('speculative-chart');
  await expect(chart).toHaveAccessibleName('投机解码目标与草稿分布');
  await expect(chart).toHaveAccessibleDescription(/总接受率 0\.9/);
  await expect(chart).toHaveAccessibleDescription(/残差分布 \[1, 0, 0\]/);
  await expect(page.getByTestId('speculative-probability-bars').locator(':scope > g')).toHaveCount(3);

  await page.getByTestId('speculative-preset').selectOption('unreachable');
  await expectMetric(page, 'speculative-total-acceptance', 0.6);
  await expectMetric(page, 'speculative-acceptance-A', 2 / 3);
  await expect(page.getByTestId('speculative-acceptance-B')).toHaveText('null（q = 0，草稿不可达）');
  await expect(page.getByTestId('speculative-acceptance-B')).not.toHaveAttribute('data-value', /.+/);
  await expectMetric(page, 'speculative-acceptance-C', 0.5);
  for (const [token, residual] of [['A', 0], ['B', 1], ['C', 0]]) {
    await expectMetric(page, `speculative-residual-${token}`, residual);
  }
  await expect(chart).toHaveAccessibleDescription(/总接受率 0\.6/);
  await expect(chart).toHaveAccessibleDescription(/B.*null/);
  await expect(chart).toHaveAccessibleDescription(/残差分布 \[0, 1, 0\]/);

  await page.getByTestId('speculative-preset').selectOption('identical');
  await expectMetric(page, 'speculative-total-acceptance', 1);
  for (const token of ['A', 'B', 'C']) {
    await expectMetric(page, `speculative-acceptance-${token}`, 1);
    await expect(page.getByTestId(`speculative-residual-${token}`)).toHaveText('—');
    await expect(page.getByTestId(`speculative-residual-${token}`)).not.toHaveAttribute('data-value', /.+/);
  }
  await expect(page.getByTestId('speculative-residual-state')).toContainText('null');
  await expect(chart).toHaveAccessibleDescription(/无拒绝质量/);

  const registration = await page.evaluate(async () => {
    const module = await import('./assets/js/components/speculative-decoding.js');
    module.defineSpeculativeDecodingElement();
    module.defineSpeculativeDecodingElement();
    return customElements.get('llm-speculative-decoding') !== undefined;
  });
  expect(registration).toBe(true);
  const resourcePaths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(resourcePaths).toContain('/llm-book/assets/js/components/speculative-decoding.js');
  expectCleanResources(monitor);
});

test('invalid UI input exposes an accessible error and clears every stale result', async ({ page }) => {
  const monitor = await openChapter(page);
  const targetA = page.getByTestId('speculative-input-target-A');
  const error = page.getByTestId('speculative-error');

  await targetA.fill('0.6');
  await expect(page.getByTestId('speculative-form')).toHaveAttribute('data-valid', 'false');
  await expect(targetA).toHaveAttribute('aria-invalid', 'true');
  await expect(error).toBeVisible();
  await expect(error).toContainText('目标分布的概率之和必须为 1');
  await expect(targetA).toHaveAccessibleDescription(/目标分布的概率之和必须为 1/);
  await expectClearedResults(page);

  await targetA.fill('0.5');
  await expect(page.getByTestId('speculative-form')).toHaveAttribute('data-valid', 'true');
  await expect(error).toBeHidden();
  await expectMetric(page, 'speculative-total-acceptance', 0.9);

  const draftA = page.getByTestId('speculative-input-draft-A');
  await draftA.fill('1.1');
  await expect(draftA).toHaveAttribute('aria-invalid', 'true');
  await expect(error).toContainText('0 到 1');
  await expect(draftA).toHaveAccessibleDescription(/0 到 1/);
  await expectClearedResults(page);
  await expect(page.locator('llm-speculative-decoding')).not.toContainText('NaN');

  await draftA.fill('0.4');
  await expectMetric(page, 'speculative-total-acceptance', 0.9);
  expectCleanResources(monitor);
});

test('native controls are labeled, keyboard operable, unique, and motion free at 390px', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const monitor = await openChapter(page);
  const host = page.locator('llm-speculative-decoding');
  const controls = host.locator('select, input');
  await expect(controls).toHaveCount(7);
  await expect(host.getByRole('textbox')).toHaveCount(6);

  for (let index = 0; index < 7; index += 1) {
    const control = controls.nth(index);
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

  const preset = page.getByTestId('speculative-preset');
  await preset.focus();
  await expect(preset).toBeFocused();

  const targetA = page.getByTestId('speculative-input-target-A');
  await targetA.focus();
  await targetA.press('End');
  await targetA.pressSequentially('5');
  await expect(targetA).toHaveValue('0.55');
  await targetA.fill('0.5');
  await expectMetric(page, 'speculative-total-acceptance', 0.9);

  const duplicateIds = await host.evaluate((root) => {
    const ids = [root.id, ...[...root.querySelectorAll('[id]')].map((node) => node.id)];
    return ids.filter((id, index) => id && ids.indexOf(id) !== index);
  });
  expect(duplicateIds).toEqual([]);

  const motion = await host.evaluate((root) => {
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

test('speculative decoding widget passes axe and has no 390px clipping', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const monitor = await openChapter(page);
  await expect(page.locator('llm-speculative-decoding')).toBeVisible();
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);

  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    controlOverflow: [...document.querySelectorAll('llm-speculative-decoding .interactive-control')]
      .map((node, index) => ({ index, clientWidth: node.clientWidth, scrollWidth: node.scrollWidth }))
      .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
    widgetOverflow: [...document.querySelectorAll('llm-speculative-decoding .interactive-widget')]
      .map((node, index) => ({ index, clientWidth: node.clientWidth, scrollWidth: node.scrollWidth }))
      .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
    tableOverflow: [...document.querySelectorAll('llm-speculative-decoding table')]
      .map((node, index) => ({ index, clientWidth: node.clientWidth, scrollWidth: node.scrollWidth }))
      .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
  }));
  expect(dimensions.scrollWidth, JSON.stringify(dimensions)).toBeLessThanOrEqual(dimensions.clientWidth);
  expect(dimensions.controlOverflow, JSON.stringify(dimensions)).toEqual([]);
  expect(dimensions.widgetOverflow, JSON.stringify(dimensions)).toEqual([]);
  expect(dimensions.tableOverflow, JSON.stringify(dimensions)).toEqual([]);
  expectCleanResources(monitor);
});

test.describe('speculative decoding without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows the static distribution figure and complete worked-example results', async ({ page }) => {
    const monitor = await openChapter(page);
    const figure = page.locator('noscript img[src="assets/figures/speculative-decoding.svg"]');
    await expect(figure).toBeVisible();
    await expect(figure).toHaveAttribute('alt', /目标分布.*草稿分布/);
    await expect(figure).toHaveAttribute('alt', /0\.9/);
    await expect(figure).toHaveAttribute('alt', /残差.*1, 0, 0/);
    await expect(page.locator('#fig-speculative-decoding')).toBeVisible();

    const results = page.locator('noscript [data-testid="speculative-static-results"]');
    await expect(results).toHaveCount(1);
    await expect(results.locator('li')).toHaveCount(4);
    await expect(results).toContainText('投机解码静态分布算例');
    await expect(results).toContainText('0.75');
    await expect(results).toContainText('总接受率');
    await expect(results).toContainText('0.9');
    const fallbackNote = page.locator('#sec-speculative-distribution-demo noscript p').last();
    await expect(fallbackNote).toContainText('保持目标分布');
    await expect(fallbackNote).toContainText('不是测试时搜索');
    await expect(page.locator('llm-speculative-decoding input')).toHaveCount(0);
    await expect(page.locator('llm-speculative-decoding').getByRole('textbox')).toHaveCount(0);
    await expect(page.getByRole('combobox', { name: '分布预设' })).toHaveCount(0);
    expectCleanResources(monitor);
  });
});

test('Typst PDF retains the speculative decoding figure and static distribution fallback', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  const compact = text.replace(/\s+/g, '');
  for (const requiredText of [
    '草稿提出，目标验证',
    '投机解码静态分布算例',
    '0.75',
    '总接受率',
    '0.9',
    '拒绝后归一化残差',
    '不是测试时搜索',
  ]) {
    expect(compact).toContain(requiredText);
  }
});
