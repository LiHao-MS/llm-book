import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const chapterPath = '/llm-book/attention.html';
const localOrigin = 'http://127.0.0.1:4173';
const projectRoot = fileURLToPath(new URL('../../', import.meta.url));
const pdfPath = join(projectRoot, '_book', 'llm-book.pdf');

const originalScores = [
  [2, 1, 3],
  [1, 1, 1],
  [3, 1, 2],
];
const causalWeights = [
  [1, 0, 0],
  [0.5, 0.5, 0],
  [0.6652409558, 0.0900305732, 0.2447284711],
];
const unmaskedWeights = [
  [0.2447284711, 0.0900305732, 0.6652409558],
  [1 / 3, 1 / 3, 1 / 3],
  [0.6652409558, 0.0900305732, 0.2447284711],
];

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

function expectClose(actual, expected, tolerance = 2e-9) {
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tolerance);
}

async function matrixValues(table) {
  return table.locator('td[data-row][data-column]').evaluateAll((cells) => {
    const rows = [];
    for (const cell of cells) {
      const row = Number(cell.dataset.row);
      const column = Number(cell.dataset.column);
      rows[row] ??= [];
      rows[row][column] = Number(cell.textContent);
    }
    return rows;
  });
}

async function matrixText(table) {
  return table.locator('td[data-row][data-column]').evaluateAll((cells) => {
    const rows = [];
    for (const cell of cells) {
      const row = Number(cell.dataset.row);
      const column = Number(cell.dataset.column);
      rows[row] ??= [];
      rows[row][column] = cell.textContent.trim();
    }
    return rows;
  });
}

function expectMatrixClose(actual, expected, tolerance = 2e-9) {
  expect(actual.length).toBe(expected.length);
  actual.forEach((row, rowIndex) => {
    expect(row.length).toBe(expected[rowIndex].length);
    row.forEach((value, columnIndex) => {
      expectClose(value, expected[rowIndex][columnIndex], tolerance);
    });
  });
}

async function expectNormalizedRows(weights) {
  for (const row of weights) {
    expectClose(row.reduce((sum, value) => sum + value, 0), 1);
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

test('causal mask matches the static values and disabling it restores raw row Softmax', async ({ page }) => {
  const monitor = await openChapter(page);
  const root = page.locator('llm-causal-mask-explorer');
  const toggle = page.getByTestId('causal-mask-enabled');
  const rawTable = page.getByTestId('causal-raw-scores');
  const maskedTable = page.getByTestId('causal-masked-scores');
  const weightTable = page.getByTestId('causal-softmax-weights');
  const chart = page.getByTestId('causal-mask-chart');

  await expect(root).toBeVisible();
  await expect(toggle).toBeChecked();
  await expect(page.getByTestId('causal-query-row')).toHaveValue('0');
  expectMatrixClose(await matrixValues(rawTable), originalScores);
  expect(await matrixText(maskedTable)).toEqual([
    ['2', '−∞', '−∞'],
    ['1', '1', '−∞'],
    ['3', '1', '2'],
  ]);
  const initialWeights = await matrixValues(weightTable);
  expectMatrixClose(initialWeights, causalWeights);
  await expectNormalizedRows(initialWeights);
  for (let row = 0; row < 3; row += 1) {
    await expect(page.getByTestId(`causal-row-sum-${row}`)).toHaveText('1');
  }
  await expect(chart).toHaveAttribute('role', 'img');
  await expect(chart).toHaveAccessibleName('当前查询行的 Softmax 权重');
  await expect(chart).toHaveAccessibleDescription(/因果掩码已开启.*查询行 q0.*未来位置权重为 0.*行和为 1/);
  const initialDescription = await chart.locator('desc').textContent();

  await toggle.uncheck();
  expectMatrixClose(await matrixValues(rawTable), originalScores);
  expectMatrixClose(await matrixValues(maskedTable), originalScores);
  const rawSoftmaxWeights = await matrixValues(weightTable);
  expectMatrixClose(rawSoftmaxWeights, unmaskedWeights);
  await expectNormalizedRows(rawSoftmaxWeights);
  await expect(chart).toHaveAccessibleDescription(/因果掩码已关闭.*查询行 q0.*原始行 Softmax.*行和为 1/);
  expect(await chart.locator('desc').textContent()).not.toBe(initialDescription);

  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toEqual(expect.arrayContaining([
    '/llm-book/assets/js/components/causal-mask-explorer.js',
    '/llm-book/assets/js/lib/attention.js',
  ]));
  expectCleanResources(monitor);
});

test('query selection highlights that row in all three matrices and updates the SVG', async ({ page }) => {
  const monitor = await openChapter(page);
  const root = page.locator('llm-causal-mask-explorer');
  const query = page.getByTestId('causal-query-row');
  const chart = page.getByTestId('causal-mask-chart');

  await query.selectOption('2');
  const selectedRows = await root.locator('tbody tr.is-current-query').evaluateAll((rows) =>
    rows.map((row) => ({
      table: row.closest('table').dataset.testid,
      row: row.dataset.row,
      background: getComputedStyle(row.cells[1]).backgroundColor,
    })),
  );
  expect(selectedRows).toEqual([
    { table: 'causal-raw-scores', row: '2', background: 'rgb(220, 236, 244)' },
    { table: 'causal-masked-scores', row: '2', background: 'rgb(220, 236, 244)' },
    { table: 'causal-softmax-weights', row: '2', background: 'rgb(220, 236, 244)' },
  ]);
  expect(await root.locator('tbody tr.is-current-query').count()).toBe(3);
  await expect(chart).toHaveAccessibleDescription(
    /查询行 q2.*0\.6652409558, 0\.0900305732, 0\.2447284711.*行和为 1/,
  );
  const barWeights = await page.getByTestId('causal-weight-bar').evaluateAll((bars) =>
    bars.map((bar) => Number(bar.dataset.weight)),
  );
  expectMatrixClose([barWeights], [causalWeights[2]]);
  expectCleanResources(monitor);
});

test('controls are labelled, keyboard operable, focused, motion-free, and contained at 390px', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const monitor = await openChapter(page);
  const root = page.locator('llm-causal-mask-explorer');
  const query = page.getByRole('combobox', { name: '查询行 i' });
  const toggle = page.getByRole('checkbox', { name: '启用因果掩码' });

  await expect(query).toHaveAttribute('data-testid', 'causal-query-row');
  await expect(toggle).toHaveAttribute('data-testid', 'causal-mask-enabled');
  for (const control of [query, toggle]) {
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

  await query.focus();
  await query.press('q');
  await expect(query).toHaveValue('1');
  await expect(root.locator('tbody tr.is-current-query')).toHaveCount(3);
  await expect(page.getByTestId('causal-mask-chart')).toHaveAccessibleDescription(/查询行 q1/);
  await query.press('q');
  await expect(query).toHaveValue('2');
  await query.press('q');
  await expect(query).toHaveValue('0');

  await toggle.focus();
  await toggle.press('Space');
  await expect(toggle).not.toBeChecked();
  expectMatrixClose(await matrixValues(page.getByTestId('causal-softmax-weights')), unmaskedWeights);
  await toggle.press('Space');
  await expect(toggle).toBeChecked();
  expectMatrixClose(await matrixValues(page.getByTestId('causal-softmax-weights')), causalWeights);

  for (const name of ['因果原始分数', '因果掩码后分数', '因果 Softmax 权重']) {
    const region = page.getByRole('region', { name, exact: true });
    await expect(region).toHaveCount(1);
    await expect(region).toHaveAttribute('tabindex', '0');
    await region.focus();
    const outline = await region.evaluate((element) => {
      const style = getComputedStyle(element);
      return { style: style.outlineStyle, width: Number.parseFloat(style.outlineWidth) };
    });
    expect(outline.style).not.toBe('none');
    expect(outline.width).toBeGreaterThan(0);
  }

  const duplicateIds = await root.evaluate((element) => {
    const ids = [element.id, ...[...element.querySelectorAll('[id]')].map((node) => node.id)];
    return ids.filter((id, index) => id && ids.indexOf(id) !== index);
  });
  expect(duplicateIds).toEqual([]);
  await expect(page.locator('#fig-causal-mask')).toHaveCount(1);

  const motion = await root.evaluate((element) => {
    const elements = [element, ...element.querySelectorAll('*')];
    return elements.flatMap((node) => {
      const style = getComputedStyle(node);
      const animationDurations = style.animationDuration.split(',').map(Number.parseFloat);
      const transitionDurations = style.transitionDuration.split(',').map(Number.parseFloat);
      const hasMotion = (style.animationName !== 'none' && animationDurations.some((duration) => duration > 0))
        || transitionDurations.some((duration) => duration > 0);
      return [
        ...node.getAnimations().map((animation) => animation.playState),
        ...(hasMotion ? [style.animationName || style.transitionProperty] : []),
      ];
    });
  });
  expect(motion).toEqual([]);

  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
    widgetOverflow: [...document.querySelectorAll('llm-causal-mask-explorer .interactive-widget')]
      .map((element) => ({ clientWidth: element.clientWidth, scrollWidth: element.scrollWidth }))
      .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
    controlOverflow: [...document.querySelectorAll('llm-causal-mask-explorer .interactive-control')]
      .map((element) => ({ clientWidth: element.clientWidth, scrollWidth: element.scrollWidth }))
      .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
  }));
  expect(dimensions.scrollWidth, JSON.stringify(dimensions)).toBeLessThanOrEqual(dimensions.clientWidth);
  expect(dimensions.widgetOverflow).toEqual([]);
  expect(dimensions.controlOverflow).toEqual([]);
  expectCleanResources(monitor);
});

test('causal mask explorer passes a scoped WCAG 2.1 AA axe scan', async ({ page }) => {
  await openChapter(page);
  await expect(page.locator('llm-causal-mask-explorer')).toBeVisible();
  const results = await new AxeBuilder({ page })
    .include('llm-causal-mask-explorer')
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();
  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows the original static causal figure with a unique anchor and no controls', async ({ page }) => {
    const monitor = await openChapter(page);
    const image = page.locator('noscript img[src="assets/figures/causal-mask.svg"]');
    await expect(image).toBeVisible();
    await expect(image).toHaveAttribute('alt', /三乘三因果掩码数值图/);
    await expect(image).toHaveAttribute('alt', /0\.6652,0\.0900,0\.2447/);
    await expect(page.locator('#fig-causal-mask')).toHaveCount(1);
    await expect(page.getByTestId('causal-mask-enabled')).toHaveCount(0);
    await expect(page.getByTestId('causal-query-row')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});

test('Typst PDF retains the original static causal mask figure', () => {
  const text = execFileSync('pdftotext', ['-layout', pdfPath, '-'], {
    encoding: 'utf8',
    maxBuffer: 10 * 1024 * 1024,
  });
  const compact = text.replace(/\s+/g, '');
  for (const requiredText of [
    '因果掩码：未来权重必须在Softmax前消失',
    '原始分数S',
    '掩码后S+M',
    '注意力权重A',
    '0.6652',
    '0.0900',
    '0.2447',
    '三行权重和均为1',
  ]) {
    expect(compact).toContain(requiredText);
  }
});
