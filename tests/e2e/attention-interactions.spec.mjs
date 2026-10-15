import { expect, test } from '@playwright/test';

const chapterPath = '/llm-book/attention.html';
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

function expectMatrixClose(actual, expected, tolerance = 1e-9) {
  expect(actual.length).toBe(expected.length);
  actual.forEach((row, rowIndex) => {
    expect(row.length).toBe(expected[rowIndex].length);
    row.forEach((value, columnIndex) => {
      expectClose(value, expected[rowIndex][columnIndex], tolerance);
    });
  });
}

function expectVectorClose(text, expected, tolerance = 1e-9) {
  const actual = JSON.parse(text.replace('(', '[').replace(')', ']'));
  expect(actual.length).toBe(expected.length);
  actual.forEach((value, index) => expectClose(value, expected[index], tolerance));
}

test('attention controls recompute real matrices, normalized rows, output, and SVG bars', async ({ page }) => {
  const monitor = await openChapter(page);
  const query = page.getByTestId('attention-query-index');
  const scale = page.getByTestId('attention-scale');
  const scores = page.getByTestId('attention-scores');
  const weights = page.getByTestId('attention-weights');
  const output = page.getByTestId('attention-output');
  const distribution = page.getByTestId('attention-distribution');
  const outputChart = page.getByTestId('attention-output-chart');

  await expect(query).toHaveValue('0');
  await expect(scale).toHaveValue('0.707106781');
  expectMatrixClose(await matrixValues(scores), [
    [Math.SQRT1_2, 0, Math.SQRT1_2],
    [0, Math.SQRT1_2, Math.SQRT1_2],
    [Math.SQRT1_2, Math.SQRT1_2, Math.sqrt(2)],
  ], 3e-9);
  const expectedWeights = [
    [0.4011120927, 0.1977758146, 0.4011120927],
    [0.1977758146, 0.4011120927, 0.4011120927],
    [0.2482550783, 0.2482550783, 0.5034898435],
  ];
  const actualWeights = await matrixValues(weights);
  expectMatrixClose(actualWeights, expectedWeights);
  for (const row of actualWeights) {
    expectClose(row.reduce((sum, value) => sum + value, 0), 1, 2e-9);
  }
  for (let row = 0; row < 3; row += 1) {
    await expect(page.getByTestId(`attention-row-sum-${row}`)).toHaveText('1');
  }
  expectVectorClose(await output.textContent(), [0.8022241854, 0.5988879073]);
  await expect(page.getByTestId('attention-weighted-result')).toContainText('0.4011120927×(1, 0)');
  await expect(page.getByTestId('attention-weighted-result')).toContainText('= (0.8022241853, 0.5988879073)');

  const initialDistribution = await distribution.getAttribute('data-signature');
  const initialOutputChart = await outputChart.getAttribute('data-signature');
  await query.fill('1');
  expectVectorClose(await output.textContent(), [0.5988879073, 0.8022241854]);
  expect(await distribution.getAttribute('data-signature')).not.toBe(initialDistribution);
  expect(await outputChart.getAttribute('data-signature')).not.toBe(initialOutputChart);

  await query.fill('0');
  await scale.fill('1');
  expectMatrixClose(await matrixValues(scores), [
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 2],
  ]);
  expectMatrixClose(await matrixValues(weights), [
    [0.4223187983, 0.1553624035, 0.4223187983],
    [0.1553624035, 0.4223187983, 0.4223187983],
    [0.2119415576, 0.2119415576, 0.5761168848],
  ]);
  await expect(output).toHaveText('(0.8446375965, 0.5776812017)');
  const firstWeightBar = page.getByTestId('attention-weight-bar').first();
  expectClose(Number(await firstWeightBar.getAttribute('data-weight')), 0.4223187983);
  expectClose(Number(await firstWeightBar.getAttribute('height')), 140 * 0.4223187983, 1e-6);
  const firstOutputBar = page.getByTestId('attention-output-bar').first();
  expectClose(Number(await firstOutputBar.getAttribute('data-output')), 0.8446375965);
  expectClose(Number(await firstOutputBar.getAttribute('height')), 140 * 0.8446375965, 1e-6);
  expectCleanResources(monitor);
});

test('scale supports keyboard-only direct numeric exploration with exact initial display', async ({ page }) => {
  const monitor = await openChapter(page);
  const scale = page.getByTestId('attention-scale');
  const scaleValue = page.getByTestId('attention-scale-value');
  const distribution = page.getByTestId('attention-distribution');
  const outputChart = page.getByTestId('attention-output-chart');
  const initialDistribution = await distribution.getAttribute('data-signature');
  const initialOutputChart = await outputChart.getAttribute('data-signature');

  await expect(scale).toHaveValue('0.707106781');
  await expect(scaleValue).toHaveText('0.707106781');
  await scale.focus();
  await scale.press('ControlOrMeta+A');
  await scale.pressSequentially('1');

  await expect(scale).toHaveValue('1');
  await expect(scale).toHaveAttribute('type', 'number');
  await expect(scale).toHaveAttribute('step', 'any');
  await expect(scaleValue).toHaveText('1');
  expectMatrixClose(await matrixValues(page.getByTestId('attention-scores')), [
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 2],
  ]);
  expectMatrixClose(await matrixValues(page.getByTestId('attention-weights')), [
    [0.4223187983, 0.1553624035, 0.4223187983],
    [0.1553624035, 0.4223187983, 0.4223187983],
    [0.2119415576, 0.2119415576, 0.5761168848],
  ]);
  await expect(page.getByTestId('attention-output')).toHaveText('(0.8446375965, 0.5776812017)');
  await expect(page.getByTestId('attention-weighted-result')).toContainText('= (0.8446375965, 0.5776812017)');
  expect(await distribution.getAttribute('data-signature')).not.toBe(initialDistribution);
  expect(await outputChart.getAttribute('data-signature')).not.toBe(initialOutputChart);
  expectClose(
    Number(await page.getByTestId('attention-weight-bar').first().getAttribute('data-weight')),
    0.4223187983,
  );
  expectClose(
    Number(await page.getByTestId('attention-output-bar').first().getAttribute('data-output')),
    0.8446375965,
  );
  expectCleanResources(monitor);
});

test('scale uses aligned constraints and is valid from first render', async ({ page }) => {
  const monitor = await openChapter(page);
  const scale = page.getByTestId('attention-scale');

  await expect(scale).toHaveValue('0.707106781');
  await expect(scale).toHaveAttribute('min', '0.1');
  await expect(scale).toHaveAttribute('max', '2');
  await expect(scale).toHaveAttribute('step', 'any');
  await expect(scale).toHaveAttribute('required', '');
  await expect(scale).toHaveAttribute('aria-invalid', 'false');
  const validity = await scale.evaluate((element) => ({
    valid: element.validity.valid,
    valueMissing: element.validity.valueMissing,
    stepMismatch: element.validity.stepMismatch,
    rangeUnderflow: element.validity.rangeUnderflow,
    rangeOverflow: element.validity.rangeOverflow,
  }));
  expect(validity).toEqual({
    valid: true,
    valueMissing: false,
    stepMismatch: false,
    rangeUnderflow: false,
    rangeOverflow: false,
  });
  await expect(page.getByTestId('attention-scale-error')).toBeHidden();
  expectCleanResources(monitor);
});

test('scale rejects invalid input with an accessible error and no stale results', async ({ page }) => {
  const monitor = await openChapter(page);
  const scale = page.getByTestId('attention-scale');
  const error = page.getByTestId('attention-scale-error');
  const root = page.locator('llm-attention-matrix');

  for (const value of ['0', '-1', '', '0.05', '2.01']) {
    await scale.fill(value);
    await expect(scale).toHaveValue(value);
    await expect(scale).toHaveAttribute('aria-invalid', 'true');
    await expect(error).toBeVisible();
    await expect(error).toHaveText('请输入 0.1 到 2 之间的数值。');
    const errorId = await error.getAttribute('id');
    expect(errorId).toBeTruthy();
    await expect(scale).toHaveAttribute('aria-describedby', errorId);
    await expect(scale).toHaveAccessibleDescription('请输入 0.1 到 2 之间的数值。');

    for (const testId of ['attention-scale-value', 'attention-output', 'attention-weighted-result']) {
      await expect(page.getByTestId(testId)).toHaveText('');
    }
    const cleared = await root.evaluate((element) => ({
      nonEmptyValues: [...element.querySelectorAll(
        'td[data-row][data-column], td[data-testid^="attention-row-sum-"], .attention-weight-value, .attention-output-value',
      )].map((node) => node.textContent.trim()).filter(Boolean),
      signatures: element.querySelectorAll('[data-signature]').length,
      nonZeroBars: [...element.querySelectorAll('.attention-weight-bar, .attention-output-bar')]
        .filter((bar) => Number(bar.getAttribute('height')) !== 0).length,
      descriptions: [...element.querySelectorAll('desc')].map((node) => node.textContent),
    }));
    expect(cleared).toEqual({
      nonEmptyValues: [],
      signatures: 0,
      nonZeroBars: 0,
      descriptions: ['', ''],
    });
  }

  await scale.fill('1');
  await expect(scale).toHaveAttribute('aria-invalid', 'false');
  await expect(error).toBeHidden();
  await expect(page.getByTestId('attention-output')).toHaveText('(0.8446375965, 0.5776812017)');
  expectCleanResources(monitor);
});

test('RoPE slider rotates (1, 0) counterclockwise at all quarter turns and updates the SVG', async ({ page }) => {
  const monitor = await openChapter(page);
  const angle = page.getByTestId('rope-angle');
  const cosine = page.getByTestId('rope-cos');
  const sine = page.getByTestId('rope-sin');
  const rotation = page.getByTestId('rope-rotation');
  const point = page.getByTestId('rope-point');
  const vector = page.getByTestId('rope-rotated-vector');
  const chart = page.getByTestId('rope-chart');

  await expect(angle).toHaveValue('90');
  await expect(page.getByTestId('rope-original')).toHaveText('(1, 0)');
  const cases = [
    { angle: '0', cos: '1', sin: '0', vector: '(1, 0)', x: 440, y: 180 },
    { angle: '90', cos: '0', sin: '1', vector: '(0, 1)', x: 320, y: 60 },
    { angle: '180', cos: '-1', sin: '0', vector: '(-1, 0)', x: 200, y: 180 },
    { angle: '-90', cos: '0', sin: '-1', vector: '(0, -1)', x: 320, y: 300 },
  ];
  let previousSignature = await chart.getAttribute('data-signature');
  for (const expected of cases) {
    await angle.fill(expected.angle);
    await expect(cosine).toHaveText(expected.cos);
    await expect(sine).toHaveText(expected.sin);
    await expect(rotation).toHaveText(expected.vector);
    await expect(point).toHaveAttribute('cx', String(expected.x));
    await expect(point).toHaveAttribute('cy', String(expected.y));
    await expect(vector).toHaveAttribute('x2', String(expected.x));
    await expect(vector).toHaveAttribute('y2', String(expected.y));
    const signature = await chart.getAttribute('data-signature');
    expect(signature).not.toBe(previousSignature);
    previousSignature = signature;
  }
  await expect(page.getByTestId('rope-original-vector')).toHaveAttribute('x2', '440');
  await expect(page.getByTestId('rope-original-vector')).toHaveAttribute('y2', '180');
  expectCleanResources(monitor);
});

test('attention and RoPE controls have labels, keyboard behavior, visible focus, and dynamic descriptions', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const monitor = await openChapter(page);
  const controls = [
    { testid: 'attention-query-index', name: '查询索引 i', role: 'slider' },
    { testid: 'attention-scale', name: '缩放系数 scale', role: 'spinbutton' },
    { testid: 'rope-angle', name: '旋转角度 φ', role: 'slider' },
  ];

  for (const { testid, name, role } of controls) {
    const locator = page.getByTestId(testid);
    await expect(page.getByRole(role, { name })).toHaveAttribute('data-testid', testid);
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
    await locator.press(role === 'spinbutton' ? 'ArrowUp' : 'ArrowRight');
    expect(await locator.inputValue()).not.toBe(initial);
    if (role === 'spinbutton') {
      await locator.press('ControlOrMeta+A');
      await locator.pressSequentially(initial);
    } else {
      await locator.press('ArrowLeft');
    }
    await expect(locator).toHaveValue(initial);
  }

  for (const name of ['真实缩放分数矩阵', 'Softmax 权重矩阵']) {
    const region = page.getByRole('region', { name });
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

  const distribution = page.getByTestId('attention-distribution');
  const outputChart = page.getByTestId('attention-output-chart');
  const ropeChart = page.getByTestId('rope-chart');
  await expect(distribution).toHaveAttribute('role', 'img');
  await expect(distribution).toHaveAccessibleName('当前查询的 Softmax 权重分布');
  await expect(outputChart).toHaveAccessibleName('当前查询的 Value 加权输出');
  await page.getByTestId('attention-query-index').fill('2');
  await expect(distribution).toHaveAccessibleDescription(/查询索引 2.*Softmax 权重.*行和为 1/);
  await expect(outputChart).toHaveAccessibleDescription(/查询索引 2.*0\.7517449217/);

  await expect(ropeChart).toHaveAttribute('role', 'img');
  await expect(ropeChart).toHaveAccessibleName('RoPE 二维旋转');
  await page.getByTestId('rope-angle').fill('-90');
  await expect(ropeChart).toHaveAccessibleDescription(/-90°.*\(0, -1\).*逆时针/);

  const duplicateIds = await page.locator('llm-attention-matrix, llm-rope-rotation').evaluateAll((roots) => {
    const ids = roots.flatMap((root) => [
      root.id,
      ...[...root.querySelectorAll('[id]')].map((node) => node.id),
    ]);
    return ids.filter((id, index) => id && ids.indexOf(id) !== index);
  });
  expect(duplicateIds).toEqual([]);

  const motion = await page.locator('llm-attention-matrix, llm-rope-rotation').evaluateAll((roots) => {
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
      controlOverflow: [...document.querySelectorAll('.attention-controls .interactive-control')]
        .map((element, index) => ({
          index,
          clientWidth: element.clientWidth,
          scrollWidth: element.scrollWidth,
        }))
        .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
    }));
    expect(dimensions.scrollWidth, `页面在 ${width}px 视口不应横向溢出`).toBeLessThanOrEqual(dimensions.clientWidth);
    expect(dimensions.controlOverflow, `控件在 ${width}px 视口不应横向溢出`).toEqual([]);
  }
  expectCleanResources(monitor);
});

test('custom elements mount directly and load only local attention modules', async ({ page }) => {
  const monitor = await openChapter(page);

  await expect(page.locator('.interactive-mount > llm-attention-matrix')).toHaveCount(1);
  await expect(page.locator('.interactive-mount > llm-rope-rotation')).toHaveCount(1);
  await expect(page.locator('p > llm-attention-matrix, p > llm-rope-rotation')).toHaveCount(0);
  await expect(page.locator('llm-attention-matrix#fig-attention-scores')).toBeVisible();
  await expect(page.locator('llm-rope-rotation#fig-rope-rotation')).toBeVisible();

  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toEqual(expect.arrayContaining([
    '/llm-book/assets/js/book.js',
    '/llm-book/assets/js/components/attention-matrix.js',
    '/llm-book/assets/js/components/rope-rotation.js',
    '/llm-book/assets/js/lib/attention.js',
    '/llm-book/assets/figures/causal-mask.svg',
    '/llm-book/assets/figures/multi-head.svg',
  ]));
  expect(paths.filter((path) => path === '/llm-book/assets/js/book.js')).toHaveLength(1);
  expectCleanResources(monitor);
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows both static attention figures with anchors and alternatives', async ({ page }) => {
    const monitor = await openChapter(page);
    const fallbacks = [
      {
        anchor: 'fig-attention-scores',
        src: 'assets/figures/attention-scores.svg',
        alternative: /缩放点积注意力数值图/,
      },
      {
        anchor: 'fig-rope-rotation',
        src: 'assets/figures/rope-rotation.svg',
        alternative: /RoPE 几何图/,
      },
    ];

    for (const { anchor, src, alternative } of fallbacks) {
      const image = page.locator(`noscript img[src="${src}"]`);
      await expect(image).toBeVisible();
      await expect(image).toHaveAttribute('alt', alternative);
      await expect(page.locator(`#${anchor}`)).toBeVisible();
    }
    await expect(page.getByTestId('attention-query-index')).toHaveCount(0);
    await expect(page.getByTestId('rope-angle')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});
