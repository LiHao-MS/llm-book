import { expect, test } from '@playwright/test';

const chapterPath = '/llm-book/embedding-geometry.html';
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

test('vector sliders update exact dot, cosine, and projection values at 390px', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const monitor = await openChapter(page);
  const angleA = page.getByTestId('vector-angle-a');
  const angleB = page.getByTestId('vector-angle-b');

  await expect(angleA).toHaveValue('45');
  await expect(angleB).toHaveValue('0');
  await angleA.fill('90');
  await angleB.fill('0');
  await expect(page.getByTestId('vector-dot')).toHaveText('0');
  await expect(page.getByTestId('vector-cosine')).toHaveText('0');
  await expect(page.getByTestId('vector-projection')).toHaveText('(0, 0)');

  await angleA.fill('45');
  await expect(page.getByTestId('vector-dot')).toHaveText('√2/2');
  await expect(page.getByTestId('vector-cosine')).toHaveText('√2/2');
  await expect(page.getByTestId('vector-projection')).toHaveText('(√2/2, 0)');

  const dimensions = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
  expectCleanResources(monitor);
});

test('matrix choices transform the point, basis vectors, grid, and SVG coordinates', async ({ page }) => {
  const monitor = await openChapter(page);
  const select = page.getByTestId('matrix-select');
  const originalPoint = page.getByTestId('matrix-point');
  const transformedPoint = page.getByTestId('matrix-transformed-point');
  const transformedMarker = page.getByTestId('matrix-transformed-point-marker');
  const secondBasisVector = page.getByTestId('matrix-ae2');
  const transformedGrid = page.getByTestId('matrix-transformed-grid');
  const transformedSquare = page.getByTestId('matrix-transformed-square');

  expect(await select.locator('option').evaluateAll((options) =>
    options.map((option) => option.value),
  )).toEqual([
    'identity',
    'rotate90',
    'project-x',
    'shear-x',
    'scale-xy',
  ]);
  const initialGrid = await transformedGrid.getAttribute('data-signature');
  const cases = [
    { value: 'identity', point: '(1, 0)', marker: [380, 190], ae2: [320, 130] },
    { value: 'rotate90', point: '(0, 1)', marker: [320, 130], ae2: [260, 190] },
    { value: 'project-x', point: '(1, 0)', marker: [380, 190], ae2: [320, 190] },
    { value: 'shear-x', point: '(1, 0)', marker: [380, 190], ae2: [380, 130] },
    { value: 'scale-xy', point: '(2, 0)', marker: [440, 190], ae2: [320, 160] },
  ];

  for (const { value, point, marker, ae2 } of cases) {
    await select.selectOption(value);
    await expect(originalPoint).toHaveText('(1, 0)');
    await expect(transformedPoint).toHaveText(point);
    await expect(transformedMarker).toHaveAttribute('cx', String(marker[0]));
    await expect(transformedMarker).toHaveAttribute('cy', String(marker[1]));
    await expect(secondBasisVector).toHaveAttribute('x2', String(ae2[0]));
    await expect(secondBasisVector).toHaveAttribute('y2', String(ae2[1]));
  }
  const labelsOverlap = await page.locator(
    'llm-matrix-transformer .matrix-first-basis-label, llm-matrix-transformer .matrix-transformed-point-label',
  ).evaluateAll(([basisLabel, pointLabel]) => {
    const basis = basisLabel.getBoundingClientRect();
    const point = pointLabel.getBoundingClientRect();
    return basis.left < point.right && basis.right > point.left
      && basis.top < point.bottom && basis.bottom > point.top;
  });
  expect(labelsOverlap).toBe(false);
  expect(await transformedGrid.getAttribute('data-signature')).not.toBe(initialGrid);

  await select.selectOption('rotate90');
  const firstGridSegment = transformedGrid.locator('line').first();
  await expect(firstGridSegment).toHaveAttribute('x1', '440');
  await expect(firstGridSegment).toHaveAttribute('y1', '310');
  await expect(firstGridSegment).toHaveAttribute('x2', '440');
  await expect(firstGridSegment).toHaveAttribute('y2', '70');
  await expect(transformedSquare).toHaveAttribute('points', '320,190 320,130 260,130 260,190');
  expectCleanResources(monitor);
});

test('native controls have labels, keyboard behavior, visible focus, and unique IDs', async ({ page }) => {
  const monitor = await openChapter(page);
  const controls = [
    { locator: page.getByTestId('vector-angle-a'), name: '向量 A 角度', key: 'ArrowRight', value: '46' },
    { locator: page.getByTestId('vector-angle-b'), name: '向量 B 角度', key: 'ArrowRight', value: '1' },
    { locator: page.getByTestId('matrix-select'), name: '矩阵变换', key: 'r', value: 'rotate90' },
  ];

  for (const { locator, name, key, value } of controls) {
    const expectedRole = name === '矩阵变换' ? 'combobox' : 'slider';
    await expect(page.getByRole(expectedRole, { name })).toHaveAttribute(
      'data-testid',
      await locator.getAttribute('data-testid'),
    );
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
    await locator.press(key);
    await expect(locator).toHaveValue(value);
  }

  const duplicateIds = await page.locator('llm-vector-projector, llm-matrix-transformer').evaluateAll(
    (roots) => {
      const ids = roots.flatMap((root) => [root.id, ...[...root.querySelectorAll('[id]')].map((node) => node.id)]);
      return ids.filter((id, index) => id && ids.indexOf(id) !== index);
    },
  );
  expect(duplicateIds).toEqual([]);
  expectCleanResources(monitor);
});

test('interactive SVG alternatives describe the current values', async ({ page }) => {
  const monitor = await openChapter(page);
  const vectorChart = page.getByTestId('vector-chart');
  const matrixChart = page.getByTestId('matrix-chart');

  await expect(vectorChart).toHaveAttribute('role', 'img');
  await expect(vectorChart).toHaveAccessibleName('向量投影与正交分解');
  await page.getByTestId('vector-angle-a').fill('90');
  await expect(vectorChart).toHaveAccessibleDescription(/90°.*内积为 0.*投影为 \(0, 0\)/);

  await expect(matrixChart).toHaveAttribute('role', 'img');
  await expect(matrixChart).toHaveAccessibleName('矩阵变换与基向量的像');
  await page.getByTestId('matrix-select').selectOption('rotate90');
  await expect(matrixChart).toHaveAccessibleDescription(/rotate90.*\(1, 0\).* \(0, 1\)/);
  expectCleanResources(monitor);
});

test('custom elements mount directly and all local ES modules load through the project subpath', async ({ page }) => {
  const monitor = await openChapter(page);

  await expect(page.locator('.interactive-mount > llm-vector-projector')).toHaveCount(1);
  await expect(page.locator('.interactive-mount > llm-matrix-transformer')).toHaveCount(1);
  await expect(page.locator('p > llm-vector-projector, p > llm-matrix-transformer')).toHaveCount(0);
  await expect(page.locator('mjx-container[jax="SVG"]').first()).toBeVisible();

  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toEqual(expect.arrayContaining([
    '/llm-book/assets/js/book.js',
    '/llm-book/assets/js/components/vector-projector.js',
    '/llm-book/assets/js/components/matrix-transformer.js',
    '/llm-book/assets/js/lib/linalg.js',
    '/llm-book/assets/figures/embedding-space.svg',
  ]));
  expect(paths.filter((path) => path === '/llm-book/assets/js/book.js')).toHaveLength(1);
  expectCleanResources(monitor);
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows both static geometry figures with anchors and alternatives', async ({ page }) => {
    const monitor = await openChapter(page);
    const fallbacks = [
      {
        anchor: 'fig-vector-projection',
        src: 'assets/figures/vector-projection.svg',
        alternative: /向量投影示意图/,
      },
      {
        anchor: 'fig-matrix-transform',
        src: 'assets/figures/matrix-transform.svg',
        alternative: /矩阵变换示意图/,
      },
    ];

    for (const { anchor, src, alternative } of fallbacks) {
      const image = page.locator(`noscript img[src="${src}"]`);
      await expect(image).toBeVisible();
      await expect(image).toHaveAttribute('alt', alternative);
      await expect(page.locator(`#${anchor}`)).toBeVisible();
    }
    await expect(page.getByTestId('vector-angle-a')).toHaveCount(0);
    await expect(page.getByTestId('matrix-select')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});
