import { expect, test } from '@playwright/test';

const chapterPath = '/llm-book/softmax-cross-entropy.html';
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

test('cross-entropy slider updates natural-log loss and perplexity', async ({ page }) => {
  const monitor = await openChapter(page);
  const probability = page.getByTestId('ce-probability');

  await expect(page.getByRole('slider', { name: '目标 Token 概率' })).toHaveAttribute(
    'data-testid',
    'ce-probability',
  );
  await expect(probability).toHaveValue('0.5');
  await probability.fill('0.25');

  await expect(page.getByTestId('ce-loss')).toHaveText(String(2 * Math.LN2));
  await expect(page.getByTestId('ce-perplexity')).toHaveText('4');
  expectCleanResources(monitor);
});

test('decoding Top-p keeps two tokens and normalizes in original order', async ({ page }) => {
  const monitor = await openChapter(page);
  const temperature = page.getByTestId('decoding-temperature');
  const topP = page.getByTestId('decoding-top-p');
  const distribution = page.getByTestId('decoding-distribution');

  await expect(page.getByRole('slider', { name: '温度' })).toHaveAttribute(
    'data-testid',
    'decoding-temperature',
  );
  await expect(page.getByRole('slider', { name: 'Top-p' })).toHaveAttribute(
    'data-testid',
    'decoding-top-p',
  );
  await expect(temperature).toHaveValue('1');
  await topP.fill('0.7');

  await expect(page.getByTestId('decoding-kept-count')).toHaveText('2');
  const tokens = await distribution.locator('[data-token]').evaluateAll((nodes) =>
    nodes.map((node) => node.getAttribute('data-token')),
  );
  const probabilities = await distribution.locator('[data-token]').evaluateAll((nodes) =>
    nodes.map((node) => Number(node.getAttribute('data-probability'))),
  );
  expect(tokens).toEqual(['的', '模型', '学习']);
  expect(probabilities[0]).toBe(0);
  expect(probabilities[1]).toBeCloseTo(0.625, 12);
  expect(probabilities[2]).toBeCloseTo(0.375, 12);
  expect(probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
  await expect(distribution.locator('[data-kept="false"]')).toContainText('排除');
  expectCleanResources(monitor);
});

test('native sliders work from the keyboard and charts have SVG descriptions', async ({ page }) => {
  const monitor = await openChapter(page);
  const probability = page.getByTestId('ce-probability');

  await probability.focus();
  await probability.press('ArrowRight');
  await expect(probability).toHaveValue('0.51');

  const decodingChart = page.getByTestId('decoding-distribution');
  await expect(decodingChart).toHaveAttribute('role', 'img');
  await expect(decodingChart.locator('title')).toHaveText('温度与 Top-p 解码分布');
  await expect(decodingChart.locator('desc')).not.toBeEmpty();
  expectCleanResources(monitor);
});

test('custom elements mount directly in their interactive containers', async ({ page }) => {
  const monitor = await openChapter(page);

  await expect(page.locator('.interactive-mount > llm-cross-entropy')).toHaveCount(1);
  await expect(page.locator('.interactive-mount > llm-decoding-explorer')).toHaveCount(1);
  expectCleanResources(monitor);
});

test('all local modules load through the Pages project subpath', async ({ page }) => {
  const monitor = await openChapter(page);
  await expect(page.getByTestId('ce-probability')).toBeVisible();

  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toEqual(
    expect.arrayContaining([
      '/llm-book/assets/js/book.js',
      '/llm-book/assets/js/components/cross-entropy.js',
      '/llm-book/assets/js/components/decoding-explorer.js',
      '/llm-book/assets/js/lib/probability.js',
      '/llm-book/assets/figures/next-token-pipeline.svg',
    ]),
  );
  expectCleanResources(monitor);
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows the same static cross-entropy and Top-p figures', async ({ page }) => {
    const monitor = await openChapter(page);

    await expect(page.locator('noscript img[src="assets/figures/cross-entropy-curve.svg"]')).toBeVisible();
    await expect(page.locator('noscript img[src="assets/figures/nucleus-sampling.svg"]')).toBeVisible();
    await expect(page.getByTestId('ce-probability')).toHaveCount(0);
    await expect(page.getByTestId('decoding-top-p')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});
