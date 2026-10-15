import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const homePath = '/llm-book/index.html';
const chapterPath = '/llm-book/attention.html';
const localOrigin = 'http://127.0.0.1:4173';
const mountPath = '/llm-book/';

async function openPage(page, path) {
  await page.goto(path, { waitUntil: 'networkidle' });
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

test('注意力章通过 WCAG 2.1 AA axe 扫描', async ({ page }) => {
  await openPage(page, chapterPath);
  const results = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
    .analyze();

  expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);
});

test('注意力章在 390×844 视口无页面或长公式横向裁切', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openPage(page, chapterPath);

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

  await openPage(page, chapterPath);
  expect(offOriginRequests, `发现跨源运行时请求：${offOriginRequests.join(', ')}`).toEqual([]);

  const formulas = page.locator('#quarto-document-content mjx-container[jax="SVG"]');
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
});

test('注意力章的全部站内链接、锚点及 PDF 均有效', async ({ page, request }) => {
  test.setTimeout(120_000);
  await openPage(page, chapterPath);
  const hrefs = await page.locator('a[href]').evaluateAll((links) =>
    links.map((link) => link.href),
  );
  const targets = new Map();
  for (const href of hrefs) {
    const url = new URL(href);
    if (url.origin === localOrigin && url.pathname.startsWith(mountPath)) {
      targets.set(url.href, url);
    }
  }

  const targetPaths = [...targets.values()].map((url) => url.pathname);
  expect(targetPaths).toEqual(expect.arrayContaining([
    mountPath,
    '/llm-book/softmax-cross-entropy.html',
    '/llm-book/embedding-geometry.html',
    '/llm-book/optimization.html',
    chapterPath,
  ]));
  expect(targetPaths.some((path) => path.endsWith('.pdf'))).toBe(true);

  const failures = [];
  for (const url of targets.values()) {
    const resourceUrl = new URL(url);
    resourceUrl.hash = '';
    const response = await request.get(resourceUrl.href);
    if (response.status() !== 200) {
      failures.push(`${response.status()} ${url.href}`);
      continue;
    }

    if (url.hash) {
      await page.goto(url.href, { waitUntil: 'domcontentloaded' });
      const id = decodeURIComponent(url.hash.slice(1));
      const targetExists = await page.evaluate(
        (targetId) => Boolean(document.getElementById(targetId) || document.getElementsByName(targetId).length),
        id,
      );
      if (!targetExists) {
        failures.push(`缺少锚点目标 ${url.href}`);
      }
    }
  }

  expect(failures).toEqual([]);
});

test.describe('注意力章禁用 JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('显示两幅交互后备图和两幅静态教学图', async ({ page }) => {
    await openPage(page, chapterPath);
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
    await expect(page.locator('img[src="assets/figures/multi-head.svg"]')).toBeVisible();
    await expect(page.locator('img[src="assets/figures/causal-mask.svg"]')).toBeVisible();
    await expect(page.getByTestId('attention-query-index')).toHaveCount(0);
    await expect(page.getByTestId('rope-angle')).toHaveCount(0);
  });
});

for (const query of ['注意力', 'RoPE']) {
  test(`搜索“${query}”返回注意力章`, async ({ page }) => {
    await openPage(page, homePath);
    const input = page.locator('#quarto-search .aa-Input');
    await expect(input).toBeVisible();
    await input.fill(query);
    const results = page.locator('#quarto-search-results a.search-result-link');

    await expect
      .poll(async () => results.evaluateAll(
        (links, expected) => links.some((link) => {
          const url = new URL(link.href);
          return url.pathname === expected.path && link.textContent.includes(expected.title);
        }),
        { path: chapterPath, title: '序列建模与自注意力机制' },
      ), { message: `搜索结果应包含注意力章（${chapterPath}）` })
      .toBe(true);
  });
}
