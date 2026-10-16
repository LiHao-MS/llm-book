import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

const homePath = '/llm-book/index.html';
const chapterPath = '/llm-book/softmax-cross-entropy.html';
const localOrigin = 'http://127.0.0.1:4173';
const mountPath = '/llm-book/';
const pages = [
  { name: '首页', path: homePath },
  { name: '样板章', path: chapterPath },
];

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

for (const { name, path } of pages) {
  test(`${name}通过 WCAG 2.1 AA axe 扫描`, async ({ page }) => {
    await openPage(page, path);
    const results = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
      .analyze();

    expect(results.violations, JSON.stringify(axeSummary(results.violations), null, 2)).toEqual([]);
  });

  test(`${name}在 390×844 视口无水平溢出`, async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await openPage(page, path);

    const dimensions = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
    }));
    expect(
      dimensions.scrollWidth,
      `页面宽度 ${JSON.stringify(dimensions)}`,
    ).toBeLessThanOrEqual(dimensions.clientWidth);
  });
}

test('核心公式在拒绝所有跨源请求时仍由本地资源完整渲染', async ({ page }) => {
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

  const formulas = page.locator('mjx-container[jax="SVG"]');
  await expect(formulas.first()).toBeVisible();
  expect(await formulas.count()).toBeGreaterThan(10);
  const dimensions = await formulas.first().evaluate((element) => {
    const bounds = element.getBoundingClientRect();
    return { width: bounds.width, height: bounds.height };
  });
  expect(dimensions.width).toBeGreaterThan(0);
  expect(dimensions.height).toBeGreaterThan(0);
});

test('移动端长公式不产生横向裁切', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openPage(page, chapterPath);

  const overflow = await page.locator('mjx-container[display="true"]').evaluateAll((nodes) =>
    nodes
      .map((node, index) => ({
        index,
        text: node.textContent,
        clientWidth: node.clientWidth,
        scrollWidth: node.scrollWidth,
      }))
      .filter(({ scrollWidth, clientWidth }) => scrollWidth > clientWidth + 1),
  );

  expect(overflow, `以下公式被横向裁切：${JSON.stringify(overflow)}`).toEqual([]);
});

test('禁用 JavaScript 时两幅后备图可见且有替代文本', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await openPage(page, chapterPath);

  const fallbacks = [
    {
      src: 'assets/figures/cross-entropy-curve.svg',
      alternative: /交叉熵曲线图/,
    },
    {
      src: 'assets/figures/nucleus-sampling.svg',
      alternative: /Top-p/,
    },
  ];
  for (const fallback of fallbacks) {
    const image = page.locator(`noscript img[src="${fallback.src}"]`);
    await expect(image).toBeVisible();
    await expect(image).toHaveAttribute('alt', fallback.alternative);
  }

  await context.close();
});

test('所有章节禁用 JavaScript 时在移动端无水平溢出', async ({ browser }) => {
  const context = await browser.newContext({
    javaScriptEnabled: false,
    viewport: { width: 390, height: 844 },
  });
  for (const chapter of [
    'index',
    'softmax-cross-entropy',
    'embedding-geometry',
    'optimization',
    'attention',
    'transformer-architecture',
    'pretraining-scaling',
    'inference-systems',
    'rl-foundations',
    'rlhf-alignment',
    'frontier-rl',
  ]) {
    const page = await context.newPage();
    await openPage(page, `${mountPath}${chapter}.html`);
    const dimensions = await page.evaluate(() => ({
      clientWidth: document.documentElement.clientWidth,
      scrollWidth: document.documentElement.scrollWidth,
    }));
    expect(
      dimensions.scrollWidth,
      `${chapter} 在 390×844 视口的页面宽度 ${JSON.stringify(dimensions)}`,
    ).toBeLessThanOrEqual(dimensions.clientWidth);
    await page.close();
  }
  await context.close();
});

test('所有原生滑块都有显式标签和可见焦点', async ({ page }) => {
  await openPage(page, chapterPath);
  const sliders = page.getByRole('slider');
  await expect(sliders).toHaveCount(3);

  for (let index = 0; index < 3; index += 1) {
    const slider = sliders.nth(index);
    const id = await slider.getAttribute('id');
    expect(id).toBeTruthy();
    const label = page.locator(`label[for="${id}"]`);
    await expect(label).toHaveCount(1);
    await expect(label).toHaveText(/\S+/);

    await slider.focus();
    const outline = await slider.evaluate((element) => {
      const style = getComputedStyle(element);
      return { style: style.outlineStyle, width: Number.parseFloat(style.outlineWidth) };
    });
    expect(outline.style, `${id} 的焦点轮廓样式`).not.toBe('none');
    expect(outline.width, `${id} 的焦点轮廓宽度`).toBeGreaterThan(0);
  }
});

test('键盘方向键能操作所有原生滑块', async ({ page }) => {
  await openPage(page, chapterPath);
  const sliders = page.getByRole('slider');
  await expect(sliders).toHaveCount(3);

  for (let index = 0; index < 3; index += 1) {
    const slider = sliders.nth(index);
    const initial = Number(await slider.inputValue());
    const min = Number(await slider.getAttribute('min'));
    const max = Number(await slider.getAttribute('max'));
    const step = Number(await slider.getAttribute('step')) || 1;
    const forward = initial < max;
    const key = forward ? 'ArrowRight' : 'ArrowLeft';
    const reverseKey = forward ? 'ArrowLeft' : 'ArrowRight';
    const expected = Math.min(max, Math.max(min, initial + (forward ? step : -step)));

    await slider.focus();
    await slider.press(key);
    expect(Number(await slider.inputValue())).toBeCloseTo(expected, 10);
    await slider.press(reverseKey);
    expect(Number(await slider.inputValue())).toBeCloseTo(initial, 10);
  }
});

test('交互 SVG 都有有意义的名称和动态描述', async ({ page }) => {
  await openPage(page, chapterPath);
  const charts = [
    {
      locator: page.locator('llm-cross-entropy svg.interactive-chart'),
      name: '交叉熵损失曲线',
      description: /当前目标概率为 .*自然对数损失为 .*困惑度/,
    },
    {
      locator: page.getByTestId('decoding-distribution'),
      name: '温度与 Top-p 解码分布',
      description: /温度 .*Top-p .*保留 .*Token/,
    },
  ];

  for (const chart of charts) {
    await expect(chart.locator).toHaveAttribute('role', 'img');
    await expect(chart.locator).toHaveAccessibleName(chart.name);
    await expect(chart.locator).toHaveAccessibleDescription(chart.description);
  }
});

test('减少动态效果偏好下组件没有非必要动画', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await openPage(page, chapterPath);

  const motion = await page.evaluate(() => {
    const roots = [...document.querySelectorAll('llm-cross-entropy, llm-decoding-explorer')];
    const elements = roots.flatMap((root) => [root, ...root.querySelectorAll('*')]);
    const activeAnimations = elements.flatMap((element) =>
      element.getAnimations().map((animation) => ({
        element: element.tagName,
        playState: animation.playState,
      })),
    );
    const motionStyles = elements.flatMap((element) => {
      const style = getComputedStyle(element);
      const animationDurations = style.animationDuration
        .split(',')
        .map((duration) => Number.parseFloat(duration));
      const transitionDurations = style.transitionDuration
        .split(',')
        .map((duration) => Number.parseFloat(duration));
      const hasAnimation = style.animationName !== 'none'
        && animationDurations.some((duration) => duration > 0);
      const hasTransition = transitionDurations.some((duration) => duration > 0);
      return hasAnimation || hasTransition
        ? [{
            element: element.tagName,
            animationName: style.animationName,
            animationDuration: style.animationDuration,
            transitionDuration: style.transitionDuration,
          }]
        : [];
    });
    return { activeAnimations, motionStyles };
  });

  expect(motion.activeAnimations).toEqual([]);
  expect(motion.motionStyles).toEqual([]);
});

test('首页和样板章的站内链接、锚点及 PDF 均有效', async ({ page, request }) => {
  test.setTimeout(120_000);
  const targets = new Map();

  for (const { path } of pages) {
    await openPage(page, path);
    const hrefs = await page.locator('a[href]').evaluateAll((links) =>
      links.map((link) => link.href),
    );
    for (const href of hrefs) {
      const url = new URL(href);
      if (url.origin === localOrigin && url.pathname.startsWith(mountPath)) {
        targets.set(url.href, url);
      }
    }
  }

  expect([...targets.values()].some((url) => url.pathname.endsWith('.pdf'))).toBe(true);
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

const searchCases = [
  {
    query: '交叉熵',
    path: '/llm-book/softmax-cross-entropy.html',
    title: '从 Softmax 到交叉熵',
  },
  { query: '梯度裁剪', path: homePath, title: '理解大语言模型' },
  { query: '注意力', path: homePath, title: '理解大语言模型' },
  { query: '强化学习', path: homePath, title: '理解大语言模型' },
  { query: 'RLHF', path: homePath, title: '理解大语言模型' },
  { query: 'Scaling Law', path: homePath, title: '理解大语言模型' },
  {
    query: 'Transformer',
    path: '/llm-book/transformer-architecture.html',
    title: 'Transformer 架构与预训练',
  },
  {
    query: 'Scaling Laws',
    path: '/llm-book/pretraining-scaling.html',
    title: '预训练与 Scaling Laws',
  },
  {
    query: 'KV cache',
    path: '/llm-book/inference-systems.html',
    title: '推理与系统工程',
  },
  {
    query: 'RLVR',
    path: '/llm-book/frontier-rl.html',
    title: '前沿 RL、Agentic RL 与开放问题',
  },
  {
    query: 'Agentic RL',
    path: '/llm-book/frontier-rl.html',
    title: '前沿 RL、Agentic RL 与开放问题',
  },
  {
    query: 'reward hacking',
    path: '/llm-book/rlhf-alignment.html',
    title: '后训练、偏好优化与 RLHF',
  },
];

for (const { query, path, title } of searchCases) {
  test(`搜索“${query}”返回明确目标页`, async ({ page }) => {
    await openPage(page, homePath);
    const input = page.locator('#quarto-search .aa-Input');
    await expect(input).toBeVisible();
    await input.fill(query);
    const results = page.locator('#quarto-search-results a.search-result-link');

    await expect
      .poll(async () => {
        const matches = await results.evaluateAll(
          (links, expected) => links.some((link) => {
            const url = new URL(link.href);
            return url.pathname === expected.path && link.textContent.includes(expected.title);
          }),
          { path, title },
        );
        return matches;
      }, { message: `搜索结果应包含 ${title}（${path}）` })
      .toBe(true);
  });
}
