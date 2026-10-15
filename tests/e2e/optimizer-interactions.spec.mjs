import { expect, test } from '@playwright/test';

const chapterPath = '/llm-book/optimization.html';
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

test('optimizer controls reproduce SGD, Momentum, Adam, and AdamW updates within 40 steps', async ({ page }) => {
  const monitor = await openChapter(page);
  const algorithm = page.getByTestId('optimizer-algorithm');
  const next = page.getByTestId('optimizer-next');
  const reset = page.getByTestId('optimizer-reset');
  const step = page.getByTestId('optimizer-step');
  const point = page.getByTestId('optimizer-point');
  const loss = page.getByTestId('optimizer-loss');
  const velocity = page.getByTestId('optimizer-velocity');
  const trajectory = page.getByTestId('optimizer-trajectory-path');
  const fullTrajectory = page.getByTestId('optimizer-full-trajectory');

  expect(await algorithm.locator('option').evaluateAll((options) =>
    options.map((option) => option.value),
  )).toEqual(['sgd', 'momentum', 'adam', 'adamw']);
  await expect(step).toHaveText('0');
  await expect(point).toHaveText('(1, 1)');
  await expect(loss).toHaveText('2.5');

  await next.click();
  await expect(step).toHaveText('1');
  await expect(point).toHaveText('(0.8, 1.1)');
  await expect(loss).toHaveText('2.025');

  await algorithm.selectOption('momentum');
  await expect(velocity).toHaveText('(2, -1)');
  await next.click();
  await expect(step).toHaveText('2');
  await expect(point).toHaveText('(0.44, 1.28)');
  await expect(loss).toHaveText('1.296');
  await expect(velocity).toHaveText('(3.6, -1.8)');
  const momentumPath = await trajectory.getAttribute('d');

  await reset.click();
  await algorithm.selectOption('adam');
  await next.click();
  await expect(point).toHaveText('(0.9, 1.1)');
  const adamFullPath = await fullTrajectory.getAttribute('d');

  await algorithm.selectOption('adamw');
  await expect(step).toHaveText('1');
  await expect(point).toHaveText('(0.891, 1.089)');
  expect(await trajectory.getAttribute('d')).not.toBe(momentumPath);
  expect(await fullTrajectory.getAttribute('d')).not.toBe(adamFullPath);

  await page.getByTestId('optimizer-learning-rate').fill('0.2');
  await algorithm.selectOption('sgd');
  await expect(point).toHaveText('(0.6, 1.2)');
  await page.getByTestId('optimizer-step-input').fill('40');
  await expect(step).toHaveText('40');
  await expect(next).toBeDisabled();
  expectCleanResources(monitor);
});

test('optimizer disables controls ignored by the selected algorithm', async ({ page }) => {
  const monitor = await openChapter(page);
  const algorithm = page.getByTestId('optimizer-algorithm');
  const momentum = page.getByTestId('optimizer-momentum');
  const weightDecay = page.getByTestId('optimizer-weight-decay');

  for (const [value, momentumDisabled, weightDecayDisabled] of [
    ['sgd', true, true],
    ['momentum', false, true],
    ['adam', true, true],
    ['adamw', true, false],
  ]) {
    await algorithm.selectOption(value);
    if (momentumDisabled) {
      await expect(momentum).toBeDisabled();
    } else {
      await expect(momentum).toBeEnabled();
    }
    if (weightDecayDisabled) {
      await expect(weightDecay).toBeDisabled();
    } else {
      await expect(weightDecay).toBeEnabled();
    }
  }
  expectCleanResources(monitor);
});

test('warmup+cosine schedule and global-norm clipping show exact values', async ({ page }) => {
  const monitor = await openChapter(page);
  const stepInput = page.getByTestId('lr-step-input');
  const threshold = page.getByTestId('clip-threshold');
  const learningRate = page.getByTestId('lr-value');
  const clipFactor = page.getByTestId('clip-factor');
  const clippedGradient = page.getByTestId('clipped-gradient');
  const clippedNorm = page.getByTestId('clipped-gradient-norm');

  await expect(page.getByTestId('gradient-norm')).toHaveText('5');
  await expect(threshold).toHaveValue('2');
  await expect(clipFactor).toHaveText('0.4');
  await expect(clippedGradient).toHaveText('(1.2, 1.6)');
  await expect(clippedNorm).toHaveText('2');

  await threshold.fill('1');
  await expect(clipFactor).toHaveText('0.2');
  await expect(clippedGradient).toHaveText('(0.6, 0.8)');
  await expect(clippedNorm).toHaveText('1');

  await threshold.fill('4');
  await expect(clipFactor).toHaveText('0.8');
  await expect(clippedGradient).toHaveText('(2.4, 3.2)');
  await expect(clippedNorm).toHaveText('4');

  await threshold.fill('5');
  await expect(clipFactor).toHaveText('1');
  await expect(clippedGradient).toHaveText('(3, 4)');
  await expect(clippedNorm).toHaveText('5');
  await threshold.fill('1');
  await expect(clipFactor).toHaveText('0.2');
  await expect(clippedNorm).toHaveText('1');

  for (const [step, expected] of [
    ['0', '0'],
    ['5', '0.0005'],
    ['10', '0.001'],
    ['55', '0.00055'],
    ['100', '0.0001'],
  ]) {
    await stepInput.fill(step);
    await expect(learningRate).toHaveText(expected);
  }
  expectCleanResources(monitor);
});

test('native optimizer controls are labelled, keyboard-operable, and visibly focused', async ({ page }) => {
  const monitor = await openChapter(page);
  const controls = [
    { testid: 'optimizer-algorithm', name: '优化算法', role: 'combobox' },
    { testid: 'optimizer-learning-rate', name: '优化器学习率 η', role: 'slider' },
    { testid: 'optimizer-momentum', name: 'Momentum 系数 β', role: 'slider', requiredAlgorithm: 'momentum' },
    { testid: 'optimizer-weight-decay', name: '权重衰减 λ', role: 'slider', requiredAlgorithm: 'adamw' },
    { testid: 'optimizer-step-input', name: '优化轨迹步数', role: 'slider' },
    { testid: 'lr-step-input', name: '学习率调度步数', role: 'slider' },
    { testid: 'clip-threshold', name: '梯度裁剪阈值 τ', role: 'slider' },
  ];

  for (const { testid, name, role, requiredAlgorithm } of controls) {
    if (requiredAlgorithm) {
      await page.getByTestId('optimizer-algorithm').selectOption(requiredAlgorithm);
    }
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
    if (role === 'slider') {
      const initial = await locator.inputValue();
      await locator.press('ArrowRight');
      expect(await locator.inputValue()).not.toBe(initial);
      await locator.press('ArrowLeft');
      await expect(locator).toHaveValue(initial);
    }
  }

  const algorithm = page.getByTestId('optimizer-algorithm');
  await algorithm.press('m');
  await expect(algorithm).toHaveValue('momentum');
  await page.getByTestId('optimizer-next').press('Enter');
  await expect(page.getByTestId('optimizer-step')).toHaveText('1');
  await page.getByTestId('optimizer-reset').press('Space');
  await expect(page.getByTestId('optimizer-step')).toHaveText('0');

  const duplicateIds = await page.locator('llm-optimizer-trajectory, llm-lr-schedule').evaluateAll(
    (roots) => {
      const ids = roots.flatMap((root) => [root.id, ...[...root.querySelectorAll('[id]')].map((node) => node.id)]);
      return ids.filter((id, index) => id && ids.indexOf(id) !== index);
    },
  );
  expect(duplicateIds).toEqual([]);
  expectCleanResources(monitor);
});

test('dynamic SVG descriptions, bounded motion, mounts, and local modules stay coherent', async ({ page }) => {
  const monitor = await openChapter(page);
  const optimizerChart = page.getByTestId('optimizer-chart');
  const scheduleChart = page.getByTestId('lr-chart');

  await expect(optimizerChart).toHaveAttribute('role', 'img');
  await expect(optimizerChart).toHaveAccessibleName('二次损失上的优化轨迹');
  await page.getByTestId('optimizer-next').click();
  await expect(optimizerChart).toHaveAccessibleDescription(/SGD.*第 1 步.*\(0\.8, 1\.1\).*2\.025/);

  await expect(scheduleChart).toHaveAttribute('role', 'img');
  await expect(scheduleChart).toHaveAccessibleName('Warmup 与 cosine 学习率及全局范数裁剪');
  await page.getByTestId('lr-step-input').fill('55');
  await page.getByTestId('clip-threshold').fill('1');
  await expect(scheduleChart).toHaveAccessibleDescription(/第 55 步.*0\.00055.*阈值 1.*因子 0\.2.*范数为 1/);

  const motion = await page.locator('llm-optimizer-trajectory, llm-lr-schedule').evaluateAll((roots) => {
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

  await expect(page.locator('.interactive-mount > llm-optimizer-trajectory')).toHaveCount(1);
  await expect(page.locator('.interactive-mount > llm-lr-schedule')).toHaveCount(1);
  await expect(page.locator('p > llm-optimizer-trajectory, p > llm-lr-schedule')).toHaveCount(0);
  const paths = monitor.localResources.map((response) => new URL(response.url()).pathname);
  expect(paths).toEqual(expect.arrayContaining([
    '/llm-book/assets/js/book.js',
    '/llm-book/assets/js/components/optimizer-trajectory.js',
    '/llm-book/assets/js/components/lr-schedule.js',
    '/llm-book/assets/js/lib/optimization.js',
  ]));
  expect(paths.filter((path) => path === '/llm-book/assets/js/book.js')).toHaveLength(1);
  expectCleanResources(monitor);
});

test('optimizer state and figure prose match the active presentation', async ({ page }) => {
  const monitor = await openChapter(page);
  const velocity = (await page.getByTestId('optimizer-velocity').textContent()).trim();
  const prose = await page.locator('#sec-adamw').textContent();

  expect({
    velocity,
    distinguishesStaticAndInteractive: prose.includes(
      '静态图展示同一二次损失上的十步轨迹，HTML 交互则允许逐步查看并切换算法',
    ),
  }).toEqual({ velocity: '—', distinguishesStaticAndInteractive: true });
  expectCleanResources(monitor);
});

test.describe('without JavaScript', () => {
  test.use({ javaScriptEnabled: false });

  test('shows both static optimization figures with anchors and alternatives', async ({ page }) => {
    const monitor = await openChapter(page);
    const fallbacks = [
      {
        anchor: 'fig-optimizer-trajectory',
        src: 'assets/figures/optimizer-trajectory.svg',
        alternative: /优化器轨迹对比图/,
      },
      {
        anchor: 'fig-lr-schedule',
        src: 'assets/figures/lr-schedule.svg',
        alternative: /学习率与梯度裁剪教学图/,
      },
    ];

    for (const { anchor, src, alternative } of fallbacks) {
      const image = page.locator(`noscript img[src="${src}"]`);
      await expect(image).toBeVisible();
      await expect(image).toHaveAttribute('alt', alternative);
      await expect(page.locator(`#${anchor}`)).toBeVisible();
    }
    await expect(page.getByTestId('optimizer-algorithm')).toHaveCount(0);
    await expect(page.getByTestId('clip-threshold')).toHaveCount(0);
    expectCleanResources(monitor);
  });
});
