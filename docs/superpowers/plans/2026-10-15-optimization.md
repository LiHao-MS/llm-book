# 优化：从经验风险到 AdamW Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一章“优化：从经验风险到 AdamW”，讲清损失函数、优化器、学习率调度、梯度裁剪和权重衰减，并提供优化轨迹与学习率/裁剪两个可交互实验。

**Architecture:** 复用现有 Quarto/Typst/MathJax SVG 出版管线与测试体系。先以 JS/Python 实现可验证的优化器内核，再写静态章节和图，最后加入交互并执行全量验证与公网复核。

**Tech Stack:** Quarto 1.9.38、Typst、MathJax SVG、原生 JavaScript/Web Components/SVG、Python 3、Node.js 24、Node Test Runner、Playwright、axe-core、GitHub Actions/Pages。

**Spec:** `docs/superpowers/specs/2026-10-14-llm-book-design.md`

## Global Constraints

- 保持单一 `.qmd` 内容源，交互必须有等价的 no-JS 与 Typst 静态后备；不得为 HTML/PDF 分叉正文。
- 不引入前端框架、后端、模型 API、GPU 或第三方运行时请求；只使用原生 ES Modules 和 SVG。
- 优化器公式必须在 JS/Python 内核中实现一次，交互不得复制或另写一套公式。
- 内容严谨、面向计算机专业研究生，覆盖经验风险、SGD、Momentum、Adam/AdamW、学习率调度、梯度裁剪、权重衰减与数值稳定性，不用产品教程替代原理。
- 保持现有章节、交互、搜索、离线和可访问性行为；本需求只创建一个 commit，验证和独立审查通过后再推送。

## Review Focus

- 空梯度、零学习率、非法超参数、除零与非有限输入必须明确处理，不能静默产生 NaN；由 Task 1 的单元测试覆盖。
- SGD/Momentum/Adam 的更新方向、偏差校正、衰减项和全局梯度裁剪公式必须逐字正确；由 Task 1 的固定数值用例与 Task 3 的交互用例覆盖。
- Warmup、cosine schedule、AdamW 的 decoupled weight decay 与 L2 正则化不能混为一谈；由 Task 2 的公式/文本检查和 Task 3 的浏览器检查覆盖。
- 交互图必须显示实际更新轨迹而不是装饰动画，PDF/no-JS 需保留等价静态解释；由 Task 3/4 的后备和静态图检查覆盖。
- 移动端长公式、跨源请求、可访问性和历史章节回归不得被破坏；由 Task 4 的全量质量测试覆盖。

---

### Task 1: 优化器数值内核

**Files:**
- Create: `assets/js/lib/optimization.js`
- Create: `examples/optimization.py`
- Create: `tests/unit/optimization.test.mjs`
- Create: `tests/test_optimization.py`

**Interfaces:**
- Produces `quadratic(point, center)` 与 `quadraticGradient(point, center)`，对应 `J(x,y)=0.5((x-cx)^2+(y-cy)^2)` 及梯度 `(x-cx, y-cy)`。
- Produces `sgdStep(point, gradient, learningRate)`、`momentumStep(point, gradient, state)`、`adamStep(point, gradient, state)`、`adamwStep(point, gradient, state, weightDecay)`，均使用一维二元数组并返回新状态，不修改输入。
- `momentumStep` 维护速度 `v=βv_prev+g`，`adamStep` 维护一阶/二阶矩、时间步与偏差校正，`adamwStep` 在 Adam 更新后额外应用 decoupled weight decay：`θ <- θ(1-λ)`。
- Produces `clipGlobalNorm(gradient, threshold)` 和 `cosineLearningRate(step, totalSteps, minRate, maxRate)`，采用 `c = min(1, τ/||g||` 的缩放以及 `min + 0.5(max-min)(1+cos(πt/T)`。
- 非法维度、非有限值、负学习率、负阈值、非正总步数、越界动量参数均明确抛出 `RangeError`（Python 为 `ValueError`）；零梯度与零学习率合法。`step` 超出 `[0,totalSteps]` 同样抛出 `RangeError`。

- [ ] **Step 1: 先写固定失败测试**

至少覆盖：

```js
assert.deepEqual(sgdStep([1, 1], [2, -1], 0.1), [0.8, 1.1]);
assert.deepEqual(momentumStep([1, 1], [2, -1], 0.1, { velocity: [1, 1], beta: 0.5 }).point, [0.75, 1.05]);
assert.equal(adamStep([1, 1], [2, -1], 0.1, { m: [0, 0], v: [0, 0], t: 0, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }).point, [0.9, 1.1]);
assert.deepEqual(adamwStep([1, 1], [2, -1], 0.1, { m: [0, 0], v: [0, 0], t: 0, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }, 0.01).point, [0.891, 1.089]);
assert.deepEqual(clipGlobalNorm([3, 4], 1), [0.6, 0.8]);
assert.equal(cosineLearningRate(0, 10, 0.2, 1), 1);
assert.equal(cosineLearningRate(5, 10, 0, 1), 0.5);
assert.equal(cosineLearningRate(10, 10, 0, 1), 0);
assert.equal(cosineLearningRate(10, 10, 0.2, 1), 0.2);
assert.throws(() => cosineLearningRate(0, 0, 0.2, 1), RangeError);

// 大有限梯度首步：期望 point[0] ≈ 0.9，且 v[0] ≈ 2.25e305。
assertClose(adamStep([1, 1], [1.5e154, 0], 0.1, initialAdamState).point[0], 0.9);
// 非平凡前态第二步：m=[0.00015,0.0003], v=[1.5e-8,6e-8], t=1, beta1=beta2=0.5, 零梯度。
// 保留 epsilon 的精确结果约为 [0.9000099990001,-1.0999950002499875]，按一位小数为 [0.9,-1.1]。
assert.deepEqual(adamStep([1, -1], [0, 0], 0.1, secondAdamState).point.map((x) => Number(x.toFixed(1))), [0.9, -1.1]);
```

Python 使用相同输入与预期，另测零梯度不更新、非法阈值和空向量均报错。检查输入数组未被修改。

- [ ] **Step 2: 运行并确认预期失败**

Run: `node --test tests/unit/optimization.test.mjs`  
Expected: FAIL，因 `assets/js/lib/optimization.js` 不存在。

Run: `python3 -m unittest tests.test_optimization -v`  
Expected: FAIL，因 `examples/optimization.py` 不存在。

- [ ] **Step 3: 实现最小优化器与调度函数并复测**

使用直接的数值计算，不引入依赖；实现 Adam 的 `m`、`v`、`t` 更新与 `mhat/vhat` 偏差校正，AdamW 在同一接口内解耦衰减，并确保所有返回状态为不可变数据。

Run: `node --test tests/unit/optimization.test.mjs && python3 -m unittest tests.test_optimization -v`  
Expected: 全部 PASS，且输入状态未被修改。

### Task 2: 静态章节、图示与参考文献

**Files:**
- Create: `optimization.qmd`
- Create: `assets/figures/loss-surface.svg`
- Create: `assets/figures/optimizer-trajectory.svg`
- Create: `assets/figures/lr-schedule.svg`
- Modify: `_quarto.yml`
- Modify: `references.bib`
- Modify: `tests/artifacts/artifacts.test.mjs`

**Interfaces:**
- Produces `_book/optimization.html` 与 PDF 新章，章节锚点为 `#sec-empirical-risk`、`#sec-sgd`、`#sec-momentum`、`#sec-adam`、`#sec-schedule`、`#sec-adamw`。
- 图锚点为 `#fig-loss-surface`、`#fig-optimizer-trajectory`、`#fig-lr-schedule`。
- 文献键为 `kingma2014adam`、`loshchilov2017decoupled`、`goodfellow2016deep`。

- [ ] **Step 1: 先扩展失败产物测试**

断言章节 HTML 存在，含“优化：从经验风险到 AdamW”“经验风险”“Momentum”“AdamW”“梯度裁剪”“权重衰减”，包含三幅 SVG 和全部章节/图锚点及三个文献键，且无未解析 `[?]`。PDF 文本含章名、梯度裁剪和 decoupled weight decay。

Run: `npm run build && npm run test:artifacts`  
Expected: FAIL，因章节、图和文献尚未创建。

- [ ] **Step 2: 撰写完整静态章节**

依次讲清：经验风险与损失函数、梯度下降、SGD 与 mini-batch、Momentum、Adam 的矩估计与偏差校正、学习率 warmup/cosine schedule、全局范数梯度裁剪、L2 与 decoupled weight decay、AdamW、混合精度与数值稳定性、常见误解和至少六道练习。明确比较 `θ <- θ - ηg`、Momentum 的速度累积、Adam 的 m/v 与 `mhat/vhat`、AdamW 的解耦衰减，避免把 L2 与 AdamW 当成同一件事。

- [ ] **Step 3: 创建三幅可打印 SVG 教学图**

分别表达二维损失曲面与梯度方向、不同优化器轨迹、warmup+cosine 学习率和全局梯度裁剪。包含可核对的参数、箭头与中文说明，并写具体 `fig-alt`。

- [ ] **Step 4: 构建并检查静态章**

Run: `npm run build && npm run test:artifacts && pdftotext -layout _book/llm-book.pdf /tmp/optimization.txt`  
Expected: 全部断言通过，PDF 文本含章名、Momentum、AdamW、梯度裁剪和 decoupled weight decay。

### Task 3: 优化轨迹与学习率/裁剪交互

**Files:**
- Create: `assets/js/components/optimizer-trajectory.js`
- Create: `assets/js/components/lr-schedule.js`
- Create: `tests/e2e/optimizer-interactions.spec.mjs`
- Modify: `assets/js/book.js`
- Modify: `optimization.qmd`
- Modify: `styles/site.scss`

**Interfaces:**
- `<llm-optimizer-trajectory algorithm="sgd" learning-rate="0.1" momentum="0.9" weight-decay="0.01">`：输出 `optimizer-step`、`optimizer-point`、`optimizer-loss` 和 SVG 轨迹。
- `<llm-lr-schedule threshold="2">`：输出 `lr-value`、`clip-factor`、`clipped-gradient` 和 SVG 学习率/裁剪曲线。
- 两个组件分别导出 `defineOptimizerTrajectoryElement()` 和 `defineLrScheduleElement()`，复用 Task 1 的全部优化器、调度和裁剪函数，不复制公式。

- [ ] **Step 1: 先写浏览器失败测试**

断言单步 SGD 从 `(1,1)` 到 `(0.8,1.1)`；Momentum 后状态和速度正确；AdamW 的衰减项与 Adam 不重合。切换算法后轨迹确实重算。阈值 4 时裁剪因子为 0.8、裁剪后范数为 4；阈值 5 时因子为 1，阈值 1 时因子为 0.2 且梯度为 `(0.6,0.8)`、范数为 1。

Run: `npm run build && npx playwright test tests/e2e/optimizer-interactions.spec.mjs`  
Expected: FAIL，因交互尚未实现。

- [ ] **Step 2: 实现优化轨迹交互**

提供算法选择（SGD、Momentum、Adam、AdamW）与学习率/动量/权重衰减控件，在二维二次损失曲面 `J(x,y)=0.5[(x+1)^2+(y-2)^2]` 上从 `(1,1)` 逐步更新，显示真实轨迹、当前点和损失。预计算最多 40 步，停止在有限范围内，禁止无界动画。

- [ ] **Step 3: 实现学习率/裁剪交互**

绘制 warmup+cosine 学习率曲线和当前步数，显示原梯度范数、裁剪因子和裁剪后范数；同时提供全局范数裁剪公式与静态图后备。控件只改变必要状态，结果由 Task 1 函数计算。

- [ ] **Step 4: 接入条件内容并复测**

保留静态图作为 no-JS 与 Typst 后备，HTML 使用 `{=html}` 挂载，添加可访问标签、焦点和动态 SVG 描述。

Run: `npm run build && npm run test:js && npx playwright test tests/e2e/optimizer-interactions.spec.mjs`  
Expected: 全部 PASS，无脚本异常、越界动画或本地资源 404。

### Task 4: 全量质量验证与公网发布

**Files:**
- Create: `tests/e2e/optimizer-quality.spec.mjs`
- Modify: `tests/artifacts/artifacts.test.mjs`

**Interfaces:**
- 保持 `verify` 顺序与 CI 配置不变，新增质量测试必须验证真实 HTML/PDF、静态服务器和浏览器行为。

- [ ] **Step 1: 增加质量测试并记录真实结果**

覆盖新章节 axe、390px 溢出、公式 SVG、全部链接、搜索“优化”“梯度裁剪”“AdamW”、no-JS 后备、跨源拒绝及既有测试回归。

Run: `npx playwright test tests/e2e/optimizer-quality.spec.mjs`  
Expected: 新测试能发现缺口则修复；已有正确行为可直接通过。

- [ ] **Step 2: 修复真实问题并完整验证**

仅修复真实失败。随后删除 `_book/` 并运行 `npm run verify`，检查 PDF 的公式、图、练习和中文标签，再做一次独立代码/内容审查。

Expected: 全部测试和审查通过，无跨源运行时依赖。

- [ ] **Step 3: 创建唯一提交并推送**

```bash
git add .
git commit -m "feat: add optimization chapter"
git push origin main
```

- [ ] **Step 4: 验证公网部署**

用 `gh run watch` 确认 CI/CD 成功，再以真实浏览器检查两个交互、SVG 公式、移动端无溢出和零跨源请求，确认 PDF 可下载且含新章。
