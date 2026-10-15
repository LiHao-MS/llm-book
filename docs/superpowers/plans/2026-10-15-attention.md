# 序列建模与自注意力机制 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一章“序列建模与自注意力机制”，讲清 RNN 局限、注意力、缩放点积注意力、多头注意力、因果掩码和 RoPE，并提供注意力矩阵与 RoPE 旋转两个交互。

**Architecture:** 先实现并验证共享注意力内核，再撰写静态章节和图，最后加入注意力矩阵与 RoPE 旋转交互。所有交互均复用 `probability.js` 的 Softmax 与数值函数，保持单一 `.qmd` 源稿和 HTML/Typst 双输出。

**Tech Stack:** Quarto 1.9.38、Typst、MathJax SVG、原生 JavaScript/Web Components/SVG、Python 3、Node.js 24、Node Test Runner、Playwright、axe-core、GitHub Actions/Pages。

**Spec:** `docs/superpowers/specs/2026-10-14-llm-book-design.md`

## Global Constraints

- 保持单一 `.qmd` 源稿；交互必须有语义等价的 no-JS 与 Typst 静态后备，不引入框架、后端、模型 API、GPU 或第三方运行时请求。
- 注意力与 RoPE 公式必须在 JS/Python 内核中实现一次，交互不得复制或另写实现。
- 保留 MathJax SVG、既有章节、交互、搜索和可访问性；本需求只创建一个 commit，验证与独立审查通过后再推送。
- 内容严谨且面向计算机专业研究生，覆盖 RNN 局限、注意力、缩放点积注意力、多头注意力、因果掩码、RoPE 与相对位置，不把产品教程当原理。

## Review Focus

- 维度不一致、空输入、非有限值、零缩放、非方阵掩码和非法 RoPE 角度必须被明确拒绝；由 Task 1 的 JS/Python 单测覆盖。
- Softmax 的行归一化、缩放位置、未来 Token 掩码和多头拆分必须正确；由 Task 1 的固定分数、掩码和多头输出覆盖。
- RoPE 的旋转方向、奇偶维配对和相对角度必须正确；由 Task 1 的 0°/90°/180° 用例和 Task 3 的交互检查覆盖。
- 交互图必须表达真实注意力分布与几何旋转，而不是装饰动画；由 Task 3 的浏览器断言和静态图后备覆盖。
- 公式、图和 PDF 不得在移动端溢出，跨源请求和可访问性不得退化；由 Task 4 的全量质量测试覆盖。

---

### Task 1: 注意力与 RoPE 数值内核

**Files:**
- Create: `assets/js/lib/attention.js`
- Create: `examples/attention.py`
- Create: `tests/unit/attention.test.mjs`
- Create: `tests/test_attention.py`

**Interfaces:**
- Produces `scaledDotProductAttention(q, k, v, scale)`，使用现有 `softmax` 计算每个查询的分布并输出值加权结果；返回 `{ scores, weights, output }`。
- Produces `causalMask(scores)`，将上三角未来位置（含主对角线之上的位置）设为 `-Infinity`，不修改输入。
- Produces `multiHeadAttention(q, k, v, headCount)`，按末维均分头维，返回 `{ heads: [...], output: [...] }`。
- Produces `rotatePair(vector, angle)`，实现 RoPE 的二维旋转：`(x1 cosθ - x2 sinθ, x1 sinθ + x2 cosθ)`；角度以弧度表示，并要求偶数维向量。
- 空输入、维度不一致、非有限值、零或负缩放、非方阵掩码、非偶数维 RoPE 向量、非有限角度均明确抛出 `RangeError`（Python 为 `ValueError`）。

- [ ] **Step 1: 先写固定失败测试**

使用 `q=k=v=[[1,0],[0,1],[1,1]]`、`scale=1/√2` 和分数 `[[2,1,3],[1,1,1],[3,1,2]]` 验证基础结果，并使用独立 K/V 夹具排除分数误用 V 或输出误加权 K：

```js
const masked = causalMask(scores);
assert.deepEqual(masked[0][1], -Infinity);
assert.deepEqual(masked[1][2], -Infinity);
assert.deepEqual(masked[0][2], -Infinity);
assert.deepEqual(masked[2][2], 2);
assertClose(weights[0].reduce((a, b) => a + b, 0), 1);
assertClose(weights[1].reduce((a, b) => a + b, 0), 1);
assertClose(weights[2].reduce((a, b) => a + b, 0), 1);
assertClose(output[0][0], 0.8022241854);
assertClose(output[0][1], 0.5988879073);

// q·k = [1, 0]，Softmax 后对不同的 V=[10,20] 加权。
const distinct = scaledDotProductAttention([[1, 0]], [[1, 0], [0, 1]], [[10], [20]], 1);
assert.deepEqual(distinct.scores, [[1, 0]]);
assertClose(distinct.weights[0][0], 0.7310585786);
assertClose(distinct.weights[0][1], 0.2689414214);
assertClose(distinct.output[0][0], 12.6894142137);

// 两头拆分 Q/K；V 的两个相同特征列使每头仍对 [10,20] 加权。
const multi = multiHeadAttention([[1, 0]], [[1, 0], [0, 1]], [[10, 10], [20, 20]], 2);
assert.deepEqual(multi.heads[0].scores, [[1, 0]]);
assert.deepEqual(multi.heads[1].scores, [[0, 0]]);
assertClose(multi.heads[0].output[0][0], 12.6894142137);
assertClose(multi.heads[1].output[0][0], 15);
assertClose(multi.output[0][0], 12.6894142137);
assertClose(multi.output[0][1], 15);
assertVectorClose(rotatePair([1, 0], Math.PI / 2), [0, 1]);
assert.deepEqual(rotatePair([1, 0], 0), [1, 0]);
assertVectorClose(rotatePair([1, 0], Math.PI), [-1, 0]);
assert.throws(() => rotatePair([1, 2, 3], 0), RangeError);
```

Python 使用相同输入与预期，另测 `headCount=2` 时两头独立输出、Softmax 行和为 1，以及 `scale=0` 时明确报错。`headCount=1` 时 `output` 与单头输出一致。

- [ ] **Step 2: 运行并确认预期失败**

Run: `node --test tests/unit/attention.test.mjs`  
Expected: FAIL，因 `assets/js/lib/attention.js` 不存在。

Run: `python3 -m unittest tests.test_attention -v`  
Expected: FAIL，因 `examples/attention.py` 不存在。

- [ ] **Step 3: 实现最小注意力、掩码、多头与 RoPE 并复测**

`scaledDotProductAttention` 只对最后一维计算 `q·k`，复用现有 Softmax，并输出 `{ scores, weights, output }`。掩码保持上三角未来位置为 `-Infinity`。RoPE 将相邻维度视为一对，使用 `Math.cos`/`Math.sin` 计算旋转，保持输入不变并返回新数组。

Run: `node --test tests/unit/attention.test.mjs && python3 -m unittest tests.test_attention -v`  
Expected: 全部 PASS，且 Softmax 每行和为 1、旋转方向与输入不变性正确。

### Task 2: 静态章节、图示与参考文献

**Files:**
- Create: `attention.qmd`
- Create: `assets/figures/attention-scores.svg`
- Create: `assets/figures/causal-mask.svg`
- Create: `assets/figures/multi-head.svg`
- Create: `assets/figures/rope-rotation.svg`
- Modify: `_quarto.yml`
- Modify: `references.bib`
- Modify: `tests/artifacts/artifacts.test.mjs`

**Interfaces:**
- Produces `_book/attention.html` 与 PDF 新章，锚点为 `#sec-rnn-limits`、`#sec-attention`、`#sec-scaled-attention`、`#sec-multi-head`、`#sec-causal-mask`、`#sec-rope`。
- 图锚点为 `#fig-attention-scores`、`#fig-causal-mask`、`#fig-multi-head`、`#fig-rope-rotation`。
- 文献键为 `bahdanau2014neural`、`luong2015effective`、`vaswani2017attention`、`su2021roformer`。

- [ ] **Step 1: 先扩展失败产物测试**

断言章节 HTML 存在，含“序列建模与自注意力机制”“RNN”“缩放点积注意力”“多头注意力”“因果掩码”“RoPE”，包含四幅 SVG 和全部章节/图锚点及四个文献键，且无未解析 `[?]`。PDF 文本含章名、Softmax、RoPE 和因果掩码。

Run: `npm run build && npm run test:artifacts`  
Expected: FAIL，因章节、图和参考文献尚未创建。

- [ ] **Step 2: 撰写完整静态章节**

依次说明：序列建模任务与 RNN 的顺序计算/长依赖瓶颈、注意力作为值加权平均、Q/K/V、缩放点积注意力、Softmax 行归一化、为什么要除以 `sqrt(d_k)`、多头注意力如何并行表示不同子空间、因果掩码如何保证自回归训练不看到未来、RoPE 的旋转几何与相对位置、常见误解和至少六道练习。特别说明注意力不是“给每个 Token 一个固定标签”，而是根据当前查询动态计算权重。

- [ ] **Step 3: 创建四幅可打印 SVG 教学图**

分别表达注意力分数与加权输出、因果掩码的三角结构、两个头的不同注意力模式，以及 RoPE 对二维向量的旋转和相对角度。包含可核对的数值、方向、矩阵单元格和具体 `fig-alt`。

- [ ] **Step 4: 构建并检查静态章**

Run: `npm run build && npm run test:artifacts && pdftotext -layout _book/llm-book.pdf /tmp/attention.txt`  
Expected: 全部断言通过，PDF 文本含章名、Softmax、RoPE 和因果掩码。

### Task 3: 注意力矩阵与 RoPE 旋转交互

**Files:**
- Create: `assets/js/components/attention-matrix.js`
- Create: `assets/js/components/rope-rotation.js`
- Create: `tests/e2e/attention-interactions.spec.mjs`
- Modify: `assets/js/book.js`
- Modify: `attention.qmd`
- Modify: `styles/site.scss`

**Interfaces:**
- `<llm-attention-matrix query-index="0" scale="0.707106781">`：输出 `attention-scores`、`attention-weights`、`attention-output` 和 SVG 分布/输出图。
- `<llm-rope-rotation angle="90">`：输出 `rope-cos`、`rope-sin`、`rope-rotation` 和 SVG 旋转图。
- 两个组件分别导出 `defineAttentionMatrixElement()` 和 `defineRopeRotationElement()`，复用 Task 1 的注意力与 RoPE 实现，不复制公式。

- [ ] **Step 1: 先写浏览器失败测试**

将查询索引设为 0 时，断言三行 Softmax 权重和为 1，首行输出的两个分量分别为 `0.8022241854`、`0.5988879073`。将 RoPE 角度从 `0` 调到 `90`，断言点从 `(1,0)` 变为 `(0,1)`，且 cos 为 0、sin 为 1。切换查询索引和角度后，分布与旋转图同步更新。

Run: `npm run build && npx playwright test tests/e2e/attention-interactions.spec.mjs`  
Expected: FAIL，因交互尚未实现。

- [ ] **Step 2: 实现注意力矩阵交互**

提供查询索引与缩放系数控件，展示真实分数矩阵、归一化权重、输出向量和加权结果；显示 `q·k`、softmax 行和与输出，保证行和为 1，不把权重画成固定标签。

- [ ] **Step 3: 实现 RoPE 旋转交互**

提供角度滑块，显示原始向量、旋转后向量、cos/sin 值和几何旋转图。至少支持 0°、90°、180°、-90°，保证旋转方向与公式一致，并保留可访问描述。

- [ ] **Step 4: 接入条件内容并复测**

HTML 使用 `{=html}` 挂载，no-JS 与 Typst 显示同一组静态图，保留图锚点、图注和本地模块路径。加入可访问标签、焦点样式和动态描述。

Run: `npm run build && npm run test:js && npx playwright test tests/e2e/attention-interactions.spec.mjs`  
Expected: 全部 PASS，无脚本异常、越界动画或本地资源 404。

### Task 4: 全量质量验证与公网发布

**Files:**
- Create: `tests/e2e/attention-quality.spec.mjs`
- Modify: `tests/artifacts/artifacts.test.mjs`

**Interfaces:**
- 保持 `verify` 顺序和 CI 配置不变，新增质量测试必须验证真实 HTML/PDF、静态服务器和浏览器行为。

- [ ] **Step 1: 增加质量测试并记录真实结果**

覆盖新章节 axe、390px 溢出、公式 SVG、全部链接、搜索“注意力”“RoPE”、no-JS 后备、跨源拒绝及既有章节回归。

Run: `npx playwright test tests/e2e/attention-quality.spec.mjs`  
Expected: 新测试能发现缺口则修复；已有正确行为可直接通过。

- [ ] **Step 2: 修复真实问题并完整验证**

仅修复真实失败。删除 `_book/` 后运行 `npm run verify`，再检查 PDF 的公式、图、练习和中文标签，并做独立代码/内容审查。

Expected: 全部测试和审查通过，无跨源运行时依赖。

- [ ] **Step 3: 创建唯一提交并推送**

```bash
git add .
git commit -m "feat: add attention chapter"
git push origin main
```

- [ ] **Step 4: 验证公网部署**

用 `gh run watch` 确认 CI/CD 成功，再以真实浏览器检查两个交互、SVG 公式、移动端无溢出和零跨源请求，确认 PDF 可下载且包含新章。
