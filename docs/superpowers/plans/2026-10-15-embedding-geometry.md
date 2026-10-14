# 向量、矩阵与 Embedding 几何 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一章“向量、矩阵与 Embedding 几何”，用同一源稿生成正式 HTML/PDF，并以两个原生 JS 交互帮助读者建立内积、投影、矩阵变换和 Embedding 的直觉。

**Architecture:** 复用现有 Quarto 1.9.38 Book、Typst、MathJax SVG、原生 ES Modules 和 CI/CD。先以 JS/Python 共享数值内核保证数学定义正确，再写静态章和图，最后接入两个交互并执行完整验证与公网复核。

**Tech Stack:** Quarto 1.9.38、Typst、Pandoc Markdown、MathJax SVG、原生 JavaScript/Web Components/SVG、Python 3、Node.js 24、Node Test Runner、Playwright、axe-core、GitHub Actions/Pages。

**Spec:** `docs/superpowers/specs/2026-10-14-llm-book-design.md`

## Global Constraints

- 保持同一份 `.qmd` 源稿，不为 HTML/PDF 复制正文；交互必须有语义等价的 no-JS 与 Typst 静态后备。
- 不引入 React/Vue、后端、模型 API、GPU 或第三方运行时请求；公式继续使用本地 MathJax SVG。
- 只使用原生 ES Modules 和 SVG，并复用本章 JS/Python 数值函数，不在组件中另写数学公式。
- 内容面向计算机专业研究生，严谨但生动；保留英文术语、练习和原始参考文献，不虚构出处或结论。
- 不破坏既有章节、交互、搜索、可访问性、离线和 Pages 子路径行为；已知上游 Pandoc 翻译警告不作为修改 `lang: zh-CN` 的理由。
- 本需求只创建一个新的 commit，不改写已推送历史；完成验证和独立审查后推送 `main`，未通过不得宣称发布完成。

## Review Focus

- 零向量、维度不一致、空向量和非有限输入必须被明确拒绝，不能静默产生 NaN；由 Task 1 的 JS/Python 单测覆盖。
- 投影、余弦相似度和矩阵变换的符号、分母和基向量顺序必须正确；由 Task 1 的固定数值用例和 Task 3 的浏览器用例覆盖。
- 交互在 no-JS、PDF、390px 移动端和 `/llm-book/` 子路径下仍须可用；由 Task 3/4 的后备、溢出、离线和资源测试覆盖。
- 新章节必须保持正式 MathJax SVG 公式、可访问标签、中文搜索和跨章链接；由 Task 4 的真实浏览器检查覆盖。
- HTML 与 PDF 的图形、公式和参考文献不得互相矛盾；由 Task 2/4 的产物断言和 PDF 文本/页面检查覆盖。

---

### Task 1: 向量与矩阵数值内核

**Files:**
- Create: `assets/js/lib/linalg.js`
- Create: `examples/vector_geometry.py`
- Create: `tests/unit/linalg.test.mjs`
- Create: `tests/test_vector_geometry.py`

**Interfaces:**
- Produces JS exports: `add(a, b)`、`dot(a, b)`、`norm(a)`、`cosine(a, b)`、`projection(a, b)`、`transform(matrix, vector)`。
- Produces Python equivalents: `add`、`dot`、`norm`、`cosine`、`projection`、`transform`。
- `projection(a, b)` 计算 `(a·b)/(b·b) b`，`transform(A, x)` 计算 `A x`；向量用一维数组表示，矩阵用行主序二维数组表示。
- 空向量、维度不一致、非有限值、非矩形矩阵、矩阵维度不匹配，以及余弦/投影中的零向量均明确抛出 `RangeError`（Python 为 `ValueError`）。

- [ ] **Step 1: 先写 JS/Python 失败测试**

固定以下断言，浮点容差为 `1e-9`：

```js
assert.deepEqual(add([1, 2], [3, 4]), [4, 6]);
assert.equal(dot([1, 2], [3, 4]), 11);
assert.equal(norm([3, 4]), 5);
assertClose(cosine([1, 0], [1, 1]), Math.SQRT1_2);
assert.deepEqual(projection([2, 2], [1, 0]), [2, 0]);
assert.deepEqual(transform([[0, -1], [1, 0]], [1, 0]), [0, 1]);
assert.throws(() => cosine([0, 0], [1, 0]), RangeError);
assert.throws(() => projection([1, 0], [0, 0]), RangeError);
assert.throws(() => transform([[1, 2], [3, 4]], [1, 2, 3]), RangeError);
```

Python 使用相同输入与预期，并额外验证 `add([1], [1, 2])`、`norm([])` 和 `transform([[1, 2], [3, 4]], [])` 抛出 `ValueError`。

- [ ] **Step 2: 运行并确认预期失败**

Run: `node --test tests/unit/linalg.test.mjs`  
Expected: FAIL，因 `assets/js/lib/linalg.js` 不存在。

Run: `python3 -m unittest tests.test_vector_geometry -v`  
Expected: FAIL，因 `examples/vector_geometry.py` 不存在。

- [ ] **Step 3: 实现最小纯函数并复测**

使用普通循环和 `Math.hypot`/Python `math.sqrt`，不引入第三方依赖，也不修改现有 `probability.js`。Python 增加 `main()` 输出 `projection([2, 2], [1,0])` 与 `transform([[0,-1],[1,0]],[1,0])`。

Run: `node --test tests/unit/linalg.test.mjs && python3 -m unittest tests.test_vector_geometry -v`  
Expected: 全部 PASS。

### Task 2: 静态章节、图示与参考文献

**Files:**
- Create: `embedding-geometry.qmd`
- Create: `assets/figures/vector-projection.svg`
- Create: `assets/figures/matrix-transform.svg`
- Create: `assets/figures/embedding-space.svg`
- Modify: `_quarto.yml`
- Modify: `references.bib`
- Modify: `tests/artifacts/artifacts.test.mjs`

**Interfaces:**
- Produces `_book/embedding-geometry.html` 和 `_book/llm-book.pdf` 中的新章节，锚点为 `#sec-dot-product`、`#sec-matrix-transform`、`#sec-svd`、`#sec-embeddings`。
- 图锚点为 `#fig-vector-projection`、`#fig-matrix-transform`、`#fig-embedding-space`。
- 文献键为 `axler2015linear`、`mikolov2013efficient`、`vaswani2017attention`。

- [ ] **Step 1: 先扩展失败产物测试**

断言新章节 HTML 存在，包含“向量、矩阵与 Embedding 几何”“内积”“投影”“SVD”“常见误解”“练习”，包含三幅 SVG 和全部章节/图锚点及三个文献键，且无未解析 `[?]`。同时断言 PDF 文本含章名和三幅图文字。

Run: `npm run build && npm run test:artifacts`  
Expected: FAIL，因新章节和图示尚不存在。

- [ ] **Step 2: 撰写完整静态章节**

章节依次覆盖：向量及其运算、内积与范数、余弦相似度、向量投影、矩阵作为线性变换、特征值与特征向量、SVD 和低秩近似、Embedding 的几何意义、与 Q/K/V 的关系、常见误解和至少六道练习。通过 include 复用 `examples/vector_geometry.py`，不复制实现。

必须明确：投影分母是 `b·b`、余弦不是距离或概率、矩阵列向量是基向量的像、SVD 不用于“压缩一切数据”的万能证明、注意力分数的缩放点积不等同于语义相似度。引用 Axler 的线性代数教材、Mikolov 词向量原论文和 Vaswani 的 Transformer 原论文，并核对真实书目信息。

- [ ] **Step 3: 创建三幅可打印 SVG 教学图**

分别表达向量投影、矩阵对单位网格/单位方块的变换、低秩子空间与 Embedding 几何。图中符号使用通用数学记号和中文说明，浅色/深色模式与 PDF 均可读，并写具体 `fig-alt`。

- [ ] **Step 4: 构建并检查静态章**

Run: `npm run build && npm run test:artifacts && pdftotext -layout _book/llm-book.pdf /tmp/embedding-geometry.txt`  
Expected: 全部断言通过，PDF 文本含章名、投影、SVD 和三幅图文字，无未解析引用。

### Task 3: 向量投影与矩阵变换交互

**Files:**
- Create: `assets/js/components/vector-projector.js`
- Create: `assets/js/components/matrix-transformer.js`
- Create: `tests/e2e/geometry-interactions.spec.mjs`
- Modify: `assets/js/book.js`
- Modify: `embedding-geometry.qmd`
- Modify: `styles/site.scss`
- Modify: `tests/artifacts/artifacts.test.mjs`

**Interfaces:**
- `<llm-vector-projector angle-a="45" angle-b="0">`：两个原生角度 range input；输出 `vector-dot`、`vector-cosine`、`vector-projection` 和 SVG 向量图。
- `<llm-matrix-transformer matrix="identity">`：原生 select 提供 `identity`、`rotate90`、`project-x`、`shear-x`、`scale-xy`；输出 `matrix-point`、`matrix-transformed-point` 和 SVG 网格/基向量图。
- 两个组件分别导出 `defineVectorProjectorElement()` 和 `defineMatrixTransformerElement()`，并复用 Task 1 的 `dot`、`cosine`、`projection`、`transform`。
- no-JS 和 Typst 使用 Task 2 的静态图；自定义元素以 `{=html}` 挂载，避免被 Pandoc 包进 `<p>`。

- [ ] **Step 1: 先写浏览器失败测试**

在 390px 视口中，将向量 A 的角度设为 `90`、B 设为 `0`，断言内积为 `0`、余弦为 `0`、投影为 `(0, 0)`；改为 `45` 后，断言内积为 `√2/2`、余弦为 `√2/2`、投影为 `(√2/2, 0)`。矩阵选择 `rotate90` 后，断言点 `(1, 0)` 变为 `(0, 1)`，且 SVG 坐标同步更新。

同时断言两个交互的标签、键盘行为、动态描述、no-JS 后备和本地模块请求均正确。

Run: `npm run build && npx playwright test tests/e2e/geometry-interactions.spec.mjs`  
Expected: FAIL，因两个交互尚未实现。

- [ ] **Step 2: 实现向量投影交互**

使用角度计算单位向量，以 Task 1 的 `dot`、`cosine`、`projection` 计算结果，并用 SVG 画出两个向量、投影和垂线。所有输出使用固定的 test id 和可访问文字描述，不复制投影公式。

- [ ] **Step 3: 实现矩阵变换交互**

以 Task 1 的 `transform` 计算基向量和测试点变换，使用 SVG 显示原/变换网格与单位方块。矩阵选项固定为 identity、rotate90、project-x、shear-x（`[[1,1],[0,1]]`）和 scale-xy（`[[2,0],[0,0.5]]`），并保留可核对的输出精度。

- [ ] **Step 4: 接入 Quarto 条件内容与后备并复测**

HTML 显示两个交互，no-JS 与 Typst 显示同一组静态图，保留图锚点和图注。章末只加载一次现有的 `assets/js/book.js`。

Run: `npm run build && npm run test:js && npx playwright test tests/e2e/geometry-interactions.spec.mjs`  
Expected: 全部 PASS，无页面脚本异常或本地资源 404。

### Task 4: 全量质量验证与公网发布

**Files:**
- Create: `tests/e2e/geometry-quality.spec.mjs`
- Modify: `tests/artifacts/artifacts.test.mjs`

**Interfaces:**
- 保持现有 `verify` 顺序和 CI 工作流不变；新增质量测试必须通过真实渲染产物、静态服务器和 PDF 文本检查。

- [ ] **Step 1: 增加新章节质量测试**

覆盖新章节 axe、移动端溢出、SVG 数学渲染、全部站内链接、搜索“向量”“投影”“SVD”、no-JS 后备和拒绝跨源请求；同时断言既有 24 项浏览器测试不回退。

Run: `npx playwright test tests/e2e/geometry-quality.spec.mjs`  
Expected: 新测试至少先证明缺口；若已有正确行为可直接通过，不为制造失败而改动。

- [ ] **Step 2: 修复真实问题并执行完整验证**

只修复测试证明的问题。随后删除 `_book/` 并运行 `npm run verify`，再执行 `pdftotext -layout _book/llm-book.pdf /tmp/llm-book-final.txt`，检查新章公式、图、练习和中文标签。

Expected: 全部 JS/Python/artifact/Playwright 测试通过，无跨源运行时请求，PDF 含新章且无未解析引用。

- [ ] **Step 3: 独立审查、提交并推送**

先检查 `git diff --check` 和变更范围，再安排独立代码/内容审查。仅当审查无 Critical/Important 问题时，创建一个 commit：

```bash
git add .
git commit -m "feat: add embedding geometry chapter"
git push origin main
```

- [ ] **Step 4: 验证远程 CI 与公网结果**

使用 `gh run watch` 等待 `Verify and deploy book` 成功。随后以真实浏览器检查新章节两个交互、SVG 公式、移动端无溢出和零跨源请求，并确认公网 PDF 可下载且包含新章节。只有这些检查全部通过才报告发布完成。
