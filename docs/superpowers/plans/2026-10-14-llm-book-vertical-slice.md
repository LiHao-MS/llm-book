# LLM 数字教材首个纵向样板 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一章可公开访问的 Quarto 数字教材纵向样板，以同一 `.qmd` 源稿生成 HTML 与 Typst PDF，并包含两个低资源交互及其完整质量验证。

**Architecture:** Quarto 1.9 Book 是唯一出版管线，HTML 是主输出，内建 orange-book Typst 格式生成 PDF；原生 ES Modules/Web Components 复用同一套概率函数实现交互，Python 示例与 JavaScript 数值结果互相校验。GitHub Actions 在干净环境完成构建和测试后，将 `_book/` 作为静态产物部署到 GitHub Pages。

**Tech Stack:** Quarto 1.9.x、Pandoc Markdown、Typst、原生 JavaScript（ES Modules、SVG、Web Components）、Python 3、Node.js 24、Node Test Runner、Playwright、axe-core、GitHub Actions、GitHub Pages。

**Spec:** `docs/superpowers/specs/2026-10-14-llm-book-design.md`

## Global Constraints

- 只维护一份 `.qmd` 内容源；不得为 HTML 和 PDF 复制正文。
- 核心学习不依赖后端、模型 API、GPU、大模型权重或大文件下载。
- 前端不使用 React/Vue 或大型可视化框架；交互使用原生 ES Modules 和 SVG。
- 任何交互中的关键知识都必须有语义等价的静态图或文字后备，PDF 不得缺内容。
- 首个样板章至少包含两个交互、一个最小 Python 示例、练习和原始参考文献。
- 首个样板使用 Quarto 1.9 内建 orange-book；只有出现具体版式需求时才新建 Typst 模板，若 Typst 不能达到正式 PDF 要求，按 spec 使用 LaTeX 后备而不分叉源稿。
- 站点只部署到 GitHub Pages；`_book/`、`.quarto/`、`node_modules/` 和测试报告不纳入版本控制。
- 按用户偏好，本纵向样板作为一个需求只保留一个 commit：Task 1 创建首个本地提交，后续任务只 amend 这个尚未推送的提交以提供逐任务审查 diff；未经明确确认不得 push 或修改远程仓库。
- 正式书名、仓库名、许可证、自定义域名和外部协作策略不在本计划中擅自决定。

## Review Focus

- 极端 Logits、零/越界概率、非法温度和 Top-p 边界必须产生确定结果或明确异常，不能静默出现 NaN；由 Task 1 的数值单元测试覆盖。
- 项目部署在 `/llm-book/` 这类子路径时，脚本、图片、PDF 链接和站内导航仍须可用；由 Task 4 的子路径 Playwright 测试覆盖。
- 禁用 JavaScript 或导出 PDF 时，读者仍须看到等价静态解释；由 Task 4 的 no-JS 测试和 Task 3/4 的产物测试覆盖。
- 中文与中英混合搜索必须能找到目标章节；由 Task 5 的“交叉熵”“梯度裁剪”“注意力”“强化学习”“RLHF”“Scaling Law”搜索测试覆盖。
- 中文字体、公式和图表必须在真实 PDF 中正确呈现，键盘、移动端和减少动态效果用户不能被交互排除；由 Task 5 的 axe/响应式/键盘测试及 Task 6 的 PDF 检查覆盖。

---

### Task 1: JS 与 Python 数值内核

**Files:**
- Create: `assets/js/lib/probability.js`
- Create: `examples/__init__.py`
- Create: `examples/softmax_cross_entropy.py`
- Create: `tests/__init__.py`
- Create: `tests/unit/probability.test.mjs`
- Create: `tests/test_probability.py`
- Create: `package.json`
- Create: `package-lock.json`

**Interfaces:**
- Produces JS exports: `softmax(logits: number[]) -> number[]`、`crossEntropy(probability: number) -> number`、`perplexity(probability: number) -> number`、`temperatureProbabilities(probabilities: number[], temperature: number) -> number[]`、`topP(probabilities: number[], threshold: number) -> { probabilities: number[], kept: boolean[] }`。
- Produces Python functions: `softmax(logits: Sequence[float]) -> list[float]`、`cross_entropy(probability: float) -> float`、`perplexity(probability: float) -> float`。
- `topP` 保留达到阈值的最小高概率集合并重新归一化；`kept` 按原 Token 顺序标记，返回值不得重排。
- 非有限数、空输入、负数概率、概率和不为正、`probability <= 0 || probability > 1`、`temperature <= 0`、`threshold <= 0 || threshold > 1` 均明确抛错（JS `RangeError`，Python `ValueError`）。

- [ ] **Step 1: 先写 JS 与 Python 失败测试**

JS 测试至少固定以下断言，并使用 `1e-9` 的浮点容差：

```js
assertClose(softmax([1, 2, 3])[2], 0.6652409558);
assertClose(softmax([1000, 1001])[0], 0.2689414214);
assertClose(crossEntropy(0.5), Math.LN2);
assertClose(perplexity(0.5), 2);
assert.deepEqual(topP([0.2, 0.5, 0.3], 0.7).kept, [false, true, true]);
assertClose(topP([0.2, 0.5, 0.3], 0.7).probabilities[1], 0.625);
assert.throws(() => crossEntropy(0), RangeError);
assert.throws(() => temperatureProbabilities([0.5, 0.5], 0), RangeError);
```

Python 测试固定相同数值：`softmax([1, 2, 3])[2] == 0.6652409558`、`cross_entropy(0.5) == log(2)`、`perplexity(0.25) == 4`，并验证 `cross_entropy(0)` 与包含 NaN/Infinity 的 Logits 数组会抛错；负 Logits 本身必须合法。另在 JS 测试中验证温度结果归一化且高温后分布更均匀。

- [ ] **Step 2: 运行测试并确认失败原因**

Run: `node --test tests/unit/probability.test.mjs`  
Expected: FAIL，原因是 `assets/js/lib/probability.js` 尚不存在。

Run: `python3 -m unittest tests.test_probability -v`  
Expected: FAIL，原因是 `examples/softmax_cross_entropy.py` 尚不存在。

- [ ] **Step 3: 实现最小数值函数**

在 JS 中使用减去最大 Logit 的稳定 Softmax；温度变换采用 `p_i ** (1 / T)` 后归一化；Top-p 按降序累计但将结果映射回原顺序。Python 实现与 JS 保持相同定义，不引入第三方依赖，并包含直接运行文件时打印 `[1, 2, 3]` 示例结果的 `main()`。

- [ ] **Step 4: 建立脚本并运行全部数值测试**

在 `package.json` 中声明 `"type": "module"`，添加 `test:js`、`test:python` 和后续共用的脚本入口，生成锁文件但不添加运行时依赖。

Run: `node --test tests/unit/probability.test.mjs && python3 -m unittest tests.test_probability -v`  
Expected: 全部 PASS，且 JS/Python 的共享数值断言一致。

### Task 2: Quarto HTML/Typst 出版骨架

**Files:**
- Create: `_quarto.yml`
- Create: `_brand.yml`
- Create: `index.qmd`
- Create: `references.bib`
- Create: `styles/site.scss`
- Create: `.gitignore`
- Create: `tests/artifacts/artifacts.test.mjs`
- Modify: `package.json`

**Interfaces:**
- `npm run build` 执行 `quarto render`，输出 `_book/index.html` 与 `_book/llm-book.pdf`。
- `npm run test:artifacts` 执行 `node --test tests/artifacts/artifacts.test.mjs`。
- `_quarto.yml` 固定 `output-file: llm-book`、`lang: zh-CN`、`downloads: [pdf]`、全文搜索和 `assets/**`、`examples/**` 资源复制。

- [ ] **Step 1: 先写最小产物测试**

测试断言 `_book/index.html` 存在且包含“理解大语言模型”，`_book/llm-book.pdf` 存在、大于 100 KB 且以 `%PDF-` 开头；递归计算 `_book/` 总大小并断言低于 GitHub Pages 的 1 GB 上限。

- [ ] **Step 2: 运行产物测试并确认失败**

Run: `node --test tests/artifacts/artifacts.test.mjs`  
Expected: FAIL，因为 Quarto 配置与 `_book/` 尚不存在。

- [ ] **Step 3: 创建 Quarto Book 配置与首页**

配置 `project.type: book`、`project.output-dir: _book`；HTML 使用 `styles/site.scss`、代码复制和目录，Typst 使用 Quarto 1.9 内建 orange-book、目录、章节编号及 `Noto Serif CJK SC`。`book` 包含工作标题、`output-file: llm-book`、`downloads: [pdf]`、`search: true` 和首页。`_brand.yml` 只定义一套克制的学术配色，不新建 Typst 扩展。

首页写清目标读者、七部分路线、低资源阅读方式和当前“纵向样板”状态，不放空泛占位文字；路线说明中明确出现“梯度裁剪”“注意力”“强化学习”“RLHF”“Scaling Law”，供中文搜索验收定位。`.gitignore` 排除所有生成物。

- [ ] **Step 4: 构建并验证双输出**

Run: `quarto render && node --test tests/artifacts/artifacts.test.mjs`  
Expected: HTML 与 PDF 均生成，测试 PASS。

Run: `pdftotext _book/llm-book.pdf /tmp/llm-book-index.txt`  
Expected: `/tmp/llm-book-index.txt` 能提取“理解大语言模型”，不出现整页乱码；若字体缺失，先安装 Noto CJK 字体重跑，不通过删除中文来规避。

### Task 3: 样板章静态正文、代码与正式 PDF

**Files:**
- Create: `softmax-cross-entropy.qmd`
- Create: `assets/figures/next-token-pipeline.svg`
- Create: `assets/figures/cross-entropy-curve.svg`
- Create: `assets/figures/nucleus-sampling.svg`
- Modify: `_quarto.yml`
- Modify: `references.bib`
- Modify: `tests/artifacts/artifacts.test.mjs`

**Interfaces:**
- Produces `_book/softmax-cross-entropy.html`，章内锚点为 `#sec-softmax`、`#sec-cross-entropy`、`#sec-decoding`。
- 图编号为 `#fig-next-token`、`#fig-cross-entropy`、`#fig-top-p`，可在正文中交叉引用。
- 文献键固定为 `bengio2003neural`、`goodfellow2016deep`、`holtzman2019curious`。

- [ ] **Step 1: 扩展产物测试并确认先失败**

新增断言：章节 HTML 存在；包含“从 Softmax 到交叉熵”“Teacher Forcing”“常见误解”“练习”；包含三幅静态 SVG 的引用；存在 `data-cites` 中的三个文献键；不包含未解析的 `[?]`。PDF 的章节内容在 Step 4 通过文本提取验证，不依赖易波动的文件大小或修改时间断言。

Run: `npm run build && npm run test:artifacts`  
Expected: FAIL，因为样板章和三幅图尚未创建。

- [ ] **Step 2: 写完整静态样板章**

章节依次覆盖：学习目标和先修知识、Logits、稳定 Softmax、NLL/MLE 与交叉熵、单 Token 困惑度与语料平均 NLL 的区别、Teacher Forcing、温度与 Top-p、常见误解、总结和至少五道练习。在 Python fenced code block 中使用 `{{< include examples/softmax_cross_entropy.py >}}` 复用 Task 1 代码，不复制实现。

至少引用 Bengio 神经语言模型、Goodfellow 等《Deep Learning》和 Holtzman 等 Nucleus Sampling 原始文献；不要虚构页码、结论或 DOI。

- [ ] **Step 3: 创建三幅可打印 SVG 教学图**

分别表达“上下文到下一 Token 分布”的流水线、交叉熵随目标概率变化的曲线、Top-p 截断与重归一化。图内文字使用中文和通用数学符号，SVG 在浅色/深色页面及 PDF 中均可读，并在 `.qmd` 中写具体 `fig-alt`。

- [ ] **Step 4: 渲染并检查正文和 PDF**

Run: `npm run build && npm run test:artifacts && npm run test:python && pdftotext _book/llm-book.pdf /tmp/llm-book-chapter.txt`  
Expected: 全部测试 PASS，HTML 无未解析引用，`/tmp/llm-book-chapter.txt` 包含“从 Softmax 到交叉熵”和“练习”。

人工检查公式中的 Softmax 分母、交叉熵负号、温度指数 `1/T`、Top-p 重归一化和困惑度定义；这些数学正确性不能由“构建成功”代替。

### Task 4: 两个原生 JS 交互及静态后备

**Files:**
- Create: `assets/js/components/cross-entropy.js`
- Create: `assets/js/components/decoding-explorer.js`
- Create: `assets/js/book.js`
- Create: `playwright.config.mjs`
- Create: `tests/support/static-server.mjs`
- Create: `tests/e2e/interactions.spec.mjs`
- Modify: `softmax-cross-entropy.qmd`
- Modify: `styles/site.scss`
- Modify: `package.json`
- Modify: `package-lock.json`

**Interfaces:**
- `assets/js/book.js` 注册 `<llm-cross-entropy probability="0.5">` 与 `<llm-decoding-explorer tokens="的,模型,学习" probabilities="0.2,0.5,0.3" temperature="1" top-p="1">`。
- 两个组件分别导出 `defineCrossEntropyElement()` 和 `defineDecodingExplorerElement()`，并复用 Task 1 的 JS 函数。
- 可访问输出使用固定 test id：`ce-probability`、`ce-loss`、`ce-perplexity`、`decoding-temperature`、`decoding-top-p`、`decoding-kept-count`、`decoding-distribution`。
- 测试服务器将 `_book/` 挂载到 `http://127.0.0.1:4173/llm-book/`，以验证 Pages 项目子路径；`npm run test:e2e` 执行 Playwright。

- [ ] **Step 1: 配置浏览器测试并先写失败用例**

添加 `@playwright/test` 开发依赖，并执行 `npx playwright install chromium` 安装测试浏览器。编写测试访问 `/llm-book/softmax-cross-entropy.html`，将 `ce-probability` 从 `0.5` 调到 `0.25`，断言 `ce-loss == 2 * Math.LN2`、`ce-perplexity == 4`；将解码组件保持 `temperature=1`、设置 `top-p=0.7`，断言保留 2 个 Token、分布归一化且原顺序为“模型”后“学习”。同时断言所有本地脚本和 SVG 请求返回 200。

Run: `npm run build && npx playwright test tests/e2e/interactions.spec.mjs`  
Expected: FAIL，因为两个自定义元素尚未注册和实现。

- [ ] **Step 2: 实现交叉熵交互**

用带明确标签的 range input 控制目标 Token 概率，用 SVG 绘制 `-log(p)` 曲线和当前点，同时输出自然对数损失与单 Token 困惑度。只读取 Task 1 的函数结果，不在组件中另写一份公式。

- [ ] **Step 3: 实现解码分布交互**

用两个带标签的 range input 控制温度和 Top-p，使用 Task 1 的温度变换与截断结果绘制 SVG 概率条；被截断 Token 保留可见但标记为排除，显示保留数量和归一化后的概率。初始输入固定为接口中的三个 Token，保证测试和学习示例可复现。

- [ ] **Step 4: 在 Quarto 中接入条件内容和 no-JS 后备**

HTML 格式显示自定义元素，并在章末以 `type="module"` 加载相对路径 `assets/js/book.js`；`unless-format="html"` 分支继续显示 Task 3 的静态图。HTML 的 `<noscript>` 中也放同一静态图，避免禁用 JS 时只剩空容器。

- [ ] **Step 5: 构建并验证交互与后备**

Run: `npm run build && npm run test:js && npx playwright test tests/e2e/interactions.spec.mjs`  
Expected: 数值测试和浏览器测试全部 PASS，无页面脚本异常，子路径资源无 404。

### Task 5: 中文搜索、可访问性与边界体验

**Files:**
- Create: `tests/e2e/quality.spec.mjs`
- Modify: `package.json`
- Modify: `package-lock.json`
- Modify: `styles/site.scss`
- Modify: `assets/js/components/cross-entropy.js`
- Modify: `assets/js/components/decoding-explorer.js`
- Conditionally create only if Quarto search fails: `scripts/build-search-index.mjs`, `assets/js/search.js`

**Interfaces:**
- `@axe-core/playwright` 的 WCAG 2.1 AA 扫描结果必须为 `violations: []`。
- 若默认搜索不合格，`buildSearchIndex({ bookDir: '_book' })` 在构建后生成静态 `search-index.json`，搜索仍不依赖后端；只有实际触发该条件才实现此接口。

- [ ] **Step 1: 先写质量测试**

添加 `@axe-core/playwright` 开发依赖。覆盖以下行为：禁用 JS 时两幅后备图可见；`390×844` 视口无水平溢出；键盘方向键能操作两个滑块；`prefers-reduced-motion: reduce` 下组件没有非必要动画；axe 的 WCAG 2.1 AA 扫描无违规；遍历首页和样板章的站内链接及 PDF 下载链接，均返回 200 且目标存在；搜索“交叉熵”“梯度裁剪”“注意力”“强化学习”“RLHF”“Scaling Law”均返回明确目标页，其中后五者可指向首页相应路线说明。

- [ ] **Step 2: 运行质量测试并记录真实结果**

Run: `npx playwright test tests/e2e/quality.spec.mjs`  
Expected: 逐项记录组件、CSS、后备内容和搜索结果；已经正确的行为允许直接通过，不强行制造失败，失败用例不得删除或改成宽松断言。

- [ ] **Step 3: 只修复实际失败项**

针对失败补齐表单标签、SVG 标题/描述、焦点样式、移动布局或减少动态效果规则。若仅默认中文搜索失败，先用页面标题、章节标题和正文的规范化子串匹配实现静态索引，中文不按空格分词，英文查询统一小写；把索引脚本接到 `npm run build` 的 `quarto render` 之后，不引入搜索服务；若全部通过则不做无行为增益的改动。

- [ ] **Step 4: 重跑所有浏览器测试**

Run: `npm run build && npx playwright test`  
Expected: 所有交互与质量测试 PASS，axe 无 WCAG 2.1 AA 违规。

### Task 6: 一键验证、CI、提交与 Pages 发布

**Files:**
- Create: `.github/workflows/book.yml`
- Modify: `package.json`
- Modify: `.gitignore`

**Interfaces:**
- `test:js` 固定为 `node --test tests/unit/*.test.mjs`（不含生成物测试），`test:python` 固定为 `python3 -m unittest discover -s tests -p 'test_*.py'`，`test:artifacts` 固定为 `node --test tests/artifacts/artifacts.test.mjs`。
- `npm run verify` 固定为 `npm run build && npm run test:js && npm run test:python && npm run test:artifacts && npm run test:e2e`，覆盖 HTML/PDF、JS、Python、产物和全部浏览器测试。
- GitHub Actions 的 `verify` job 复现同一命令并上传 `_book/`；`deploy` job 仅在 `main` 分支验证成功后部署。

- [ ] **Step 1: 编写 GitHub Actions 工作流**

工作流在 push、pull request 和手动触发时运行。使用 `actions/checkout@v6`、`quarto-dev/quarto-actions/setup@v2`、`actions/setup-python@v7`（Python 3.14）、`actions/setup-node@v6`（Node 24），安装 `fonts-noto-cjk` 与 `poppler-utils`，执行 `npm ci`、`npx playwright install --with-deps chromium`、`npm run verify`。Quarto setup 的 `version` 固定为本地实际验证过的 1.9.x 补丁版本，不使用 `latest` 或预发布版。

验证成功后使用 `actions/configure-pages@v5`、`actions/upload-pages-artifact@v4`（路径 `_book/`）和 `actions/deploy-pages@v4`。`deploy` 必须 `needs: verify`，仅在非 PR 的 `main` 引用运行，并设置 `environment.name: github-pages` 和部署 URL 输出；权限最小化为 `contents: read`、`pages: write`、`id-token: write`，Pages 并发部署取消旧任务。

- [ ] **Step 2: 在本地执行完整干净验证**

删除可再生成的 `_book/` 后运行 `npm run verify`，避免把上一任务的旧产物误当成本次结果。  
Expected: HTML/PDF 构建、全部单元测试、产物测试和浏览器测试均 PASS。

Run: `pdftotext _book/llm-book.pdf /tmp/llm-book.txt`  
Expected: 能提取“交叉熵”“Teacher Forcing”和“练习”；再打开 PDF 人工检查中文字体、公式、三幅图、代码换行、引用和分页，不能用文件存在代替正式排版验收。

- [ ] **Step 3: 完成本需求的唯一本地提交**

确认 `git status` 只包含本设计、计划、源码、测试和工作流，不包含 `_book/` 或测试报告。若当前目录仍不是 Git 仓库，执行 `git init -b main`；随后暂存全部目标文件，已有提交时只 amend 本需求创建的未推送提交：

```bash
git add .
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  git commit --amend -m "feat: add LLM book vertical slice"
else
  git commit -m "feat: add LLM book vertical slice"
fi
git rev-list --count HEAD
```

Expected: `git rev-list --count HEAD` 输出 `1`，提交包含完整纵向样板且工作树干净。

- [ ] **Step 4: 经用户明确授权后发布并验证公网结果**

先向用户取得公开仓库地址、Pages URL 和 push 确认；未取得时停止于此并明确报告“本地完成、尚未发布”，不得猜仓库名或自行创建远程仓库。获得确认后再关联远程并 push，确保仓库 Pages 的 Source 为 GitHub Actions，使用 `gh run watch` 等待工作流完成。

最后用 `curl -fsSL <pages-url>/softmax-cross-entropy.html` 检查章节可公开访问，用浏览器实际点击两个交互并打开 `<pages-url>/llm-book.pdf`。只有这些远程检查通过后才能宣称 GitHub Pages 发布完成。
