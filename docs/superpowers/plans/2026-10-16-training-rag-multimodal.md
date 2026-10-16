# Distributed Training, RAG, Multimodal, and Visualization Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 11 页教材上新增分布式训练、检索/RAG、多模态三章，并为因果掩码、PPO 截断、投机解码增加可复核的交互可视化，最终形成 14 页的 HTML/Typst 双输出教材。

**Architecture:** 继续保持 Quarto `.qmd` 为唯一内容源；每章使用原生 Web Component、内联 SVG 和小型内置数据，交互只负责探索正文已经解释的数值模型。所有新增交互都有 noscript 静态图或表格及 PDF 等价内容；六个内容工作流只拥有各自文件，共享导航、参考文献和跨章测试最后统一集成。

**Tech Stack:** Quarto 1.9.38、Pandoc Markdown、Typst、原生 JavaScript ES Modules、SVG、Node.js 内置测试、Playwright、axe-core。

**Spec:** `docs/superpowers/specs/2026-10-14-llm-book-design.md`

## Global Constraints

- 不引入 React、Vue、图表库、后端、模型 API、权重文件或新的运行时依赖；资源必须从 `/llm-book/` 子路径本地加载。
- 每个可视化必须回答一个明确教学问题；数值例子标注简化假设，不把玩具估算写成真实模型或集群的保证。
- 所有交互具备原生可标签控件、键盘操作、可见焦点、动态 SVG 描述、390×844 无溢出和 reduced-motion 支持。
- 禁用 JavaScript 与 PDF 中不得丢失关键知识；静态 SVG 包含 `title/desc`，图像具有有意义的 `alt/fig-alt`。
- 新正文使用简体中文、首次出现保留英文术语，含学习目标、必要推导、常见误解、总结、六道练习和原始论文/官方资料；不重复已有章节的基础定义。
- 轻量数值内核先写失败测试再实现；最终必须通过完整 `npm run verify`，不能通过放宽既有断言消错。
- 本轮只创建一个 Git 提交；中间工作不逐个提交，不用 `git commit --amend` 修改上一轮已推送提交。

## Review Focus

- 分布式显存与多模态成本是简化账目：正文必须列出字节数、均匀切分等假设，测试固定玩具输入而不是冒充真实基准（Tasks 1、3）。
- 检索分数不是概率：点积/余弦切换、top-k 与上下文 Token 预算必须分别计算，不能被 Top-p 归一化语义污染（Task 2）。
- 投机解码在 `q_i=0`、完全接受和无拒绝质量时必须有定义明确的边界行为，无效输入不得保留陈旧结果（Task 6）。
- 插入新章会改变后续 PDF 章节号：导航后处理、页面计数、PDF 标题/练习断言和搜索列表必须同步，不能保留“11 个章节页面”（Task 7）。
- 交互默认状态、静态后备和正文数值必须可互相复核；无脚本、PDF、移动端及跨源拒绝场景都需真实浏览器或产物测试（Tasks 1–8）。

---

### Task 1: 分布式训练与显存优化章节

**Files:**
- Create: `distributed-training.qmd`
- Create: `assets/js/components/distributed-training.js`
- Create: `assets/figures/distributed-training.svg`
- Create: `tests/unit/distributed-training.test.mjs`
- Create: `tests/e2e/distributed-training.spec.mjs`

**Interfaces:**
- Produces: `defineDistributedTrainingElement()`，注册 `<llm-distributed-training>`。
- Produces: 纯函数 `estimateStaticStates({ parameters, layers, dataParallel, tensorParallel, pipelineParallel, zeroStage })`，返回每设备权重、梯度、优化器与总静态状态字节数；`layers` 用于验证均匀 PP 切分。
- Consumes: 既有 `pretraining-scaling.qmd` 的 FLOPs/混合精度口径和 `transformer-architecture.qmd` 的 Block 结构；不重讲 Scaling 或优化器。

- [ ] **Step 1: 写数值内核失败测试。** 固定玩具模型 `parameters=1_000_000_000,layers=12`，明确每个参数 16 字节静态状态（2 字节权重、2 字节梯度、4 字节主权重、8 字节 Adam 一阶/二阶状态）；DP 不切分状态，TP/PP 均匀切分，ZeRO-3 再按 DP 度切分。测试 `DP=8,TP=PP=1,zeroStage=0` 总量为 16 GB（十进制字节）以及 `DP=2,TP=2,PP=2,zeroStage=3` 总量为 2 GB，并断言 `pipelineParallel=5` 不能均匀切分 12 层时抛错，同时覆盖不合法并行度。
- [ ] **Step 2: 运行 `node --test tests/unit/distributed-training.test.mjs`，确认因函数不存在而失败；随后实现最小纯函数并 rerun 至通过。** 非均匀切分必须显式报错，不能静默生成负值或 NaN。
- [ ] **Step 3: 写章节和交互。** 章节依次覆盖梯度累积、DP、TP、PP、ZeRO/FSDP、激活重计算及三维并行；用同一个小模型解释“数据切分、张量切分、层切分、状态切分”不是同一回事。交互切换策略与并行度，显示静态状态账目和通信事件；固定四阶段/四 microbatch 图展示流水线气泡，不声称模拟真实吞吐。
- [ ] **Step 4: 添加静态后备和 E2E。** `distributed-training.svg` 同时给出 DP/TP/PP 数据流和流水线时间线；E2E 核对上述数值、控件标签、动态描述、无脚本/PDF 后备、axe、390px、本地资源和恰好六道练习。
- [ ] **Step 5: 集成后运行 `npx playwright test tests/e2e/distributed-training.spec.mjs` 与该单元测试，确认通过。**

### Task 2: 检索、RAG 与上下文管理章节

**Files:**
- Create: `retrieval-rag.qmd`
- Create: `assets/js/components/retrieval-rag.js`
- Create: `assets/figures/retrieval-rag.svg`
- Create: `tests/unit/retrieval-rag.test.mjs`
- Create: `tests/e2e/retrieval-rag.spec.mjs`

**Interfaces:**
- Produces: `defineRetrievalRagElement()`，注册 `<llm-retrieval-rag>`。
- Produces: `rankDocuments(query, documents, metric)` 与 `selectContext(rankedDocuments, tokenBudget)`；文档对象含 `id/vector/text/tokens`，排序结果保留原始分数和排名。
- Consumes: `embedding-geometry.qmd` 的内积/余弦、`inference-systems.qmd` 的上下文成本；不把近重复去重当作 RAG 已讲内容。

- [ ] **Step 1: 写失败测试。** 用 `query=[1,0]`、长偏离文档 `[10,1]` 和短对齐文档 `[1,0]`，断言点积优先前者而余弦优先后者；测试稳定 tie-break、空输入/非有限值，以及 Token 预算只按顺序保留完整文档、超预算文档不产生半条上下文。
- [ ] **Step 2: 运行单元测试确认 RED，实现两个纯函数并确认 GREEN。** 相似度不归一化成概率；返回新数组，不修改调用方文档。
- [ ] **Step 3: 写章节和交互。** 覆盖 chunking、向量检索、ANN 的召回/成本权衡、metadata filter、top-k、重排、上下文组装、引用与 groundedness、RAG 评测，以及“扩大上下文不等于有效利用上下文”。交互使用固定二维文档集，可切换点积/余弦、top-k 和 Token 预算，联动二维点图、排名条和最终上下文；明确这是几何玩具，不运行真实语义模型。
- [ ] **Step 4: 添加 `retrieval-rag.svg`（检索→重排→预算→生成/引用流程）和 E2E。** E2E 核对排名反转、预算、动态描述、无脚本表格/图、axe、390px、本地资源和六道练习。
- [ ] **Step 5: 集成后运行本章 Playwright 与单元测试。**

### Task 3: 多模态模型与统一表示章节

**Files:**
- Create: `multimodal-models.qmd`
- Create: `assets/js/components/multimodal-patch.js`
- Create: `assets/figures/multimodal-models.svg`
- Create: `tests/unit/multimodal-patch.test.mjs`
- Create: `tests/e2e/multimodal-models.spec.mjs`

**Interfaces:**
- Produces: `defineMultimodalPatchElement()`，注册 `<llm-multimodal-patch>`。
- Produces: `computePatchLayout({ width, height, patchSize, textTokens })`，返回行列 patch 数、视觉 Token 数、总序列长和玩具注意力分数数量。
- Consumes: `attention.qmd` 的序列/掩码与 `inference-systems.qmd` 的二次成本直觉；不重讲完整 ViT 或 Transformer Block。

- [ ] **Step 1: 写失败测试。** 固定 224×224 图像：patch 14 得到 16×16=256 个视觉 Token，patch 16 得到 196，patch 32 得到 49；加入固定 `textTokens=20` 后核对总序列与 `sequenceLength²`。覆盖不能整除时右/下边缘补零并按 ceil 计数的语义、非正尺寸和非整数 patch。
- [ ] **Step 2: 运行测试确认 RED，实现纯函数并确认 GREEN。**
- [ ] **Step 3: 写章节和交互。** 覆盖图像 patch/音频帧、模态编码器、对比对齐（CLIP）、投影/Q-Former、early fusion 与 cross-attention、视觉指令微调、统一表示的边界，以及幻觉/模态偏置/评测。交互调整 patch size，联动网格、视觉/文本 Token 数和二次分数成本；明确只核算教学序列账，不声称某一真实模型结构。
- [ ] **Step 4: 添加 `multimodal-models.svg`（视觉编码→投影/融合→共享 LLM）和 E2E。** 核对三组数值、网格、动态描述、静态后备、axe、390px、本地资源和六道练习。
- [ ] **Step 5: 集成后运行本章 Playwright 与单元测试。**

### Task 4: 因果掩码交互升级

**Files:**
- Modify: `attention.qmd` 的 `#sec-causal-mask`
- Create: `assets/js/components/causal-mask-explorer.js`
- Create: `tests/e2e/causal-mask-explorer.spec.mjs`
- Reuse: `assets/figures/causal-mask.svg`、`assets/js/lib/attention.js`

**Interfaces:**
- Consumes: 现有 `causalMask(scores)`（返回已加掩码的分数矩阵）、逐行 `maskedSoftmax(logits)` 与既有三乘三固定分数矩阵。
- Produces: `defineCausalMaskExplorerElement()`，注册 `<llm-causal-mask-explorer>`；宿主继续使用 `#fig-causal-mask`。

- [ ] **Step 1: 先写 E2E 数值断言。** 默认状态与静态图一致：未来权重为 0、每行和为 1；关闭因果掩码后得到原始行 Softmax，且动态 `desc` 同步改变。
- [ ] **Step 2: 实现三矩阵联动。** 展示“原始分数→掩码后分数→Softmax 权重”，提供因果开关和查询行选择，突出当前行；不改变既有单头注意力组件。
- [ ] **Step 3: 保留原静态图作为 noscript/PDF 后备，核对锚点唯一、键盘操作、动态描述、axe 与 390px。** 运行新 spec 和既有 `attention-interactions.spec.mjs`、`attention-quality.spec.mjs`。

### Task 5: PPO Clipping 交互升级

**Files:**
- Modify: `rl-foundations.qmd` 的 `#sec-ppo`
- Create: `assets/js/components/ppo-clipping.js`
- Create: `assets/figures/ppo-clipping.svg`
- Create: `tests/unit/ppo-clipping.test.mjs`
- Create: `tests/e2e/ppo-clipping.spec.mjs`

**Interfaces:**
- Produces: `ppoClippedObjective(ratio, advantage, epsilon)` 与 `definePpoClippingElement()`，注册 `<llm-ppo-clipping>`。

- [ ] **Step 1: 写失败单元测试。** 固定 `epsilon=0.2`，断言 `ratio=1.4,advantage=1` 返回 1.2，`ratio=0.6,advantage=-1` 返回 -0.8；覆盖比率零点、正负优势和非有限/负 epsilon。
- [ ] **Step 2: 确认 RED 后实现纯函数并确认 GREEN。**
- [ ] **Step 3: 添加交互曲线。** 横轴为概率比，展示未截断项、截断项和二者 min；可切正负优势、调整 ratio/epsilon，标出当前点，不把 clip 描述成参数硬约束。
- [ ] **Step 4: 添加静态曲线后备和 E2E，运行新旧 RL 测试，检查数值、动态描述、axe、390px 和无脚本/PDF。**

### Task 6: 投机解码分布交互升级

**Files:**
- Modify: `inference-systems.qmd` 的 `#sec-speculative-decoding`
- Create: `assets/js/components/speculative-decoding.js`
- Create: `assets/figures/speculative-decoding.svg`
- Create: `tests/unit/speculative-decoding.test.mjs`
- Create: `tests/e2e/speculative-decoding.spec.mjs`

**Interfaces:**
- Produces: `speculativeAcceptance(p, q)`（返回 `{ perToken, total }`，`q_i=0` 时对应 `perToken[i]=null`）、`speculativeResidual(p, q)`（返回残差数组；无拒绝质量时返回 `null`）与 `defineSpeculativeDecodingElement()`，注册 `<llm-speculative-decoding>`。

- [ ] **Step 1: 写失败单元测试。** 固定 `p=[0.5,0.3,0.2]`、`q=[0.4,0.4,0.2]`，逐 Token 接受率为 `[1,0.75,1]`，总接受率为 0.9，残差分布为 `[1,0,0]`；覆盖 `p=q` 时总接受率 1 且残差为 `null`、`q_i=0` 的不可达 Token、非归一化和维度不匹配。
- [ ] **Step 2: 确认 RED 后实现纯函数并确认 GREEN。** 当 `q_i>0` 时单 Token 接受率为 `min(1,p_i/q_i)`；`q_i=0` 表示草稿不会采样该 Token，显示 `null` 且对总接受率贡献 0。总接受率为 `sum(min(p_i,q_i))`；禁止除以零和静默返回 NaN。
- [ ] **Step 3: 添加分布交互。** 用少量 Token 的目标/草稿分布预设或受约束控件，显示两组概率条、逐 Token 接受率、总接受率和拒绝后的归一化残差；明确它减少串行步数且保持目标分布，不与测试时搜索混同。
- [ ] **Step 4: 添加静态分布图和 E2E，运行新旧推理测试，覆盖默认数值、边界状态、可访问错误、动态描述、390px 和静态后备。**

### Task 7: 共享集成、参考文献与跨章回归

**Files:**
- Modify: `_quarto.yml`、`index.qmd`、`assets/js/book.js`、`references.bib`
- Modify: `scripts/strip-embedded-navigation.mjs`
- Modify: `tests/artifacts/artifacts.test.mjs`、`tests/e2e/quality.spec.mjs`
- Create: `docs/superpowers/plans/2026-10-16-training-rag-multimodal.md`（保存本计划）

**Interfaces:**
- Consumes: Tasks 1–6 的定义函数、组件标签、文件锚点和测试。
- Produces: 最终章节顺序为 `index → softmax-cross-entropy → embedding-geometry → optimization → attention → transformer-architecture → pretraining-scaling → distributed-training → inference-systems → rl-foundations → rlhf-alignment → frontier-rl → retrieval-rag → multimodal-models`。

- [ ] **Step 1: 接线和导航。** 按上述顺序更新 `_quarto.yml` 与导航后处理数组；在 `book.js` 注册六个新组件，首页改为“14 个章节页面”，仅把仍缺失的主题列为待写。
- [ ] **Step 2: 补充原始参考文献。** 优先复用已有 BibTeX key；新增 ZeRO、Megatron-LM、GPipe/流水线、RAG/DPR、HNSW/FAISS、Lost in the Middle、ViT、CLIP、BLIP-2/Flamingo/LLaVA 等正文实际引用条目，不堆砌未引用文献。
- [ ] **Step 3: 更新跨章测试。** 把三个新 HTML 页加入本地资源、页面体积、无脚本移动端和搜索用例；更新 PDF 新章节号（推理 9、RL 10、RLHF 11、前沿 RL 12、RAG 13、多模态 14）、练习、图文本和引用名称断言。
- [ ] **Step 4: 运行 `npm run build && npm run test:js && npm run test:python && npm run test:artifacts`。** 预期全部通过；若有失败，修正源稿或实现，不删除新章数学/后备断言。
- [ ] **Step 5: 运行完整 `npx playwright test` 和 `git diff --check`。** 检查没有根目录临时 HTML、`site_libs/`、`_book/` 或其他生成物进入暂存区。

### Task 8: 独立审查、发布与线上验收

**Files:**
- Modify only: 审查发现的本轮源文件和测试；最终提交 `main`。

- [ ] **Step 1: 独立审查。** 由未参与实现的 reviewer 核对数学正确性、章节依赖、可视化与正文一致性、静态后备和共享集成；Critical/Important 问题先写能复现的回归测试再修复。
- [ ] **Step 2: 最终完整验证。** 运行 `npm run verify`，确认构建、全部 JS/Python/artifact/Playwright 测试通过，并检查最终 diff 只包含本轮范围。
- [ ] **Step 3: 单提交发布。** 执行 `git add .`、`git diff --cached --check`、`git commit -m "feat: add training and multimodal chapters"`、`git push origin main`。
- [ ] **Step 4: 等待 `Verify and deploy book` 成功。** 用 `gh run watch <id> --exit-status` 核对 verify 和 deploy 两个 job。
- [ ] **Step 5: 线上验收。** 用真实 Chromium 检查公开首页为 14 页版本，至少操作分布式并行、RAG 排名、多模态 patch、PPO clipping 和投机解码交互；14 页在 390px 无页面溢出、无资源错误，`curl` 实际下载并验证 `llm-book.pdf` 文件头。
