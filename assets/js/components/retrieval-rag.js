import { cosine, dot } from '../lib/linalg.js';

function validateQuery(query) {
  if (!Array.isArray(query) || query.length === 0) {
    throw new RangeError('query 必须是非空向量。');
  }
  for (const value of query) {
    if (!Number.isFinite(value)) {
      throw new RangeError('query 必须只含有限数值。');
    }
  }
}

export function rankDocuments(query, documents, metric) {
  validateQuery(query);
  if (!Array.isArray(documents)) {
    throw new RangeError('documents 必须是数组。');
  }
  if (!['dot', 'cosine'].includes(metric)) {
    throw new RangeError('metric 必须是 dot 或 cosine。');
  }

  const scored = documents.map((document, index) => {
    const score = metric === 'dot'
      ? dot(query, document.vector)
      : cosine(query, document.vector);
    if (!Number.isFinite(score)) {
      throw new RangeError('相似度分数必须是有限数值。');
    }
    return { document, index, score };
  });

  scored.sort((a, b) => b.score - a.score || a.index - b.index);
  return scored.map(({ document, score }, index) => ({
    ...document,
    score,
    rank: index + 1,
  }));
}

export function selectContext(rankedDocuments, tokenBudget) {
  if (!Array.isArray(rankedDocuments)) {
    throw new RangeError('rankedDocuments 必须是数组。');
  }
  if (!Number.isSafeInteger(tokenBudget) || tokenBudget < 0) {
    throw new RangeError('tokenBudget 必须是非负整数。');
  }
  for (const document of rankedDocuments) {
    if (!Number.isSafeInteger(document.tokens) || document.tokens < 0) {
      throw new RangeError('文档 tokens 必须是非负整数。');
    }
  }

  const selected = [];
  let usedTokens = 0;
  for (const document of rankedDocuments) {
    if (usedTokens + document.tokens <= tokenBudget) {
      selected.push(document);
      usedTokens += document.tokens;
    }
  }
  return selected;
}

const QUERY = [1, 0];
const DOCUMENTS = [
  { id: 'D1', vector: [10, 1], text: '长偏离文档：缓存部署的历史说明', tokens: 8 },
  { id: 'D2', vector: [1, 0], text: '短对齐文档：余弦检索的定义', tokens: 3 },
  { id: 'D3', vector: [2, 2], text: '斜向文档：元数据过滤的笔记', tokens: 4 },
  { id: 'D4', vector: [0, 1], text: '正交文档：与查询无关的背景', tokens: 2 },
  { id: 'D5', vector: [3, 0.5], text: '中等偏离文档：索引参数说明', tokens: 6 },
];
const LABEL_OFFSETS = {
  D1: { dx: -12, dy: -14, anchor: 'end' },
  D2: { dx: 10, dy: 25, anchor: 'start' },
  D3: { dx: 10, dy: -12, anchor: 'start' },
  D4: { dx: 10, dy: -12, anchor: 'start' },
  D5: { dx: 10, dy: -10, anchor: 'start' },
};
const STATE_COLORS = {
  selected: '#1F648C',
  'budget-skipped': '#C65D21',
  'outside-top-k': '#8B969C',
};
const svgNamespace = 'http://www.w3.org/2000/svg';
const HTMLElementBase = globalThis.HTMLElement ?? class {};
let instanceCount = 0;

const componentStyles = `
  <style>
    llm-retrieval-rag { display: block; min-inline-size: 0; }
    .retrieval-widget { min-inline-size: 0; }
    .retrieval-toy-note { margin: 0 0 0.75rem; font-weight: 600; }
    .retrieval-controls { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .retrieval-controls .interactive-control {
      grid-template-columns: minmax(0, 1fr);
      min-inline-size: 0;
    }
    .retrieval-controls select {
      inline-size: 100%;
      min-inline-size: 0;
      min-block-size: 2.5rem;
      padding: 0.35rem 0.5rem;
      border: 1px solid #8b969c;
      border-radius: 0.25rem;
      background: #ffffff;
      color: #172b3a;
      font: inherit;
    }
    .retrieval-controls select:focus-visible,
    .retrieval-text-equivalent summary:focus-visible {
      outline: 3px solid #8c421f;
      outline-offset: 3px;
    }
    .retrieval-text-equivalent { margin-top: 0.75rem; min-inline-size: 0; }
    .retrieval-text-equivalent summary { min-block-size: 2rem; cursor: pointer; }
    .retrieval-text-equivalent table {
      inline-size: 100%;
      table-layout: fixed;
      border-collapse: collapse;
      font-size: 0.8rem;
      font-variant-numeric: tabular-nums;
    }
    .retrieval-text-equivalent caption,
    .retrieval-text-equivalent th,
    .retrieval-text-equivalent td { overflow-wrap: anywhere; }
    .retrieval-text-equivalent caption { padding: 0.4rem 0; text-align: left; }
    .retrieval-text-equivalent th,
    .retrieval-text-equivalent td {
      padding: 0.35rem 0.25rem;
      border-bottom: 1px solid #dde3e7;
      text-align: left;
      vertical-align: top;
    }
    .retrieval-context-section { min-inline-size: 0; }
    .retrieval-context-section h3 { margin-bottom: 0.5rem; font-size: 1rem; }
    .retrieval-context { margin: 0; padding-inline-start: 1.5rem; overflow-wrap: anywhere; }
    @media (max-width: 48rem) {
      .retrieval-controls { grid-template-columns: minmax(0, 1fr); }
    }
  </style>
`;

function createSvgElement(name, attributes = {}) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [attribute, value] of Object.entries(attributes)) {
    element.setAttribute(attribute, String(value));
  }
  return element;
}

function formatScore(score, metric) {
  return metric === 'cosine' ? score.toFixed(6) : String(Number(score.toFixed(6)));
}

function statusFor(document, candidateIds, selectedIds) {
  if (selectedIds.has(document.id)) {
    return { key: 'selected', text: '进入上下文' };
  }
  if (candidateIds.has(document.id)) {
    return { key: 'budget-skipped', text: '预算不足，跳过完整文档' };
  }
  return { key: 'outside-top-k', text: 'Top-k 外' };
}

function geometryPoint(vector) {
  return [70 + vector[0] * 30, 340 - vector[1] * 110];
}

class RetrievalRagElement extends HTMLElementBase {
  #els;

  connectedCallback() {
    if (this.#els) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'retrieval-rag'}-${instanceCount}`;

    this.innerHTML = `
      ${componentStyles}
      <figure class="interactive-widget retrieval-widget">
        <p class="retrieval-toy-note">这是固定二维几何玩具，不运行真实语义模型。</p>
        <div class="interactive-controls retrieval-controls">
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-metric">相似度度量</label>
            <select id="${prefix}-metric" data-testid="retrieval-metric">
              <option value="dot">点积（保留模长影响）</option>
              <option value="cosine">余弦（只比较方向）</option>
            </select>
          </div>
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-top-k">Top-k 文档数</label>
            <select id="${prefix}-top-k" data-testid="retrieval-top-k">
              <option value="1">1</option><option value="2">2</option>
              <option value="3">3</option><option value="4">4</option>
              <option value="5">5</option>
            </select>
          </div>
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-budget">上下文 Token 预算</label>
            <select id="${prefix}-budget" data-testid="retrieval-token-budget">
              <option value="0">0</option><option value="3">3</option>
              <option value="4">4</option><option value="5">5</option>
              <option value="8">8</option><option value="11">11</option>
              <option value="12">12</option><option value="15">15</option>
              <option value="18">18</option><option value="23">23</option>
            </select>
          </div>
        </div>
        <dl class="interactive-results retrieval-results">
          <div><dt>已用 Token</dt><dd><output data-testid="retrieval-used-tokens"></output></dd></div>
          <div><dt>最终文档</dt><dd><output data-testid="retrieval-context-ids"></output></dd></div>
        </dl>
        <svg class="interactive-chart" data-testid="retrieval-chart" viewBox="0 0 960 430"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">二维几何与原始检索分数排名</title>
          <desc id="${prefix}-chart-desc"></desc>
          <defs>
            <marker id="${prefix}-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3"
              orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L8,3 z" fill="#172B3A" />
            </marker>
          </defs>
          <text x="45" y="35" font-size="16" font-weight="700">固定二维文档与查询 q=(1,0)</text>
          <text x="490" y="35" font-size="16" font-weight="700">原始分数排名（条长仅作相对比较）</text>
          <g data-testid="retrieval-geometry"></g>
          <g data-testid="retrieval-ranking-bars"></g>
          <g font-size="12">
            <rect x="490" y="378" width="14" height="14" fill="#1F648C" />
            <text x="512" y="390">进入上下文</text>
            <rect x="635" y="378" width="14" height="14" fill="#C65D21" />
            <text x="657" y="390">Top-k 内但预算跳过</text>
            <rect x="835" y="378" width="14" height="14" fill="#8B969C" />
            <text x="857" y="390">Top-k 外</text>
          </g>
          <text x="45" y="408" font-size="12" fill="#4C5F6A">几何位置是教学坐标；分数不是概率，也不表示答案置信度。</text>
        </svg>
        <details class="retrieval-text-equivalent" data-testid="retrieval-text-equivalent">
          <summary>当前排名与预算的文字数据</summary>
          <p>查询 q=(1,0)。先取排名前的 Top-k，再按顺序装入完整文档；超剩余预算者跳过，不截断文本。</p>
          <table>
            <caption>固定二维文档的当前排名与上下文状态</caption>
            <thead><tr><th scope="col">排名</th><th scope="col">文档与向量</th><th scope="col">原始分数</th><th scope="col">Token</th><th scope="col">状态</th></tr></thead>
            <tbody data-testid="retrieval-ranking-body"></tbody>
          </table>
        </details>
        <section class="retrieval-context-section" aria-labelledby="${prefix}-context-title">
          <h3 id="${prefix}-context-title">最终上下文（仅完整文档）</h3>
          <ol data-testid="retrieval-context"></ol>
        </section>
        <figcaption>点积与余弦使用同一固定文档集。点图、排名条、文字表和最终上下文同步更新；这里不模拟 ANN、重排模型或生成模型。</figcaption>
      </figure>
    `;

    const byTestId = (testId) => this.querySelector(`[data-testid="${testId}"]`);
    this.#els = {
      prefix,
      metric: byTestId('retrieval-metric'),
      topK: byTestId('retrieval-top-k'),
      budget: byTestId('retrieval-token-budget'),
      usedTokens: byTestId('retrieval-used-tokens'),
      contextIds: byTestId('retrieval-context-ids'),
      chart: byTestId('retrieval-chart'),
      description: this.querySelector(`#${prefix}-chart-desc`),
      geometry: byTestId('retrieval-geometry'),
      bars: byTestId('retrieval-ranking-bars'),
      rankingBody: byTestId('retrieval-ranking-body'),
      context: byTestId('retrieval-context'),
    };
    this.#els.metric.value = this.getAttribute('metric') ?? 'dot';
    this.#els.topK.value = this.getAttribute('top-k') ?? '3';
    this.#els.budget.value = this.getAttribute('token-budget') ?? '12';

    const controls = [this.#els.metric, this.#els.topK, this.#els.budget];
    const controlIds = controls.map((control) => control.id).join(' ');
    for (const output of this.querySelectorAll('.interactive-results output')) {
      output.setAttribute('for', controlIds);
    }
    for (const control of controls) {
      control.addEventListener('change', () => this.#update());
      control.addEventListener('keydown', (event) => {
        if (!['ArrowDown', 'ArrowUp'].includes(event.key)) {
          return;
        }
        event.preventDefault();
        const direction = event.key === 'ArrowDown' ? 1 : -1;
        const nextIndex = Math.min(
          Math.max(control.selectedIndex + direction, 0),
          control.options.length - 1,
        );
        if (nextIndex !== control.selectedIndex) {
          control.selectedIndex = nextIndex;
          this.#update();
        }
      });
    }
    this.#update();
  }

  #update() {
    const metric = this.#els.metric.value;
    const topK = Number(this.#els.topK.value);
    const tokenBudget = Number(this.#els.budget.value);
    const ranked = rankDocuments(QUERY, DOCUMENTS, metric);
    const candidates = ranked.slice(0, topK);
    const selected = selectContext(candidates, tokenBudget);
    const candidateIds = new Set(candidates.map((document) => document.id));
    const selectedIds = new Set(selected.map((document) => document.id));
    const usedTokens = selected.reduce((total, document) => total + document.tokens, 0);
    const signature = [
      metric,
      topK,
      tokenBudget,
      ranked.map((document) => document.id).join(','),
      selected.map((document) => document.id).join(','),
      usedTokens,
    ].join('|');

    this.#els.usedTokens.dataset.value = String(usedTokens);
    this.#els.usedTokens.textContent = `${usedTokens} / ${tokenBudget} Token`;
    this.#els.contextIds.textContent = selected.length > 0
      ? selected.map((document) => document.id).join('、')
      : '无';
    this.#els.chart.dataset.signature = signature;
    this.#renderChart(ranked, candidates, selected, candidateIds, selectedIds, {
      metric,
      topK,
      tokenBudget,
      usedTokens,
    });
    this.#renderTable(ranked, candidateIds, selectedIds, metric);
    this.#renderContext(selected);
  }

  #renderChart(ranked, candidates, selected, candidateIds, selectedIds, state) {
    const geometryNodes = [
      createSvgElement('line', { x1: 40, y1: 340, x2: 405, y2: 340, stroke: '#5E6A70', 'stroke-width': 1.5 }),
      createSvgElement('line', { x1: 70, y1: 365, x2: 70, y2: 75, stroke: '#5E6A70', 'stroke-width': 1.5 }),
      createSvgElement('text', { x: 407, y: 364, 'font-size': 12 }),
      createSvgElement('text', { x: 45, y: 85, 'font-size': 12 }),
    ];
    geometryNodes[2].textContent = 'x₁';
    geometryNodes[3].textContent = 'x₂';

    const [queryX, queryY] = geometryPoint(QUERY);
    const queryLine = createSvgElement('line', {
      x1: 70,
      y1: 340,
      x2: queryX,
      y2: queryY,
      stroke: '#172B3A',
      'stroke-width': 3,
      'marker-end': `url(#${this.#els.prefix}-arrow)`,
    });
    const queryLabel = createSvgElement('text', { x: 82, y: 326, 'font-size': 12, 'font-weight': 700 });
    queryLabel.textContent = 'q';
    geometryNodes.push(queryLine, queryLabel);

    for (const document of ranked) {
      const [x, y] = geometryPoint(document.vector);
      const status = statusFor(document, candidateIds, selectedIds);
      const vectorLine = createSvgElement('line', {
        x1: 70,
        y1: 340,
        x2: x,
        y2: y,
        stroke: STATE_COLORS[status.key],
        'stroke-width': 1.5,
        'stroke-dasharray': status.key === 'selected' ? 'none' : '5 4',
      });
      const point = createSvgElement('circle', {
        cx: x,
        cy: y,
        r: status.key === 'selected' ? 8 : 6,
        fill: STATE_COLORS[status.key],
        stroke: '#FFFFFF',
        'stroke-width': 1.5,
        'data-testid': `retrieval-point-${document.id}`,
        'data-state': status.key,
      });
      const offset = LABEL_OFFSETS[document.id];
      const label = createSvgElement('text', {
        x: x + offset.dx,
        y: y + offset.dy,
        'font-size': 12,
        'font-weight': 700,
        'text-anchor': offset.anchor,
      });
      label.textContent = document.id;
      geometryNodes.push(vectorLine, point, label);
    }
    this.#els.geometry.replaceChildren(...geometryNodes);

    const maximumScore = Math.max(...ranked.map((document) => document.score), 0);
    const barGroups = ranked.map((document, index) => {
      const y = 62 + index * 62;
      const status = statusFor(document, candidateIds, selectedIds);
      const group = createSvgElement('g', {
        'data-document-id': document.id,
        'data-rank': document.rank,
        'data-state': status.key,
      });
      const label = createSvgElement('text', { x: 490, y: y + 17, 'font-size': 13, 'font-weight': 700 });
      label.textContent = `${document.rank}. ${document.id}`;
      const track = createSvgElement('rect', {
        x: 550,
        y,
        width: 180,
        height: 22,
        rx: 3,
        fill: '#EEF1F2',
      });
      const bar = createSvgElement('rect', {
        x: 550,
        y,
        width: maximumScore > 0 ? 180 * Math.max(0, document.score) / maximumScore : 0,
        height: 22,
        rx: 3,
        fill: STATE_COLORS[status.key],
        'data-score': document.score,
      });
      const score = createSvgElement('text', { x: 742, y: y + 17, 'font-size': 12 });
      score.textContent = formatScore(document.score, state.metric);
      const statusLabel = createSvgElement('text', { x: 550, y: y + 41, 'font-size': 11, fill: '#4C5F6A' });
      statusLabel.textContent = status.text;
      group.append(label, track, bar, score, statusLabel);
      return group;
    });
    this.#els.bars.replaceChildren(...barGroups);

    const metricName = state.metric === 'dot' ? '点积' : '余弦';
    const candidateDescription = candidates.map((document) =>
      `${document.id}（原始分数 ${formatScore(document.score, state.metric)}，${document.tokens} Token）`,
    ).join(' → ');
    const contextDescription = selected.length > 0
      ? selected.map((document) => document.id).join('、')
      : '空';
    this.#els.description.textContent = `几何玩具，不运行真实语义模型。查询 q=(1,0)，当前使用${metricName}，Top-k=${state.topK}，Token 预算 ${state.tokenBudget}。候选排名：${candidateDescription}。最终上下文：${contextDescription}，共 ${state.usedTokens} Token。Top-k 内放不下的完整文档被跳过，Top-k 外文档不参与组装；所有分数都不是概率。`;
  }

  #renderTable(ranked, candidateIds, selectedIds, metric) {
    const rows = ranked.map((rankedDocument) => {
      const status = statusFor(rankedDocument, candidateIds, selectedIds);
      const tableRow = document.createElement('tr');
      tableRow.dataset.documentId = rankedDocument.id;
      tableRow.dataset.testid = `retrieval-row-${rankedDocument.id}`;

      const rankCell = document.createElement('td');
      rankCell.textContent = String(rankedDocument.rank);
      const documentCell = document.createElement('th');
      documentCell.scope = 'row';
      documentCell.textContent = `${rankedDocument.id} (${rankedDocument.vector.join(', ')})`;
      const scoreCell = document.createElement('td');
      scoreCell.dataset.testid = 'retrieval-score';
      scoreCell.dataset.value = String(rankedDocument.score);
      scoreCell.textContent = formatScore(rankedDocument.score, metric);
      const tokenCell = document.createElement('td');
      tokenCell.textContent = String(rankedDocument.tokens);
      const statusCell = document.createElement('td');
      statusCell.textContent = status.text;
      tableRow.append(rankCell, documentCell, scoreCell, tokenCell, statusCell);
      return tableRow;
    });
    this.#els.rankingBody.replaceChildren(...rows);
  }

  #renderContext(selected) {
    if (selected.length === 0) {
      const empty = document.createElement('li');
      empty.textContent = '无文档进入上下文；不能把空上下文当作有依据的回答。';
      this.#els.context.replaceChildren(empty);
      return;
    }
    const items = selected.map((selectedDocument) => {
      const item = document.createElement('li');
      item.dataset.documentId = selectedDocument.id;
      item.textContent = `[${selectedDocument.id}] ${selectedDocument.text}（${selectedDocument.tokens} Token）`;
      return item;
    });
    this.#els.context.replaceChildren(...items);
  }
}

export function defineRetrievalRagElement() {
  if (!customElements.get('llm-retrieval-rag')) {
    customElements.define('llm-retrieval-rag', RetrievalRagElement);
  }
}
