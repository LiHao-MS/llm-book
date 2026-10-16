import { causalMask, maskedSoftmax } from '../lib/attention.js';

const scores = [[2, 1, 3], [1, 1, 1], [3, 1, 2]];
const baseline = 210;
const maximumBarHeight = 140;
let instanceCount = 0;

const componentStyles = `
  <style>
    llm-causal-mask-explorer { display: block; min-inline-size: 0; }
    .causal-mask-explorer-widget { min-inline-size: 0; }
    .causal-mask-explorer-widget summary { color: #172b3a; }
    .causal-mask-controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .causal-mask-controls .interactive-control { min-inline-size: 0; }
    .causal-mask-controls select {
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
    .causal-mask-controls select:focus-visible,
    .causal-mask-toggle input:focus-visible {
      outline: 3px solid #8c421f;
      outline-offset: 3px;
    }
    .causal-mask-toggle { display: flex; gap: 0.5rem; align-items: center; }
    .causal-mask-toggle input {
      inline-size: 1.25rem;
      block-size: 1.25rem;
      flex: 0 0 auto;
      margin: 0;
      accent-color: #1f648c;
    }
    .causal-mask-matrices { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .causal-mask-matrices > * { min-inline-size: 0; }
    .causal-mask-explorer-widget tr.is-current-query > * { background-color: #dcecf4; }
    .causal-mask-weight-bar { fill: #2f6f8f; }
    @media (max-width: 60rem) {
      .causal-mask-matrices { grid-template-columns: minmax(0, 1fr); }
    }
    @media (max-width: 36rem) {
      .causal-mask-controls { grid-template-columns: minmax(0, 1fr); }
    }
  </style>
`;

function formatNumber(value) {
  if (value === -Infinity) {
    return '−∞';
  }
  const cleaned = Math.abs(value) < 1e-12 ? 0 : value;
  return String(Number(cleaned.toFixed(10)));
}

function formatList(vector) {
  return `[${vector.map(formatNumber).join(', ')}]`;
}

function matrixRows(includeRowSum = false) {
  return scores.map((_, rowIndex) => {
    const cells = scores.map((__, columnIndex) =>
      `<td data-row="${rowIndex}" data-column="${columnIndex}"></td>`,
    ).join('');
    const rowSum = includeRowSum
      ? `<td data-testid="causal-row-sum-${rowIndex}"></td>`
      : '';
    return `<tr data-row="${rowIndex}"><th scope="row">q${rowIndex}</th>${cells}${rowSum}</tr>`;
  }).join('');
}

function matrixTable(testId, caption, regionName, includeRowSum = false) {
  const rowSumHeader = includeRowSum ? '<th scope="col">行和</th>' : '';
  return `
    <div class="attention-table-wrap" tabindex="0" role="region" aria-label="${regionName}">
      <table class="attention-table" data-testid="${testId}">
        <caption>${caption}</caption>
        <thead>
          <tr><th scope="col">q（行）/ k（列）</th><th scope="col">k₀</th><th scope="col">k₁</th><th scope="col">k₂</th>${rowSumHeader}</tr>
        </thead>
        <tbody>${matrixRows(includeRowSum)}</tbody>
      </table>
    </div>
  `;
}

function weightBars() {
  return scores.map((_, index) => {
    const x = 100 + index * 180;
    return `
      <rect class="causal-mask-weight-bar" data-testid="causal-weight-bar" data-index="${index}"
        x="${x}" y="${baseline}" width="90" height="0" />
      <text class="chart-label causal-mask-weight-value" data-index="${index}" x="${x + 45}" y="${baseline - 8}" text-anchor="middle"></text>
      <text class="chart-label" x="${x + 45}" y="238" text-anchor="middle">k${index}</text>
    `;
  }).join('');
}

class CausalMaskExplorerElement extends HTMLElement {
  #toggle;
  #query;
  #rawCells;
  #maskedCells;
  #weightCells;
  #rowSums;
  #rows;
  #description;
  #weightBars;
  #weightLabels;

  connectedCallback() {
    if (this.#toggle) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'causal-mask-explorer'}-${instanceCount}`;

    this.innerHTML = `
      ${componentStyles}
      <figure class="interactive-widget causal-mask-explorer-widget">
        <div class="interactive-controls causal-mask-controls">
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-query">查询行 i</label>
            <select id="${prefix}-query" data-testid="causal-query-row">
              <option value="0">q0（第 1 行）</option>
              <option value="1">q1（第 2 行）</option>
              <option value="2">q2（第 3 行）</option>
            </select>
          </div>
          <div class="interactive-control causal-mask-toggle">
            <input id="${prefix}-enabled" data-testid="causal-mask-enabled" type="checkbox" checked>
            <label for="${prefix}-enabled">启用因果掩码</label>
          </div>
        </div>
        <div class="attention-matrices causal-mask-matrices">
          ${matrixTable('causal-raw-scores', '原始分数 S', '因果原始分数')}
          ${matrixTable('causal-masked-scores', '掩码后分数 S + M（Softmax 前）', '因果掩码后分数')}
          ${matrixTable('causal-softmax-weights', 'Softmax 权重 A（逐行归一化）', '因果 Softmax 权重', true)}
        </div>
        <svg class="interactive-chart" data-testid="causal-mask-chart" viewBox="0 0 640 260"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">当前查询行的 Softmax 权重</title>
          <desc id="${prefix}-chart-desc"></desc>
          <path class="chart-axes" d="M 60 210 H 580" />
          ${weightBars()}
        </svg>
        <details>
          <summary>对照原静态图</summary>
          <img class="interactive-chart" src="assets/figures/causal-mask.svg" alt="三乘三因果掩码数值图。原始分数矩阵为 [[2,1,3],[1,1,1],[3,1,2]]；严格上三角在 Softmax 前变为 −∞，得到 [[2,−∞,−∞],[1,1,−∞],[3,1,2]]。逐行 Softmax 后权重为 [[1,0,0],[0.5,0.5,0],[0.6652,0.0900,0.2447]]，每行和为一，主对角线保留。">
        </details>
        <figcaption>三个矩阵依次展示原始分数、Softmax 前的分数和逐行权重；高亮行为当前查询，柱图同步显示该行权重。</figcaption>
      </figure>
    `;

    this.#toggle = this.querySelector('[data-testid="causal-mask-enabled"]');
    this.#query = this.querySelector('[data-testid="causal-query-row"]');
    this.#rawCells = [...this.querySelectorAll('[data-testid="causal-raw-scores"] td[data-row][data-column]')];
    this.#maskedCells = [...this.querySelectorAll('[data-testid="causal-masked-scores"] td[data-row][data-column]')];
    this.#weightCells = [...this.querySelectorAll('[data-testid="causal-softmax-weights"] td[data-row][data-column]')];
    this.#rowSums = [...this.querySelectorAll('[data-testid^="causal-row-sum-"]')];
    this.#rows = [...this.querySelectorAll('tbody tr[data-row]')];
    this.#description = this.querySelector(`#${prefix}-chart-desc`);
    this.#weightBars = [...this.querySelectorAll('[data-testid="causal-weight-bar"]')];
    this.#weightLabels = [...this.querySelectorAll('.causal-mask-weight-value')];

    this.#query.value = this.getAttribute('query-index') ?? '0';
    this.#toggle.addEventListener('change', () => this.#update());
    this.#query.addEventListener('input', () => this.#update());
    this.#update();
  }

  #update() {
    const queryIndex = Number(this.#query.value);
    const causalEnabled = this.#toggle.checked;
    const maskedScores = causalEnabled
      ? causalMask(scores)
      : scores.map((row) => row.slice());
    const weights = maskedScores.map((row) => maskedSoftmax(row));

    for (const cell of this.#rawCells) {
      cell.textContent = formatNumber(scores[Number(cell.dataset.row)][Number(cell.dataset.column)]);
    }
    for (const cell of this.#maskedCells) {
      cell.textContent = formatNumber(maskedScores[Number(cell.dataset.row)][Number(cell.dataset.column)]);
    }
    for (const cell of this.#weightCells) {
      cell.textContent = formatNumber(weights[Number(cell.dataset.row)][Number(cell.dataset.column)]);
    }
    this.#rowSums.forEach((cell, rowIndex) => {
      const sum = weights[rowIndex].reduce((total, value) => total + value, 0);
      cell.textContent = formatNumber(sum);
    });
    for (const row of this.#rows) {
      row.classList.toggle('is-current-query', Number(row.dataset.row) === queryIndex);
    }

    const selectedWeights = weights[queryIndex];
    selectedWeights.forEach((weight, index) => {
      const height = Number((maximumBarHeight * weight).toFixed(6));
      const y = Number((baseline - height).toFixed(6));
      const bar = this.#weightBars[index];
      bar.setAttribute('y', y);
      bar.setAttribute('height', height);
      bar.dataset.weight = formatNumber(weight);
      this.#weightLabels[index].setAttribute('y', y - 8);
      this.#weightLabels[index].textContent = formatNumber(weight);
    });

    const selectedSum = selectedWeights.reduce((total, value) => total + value, 0);
    const mode = causalEnabled ? '因果掩码已开启' : '因果掩码已关闭';
    const scoreLabel = causalEnabled ? '掩码后分数' : '原始分数';
    const weightLabel = causalEnabled ? 'Softmax 权重' : '原始行 Softmax 权重';
    const futureDescription = causalEnabled ? '；未来位置权重为 0' : '';
    this.#description.textContent = `${mode}；当前查询行 q${queryIndex}；${scoreLabel}为 ${formatList(maskedScores[queryIndex])}；${weightLabel}为 ${formatList(selectedWeights)}${futureDescription}；行和为 ${formatNumber(selectedSum)}。`;
  }
}

export function defineCausalMaskExplorerElement() {
  if (!customElements.get('llm-causal-mask-explorer')) {
    customElements.define('llm-causal-mask-explorer', CausalMaskExplorerElement);
  }
}
