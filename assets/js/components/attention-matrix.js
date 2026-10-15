import { scaledDotProductAttention } from '../lib/attention.js';

const queries = [[1, 0], [0, 1], [1, 1]];
const keys = [[1, 0], [0, 1], [1, 1]];
const values = [[1, 0], [0, 1], [1, 1]];
const baseline = 210;
const maximumBarHeight = 140;
let instanceCount = 0;

function formatNumber(value) {
  const cleaned = Math.abs(value) < 1e-12 ? 0 : value;
  return String(Number(cleaned.toFixed(10)));
}

function formatVector(vector) {
  return `(${vector.map(formatNumber).join(', ')})`;
}

function formatList(vector) {
  return `[${vector.map(formatNumber).join(', ')}]`;
}

function matrixRows(includeRowSum = false) {
  return queries.map((_, rowIndex) => {
    const cells = keys.map((__, columnIndex) =>
      `<td data-row="${rowIndex}" data-column="${columnIndex}"></td>`,
    ).join('');
    const rowSum = includeRowSum
      ? `<td data-testid="attention-row-sum-${rowIndex}"></td>`
      : '';
    return `<tr><th scope="row">q${rowIndex}</th>${cells}${rowSum}</tr>`;
  }).join('');
}

function matrixTable(testId, caption, includeRowSum = false, regionName = caption) {
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
  return keys.map((_, index) => {
    const x = 100 + index * 180;
    return `
      <rect class="attention-weight-bar" data-testid="attention-weight-bar" data-index="${index}"
        x="${x}" y="${baseline}" width="90" height="0" />
      <text class="chart-label attention-weight-value" data-index="${index}" x="${x + 45}" y="${baseline - 8}" text-anchor="middle"></text>
      <text class="chart-label" x="${x + 45}" y="238" text-anchor="middle">k${index}</text>
    `;
  }).join('');
}

function outputBars() {
  return [0, 1].map((index) => {
    const x = 170 + index * 200;
    return `
      <rect class="attention-output-bar" data-testid="attention-output-bar" data-dimension="${index}"
        x="${x}" y="${baseline}" width="100" height="0" />
      <text class="chart-label attention-output-value" data-dimension="${index}" x="${x + 50}" y="${baseline - 8}" text-anchor="middle"></text>
      <text class="chart-label" x="${x + 50}" y="238" text-anchor="middle">o${index}</text>
    `;
  }).join('');
}

class AttentionMatrixElement extends HTMLElement {
  #query;
  #scale;
  #queryValue;
  #scaleValue;
  #scaleError;
  #scoreCells;
  #weightCells;
  #rowSums;
  #output;
  #weightedResult;
  #distribution;
  #distributionDescription;
  #outputChart;
  #outputDescription;
  #weightBars;
  #weightLabels;
  #outputBars;
  #outputLabels;

  connectedCallback() {
    if (this.#query) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'attention-matrix'}-${instanceCount}`;

    this.innerHTML = `
      <figure class="interactive-widget attention-matrix-widget">
        <div class="interactive-controls attention-controls">
          <div class="interactive-control">
            <label for="${prefix}-query">查询索引 i</label>
            <input id="${prefix}-query" data-testid="attention-query-index" type="range" min="0" max="2" step="1">
            <output id="${prefix}-query-value" data-testid="attention-query-value" for="${prefix}-query"></output>
          </div>
          <div class="interactive-control attention-scale-control">
            <label for="${prefix}-scale">缩放系数 scale</label>
            <input id="${prefix}-scale" data-testid="attention-scale" type="number" min="0.1" max="2" step="any" required
              aria-describedby="${prefix}-scale-error">
            <output id="${prefix}-scale-value" data-testid="attention-scale-value" for="${prefix}-scale"></output>
            <p id="${prefix}-scale-error" class="interactive-control-error" data-testid="attention-scale-error" role="alert" hidden></p>
          </div>
        </div>
        <div class="attention-matrices">
          ${matrixTable('attention-scores', '真实缩放分数 S = scale × QKᵀ', false, '真实缩放分数矩阵')}
          ${matrixTable('attention-weights', 'Softmax 权重 A（逐行归一化）', true, 'Softmax 权重矩阵')}
        </div>
        <dl class="interactive-results">
          <div>
            <dt>当前查询的输出向量</dt>
            <dd><output data-testid="attention-output" for="${prefix}-query ${prefix}-scale"></output></dd>
          </div>
          <div>
            <dt>Value 加权结果</dt>
            <dd><output data-testid="attention-weighted-result" for="${prefix}-query ${prefix}-scale"></output></dd>
          </div>
        </dl>
        <div class="attention-charts">
          <svg class="interactive-chart" data-testid="attention-distribution" viewBox="0 0 640 260"
            role="img" aria-labelledby="${prefix}-distribution-title" aria-describedby="${prefix}-distribution-desc">
            <title id="${prefix}-distribution-title">当前查询的 Softmax 权重分布</title>
            <desc id="${prefix}-distribution-desc"></desc>
            <path class="chart-axes" d="M 60 210 H 580" />
            ${weightBars()}
          </svg>
          <svg class="interactive-chart" data-testid="attention-output-chart" viewBox="0 0 640 260"
            role="img" aria-labelledby="${prefix}-output-title" aria-describedby="${prefix}-output-desc">
            <title id="${prefix}-output-title">当前查询的 Value 加权输出</title>
            <desc id="${prefix}-output-desc"></desc>
            <path class="chart-axes" d="M 60 210 H 580" />
            ${outputBars()}
          </svg>
        </div>
        <figcaption>表格给出全部查询的真实分数和权重；图形随当前查询与缩放系数更新，输出只对 Value 加权。</figcaption>
      </figure>
    `;

    this.#query = this.querySelector('[data-testid="attention-query-index"]');
    this.#scale = this.querySelector('[data-testid="attention-scale"]');
    this.#queryValue = this.querySelector(`#${prefix}-query-value`);
    this.#scaleValue = this.querySelector(`#${prefix}-scale-value`);
    this.#scaleError = this.querySelector(`#${prefix}-scale-error`);
    this.#scoreCells = [...this.querySelectorAll('[data-testid="attention-scores"] td[data-row][data-column]')];
    this.#weightCells = [...this.querySelectorAll('[data-testid="attention-weights"] td[data-row][data-column]')];
    this.#rowSums = [...this.querySelectorAll('[data-testid^="attention-row-sum-"]')];
    this.#output = this.querySelector('[data-testid="attention-output"]');
    this.#weightedResult = this.querySelector('[data-testid="attention-weighted-result"]');
    this.#distribution = this.querySelector('[data-testid="attention-distribution"]');
    this.#distributionDescription = this.querySelector(`#${prefix}-distribution-desc`);
    this.#outputChart = this.querySelector('[data-testid="attention-output-chart"]');
    this.#outputDescription = this.querySelector(`#${prefix}-output-desc`);
    this.#weightBars = [...this.querySelectorAll('[data-testid="attention-weight-bar"]')];
    this.#weightLabels = [...this.querySelectorAll('.attention-weight-value')];
    this.#outputBars = [...this.querySelectorAll('[data-testid="attention-output-bar"]')];
    this.#outputLabels = [...this.querySelectorAll('.attention-output-value')];

    this.#query.value = this.getAttribute('query-index') ?? '0';
    this.#scale.value = this.getAttribute('scale') ?? String(1 / Math.sqrt(2));
    this.#query.addEventListener('input', () => this.#update());
    this.#scale.addEventListener('input', () => this.#update());
    this.#update();
  }

  #clearResults() {
    this.#scaleValue.textContent = '';
    for (const cell of [...this.#scoreCells, ...this.#weightCells, ...this.#rowSums]) {
      cell.textContent = '';
    }
    this.#output.textContent = '';
    this.#weightedResult.textContent = '';
    delete this.#distribution.dataset.signature;
    delete this.#outputChart.dataset.signature;
    this.#weightBars.forEach((bar, index) => {
      bar.setAttribute('y', baseline);
      bar.setAttribute('height', 0);
      delete bar.dataset.weight;
      this.#weightLabels[index].setAttribute('y', baseline - 8);
      this.#weightLabels[index].textContent = '';
    });
    this.#outputBars.forEach((bar, index) => {
      bar.setAttribute('y', baseline);
      bar.setAttribute('height', 0);
      delete bar.dataset.output;
      this.#outputLabels[index].setAttribute('y', baseline - 8);
      this.#outputLabels[index].textContent = '';
    });
    this.#distributionDescription.textContent = '';
    this.#outputDescription.textContent = '';
  }

  #update() {
    const queryIndex = Number(this.#query.value);
    const scale = Number(this.#scale.value);
    const scaleIsValid = this.#scale.value !== ''
      && this.#scale.validity.valid
      && Number.isFinite(scale);
    this.#scale.setAttribute('aria-invalid', String(!scaleIsValid));
    this.#scaleError.textContent = scaleIsValid ? '' : '请输入 0.1 到 2 之间的数值。';
    this.#scaleError.hidden = scaleIsValid;
    if (!scaleIsValid) {
      this.#clearResults();
      return;
    }
    const result = scaledDotProductAttention(queries, keys, values, scale);
    const selectedWeights = result.weights[queryIndex];
    const selectedOutput = result.output[queryIndex];

    this.#queryValue.textContent = String(queryIndex);
    this.#scaleValue.textContent = String(scale);
    for (const cell of this.#scoreCells) {
      const row = Number(cell.dataset.row);
      const column = Number(cell.dataset.column);
      cell.textContent = formatNumber(result.scores[row][column]);
    }
    for (const cell of this.#weightCells) {
      const row = Number(cell.dataset.row);
      const column = Number(cell.dataset.column);
      cell.textContent = formatNumber(result.weights[row][column]);
    }
    this.#rowSums.forEach((cell, rowIndex) => {
      const sum = result.weights[rowIndex].reduce((total, value) => total + value, 0);
      cell.textContent = formatNumber(sum);
    });

    const outputText = formatVector(selectedOutput);
    const weightedTerms = selectedWeights.map((weight, index) =>
      `${formatNumber(weight)}×${formatVector(values[index])}`,
    ).join(' + ');
    this.#output.textContent = outputText;
    this.#weightedResult.textContent = `${weightedTerms} = ${outputText}`;

    this.#distribution.dataset.signature = selectedWeights.join('|');
    this.#outputChart.dataset.signature = selectedOutput.join('|');
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
    selectedOutput.forEach((value, index) => {
      const height = Number((maximumBarHeight * value).toFixed(6));
      const y = Number((baseline - height).toFixed(6));
      const bar = this.#outputBars[index];
      bar.setAttribute('y', y);
      bar.setAttribute('height', height);
      bar.dataset.output = formatNumber(value);
      this.#outputLabels[index].setAttribute('y', y - 8);
      this.#outputLabels[index].textContent = formatNumber(value);
    });

    const selectedSum = selectedWeights.reduce((total, value) => total + value, 0);
    this.#distributionDescription.textContent = `查询索引 ${queryIndex}，缩放系数 ${scale}；当前缩放分数为 ${formatList(result.scores[queryIndex])}，Softmax 权重为 ${formatList(selectedWeights)}，行和为 ${formatNumber(selectedSum)}。`;
    this.#outputDescription.textContent = `查询索引 ${queryIndex} 的 Value 加权输出为 ${outputText}；三个 Value 按当前 Softmax 权重加权，而不是对 Key 或分数加权。`;
  }
}

export function defineAttentionMatrixElement() {
  if (!customElements.get('llm-attention-matrix')) {
    customElements.define('llm-attention-matrix', AttentionMatrixElement);
  }
}
