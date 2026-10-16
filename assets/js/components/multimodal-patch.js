function assertPositiveSafeInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} 必须是正整数。`);
  }
}

function assertExactAccount(value, name) {
  if (!Number.isSafeInteger(value)) {
    throw new RangeError(`${name} 超出安全整数范围。`);
  }
}

export function computePatchLayout({ width, height, patchSize, textTokens }) {
  assertPositiveSafeInteger(width, 'width');
  assertPositiveSafeInteger(height, 'height');
  assertPositiveSafeInteger(patchSize, 'patchSize');
  if (!Number.isSafeInteger(textTokens) || textTokens < 0) {
    throw new RangeError('textTokens 必须是非负整数。');
  }

  const patchRows = Math.ceil(height / patchSize);
  const patchColumns = Math.ceil(width / patchSize);
  const visualTokens = patchRows * patchColumns;
  const sequenceLength = visualTokens + textTokens;
  const attentionScores = sequenceLength * sequenceLength;
  assertExactAccount(visualTokens, 'visualTokens');
  assertExactAccount(sequenceLength, 'sequenceLength');
  assertExactAccount(attentionScores, 'attentionScores');
  return {
    patchRows,
    patchColumns,
    visualTokens,
    sequenceLength,
    attentionScores,
  };
}

const IMAGE_WIDTH = 224;
const IMAGE_HEIGHT = 224;
const TEXT_TOKENS = 20;
const PATCH_SIZES = [14, 16, 32];
const GRID = { x: 30, y: 60, size: 360 };
const SEQUENCE = { x: 470, y: 80, width: 420, height: 48 };
const svgNamespace = 'http://www.w3.org/2000/svg';
const HTMLElementBase = globalThis.HTMLElement ?? class {};
let instanceCount = 0;

const componentStyles = `
  <style>
    llm-multimodal-patch { display: block; min-inline-size: 0; }
    .multimodal-patch-widget { min-inline-size: 0; }
    .multimodal-controls {
      grid-template-columns: minmax(0, 1fr);
      max-inline-size: 30rem;
    }
    .multimodal-controls .interactive-control { grid-template-columns: minmax(0, 1fr); }
    .multimodal-controls select {
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
    .multimodal-controls select:focus-visible,
    .multimodal-text-equivalent summary:focus-visible {
      outline: 3px solid #8c421f;
      outline-offset: 3px;
    }
    .multimodal-account-equation { font-weight: 600; overflow-wrap: anywhere; }
    .multimodal-text-equivalent { margin-top: 0.75rem; min-inline-size: 0; }
    .multimodal-text-equivalent summary { min-block-size: 2rem; cursor: pointer; }
    .multimodal-text-equivalent table {
      inline-size: 100%;
      table-layout: fixed;
      border-collapse: collapse;
      font-size: 0.8rem;
      font-variant-numeric: tabular-nums;
    }
    .multimodal-text-equivalent caption,
    .multimodal-text-equivalent th,
    .multimodal-text-equivalent td { overflow-wrap: anywhere; }
    .multimodal-text-equivalent caption { padding: 0.4rem 0; text-align: left; }
    .multimodal-text-equivalent th,
    .multimodal-text-equivalent td {
      padding: 0.35rem 0.25rem;
      border-bottom: 1px solid #dde3e7;
      text-align: left;
      vertical-align: top;
    }
    .multimodal-text-equivalent tr[data-current="true"] { background: #eef4f7; }
  </style>
`;

function createSvgElement(name, attributes = {}) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [attribute, value] of Object.entries(attributes)) {
    element.setAttribute(attribute, String(value));
  }
  return element;
}

function formatInteger(value) {
  return value.toLocaleString('en-US');
}

function accountRows() {
  return PATCH_SIZES.map((patchSize) => ({
    patchSize,
    ...computePatchLayout({
      width: IMAGE_WIDTH,
      height: IMAGE_HEIGHT,
      patchSize,
      textTokens: TEXT_TOKENS,
    }),
  }));
}

class MultimodalPatchElement extends HTMLElementBase {
  #els;

  connectedCallback() {
    if (this.#els) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'multimodal-patch'}-${instanceCount}`;
    const rows = accountRows();

    this.innerHTML = `
      ${componentStyles}
      <figure class="interactive-widget multimodal-patch-widget">
        <p><strong>教学序列账：</strong>固定 224 × 224 图像与 20 个文本 Token，不运行真实多模态模型。</p>
        <div class="interactive-controls multimodal-controls">
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-patch-size">Patch 边长 P（像素）</label>
            <select id="${prefix}-patch-size" data-testid="multimodal-patch-size">
              <option value="14">14</option>
              <option value="16">16</option>
              <option value="32">32</option>
            </select>
          </div>
        </div>
        <dl class="interactive-results">
          <div><dt>Patch 网格</dt><dd><output data-testid="multimodal-patch-grid"></output></dd></div>
          <div><dt>视觉 Token</dt><dd><output data-testid="multimodal-visual-tokens"></output></dd></div>
          <div><dt>文本 Token（固定）</dt><dd><output data-testid="multimodal-text-tokens"></output></dd></div>
          <div><dt>拼接序列长度 L</dt><dd><output data-testid="multimodal-sequence-length"></output></dd></div>
          <div><dt>玩具全注意力分数 L²</dt><dd><output data-testid="multimodal-attention-scores"></output></dd></div>
        </dl>
        <p class="multimodal-account-equation" data-testid="multimodal-account-equation"></p>
        <svg class="interactive-chart" data-testid="multimodal-patch-chart" viewBox="0 0 960 500"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">Patch 网格、拼接序列与玩具二次注意力账</title>
          <desc id="${prefix}-chart-desc"></desc>
          <defs>
            <marker id="${prefix}-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3"
              orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L8,3 z" fill="#4C5F6A" />
            </marker>
          </defs>
          <text x="30" y="32" font-size="16" font-weight="700">224 × 224 图像：每格一个 patch</text>
          <text x="470" y="32" font-size="16" font-weight="700">视觉 + 文本序列与全注意力分数</text>
          <g data-testid="multimodal-patch-cells"></g>
          <rect x="30" y="60" width="360" height="360" fill="none" stroke="#172B3A" stroke-width="2" />
          <text data-testid="multimodal-grid-caption" x="30" y="450" font-size="14"></text>
          <rect data-testid="multimodal-sequence-visual" x="470" y="80" height="48" fill="#1F648C" />
          <rect data-testid="multimodal-sequence-text" y="80" height="48" fill="#C65D21" />
          <text x="470" y="153" font-size="13" fill="#1F648C">蓝色：视觉 Token</text>
          <text x="710" y="153" font-size="13" fill="#8C421F">橙色：20 个文本 Token</text>
          <path d="M680 166 V190" stroke="#4C5F6A" stroke-width="2" marker-end="url(#${prefix}-arrow)" />
          <rect x="550" y="200" width="260" height="220" rx="8" fill="#EEF4F7" stroke="#1F648C" stroke-width="2" />
          <path d="M550 310 H810 M680 200 V420" stroke="#8EB3C4" stroke-width="1" />
          <text data-testid="multimodal-attention-dimensions" x="680" y="270" text-anchor="middle" font-size="19" font-weight="700"></text>
          <text data-testid="multimodal-attention-total" x="680" y="355" text-anchor="middle" font-size="17" font-weight="700"></text>
          <text x="680" y="455" text-anchor="middle" font-size="12" fill="#4C5F6A">L² 只数分数，不等于完整 FLOPs、显存或时延</text>
          <text x="680" y="478" text-anchor="middle" font-size="12" fill="#4C5F6A">不另加 CLS、BOS/EOS 等特殊 Token</text>
        </svg>
        <details class="multimodal-text-equivalent" data-testid="multimodal-text-equivalent">
          <summary>三组 patch 序列账的文字数据</summary>
          <table>
            <caption>224 × 224 图像与 20 个文本 Token 的静态对照</caption>
            <thead><tr><th scope="col">P</th><th scope="col">网格</th><th scope="col">视觉 + 文本 = L</th><th scope="col">L² 分数</th></tr></thead>
            <tbody>
              ${rows.map((row) => `
                <tr data-testid="multimodal-account-row" data-patch-size="${row.patchSize}" data-current="false">
                  <th scope="row">${row.patchSize}</th>
                  <td>${row.patchRows} × ${row.patchColumns}</td>
                  <td>${row.visualTokens} + ${TEXT_TOKENS} = ${row.sequenceLength}</td>
                  <td>${row.sequenceLength}² = ${formatInteger(row.attentionScores)}</td>
                </tr>`).join('')}
            </tbody>
          </table>
        </details>
        <figcaption>
          这里假设视觉和文本 Token 进入同一个教学全自注意力序列。只核算 patch、序列长度与 L² 分数数量；不对应某一真实模型结构，也不估算完整 FLOPs、显存或时延。
        </figcaption>
      </figure>
    `;

    const byTestId = (testId) => this.querySelector(`[data-testid="${testId}"]`);
    this.#els = {
      patchSize: byTestId('multimodal-patch-size'),
      patchGrid: byTestId('multimodal-patch-grid'),
      visualTokens: byTestId('multimodal-visual-tokens'),
      textTokens: byTestId('multimodal-text-tokens'),
      sequenceLength: byTestId('multimodal-sequence-length'),
      attentionScores: byTestId('multimodal-attention-scores'),
      equation: byTestId('multimodal-account-equation'),
      chart: byTestId('multimodal-patch-chart'),
      description: this.querySelector(`#${prefix}-chart-desc`),
      cells: byTestId('multimodal-patch-cells'),
      gridCaption: byTestId('multimodal-grid-caption'),
      sequenceVisual: byTestId('multimodal-sequence-visual'),
      sequenceText: byTestId('multimodal-sequence-text'),
      attentionDimensions: byTestId('multimodal-attention-dimensions'),
      attentionTotal: byTestId('multimodal-attention-total'),
      accountRows: [...this.querySelectorAll('[data-testid="multimodal-account-row"]')],
    };
    this.#els.patchSize.value = this.getAttribute('patch-size') ?? '14';
    for (const output of this.querySelectorAll('.interactive-results output')) {
      output.setAttribute('for', this.#els.patchSize.id);
    }
    this.#els.patchSize.addEventListener('change', () => this.#update());
    this.#els.patchSize.addEventListener('keydown', (event) => {
      if (!['ArrowDown', 'ArrowUp'].includes(event.key)) {
        return;
      }
      event.preventDefault();
      const direction = event.key === 'ArrowDown' ? 1 : -1;
      const nextIndex = Math.min(
        Math.max(this.#els.patchSize.selectedIndex + direction, 0),
        this.#els.patchSize.options.length - 1,
      );
      if (nextIndex !== this.#els.patchSize.selectedIndex) {
        this.#els.patchSize.selectedIndex = nextIndex;
        this.#update();
      }
    });
    this.#update();
  }

  #update() {
    const patchSize = Number(this.#els.patchSize.value);
    const layout = computePatchLayout({
      width: IMAGE_WIDTH,
      height: IMAGE_HEIGHT,
      patchSize,
      textTokens: TEXT_TOKENS,
    });
    const outputs = [
      [this.#els.visualTokens, layout.visualTokens, String(layout.visualTokens)],
      [this.#els.textTokens, TEXT_TOKENS, String(TEXT_TOKENS)],
      [this.#els.sequenceLength, layout.sequenceLength, String(layout.sequenceLength)],
      [this.#els.attentionScores, layout.attentionScores, formatInteger(layout.attentionScores)],
    ];
    this.#els.patchGrid.textContent = `${layout.patchRows} × ${layout.patchColumns}`;
    for (const [output, value, text] of outputs) {
      output.dataset.value = String(value);
      output.textContent = text;
    }
    this.#els.equation.textContent = `${layout.visualTokens} + ${TEXT_TOKENS} = ${layout.sequenceLength}；${layout.sequenceLength}² = ${formatInteger(layout.attentionScores)}`;
    this.#els.chart.dataset.signature = [
      patchSize,
      layout.patchRows,
      layout.patchColumns,
      layout.visualTokens,
      TEXT_TOKENS,
      layout.sequenceLength,
      layout.attentionScores,
    ].join('|');
    this.#renderGrid(layout);
    this.#renderSequence(layout);
    for (const row of this.#els.accountRows) {
      row.dataset.current = String(Number(row.dataset.patchSize) === patchSize);
    }
    this.#els.description.textContent = `固定 224 × 224 图像和 ${TEXT_TOKENS} 个文本 Token。当前 patch 边长 ${patchSize}，网格 ${layout.patchRows} 行 × ${layout.patchColumns} 列，共 ${layout.visualTokens} 个视觉 Token；拼接序列长度 ${layout.sequenceLength}，教学全注意力分数 ${layout.sequenceLength}² = ${formatInteger(layout.attentionScores)} 个。只核算序列账，不另加特殊 Token，不对应某一真实模型，也不表示完整 FLOPs、显存或时延。`;
  }

  #renderGrid(layout) {
    const cellWidth = GRID.size / layout.patchColumns;
    const cellHeight = GRID.size / layout.patchRows;
    const cells = [];
    for (let row = 0; row < layout.patchRows; row += 1) {
      for (let column = 0; column < layout.patchColumns; column += 1) {
        cells.push(createSvgElement('rect', {
          x: GRID.x + column * cellWidth,
          y: GRID.y + row * cellHeight,
          width: cellWidth,
          height: cellHeight,
          fill: (row + column) % 2 === 0 ? '#DCEBF2' : '#BBD5E2',
          stroke: '#FFFFFF',
          'stroke-width': 0.8,
          'data-testid': 'multimodal-patch-cell',
          'data-row': row + 1,
          'data-column': column + 1,
        }));
      }
    }
    this.#els.cells.replaceChildren(...cells);
    this.#els.gridCaption.textContent = `${layout.patchRows} 行 × ${layout.patchColumns} 列 = ${layout.visualTokens} 个视觉 Token`;
  }

  #renderSequence(layout) {
    const visualWidth = SEQUENCE.width * layout.visualTokens / layout.sequenceLength;
    this.#els.sequenceVisual.setAttribute('width', String(visualWidth));
    this.#els.sequenceText.setAttribute('x', String(SEQUENCE.x + visualWidth));
    this.#els.sequenceText.setAttribute('width', String(SEQUENCE.width - visualWidth));
    this.#els.attentionDimensions.textContent = `${layout.sequenceLength} × ${layout.sequenceLength}`;
    this.#els.attentionTotal.textContent = `= ${formatInteger(layout.attentionScores)} 个分数`;
  }
}

export function defineMultimodalPatchElement() {
  if (!customElements.get('llm-multimodal-patch')) {
    customElements.define('llm-multimodal-patch', MultimodalPatchElement);
  }
}
