const input = [1, -2, 4];
const firstWeights = [
  [1, 0, 0],
  [-1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];
const secondWeights = [
  [1, -1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];
let instanceCount = 0;

function formatNumber(value) {
  const cleaned = Math.abs(value) < 1e-12 ? 0 : value;
  return String(Number(cleaned.toFixed(10)));
}

function formatVector(vector) {
  return `(${vector.map(formatNumber).join(', ')})`;
}

function matrixVectorProduct(matrix, vector) {
  return matrix.map((row) => row.reduce((sum, weight, index) => sum + weight * vector[index], 0));
}

function layerNorm(vector) {
  const mean = vector.reduce((sum, value) => sum + value, 0) / vector.length;
  const variance = vector.reduce((sum, value) => sum + (value - mean) ** 2, 0) / vector.length;
  return vector.map((value) => (value - mean) / Math.sqrt(variance));
}

function feedForward(vector) {
  const hidden = matrixVectorProduct(firstWeights, vector).map((value) => Math.max(0, value));
  return matrixVectorProduct(secondWeights, hidden);
}

function addVectors(left, right) {
  return left.map((value, index) => value + right[index]);
}

class TransformerBlockElement extends HTMLElement {
  #normPosition;
  #residual;
  #normalizedInput;
  #sublayerOutput;
  #mergedOutput;
  #blockOutput;
  #rule;
  #chart;
  #description;
  #inputNormLabel;
  #outputNormLabel;
  #mergeLabel;
  #residualPath;

  connectedCallback() {
    if (this.#normPosition) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'transformer-block'}-${instanceCount}`;

    this.innerHTML = `
      <style>
        llm-transformer-block { display: block; }
        llm-transformer-block .transformer-residual-control input[type='checkbox'] {
          inline-size: 2rem;
          min-inline-size: 2rem;
          min-block-size: 2rem;
          margin: 0;
          accent-color: #1f648c;
          justify-self: start;
        }
        llm-transformer-block .transformer-residual-control input[type='checkbox']:focus-visible {
          outline: 3px solid #8c421f;
          outline-offset: 3px;
        }
      </style>
      <figure class="interactive-widget transformer-block-widget">
        <div class="interactive-controls">
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-norm-position">归一化位置</label>
            <select id="${prefix}-norm-position" data-testid="transformer-norm-position">
              <option value="pre">Pre-LN：先归一化</option>
              <option value="post">Post-LN：后归一化</option>
            </select>
          </div>
          <div class="interactive-control interactive-select-control transformer-residual-control">
            <label for="${prefix}-residual">启用残差连接</label>
            <input id="${prefix}-residual" data-testid="transformer-residual" type="checkbox">
          </div>
        </div>
        <dl class="interactive-results">
          <div>
            <dt>子层输入 a</dt>
            <dd><output data-testid="transformer-normalized-input" for="${prefix}-norm-position ${prefix}-residual"></output></dd>
          </div>
          <div>
            <dt>FFN 输出 f = F(a)</dt>
            <dd><output data-testid="transformer-sublayer-output" for="${prefix}-norm-position ${prefix}-residual"></output></dd>
          </div>
          <div>
            <dt>汇合表示 z</dt>
            <dd><output data-testid="transformer-merged-output" for="${prefix}-norm-position ${prefix}-residual"></output></dd>
          </div>
          <div>
            <dt>Block 输出 y</dt>
            <dd><output data-testid="transformer-block-output" for="${prefix}-norm-position ${prefix}-residual"></output></dd>
          </div>
          <div>
            <dt>当前计算规则</dt>
            <dd><output data-testid="transformer-block-rule" for="${prefix}-norm-position ${prefix}-residual"></output></dd>
          </div>
        </dl>
        <svg class="interactive-chart" data-testid="transformer-block-chart" viewBox="0 0 1200 360"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">Transformer Block 的归一化与残差数据流</title>
          <desc id="${prefix}-chart-desc"></desc>
          <defs>
            <marker id="${prefix}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#4C5F6A"></path>
            </marker>
          </defs>
          <g stroke="#5E6A70" stroke-width="2">
            <rect x="20" y="80" width="150" height="110" rx="12" fill="#F3F4F5"/>
            <rect x="215" y="80" width="170" height="110" rx="12" fill="#DCECF4"/>
            <rect x="430" y="80" width="170" height="110" rx="12" fill="#FBF8F3"/>
            <rect x="650" y="80" width="140" height="110" rx="12" fill="#E4F0E7"/>
            <rect x="840" y="80" width="170" height="110" rx="12" fill="#DCECF4"/>
            <rect x="1050" y="80" width="130" height="110" rx="12" fill="#F3F4F5"/>
          </g>
          <g fill="none" stroke="#4C5F6A" stroke-width="3" marker-end="url(#${prefix}-arrow)">
            <path d="M 170 135 H 205"/>
            <path d="M 385 135 H 420"/>
            <path d="M 600 135 H 640"/>
            <path d="M 790 135 H 830"/>
            <path d="M 1010 135 H 1040"/>
          </g>
          <path data-testid="transformer-residual-path" d="M 95 190 V 275 H 720 V 200"
            fill="none" stroke="#8C421F" stroke-width="3" stroke-dasharray="9 6" marker-end="url(#${prefix}-arrow)"/>
          <g class="chart-label" text-anchor="middle">
            <text x="95" y="125" font-size="20" font-weight="700">输入</text>
            <text x="95" y="158">x</text>
            <text data-testid="transformer-input-normalization" x="300" y="125" font-size="20" font-weight="700"></text>
            <text x="300" y="158">a</text>
            <text x="515" y="125" font-size="20" font-weight="700">FFN</text>
            <text x="515" y="158">f = F(a)</text>
            <text data-testid="transformer-merge-label" x="720" y="125" font-size="20" font-weight="700"></text>
            <text x="720" y="158">z</text>
            <text data-testid="transformer-output-normalization" x="925" y="125" font-size="20" font-weight="700"></text>
            <text x="925" y="158">y</text>
            <text x="1115" y="125" font-size="20" font-weight="700">输出</text>
            <text x="1115" y="158">y</text>
            <text data-testid="transformer-residual-label" x="405" y="260" fill="#8C421F">残差捷径直接携带 x</text>
            <text x="600" y="330">固定 x = (1, −2, 4)，F(z) = (z₁, ReLU(z₂), ReLU(z₃))，ε = 0</text>
          </g>
        </svg>
        <figcaption>切换归一化位置或残差开关会重新计算 a、f、z 与 y；所有数值均来自上方固定权重，而非预录结果。</figcaption>
      </figure>
    `;

    this.#normPosition = this.querySelector('[data-testid="transformer-norm-position"]');
    this.#residual = this.querySelector('[data-testid="transformer-residual"]');
    this.#normalizedInput = this.querySelector('[data-testid="transformer-normalized-input"]');
    this.#sublayerOutput = this.querySelector('[data-testid="transformer-sublayer-output"]');
    this.#mergedOutput = this.querySelector('[data-testid="transformer-merged-output"]');
    this.#blockOutput = this.querySelector('[data-testid="transformer-block-output"]');
    this.#rule = this.querySelector('[data-testid="transformer-block-rule"]');
    this.#chart = this.querySelector('[data-testid="transformer-block-chart"]');
    this.#description = this.querySelector(`#${prefix}-chart-desc`);
    this.#inputNormLabel = this.querySelector('[data-testid="transformer-input-normalization"]');
    this.#outputNormLabel = this.querySelector('[data-testid="transformer-output-normalization"]');
    this.#mergeLabel = this.querySelector('[data-testid="transformer-merge-label"]');
    this.#residualPath = this.querySelector('[data-testid="transformer-residual-path"]');

    this.#normPosition.value = this.getAttribute('norm-position') === 'post' ? 'post' : 'pre';
    this.#residual.checked = this.getAttribute('residual') !== 'false';
    this.#normPosition.addEventListener('change', () => this.#update());
    this.#residual.addEventListener('change', () => this.#update());
    this.#update();
  }

  #update() {
    const isPost = this.#normPosition.value === 'post';
    const hasResidual = this.#residual.checked;
    const sublayerInput = isPost ? input : layerNorm(input);
    const sublayerOutput = feedForward(sublayerInput);
    const merged = hasResidual ? addVectors(input, sublayerOutput) : sublayerOutput;
    const output = isPost ? layerNorm(merged) : merged;
    const positionName = isPost ? 'Post-LN' : 'Pre-LN';
    const residualName = hasResidual ? '启用残差连接' : '关闭残差连接';
    const rule = isPost
      ? (hasResidual ? 'y = LN(x + F(x))' : 'y = LN(F(x))')
      : (hasResidual ? 'y = x + F(LN(x))' : 'y = F(LN(x))');

    this.#normalizedInput.textContent = formatVector(sublayerInput);
    this.#sublayerOutput.textContent = formatVector(sublayerOutput);
    this.#mergedOutput.textContent = formatVector(merged);
    this.#blockOutput.textContent = formatVector(output);
    this.#rule.textContent = rule;
    this.#inputNormLabel.textContent = isPost ? '恒等' : 'LN';
    this.#outputNormLabel.textContent = isPost ? 'LN' : '恒等';
    this.#mergeLabel.textContent = hasResidual ? 'x + f' : 'f';
    const residualVisibility = hasResidual ? 'visible' : 'hidden';
    this.#residualPath.setAttribute('visibility', residualVisibility);
    this.#chart.querySelector('[data-testid="transformer-residual-label"]')
      .setAttribute('visibility', residualVisibility);
    this.#chart.dataset.signature = `${this.#normPosition.value}:${hasResidual}:${output.join('|')}`;
    this.#description.textContent = `${positionName}，${residualName}；计算规则为 ${rule}。子层输入 a 为 ${formatVector(sublayerInput)}，FFN 输出 f 为 ${formatVector(sublayerOutput)}，汇合表示 z 为 ${formatVector(merged)}，Block 输出 y 为 ${formatVector(output)}。`;
  }
}

export function defineTransformerBlockElement() {
  if (!customElements.get('llm-transformer-block')) {
    customElements.define('llm-transformer-block', TransformerBlockElement);
  }
}
