const IRREDUCIBLE_LOSS = 1.5;
const PARAMETER_COEFFICIENT = 2;
const DATA_COEFFICIENT = 2 * Math.sqrt(20);
const OPTIMAL_RATIO = 20;
const THROUGHPUT_TFLOPS = 0.1;
const MIN_COMPUTE = 60;
const MAX_COMPUTE = 6000;
const MIN_RATIO = 5;
const MAX_RATIO = 80;
const PLOT = { left: 70, right: 680, top: 45, bottom: 260 };
let instanceCount = 0;

function assertPositiveFinite(...values) {
  if (!values.every((value) => Number.isFinite(value) && value > 0)) {
    throw new RangeError('算力与 Token/参数比必须为正有限数。');
  }
}

function toyLoss(n, d) {
  return IRREDUCIBLE_LOSS
    + PARAMETER_COEFFICIENT / Math.sqrt(n)
    + DATA_COEFFICIENT / Math.sqrt(d);
}

export function scalingConfiguration(computeTflops, ratio) {
  assertPositiveFinite(computeTflops, ratio);
  const n = Math.sqrt(computeTflops / (6 * ratio));
  const d = ratio * n;
  return {
    n,
    d,
    loss: toyLoss(n, d),
    computeTflops: 6 * n * d,
  };
}

export function optimalScalingConfiguration(computeTflops) {
  return scalingConfiguration(computeTflops, OPTIMAL_RATIO);
}

function fixed(value, digits = 6) {
  return value.toFixed(digits);
}

function millions(value) {
  return `${value.toFixed(3)} M`;
}

function pathFromPoints(points) {
  return points.map(([x, y], index) =>
    `${index === 0 ? 'M' : 'L'} ${x.toFixed(3)} ${y.toFixed(3)}`,
  ).join(' ');
}

function ratioX(ratio) {
  return PLOT.left + (ratio - MIN_RATIO) / (MAX_RATIO - MIN_RATIO)
    * (PLOT.right - PLOT.left);
}

function computeX(computeTflops) {
  const fraction = (Math.log(computeTflops) - Math.log(MIN_COMPUTE))
    / (Math.log(MAX_COMPUTE) - Math.log(MIN_COMPUTE));
  return PLOT.left + fraction * (PLOT.right - PLOT.left);
}

function computeY(loss) {
  return PLOT.bottom - (loss - 2.9) / 3.5 * (PLOT.bottom - PLOT.top);
}

function setPoint(element, x, y) {
  element.setAttribute('cx', x.toFixed(3));
  element.setAttribute('cy', y.toFixed(3));
}

class ScalingLawsElement extends HTMLElement {
  #els;

  connectedCallback() {
    if (this.#els) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    this.style.display = 'block';
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'scaling-laws'}-${instanceCount}`;

    this.innerHTML = `
      <figure class="interactive-widget scaling-laws-widget">
        <div class="interactive-controls">
          <div class="interactive-control">
            <label for="${prefix}-compute">训练算力 c（TFLOP）</label>
            <input id="${prefix}-compute" data-testid="scaling-compute"
              type="range" min="${MIN_COMPUTE}" max="${MAX_COMPUTE}" step="60">
            <output data-testid="scaling-compute-value" for="${prefix}-compute"></output>
          </div>
          <div class="interactive-control">
            <label for="${prefix}-ratio">Token/参数比 r</label>
            <input id="${prefix}-ratio" data-testid="scaling-ratio"
              type="range" min="${MIN_RATIO}" max="${MAX_RATIO}" step="5">
            <output data-testid="scaling-ratio-value" for="${prefix}-ratio"></output>
          </div>
        </div>
        <dl class="interactive-results">
          <div><dt>当前参数 N</dt><dd><output data-testid="scaling-params" for="${prefix}-compute ${prefix}-ratio"></output></dd></div>
          <div><dt>当前 Token D</dt><dd><output data-testid="scaling-tokens" for="${prefix}-compute ${prefix}-ratio"></output></dd></div>
          <div><dt>算力核对 6nd</dt><dd><output data-testid="scaling-compute-check" for="${prefix}-compute ${prefix}-ratio"></output></dd></div>
          <div><dt>当前玩具损失 L</dt><dd><output data-testid="scaling-loss" for="${prefix}-compute ${prefix}-ratio"></output></dd></div>
          <div><dt>最优玩具损失 L*</dt><dd><output data-testid="scaling-optimal-loss" for="${prefix}-compute"></output></dd></div>
          <div><dt>最优 N* / D*</dt><dd><output data-testid="scaling-optimal-config" for="${prefix}-compute"></output></dd></div>
          <div><dt>训练时间（仅 FLOPs）</dt><dd><output data-testid="scaling-training-time" for="${prefix}-compute"></output></dd></div>
          <div><dt>dense 推理（仅 2N）</dt><dd><output data-testid="scaling-inference" for="${prefix}-compute ${prefix}-ratio"></output></dd></div>
          <div><dt>16 位权重存储</dt><dd><output data-testid="scaling-memory" for="${prefix}-compute ${prefix}-ratio"></output></dd></div>
        </dl>
        <svg class="interactive-chart" data-testid="scaling-ratio-chart" viewBox="0 0 720 310"
          role="img" aria-labelledby="${prefix}-ratio-title" aria-describedby="${prefix}-ratio-desc">
          <title id="${prefix}-ratio-title">固定算力下 Token/参数比与玩具损失</title>
          <desc id="${prefix}-ratio-desc"></desc>
          <text class="chart-label" x="70" y="24" font-weight="700">实线：损失曲线；橙点：当前；绿点与虚线：最优 r = 20</text>
          <path class="chart-axes" d="M 70 45 V 260 H 680" />
          <text class="chart-label" data-testid="ratio-y-max" x="62" y="50" text-anchor="end"></text>
          <text class="chart-label" data-testid="ratio-y-mid" x="62" y="157" text-anchor="end"></text>
          <text class="chart-label" data-testid="ratio-y-min" x="62" y="264" text-anchor="end"></text>
          <text class="chart-label" x="${ratioX(5)}" y="284" text-anchor="middle">5</text>
          <text class="chart-label" x="${ratioX(20)}" y="284" text-anchor="middle">20</text>
          <text class="chart-label" x="${ratioX(40)}" y="284" text-anchor="middle">40</text>
          <text class="chart-label" x="${ratioX(80)}" y="284" text-anchor="middle">80</text>
          <text class="chart-label" x="680" y="305" text-anchor="end">r = d/n</text>
          <text class="chart-label" x="48" y="40">L</text>
          <line data-testid="scaling-ratio-guide" x1="${ratioX(OPTIMAL_RATIO)}" x2="${ratioX(OPTIMAL_RATIO)}"
            y1="45" y2="260" stroke="#71558A" stroke-width="2" stroke-dasharray="7 6" />
          <path data-testid="scaling-ratio-path" fill="none" stroke="#1F648C" stroke-width="3"
            stroke-linecap="round" stroke-linejoin="round" />
          <circle data-testid="scaling-ratio-optimum" r="7" fill="#39704A" stroke="#FFFFFF" stroke-width="2" />
          <circle data-testid="scaling-ratio-current" r="5" fill="#C65D21" stroke="#FFFFFF" stroke-width="1.5" />
        </svg>
        <svg class="interactive-chart" data-testid="scaling-compute-chart" viewBox="0 0 720 310"
          role="img" aria-labelledby="${prefix}-compute-title" aria-describedby="${prefix}-compute-desc">
          <title id="${prefix}-compute-title">最优配置的玩具损失随训练算力变化</title>
          <desc id="${prefix}-compute-desc"></desc>
          <text class="chart-label" x="70" y="24" font-weight="700">最优曲线 d* = 20n*；横轴为 log c；橙点：当前算力</text>
          <path class="chart-axes" d="M 70 45 V 260 H 680" />
          <text class="chart-label" x="62" y="${computeY(3) + 4}" text-anchor="end">3</text>
          <text class="chart-label" x="62" y="${computeY(4) + 4}" text-anchor="end">4</text>
          <text class="chart-label" x="62" y="${computeY(5) + 4}" text-anchor="end">5</text>
          <text class="chart-label" x="62" y="${computeY(6) + 4}" text-anchor="end">6</text>
          <text class="chart-label" x="${computeX(60)}" y="284" text-anchor="middle">60</text>
          <text class="chart-label" x="${computeX(480)}" y="284" text-anchor="middle">480</text>
          <text class="chart-label" x="${computeX(6000)}" y="284" text-anchor="middle">6000</text>
          <text class="chart-label" x="680" y="305" text-anchor="end">c（TFLOP）</text>
          <text class="chart-label" x="48" y="40">L</text>
          <path data-testid="scaling-compute-path" fill="none" stroke="#1F648C" stroke-width="3"
            stroke-linecap="round" stroke-linejoin="round" />
          <circle data-testid="scaling-compute-current" r="6" fill="#C65D21" stroke="#FFFFFF" stroke-width="2" />
        </svg>
        <figcaption>
          归一化为 n = N/10⁶、d = D/10⁶、c = C/10¹²，故 c = 6nd；玩具参数为 E = 1.5、A = 2、B = 2√20、α = β = 1/2。
          r = 20 是这些玩具系数构造出的最优点，不是论文拟合值。训练时间固定按有效吞吐 0.1 TFLOP/s 换算；存储按每参数 2 字节和十进制 MB 计算。
        </figcaption>
      </figure>
    `;

    const byTestId = (testId) => this.querySelector(`[data-testid="${testId}"]`);
    this.#els = {
      compute: byTestId('scaling-compute'),
      ratio: byTestId('scaling-ratio'),
      computeValue: byTestId('scaling-compute-value'),
      ratioValue: byTestId('scaling-ratio-value'),
      params: byTestId('scaling-params'),
      tokens: byTestId('scaling-tokens'),
      computeCheck: byTestId('scaling-compute-check'),
      loss: byTestId('scaling-loss'),
      optimalLoss: byTestId('scaling-optimal-loss'),
      optimalConfig: byTestId('scaling-optimal-config'),
      trainingTime: byTestId('scaling-training-time'),
      inference: byTestId('scaling-inference'),
      memory: byTestId('scaling-memory'),
      ratioChart: byTestId('scaling-ratio-chart'),
      ratioDescription: this.querySelector(`#${prefix}-ratio-desc`),
      ratioPath: byTestId('scaling-ratio-path'),
      ratioCurrent: byTestId('scaling-ratio-current'),
      ratioOptimum: byTestId('scaling-ratio-optimum'),
      ratioYMin: byTestId('ratio-y-min'),
      ratioYMid: byTestId('ratio-y-mid'),
      ratioYMax: byTestId('ratio-y-max'),
      computeChart: byTestId('scaling-compute-chart'),
      computeDescription: this.querySelector(`#${prefix}-compute-desc`),
      computePath: byTestId('scaling-compute-path'),
      computeCurrent: byTestId('scaling-compute-current'),
    };
    this.#els.compute.value = this.getAttribute('compute') ?? '480';
    this.#els.ratio.value = this.getAttribute('ratio') ?? '20';
    this.#els.compute.addEventListener('input', () => this.#update());
    this.#els.ratio.addEventListener('input', () => this.#update());

    const computePoints = Array.from({ length: 81 }, (_, index) => {
      const fraction = index / 80;
      const c = Math.exp(Math.log(MIN_COMPUTE) + fraction
        * (Math.log(MAX_COMPUTE) - Math.log(MIN_COMPUTE)));
      return [computeX(c), computeY(optimalScalingConfiguration(c).loss)];
    });
    this.#els.computePath.setAttribute('d', pathFromPoints(computePoints));
    this.#update();
  }

  #update() {
    const c = Number(this.#els.compute.value);
    const ratio = Number(this.#els.ratio.value);
    const current = scalingConfiguration(c, ratio);
    const optimal = optimalScalingConfiguration(c);
    const ratioStates = Array.from(
      { length: MAX_RATIO - MIN_RATIO + 1 },
      (_, index) => scalingConfiguration(c, MIN_RATIO + index),
    );
    const losses = ratioStates.map((state) => state.loss);
    const yMin = Math.min(...losses) - 0.01;
    const yMax = Math.max(...losses) + 0.01;
    const ratioY = (loss) => PLOT.bottom - (loss - yMin) / (yMax - yMin)
      * (PLOT.bottom - PLOT.top);
    const ratioPoints = ratioStates.map((state, index) => [
      ratioX(MIN_RATIO + index),
      ratioY(state.loss),
    ]);

    this.#els.computeValue.textContent = String(c);
    this.#els.ratioValue.textContent = String(ratio);
    this.#els.params.textContent = millions(current.n);
    this.#els.tokens.textContent = millions(current.d);
    this.#els.computeCheck.textContent = `${fixed(current.computeTflops, 3)} TFLOP`;
    this.#els.loss.textContent = fixed(current.loss);
    this.#els.optimalLoss.textContent = fixed(optimal.loss);
    this.#els.optimalConfig.textContent = `${millions(optimal.n)} / ${millions(optimal.d)}`;
    this.#els.trainingTime.textContent = `${fixed(c / THROUGHPUT_TFLOPS / 3600)} h`;
    this.#els.inference.textContent = `${fixed(2 * current.n)} MFLOP/Token`;
    this.#els.memory.textContent = `${fixed(2 * current.n)} MB`;

    this.#els.ratioPath.setAttribute('d', pathFromPoints(ratioPoints));
    setPoint(this.#els.ratioCurrent, ratioX(ratio), ratioY(current.loss));
    setPoint(this.#els.ratioOptimum, ratioX(OPTIMAL_RATIO), ratioY(optimal.loss));
    this.#els.ratioYMin.textContent = fixed(yMin, 3);
    this.#els.ratioYMid.textContent = fixed((yMin + yMax) / 2, 3);
    this.#els.ratioYMax.textContent = fixed(yMax, 3);
    setPoint(this.#els.computeCurrent, computeX(c), computeY(optimal.loss));

    this.#els.ratioChart.dataset.signature = `${c}|${ratio}|${current.loss}`;
    this.#els.computeChart.dataset.signature = `${c}|${optimal.loss}`;
    this.#els.ratioDescription.textContent = `当前 c=${c} TFLOP，r=${ratio}，n=${fixed(current.n)}，d=${fixed(current.d)}，当前玩具损失 ${fixed(current.loss)}；玩具模型的最优比为 20，最优损失 ${fixed(optimal.loss)}。`;
    this.#els.computeDescription.textContent = `横轴为对数训练算力。当前 c=${c} TFLOP 的最优配置为 n=${fixed(optimal.n)}、d=${fixed(optimal.d)}，玩具损失 ${fixed(optimal.loss)}；曲线使用 d=20n。`;
  }
}

export function defineScalingLawsElement() {
  if (!customElements.get('llm-scaling-laws')) {
    customElements.define('llm-scaling-laws', ScalingLawsElement);
  }
}
