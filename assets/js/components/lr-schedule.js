import { clipGlobalNorm, cosineLearningRate } from '../lib/optimization.js';

const gradient = [3, 4];
const totalSteps = 100;
const warmupSteps = 10;
const minRate = 0.0001;
const maxRate = 0.001;
let instanceCount = 0;

function formatNumber(value) {
  if (Math.abs(value) < 1e-12) {
    return '0';
  }
  return String(Number(value.toFixed(6)));
}

function formatRate(value) {
  if (value === 0) {
    return '0';
  }
  return String(Number(value.toPrecision(6)));
}

function formatVector(vector) {
  return `(${vector.map(formatNumber).join(', ')})`;
}

function learningRateAt(step) {
  if (step <= warmupSteps) {
    return maxRate * step / warmupSteps;
  }
  return cosineLearningRate(step - warmupSteps, totalSteps - warmupSteps, minRate, maxRate);
}

function schedulePoint(step, learningRate) {
  return [
    Number((60 + 3.4 * step).toFixed(6)),
    Number((280 - 220 * learningRate / maxRate).toFixed(6)),
  ];
}

class LrScheduleElement extends HTMLElement {
  #clipFactor;
  #clippedGradient;
  #clippedLine;
  #clippedNorm;
  #clippedPoint;
  #currentStep;
  #description;
  #gradientNorm;
  #learningRate;
  #learningRateMarker;
  #schedulePath;
  #stepInput;
  #threshold;
  #thresholdValue;

  connectedCallback() {
    if (this.#stepInput) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'lr-schedule'}-${instanceCount}`;

    this.innerHTML = `
      <figure class="interactive-widget lr-widget">
        <div class="interactive-controls">
          <div class="interactive-control">
            <label for="${prefix}-step">学习率调度步数</label>
            <input id="${prefix}-step" data-testid="lr-step-input"
              type="range" min="0" max="${totalSteps}" step="1" value="0">
            <output data-testid="lr-current-step" for="${prefix}-step"></output>
          </div>
          <div class="interactive-control">
            <label for="${prefix}-threshold">梯度裁剪阈值 τ</label>
            <input id="${prefix}-threshold" data-testid="clip-threshold"
              type="range" min="0" max="6" step="1">
            <output id="${prefix}-threshold-value" for="${prefix}-threshold"></output>
          </div>
        </div>
        <dl class="interactive-results">
          <div>
            <dt>当前学习率 η(t)</dt>
            <dd><output data-testid="lr-value" for="${prefix}-step"></output></dd>
          </div>
          <div>
            <dt>原梯度范数 ‖g‖₂</dt>
            <dd><output data-testid="gradient-norm" for="${prefix}-threshold"></output></dd>
          </div>
          <div>
            <dt>裁剪因子 c</dt>
            <dd><output data-testid="clip-factor" for="${prefix}-threshold"></output></dd>
          </div>
          <div>
            <dt>裁剪后梯度 g̃</dt>
            <dd><output data-testid="clipped-gradient" for="${prefix}-threshold"></output></dd>
          </div>
          <div>
            <dt>裁剪后范数 ‖g̃‖₂</dt>
            <dd><output data-testid="clipped-gradient-norm" for="${prefix}-threshold"></output></dd>
          </div>
        </dl>
        <svg class="interactive-chart" data-testid="lr-chart" viewBox="0 0 720 360"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">Warmup 与 cosine 学习率及全局范数裁剪</title>
          <desc id="${prefix}-chart-desc"></desc>
          <path class="chart-axes" d="M 60 40 V 280 H 400 M 500 260 H 680 M 540 320 V 80" />
          <path class="lr-warmup-marker" d="M 94 40 V 280" />
          <text class="chart-label" x="66" y="58">ηmax = 0.001</text>
          <text class="chart-label" x="66" y="300">0</text>
          <text class="chart-label" x="84" y="300">W = 10</text>
          <text class="chart-label" x="382" y="300">T = 100</text>
          <path class="lr-schedule-path" data-testid="lr-schedule-path" />
          <circle class="lr-current-point" data-testid="lr-current-point" r="5" />
          <line class="clip-original-gradient" x1="540" y1="260" x2="594" y2="188" />
          <line class="clip-clipped-gradient" data-testid="clip-clipped-gradient" x1="540" y1="260" />
          <circle class="clip-clipped-point" data-testid="clip-clipped-point" r="5" />
          <text class="chart-label" x="602" y="184">g = (3, 4)</text>
          <text class="chart-label" x="500" y="340">全局范数统一缩放</text>
        </svg>
        <figcaption>左侧 T = 100、W = 10、ηmin = 0.0001、ηmax = 0.001；右侧固定 g = (3,4)，只改变当前步数或阈值。</figcaption>
      </figure>
    `;

    this.#stepInput = this.querySelector('[data-testid="lr-step-input"]');
    this.#threshold = this.querySelector('[data-testid="clip-threshold"]');
    this.#currentStep = this.querySelector('[data-testid="lr-current-step"]');
    this.#learningRate = this.querySelector('[data-testid="lr-value"]');
    this.#gradientNorm = this.querySelector('[data-testid="gradient-norm"]');
    this.#clipFactor = this.querySelector('[data-testid="clip-factor"]');
    this.#clippedGradient = this.querySelector('[data-testid="clipped-gradient"]');
    this.#clippedNorm = this.querySelector('[data-testid="clipped-gradient-norm"]');
    this.#thresholdValue = this.querySelector(`#${prefix}-threshold-value`);
    this.#description = this.querySelector(`#${prefix}-chart-desc`);
    this.#schedulePath = this.querySelector('[data-testid="lr-schedule-path"]');
    this.#learningRateMarker = this.querySelector('[data-testid="lr-current-point"]');
    this.#clippedLine = this.querySelector('[data-testid="clip-clipped-gradient"]');
    this.#clippedPoint = this.querySelector('[data-testid="clip-clipped-point"]');
    this.#threshold.value = this.getAttribute('threshold') ?? '2';

    const scheduleStates = Array.from({ length: totalSteps + 1 }, (_, step) => {
      const learningRate = learningRateAt(step);
      return schedulePoint(step, learningRate);
    });
    this.#schedulePath.setAttribute('d', scheduleStates.map(([x, y], index) =>
      `${index === 0 ? 'M' : 'L'} ${x} ${y}`,
    ).join(' '));
    this.#stepInput.addEventListener('input', () => this.#update());
    this.#threshold.addEventListener('input', () => this.#update());
    this.#update();
  }

  #update() {
    const step = Number(this.#stepInput.value);
    const threshold = Number(this.#threshold.value);
    const learningRate = learningRateAt(step);
    const originalNorm = Math.hypot(...gradient);
    const clipped = clipGlobalNorm(gradient, threshold);
    const clippedNorm = Math.hypot(...clipped);
    const factor = clippedNorm / originalNorm;
    const [scheduleX, scheduleY] = schedulePoint(step, learningRate);

    this.#currentStep.textContent = String(step);
    this.#thresholdValue.textContent = this.#threshold.value;
    this.#learningRate.textContent = formatRate(learningRate);
    this.#gradientNorm.textContent = formatNumber(originalNorm);
    this.#clipFactor.textContent = formatNumber(factor);
    this.#clippedGradient.textContent = formatVector(clipped);
    this.#clippedNorm.textContent = formatNumber(clippedNorm);
    this.#learningRateMarker.setAttribute('cx', scheduleX);
    this.#learningRateMarker.setAttribute('cy', scheduleY);
    this.#clippedLine.setAttribute('x2', 540 + 18 * clipped[0]);
    this.#clippedLine.setAttribute('y2', 260 - 18 * clipped[1]);
    this.#clippedPoint.setAttribute('cx', 540 + 18 * clipped[0]);
    this.#clippedPoint.setAttribute('cy', 260 - 18 * clipped[1]);
    this.#description.textContent = `当前为第 ${step} 步，学习率 ${formatRate(learningRate)}；梯度 ${formatVector(gradient)} 的原范数为 ${formatNumber(originalNorm)}，阈值 ${threshold}，裁剪因子 ${formatNumber(factor)}，裁剪后梯度 ${formatVector(clipped)}，裁剪后范数为 ${formatNumber(clippedNorm)}。`;
  }
}

export function defineLrScheduleElement() {
  if (!customElements.get('llm-lr-schedule')) {
    customElements.define('llm-lr-schedule', LrScheduleElement);
  }
}
