import {
  adamStep,
  adamwStep,
  momentumStep,
  quadratic,
  quadraticGradient,
  sgdStep,
} from '../lib/optimization.js';

const initialPoint = [1, 1];
const center = [-1, 2];
const maxSteps = 40;
const chart = {
  originX: 360,
  originY: 200,
  scale: 45,
};
const algorithmNames = {
  sgd: 'SGD',
  momentum: 'Momentum',
  adam: 'Adam',
  adamw: 'AdamW',
};
let instanceCount = 0;

function formatNumber(value) {
  if (Math.abs(value) < 1e-12) {
    return '0';
  }
  return String(Number(value.toFixed(6)));
}

function formatVector(vector) {
  return `(${vector.map(formatNumber).join(', ')})`;
}

function screenPoint(point) {
  return [
    Number((chart.originX + chart.scale * point[0]).toFixed(6)),
    Number((chart.originY - chart.scale * point[1]).toFixed(6)),
  ];
}

function pathData(states) {
  return states.map(({ point }, index) => {
    const [x, y] = screenPoint(point);
    return `${index === 0 ? 'M' : 'L'} ${x} ${y}`;
  }).join(' ');
}

class OptimizerTrajectoryElement extends HTMLElement {
  #algorithm;
  #currentPoint;
  #description;
  #fullTrajectory;
  #learningRate;
  #learningRateValue;
  #loss;
  #momentum;
  #momentumValue;
  #next;
  #point;
  #state = [];
  #step = 0;
  #stepInput;
  #stepOutput;
  #trajectory;
  #velocity;
  #weightDecay;
  #weightDecayValue;

  connectedCallback() {
    if (this.#algorithm) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'optimizer-trajectory'}-${instanceCount}`;

    this.innerHTML = `
      <figure class="interactive-widget optimizer-widget">
        <div class="interactive-controls optimizer-controls">
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-algorithm">优化算法</label>
            <select id="${prefix}-algorithm" data-testid="optimizer-algorithm">
              <option value="sgd">SGD</option>
              <option value="momentum">Momentum</option>
              <option value="adam">Adam</option>
              <option value="adamw">AdamW</option>
            </select>
          </div>
          <div class="interactive-control">
            <label for="${prefix}-learning-rate">优化器学习率 η</label>
            <input id="${prefix}-learning-rate" data-testid="optimizer-learning-rate"
              type="range" min="0" max="0.5" step="0.01">
            <output id="${prefix}-learning-rate-value" for="${prefix}-learning-rate"></output>
          </div>
          <div class="interactive-control">
            <label for="${prefix}-momentum">Momentum 系数 β</label>
            <input id="${prefix}-momentum" data-testid="optimizer-momentum"
              type="range" min="0" max="0.99" step="0.01">
            <output id="${prefix}-momentum-value" for="${prefix}-momentum"></output>
          </div>
          <div class="interactive-control">
            <label for="${prefix}-weight-decay">权重衰减 λ</label>
            <input id="${prefix}-weight-decay" data-testid="optimizer-weight-decay"
              type="range" min="0" max="0.1" step="0.01">
            <output id="${prefix}-weight-decay-value" for="${prefix}-weight-decay"></output>
          </div>
          <div class="interactive-control">
            <label for="${prefix}-step">优化轨迹步数</label>
            <input id="${prefix}-step" data-testid="optimizer-step-input"
              type="range" min="0" max="${maxSteps}" step="1" value="0">
            <output for="${prefix}-step">${maxSteps} 步上限</output>
          </div>
        </div>
        <div class="interactive-actions">
          <button type="button" data-testid="optimizer-reset">重置轨迹</button>
          <button type="button" data-testid="optimizer-next">下一步</button>
        </div>
        <dl class="interactive-results">
          <div>
            <dt>当前步数</dt>
            <dd><output data-testid="optimizer-step" for="${prefix}-step"></output></dd>
          </div>
          <div>
            <dt>当前点 θ</dt>
            <dd><output data-testid="optimizer-point" for="${prefix}-step ${prefix}-algorithm"></output></dd>
          </div>
          <div>
            <dt>二次损失 J(θ)</dt>
            <dd><output data-testid="optimizer-loss" for="${prefix}-step ${prefix}-algorithm"></output></dd>
          </div>
          <div>
            <dt>Momentum 速度 v</dt>
            <dd><output data-testid="optimizer-velocity" for="${prefix}-step ${prefix}-algorithm ${prefix}-momentum"></output></dd>
          </div>
        </dl>
        <svg class="interactive-chart" data-testid="optimizer-chart" viewBox="0 0 720 400"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">二次损失上的优化轨迹</title>
          <desc id="${prefix}-chart-desc"></desc>
          <path class="chart-axes" d="M 40 200 H 680 M 360 20 V 380" />
          <circle class="optimizer-contour" cx="315" cy="110" r="33.75" />
          <circle class="optimizer-contour" cx="315" cy="110" r="67.5" />
          <circle class="optimizer-contour" cx="315" cy="110" r="101.25" />
          <text class="chart-label" x="326" y="103">θ* = (−1, 2)</text>
          <path class="optimizer-full-trajectory" data-testid="optimizer-full-trajectory" />
          <path class="optimizer-trajectory" data-testid="optimizer-trajectory-path" />
          <circle class="optimizer-current-point" data-testid="optimizer-current-point" r="6" />
          <text class="chart-label" x="414" y="149">θ₀ = (1, 1)</text>
        </svg>
        <figcaption>损失为 J(x,y) = ½[(x+1)²+(y−2)²]；每次修改参数都会用数值内核重算至多 40 步，不自动播放。</figcaption>
      </figure>
    `;

    this.#algorithm = this.querySelector('[data-testid="optimizer-algorithm"]');
    this.#learningRate = this.querySelector('[data-testid="optimizer-learning-rate"]');
    this.#momentum = this.querySelector('[data-testid="optimizer-momentum"]');
    this.#weightDecay = this.querySelector('[data-testid="optimizer-weight-decay"]');
    this.#stepInput = this.querySelector('[data-testid="optimizer-step-input"]');
    this.#next = this.querySelector('[data-testid="optimizer-next"]');
    this.#stepOutput = this.querySelector('[data-testid="optimizer-step"]');
    this.#point = this.querySelector('[data-testid="optimizer-point"]');
    this.#loss = this.querySelector('[data-testid="optimizer-loss"]');
    this.#velocity = this.querySelector('[data-testid="optimizer-velocity"]');
    this.#learningRateValue = this.querySelector(`#${prefix}-learning-rate-value`);
    this.#momentumValue = this.querySelector(`#${prefix}-momentum-value`);
    this.#weightDecayValue = this.querySelector(`#${prefix}-weight-decay-value`);
    this.#description = this.querySelector(`#${prefix}-chart-desc`);
    this.#fullTrajectory = this.querySelector('[data-testid="optimizer-full-trajectory"]');
    this.#trajectory = this.querySelector('[data-testid="optimizer-trajectory-path"]');
    this.#currentPoint = this.querySelector('[data-testid="optimizer-current-point"]');

    this.#algorithm.value = this.getAttribute('algorithm') ?? 'sgd';
    this.#learningRate.value = this.getAttribute('learning-rate') ?? '0.1';
    this.#momentum.value = this.getAttribute('momentum') ?? '0.9';
    this.#weightDecay.value = this.getAttribute('weight-decay') ?? '0.01';

    this.#algorithm.addEventListener('change', () => this.#recompute());
    for (const input of [this.#learningRate, this.#momentum, this.#weightDecay]) {
      input.addEventListener('input', () => this.#recompute());
    }
    this.#stepInput.addEventListener('input', () => {
      this.#step = Number(this.#stepInput.value);
      this.#render();
    });
    this.#next.addEventListener('click', () => {
      if (this.#step < maxSteps) {
        this.#step += 1;
        this.#stepInput.value = String(this.#step);
        this.#render();
      }
    });
    this.querySelector('[data-testid="optimizer-reset"]').addEventListener('click', () => {
      this.#step = 0;
      this.#stepInput.value = '0';
      this.#render();
    });
    this.#recompute();
  }

  #recompute() {
    const algorithm = this.#algorithm.value;
    this.#momentum.disabled = algorithm !== 'momentum';
    this.#weightDecay.disabled = algorithm !== 'adamw';
    const learningRate = Number(this.#learningRate.value);
    const beta = Number(this.#momentum.value);
    const weightDecay = Number(this.#weightDecay.value);
    let point = initialPoint.slice();
    let momentumState = { velocity: [0, 0], beta };
    let adamState = {
      m: [0, 0],
      v: [0, 0],
      t: 0,
      beta1: 0.9,
      beta2: 0.999,
      epsilon: 1e-8,
    };
    const states = [{
      point: point.slice(),
      loss: quadratic(point, center),
      velocity: algorithm === 'momentum' ? momentumState.velocity.slice() : null,
    }];

    for (let index = 0; index < maxSteps; index += 1) {
      const gradient = quadraticGradient(point, center);
      if (algorithm === 'sgd') {
        point = sgdStep(point, gradient, learningRate);
      } else if (algorithm === 'momentum') {
        momentumState = momentumStep(point, gradient, learningRate, momentumState);
        point = momentumState.point;
      } else if (algorithm === 'adam') {
        adamState = adamStep(point, gradient, learningRate, adamState);
        point = adamState.point;
      } else {
        adamState = adamwStep(point, gradient, learningRate, adamState, weightDecay);
        point = adamState.point;
      }
      states.push({
        point: point.slice(),
        loss: quadratic(point, center),
        velocity: algorithm === 'momentum' ? momentumState.velocity.slice() : null,
      });
    }

    this.#state = states;
    this.#learningRateValue.textContent = this.#learningRate.value;
    this.#momentumValue.textContent = this.#momentum.value;
    this.#weightDecayValue.textContent = this.#weightDecay.value;
    this.#render();
  }

  #render() {
    const current = this.#state[this.#step];
    const visibleStates = this.#state.slice(0, this.#step + 1);
    const [x, y] = screenPoint(current.point);
    this.#stepOutput.textContent = String(this.#step);
    this.#point.textContent = formatVector(current.point);
    this.#loss.textContent = formatNumber(current.loss);
    this.#velocity.textContent = current.velocity ? formatVector(current.velocity) : '—';
    this.#next.disabled = this.#step === maxSteps;
    this.#fullTrajectory.setAttribute('d', pathData(this.#state));
    this.#trajectory.setAttribute('d', pathData(visibleStates));
    this.#currentPoint.setAttribute('cx', x);
    this.#currentPoint.setAttribute('cy', y);
    this.#description.textContent = `当前算法 ${algorithmNames[this.#algorithm.value]}，第 ${this.#step} 步（最多 ${maxSteps} 步），点 ${formatVector(current.point)}，损失 ${formatNumber(current.loss)}。实线为已走轨迹，虚线为预计算的完整轨迹。`;
  }
}

export function defineOptimizerTrajectoryElement() {
  if (!customElements.get('llm-optimizer-trajectory')) {
    customElements.define('llm-optimizer-trajectory', OptimizerTrajectoryElement);
  }
}
