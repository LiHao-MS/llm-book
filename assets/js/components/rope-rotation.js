import { rotatePair } from '../lib/attention.js';

const original = [1, 0];
const chart = {
  originX: 320,
  originY: 180,
  scale: 120,
  arcRadius: 60,
};
let instanceCount = 0;

function formatNumber(value) {
  const cleaned = Math.abs(value) < 1e-12 ? 0 : value;
  return String(Number(cleaned.toFixed(10)));
}

function formatVector(vector) {
  return `(${vector.map(formatNumber).join(', ')})`;
}

function screenPoint(vector) {
  return [
    Number((chart.originX + chart.scale * vector[0]).toFixed(6)),
    Number((chart.originY - chart.scale * vector[1]).toFixed(6)),
  ];
}

class RopeRotationElement extends HTMLElement {
  #angle;
  #angleValue;
  #cosine;
  #sine;
  #rotation;
  #chart;
  #description;
  #angleLabel;
  #angleArc;
  #rotatedVector;
  #rotatedPoint;
  #rotatedLabel;

  connectedCallback() {
    if (this.#angle) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'rope-rotation'}-${instanceCount}`;

    this.innerHTML = `
      <figure class="interactive-widget rope-rotation-widget">
        <div class="interactive-controls">
          <div class="interactive-control">
            <label for="${prefix}-angle">旋转角度 φ</label>
            <input id="${prefix}-angle" data-testid="rope-angle" type="range" min="-180" max="180" step="1">
            <output id="${prefix}-angle-value" data-testid="rope-angle-value" for="${prefix}-angle"></output>
          </div>
        </div>
        <dl class="interactive-results rope-results">
          <div>
            <dt>原始向量 x</dt>
            <dd><output data-testid="rope-original">(1, 0)</output></dd>
          </div>
          <div>
            <dt>旋转后 R(φ)x</dt>
            <dd><output data-testid="rope-rotation" for="${prefix}-angle"></output></dd>
          </div>
          <div>
            <dt>cos φ</dt>
            <dd><output data-testid="rope-cos" for="${prefix}-angle"></output></dd>
          </div>
          <div>
            <dt>sin φ</dt>
            <dd><output data-testid="rope-sin" for="${prefix}-angle"></output></dd>
          </div>
        </dl>
        <svg class="interactive-chart" data-testid="rope-chart" viewBox="0 0 640 360"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">RoPE 二维旋转</title>
          <desc id="${prefix}-chart-desc"></desc>
          <path class="chart-axes" d="M 40 180 H 600 M 320 30 V 330" />
          <circle class="rope-unit-circle" cx="320" cy="180" r="120" />
          <path class="rope-angle-arc" data-testid="rope-angle-arc" />
          <line class="rope-original-vector" data-testid="rope-original-vector"
            x1="320" y1="180" x2="440" y2="180" />
          <line class="rope-rotated-vector" data-testid="rope-rotated-vector"
            x1="320" y1="180" x2="440" y2="180" />
          <circle class="rope-original-point" cx="440" cy="180" r="4" />
          <circle class="rope-rotated-point" data-testid="rope-point" cx="440" cy="180" r="6" />
          <text class="chart-label" x="454" y="185">x = (1, 0)</text>
          <text class="chart-label rope-rotated-label" x="450" y="165">R(φ)x</text>
          <text class="chart-label rope-angle-label" x="40" y="340"></text>
        </svg>
        <figcaption>相邻两个分量按同一矩阵旋转；正角度逆时针，向量范数保持不变。</figcaption>
      </figure>
    `;

    this.#angle = this.querySelector('[data-testid="rope-angle"]');
    this.#angleValue = this.querySelector(`#${prefix}-angle-value`);
    this.#cosine = this.querySelector('[data-testid="rope-cos"]');
    this.#sine = this.querySelector('[data-testid="rope-sin"]');
    this.#rotation = this.querySelector('[data-testid="rope-rotation"]');
    this.#chart = this.querySelector('[data-testid="rope-chart"]');
    this.#description = this.querySelector(`#${prefix}-chart-desc`);
    this.#angleLabel = this.querySelector('.rope-angle-label');
    this.#angleArc = this.querySelector('[data-testid="rope-angle-arc"]');
    this.#rotatedVector = this.querySelector('[data-testid="rope-rotated-vector"]');
    this.#rotatedPoint = this.querySelector('[data-testid="rope-point"]');
    this.#rotatedLabel = this.querySelector('.rope-rotated-label');

    this.#angle.value = this.getAttribute('angle') ?? '90';
    this.#angle.addEventListener('input', () => this.#update());
    this.#update();
  }

  #update() {
    const degrees = Number(this.#angle.value);
    const radians = degrees * Math.PI / 180;
    const rotated = rotatePair(original, radians);
    const [cosine, sine] = rotated;
    const [x, y] = screenPoint(rotated);
    const rotationText = formatVector(rotated);

    this.#angleValue.textContent = `${degrees}°`;
    this.#cosine.textContent = formatNumber(cosine);
    this.#sine.textContent = formatNumber(sine);
    this.#rotation.textContent = rotationText;
    this.#chart.dataset.signature = `${degrees}:${rotated.join('|')}`;
    this.#rotatedVector.setAttribute('x2', x);
    this.#rotatedVector.setAttribute('y2', y);
    this.#rotatedPoint.setAttribute('cx', x);
    this.#rotatedPoint.setAttribute('cy', y);
    this.#rotatedLabel.setAttribute('x', Math.min(560, Math.max(80, x + 10)));
    this.#rotatedLabel.setAttribute('y', Math.min(325, Math.max(45, y - 10)));

    if (degrees === 0) {
      this.#angleArc.setAttribute('d', '');
    } else {
      const arcX = Number((chart.originX + chart.arcRadius * cosine).toFixed(6));
      const arcY = Number((chart.originY - chart.arcRadius * sine).toFixed(6));
      const sweep = degrees > 0 ? 0 : 1;
      this.#angleArc.setAttribute(
        'd',
        `M ${chart.originX + chart.arcRadius} ${chart.originY} A ${chart.arcRadius} ${chart.arcRadius} 0 0 ${sweep} ${arcX} ${arcY}`,
      );
    }
    this.#angleLabel.textContent = `φ = ${degrees}°；正角度逆时针`;
    this.#description.textContent = `原始向量为 (1, 0)，角度为 ${degrees}°，cos φ 为 ${formatNumber(cosine)}，sin φ 为 ${formatNumber(sine)}，旋转后向量为 ${rotationText}；正角度按公式逆时针旋转，负角度顺时针，范数保持为 1。`;
  }
}

export function defineRopeRotationElement() {
  if (!customElements.get('llm-rope-rotation')) {
    customElements.define('llm-rope-rotation', RopeRotationElement);
  }
}
