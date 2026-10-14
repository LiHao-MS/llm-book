import { cosine, dot, projection } from '../lib/linalg.js';

const chart = {
  originX: 320,
  originY: 180,
  scale: 120,
};
let instanceCount = 0;

function cleanNumber(value) {
  return Math.abs(value) < 1e-12 ? 0 : value;
}

function formatNumber(value) {
  const cleaned = cleanNumber(value);
  if (Math.abs(cleaned - Math.SQRT1_2) < 1e-12) {
    return '√2/2';
  }
  if (Math.abs(cleaned + Math.SQRT1_2) < 1e-12) {
    return '-√2/2';
  }
  if (Math.abs(cleaned - Math.round(cleaned)) < 1e-12) {
    return String(Math.round(cleaned));
  }
  return String(Number(cleaned.toFixed(6)));
}

function formatVector(vector) {
  return `(${vector.map(formatNumber).join(', ')})`;
}

function unitVector(angle) {
  const radians = angle * Math.PI / 180;
  return [cleanNumber(Math.cos(radians)), cleanNumber(Math.sin(radians))];
}

function screenPoint(vector) {
  return [
    Number((chart.originX + chart.scale * vector[0]).toFixed(6)),
    Number((chart.originY - chart.scale * vector[1]).toFixed(6)),
  ];
}

function setLine(line, endpoint) {
  const [x, y] = screenPoint(endpoint);
  line.setAttribute('x1', chart.originX);
  line.setAttribute('y1', chart.originY);
  line.setAttribute('x2', x);
  line.setAttribute('y2', y);
}

function setCircle(circle, point) {
  const [x, y] = screenPoint(point);
  circle.setAttribute('cx', x);
  circle.setAttribute('cy', y);
}

function setSegment(line, start, end) {
  const [x1, y1] = screenPoint(start);
  const [x2, y2] = screenPoint(end);
  line.setAttribute('x1', x1);
  line.setAttribute('y1', y1);
  line.setAttribute('x2', x2);
  line.setAttribute('y2', y2);
}

function setLabel(label, point, offsetX = 10, offsetY = -10) {
  const [x, y] = screenPoint(point);
  label.setAttribute('x', x + offsetX);
  label.setAttribute('y', y + offsetY);
}

class VectorProjectorElement extends HTMLElement {
  #angleA;
  #angleAValue;
  #angleB;
  #angleBValue;
  #description;
  #dot;
  #cosine;
  #projection;
  #projectionLine;
  #projectionPoint;
  #residual;
  #vectorA;
  #vectorALabel;
  #vectorAPoint;
  #vectorB;
  #vectorBLabel;
  #vectorBPoint;
  #projectionLabel;

  connectedCallback() {
    if (this.#angleA) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'vector-projector'}-${instanceCount}`;

    this.innerHTML = `
      <figure class="interactive-widget vector-projector-widget">
        <div class="interactive-controls">
          <div class="interactive-control">
            <label for="${prefix}-angle-a">向量 A 角度</label>
            <input id="${prefix}-angle-a" data-testid="vector-angle-a" type="range"
              min="-180" max="180" step="1">
            <output id="${prefix}-angle-a-value" for="${prefix}-angle-a"></output>
          </div>
          <div class="interactive-control">
            <label for="${prefix}-angle-b">向量 B 角度</label>
            <input id="${prefix}-angle-b" data-testid="vector-angle-b" type="range"
              min="-180" max="180" step="1">
            <output id="${prefix}-angle-b-value" for="${prefix}-angle-b"></output>
          </div>
        </div>
        <dl class="interactive-results">
          <div>
            <dt>内积 a·b</dt>
            <dd><output data-testid="vector-dot" for="${prefix}-angle-a ${prefix}-angle-b"></output></dd>
          </div>
          <div>
            <dt>余弦相似度</dt>
            <dd><output data-testid="vector-cosine" for="${prefix}-angle-a ${prefix}-angle-b"></output></dd>
          </div>
          <div>
            <dt>A 在 B 上的投影</dt>
            <dd><output data-testid="vector-projection" for="${prefix}-angle-a ${prefix}-angle-b"></output></dd>
          </div>
        </dl>
        <svg class="interactive-chart" data-testid="vector-chart" viewBox="0 0 640 360"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">向量投影与正交分解</title>
          <desc id="${prefix}-chart-desc"></desc>
          <path class="chart-axes" d="M 40 180 H 600 M 320 30 V 330" />
          <circle class="vector-unit-circle" cx="320" cy="180" r="120" />
          <line class="vector-b-line" data-testid="vector-b" />
          <line class="vector-projection-line" data-testid="vector-projection-line" />
          <line class="vector-residual-line" data-testid="vector-residual" />
          <line class="vector-a-line" data-testid="vector-a" />
          <circle class="vector-b-point" data-testid="vector-b-point" r="5" />
          <circle class="vector-projection-point" data-testid="vector-projection-point" r="5" />
          <circle class="vector-a-point" data-testid="vector-a-point" r="5" />
          <text class="chart-label vector-b-label">b</text>
          <text class="chart-label vector-projection-label">projb(a)</text>
          <text class="chart-label vector-a-label">a</text>
        </svg>
        <figcaption>向量 a 分解为 b 方向上的正交投影和与 b 垂直的残差，投影分母明确写成 b 与自身的内积。</figcaption>
      </figure>
    `;

    this.#angleA = this.querySelector('[data-testid="vector-angle-a"]');
    this.#angleB = this.querySelector('[data-testid="vector-angle-b"]');
    this.#angleAValue = this.querySelector(`#${prefix}-angle-a-value`);
    this.#angleBValue = this.querySelector(`#${prefix}-angle-b-value`);
    this.#dot = this.querySelector('[data-testid="vector-dot"]');
    this.#cosine = this.querySelector('[data-testid="vector-cosine"]');
    this.#projection = this.querySelector('[data-testid="vector-projection"]');
    this.#description = this.querySelector(`#${prefix}-chart-desc`);
    this.#vectorA = this.querySelector('[data-testid="vector-a"]');
    this.#vectorB = this.querySelector('[data-testid="vector-b"]');
    this.#projectionLine = this.querySelector('[data-testid="vector-projection-line"]');
    this.#residual = this.querySelector('[data-testid="vector-residual"]');
    this.#vectorAPoint = this.querySelector('[data-testid="vector-a-point"]');
    this.#vectorBPoint = this.querySelector('[data-testid="vector-b-point"]');
    this.#projectionPoint = this.querySelector('[data-testid="vector-projection-point"]');
    this.#vectorALabel = this.querySelector('.vector-a-label');
    this.#vectorBLabel = this.querySelector('.vector-b-label');
    this.#projectionLabel = this.querySelector('.vector-projection-label');
    this.#angleA.value = this.getAttribute('angle-a') ?? '45';
    this.#angleB.value = this.getAttribute('angle-b') ?? '0';

    this.#angleA.addEventListener('input', () => this.#update());
    this.#angleB.addEventListener('input', () => this.#update());
    this.#update();
  }

  #update() {
    const angleA = Number(this.#angleA.value);
    const angleB = Number(this.#angleB.value);
    const a = unitVector(angleA);
    const b = unitVector(angleB);
    const innerProduct = dot(a, b);
    const similarity = cosine(a, b);
    const projected = projection(a, b);

    this.#angleAValue.textContent = `${angleA}°`;
    this.#angleBValue.textContent = `${angleB}°`;
    this.#dot.textContent = formatNumber(innerProduct);
    this.#cosine.textContent = formatNumber(similarity);
    this.#projection.textContent = formatVector(projected);

    setLine(this.#vectorA, a);
    setLine(this.#vectorB, b);
    setLine(this.#projectionLine, projected);
    setSegment(this.#residual, projected, a);
    setCircle(this.#vectorAPoint, a);
    setCircle(this.#vectorBPoint, b);
    setCircle(this.#projectionPoint, projected);
    setLabel(this.#vectorALabel, a);
    setLabel(this.#vectorBLabel, b);
    setLabel(this.#projectionLabel, projected, 10, 20);

    this.#description.textContent = `向量 A 角度为 ${angleA}°，向量 B 角度为 ${angleB}°；内积为 ${formatNumber(innerProduct)}，余弦相似度为 ${formatNumber(similarity)}，A 在 B 上的投影为 ${formatVector(projected)}。实线表示两个单位向量，粗线表示投影，虚线表示与 B 垂直的残差。`;
  }
}

export function defineVectorProjectorElement() {
  if (!customElements.get('llm-vector-projector')) {
    customElements.define('llm-vector-projector', VectorProjectorElement);
  }
}
