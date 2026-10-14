import { transform } from '../lib/linalg.js';

const svgNamespace = 'http://www.w3.org/2000/svg';
const chart = {
  originX: 320,
  originY: 190,
  scale: 60,
};
const matrices = {
  identity: [[1, 0], [0, 1]],
  rotate90: [[0, -1], [1, 0]],
  'project-x': [[1, 0], [0, 0]],
  'shear-x': [[1, 1], [0, 1]],
  'scale-xy': [[2, 0], [0, 0.5]],
};
const originalPoint = [1, 0];
const firstBasisVector = [1, 0];
const secondBasisVector = [0, 1];
const unitSquare = [[0, 0], [1, 0], [1, 1], [0, 1]];
let instanceCount = 0;

function createSvgElement(name, attributes = {}) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [attribute, value] of Object.entries(attributes)) {
    element.setAttribute(attribute, String(value));
  }
  return element;
}

function formatNumber(value) {
  if (Math.abs(value) < 1e-12) {
    return '0';
  }
  return String(Number(value.toFixed(6)));
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

function setLabel(label, point, offsetY = -9) {
  const [x, y] = screenPoint(point);
  label.setAttribute('x', x + 9);
  label.setAttribute('y', y + offsetY);
}

function gridSegments(matrix) {
  const segments = [];
  for (let coordinate = -2; coordinate <= 2; coordinate += 1) {
    segments.push([[-2, coordinate], [2, coordinate]]);
    segments.push([[coordinate, -2], [coordinate, 2]]);
  }
  if (!matrix) {
    return segments;
  }
  return segments.map(([start, end]) => [
    transform(matrix, start),
    transform(matrix, end),
  ]);
}

function renderGrid(group, matrix) {
  const signature = [];
  const lines = gridSegments(matrix).map(([start, end]) => {
    const [x1, y1] = screenPoint(start);
    const [x2, y2] = screenPoint(end);
    signature.push(`${x1},${y1},${x2},${y2}`);
    return createSvgElement('line', { x1, y1, x2, y2 });
  });
  group.dataset.signature = signature.join(';');
  group.replaceChildren(...lines);
}

function renderSquare(polygon, matrix) {
  const points = unitSquare.map((point) => {
    const transformed = matrix ? transform(matrix, point) : point;
    return screenPoint(transformed).join(',');
  });
  polygon.setAttribute('points', points.join(' '));
}

class MatrixTransformerElement extends HTMLElement {
  #description;
  #firstBasisImage;
  #firstBasisLabel;
  #originalPointMarker;
  #originalSquare;
  #point;
  #secondBasisImage;
  #secondBasisLabel;
  #select;
  #transformedGrid;
  #transformedPoint;
  #transformedPointLabel;
  #transformedPointMarker;
  #transformedSquare;

  connectedCallback() {
    if (this.#select) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'matrix-transformer'}-${instanceCount}`;

    this.innerHTML = `
      <figure class="interactive-widget matrix-transformer-widget">
        <div class="interactive-control interactive-select-control">
          <label for="${prefix}-select">矩阵变换</label>
          <select id="${prefix}-select" data-testid="matrix-select">
            <option value="identity">identity（恒等）</option>
            <option value="rotate90">rotate90（旋转 90°）</option>
            <option value="project-x">project-x（投影到 x 轴）</option>
            <option value="shear-x">shear-x（x 方向剪切）</option>
            <option value="scale-xy">scale-xy（x/y 缩放）</option>
          </select>
        </div>
        <dl class="interactive-results">
          <div>
            <dt>原测试点 x</dt>
            <dd><output data-testid="matrix-point" for="${prefix}-select"></output></dd>
          </div>
          <div>
            <dt>变换后的点 Ax</dt>
            <dd><output data-testid="matrix-transformed-point" for="${prefix}-select"></output></dd>
          </div>
        </dl>
        <svg class="interactive-chart" data-testid="matrix-chart" viewBox="0 0 640 380"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">矩阵变换与基向量的像</title>
          <desc id="${prefix}-chart-desc"></desc>
          <g class="matrix-grid matrix-original-grid" data-testid="matrix-original-grid"></g>
          <g class="matrix-grid matrix-transformed-grid" data-testid="matrix-transformed-grid"></g>
          <path class="chart-axes" d="M 40 190 H 600 M 320 40 V 340" />
          <polygon class="matrix-original-square" data-testid="matrix-original-square" />
          <polygon class="matrix-transformed-square" data-testid="matrix-transformed-square" />
          <line class="matrix-original-basis" x1="320" y1="190" x2="380" y2="190" />
          <line class="matrix-original-basis" x1="320" y1="190" x2="320" y2="130" />
          <line class="matrix-first-basis-image" data-testid="matrix-ae1" />
          <line class="matrix-second-basis-image" data-testid="matrix-ae2" />
          <circle class="matrix-original-point" data-testid="matrix-original-point-marker" r="5" />
          <circle class="matrix-transformed-point" data-testid="matrix-transformed-point-marker" r="6" />
          <text class="chart-label matrix-first-basis-label">Ae1</text>
          <text class="chart-label matrix-second-basis-label">Ae2</text>
          <text class="chart-label matrix-transformed-point-label">Ax</text>
        </svg>
        <figcaption>矩阵的两列是标准基向量的像；原/变换网格与单位方块同步显示。</figcaption>
      </figure>
    `;

    this.#select = this.querySelector('[data-testid="matrix-select"]');
    this.#point = this.querySelector('[data-testid="matrix-point"]');
    this.#transformedPoint = this.querySelector('[data-testid="matrix-transformed-point"]');
    this.#description = this.querySelector(`#${prefix}-chart-desc`);
    this.#transformedGrid = this.querySelector('[data-testid="matrix-transformed-grid"]');
    this.#originalSquare = this.querySelector('[data-testid="matrix-original-square"]');
    this.#transformedSquare = this.querySelector('[data-testid="matrix-transformed-square"]');
    this.#firstBasisImage = this.querySelector('[data-testid="matrix-ae1"]');
    this.#secondBasisImage = this.querySelector('[data-testid="matrix-ae2"]');
    this.#originalPointMarker = this.querySelector('[data-testid="matrix-original-point-marker"]');
    this.#transformedPointMarker = this.querySelector('[data-testid="matrix-transformed-point-marker"]');
    this.#firstBasisLabel = this.querySelector('.matrix-first-basis-label');
    this.#secondBasisLabel = this.querySelector('.matrix-second-basis-label');
    this.#transformedPointLabel = this.querySelector('.matrix-transformed-point-label');
    this.#select.value = this.getAttribute('matrix') ?? 'identity';

    renderGrid(this.querySelector('[data-testid="matrix-original-grid"]'));
    renderSquare(this.#originalSquare);
    setCircle(this.#originalPointMarker, originalPoint);
    this.#select.addEventListener('change', () => this.#update());
    this.#update();
  }

  #update() {
    const matrixName = this.#select.value;
    const matrix = matrices[matrixName];
    const transformedPoint = transform(matrix, originalPoint);
    const firstBasisImage = transform(matrix, firstBasisVector);
    const secondBasisImage = transform(matrix, secondBasisVector);

    this.#point.textContent = formatVector(originalPoint);
    this.#transformedPoint.textContent = formatVector(transformedPoint);
    renderGrid(this.#transformedGrid, matrix);
    renderSquare(this.#transformedSquare, matrix);
    setLine(this.#firstBasisImage, firstBasisImage);
    setLine(this.#secondBasisImage, secondBasisImage);
    setCircle(this.#transformedPointMarker, transformedPoint);
    setLabel(this.#firstBasisLabel, firstBasisImage);
    setLabel(this.#secondBasisLabel, secondBasisImage);
    setLabel(this.#transformedPointLabel, transformedPoint, 20);

    this.#description.textContent = `当前矩阵为 ${matrixName}，点 ${formatVector(originalPoint)} 变换为 ${formatVector(transformedPoint)}；第一列 Ae1 为 ${formatVector(firstBasisImage)}，第二列 Ae2 为 ${formatVector(secondBasisImage)}。浅色网格和单位方块为原图，彩色网格和多边形为变换结果。`;
  }
}

export function defineMatrixTransformerElement() {
  if (!customElements.get('llm-matrix-transformer')) {
    customElements.define('llm-matrix-transformer', MatrixTransformerElement);
  }
}
