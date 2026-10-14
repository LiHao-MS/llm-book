import { temperatureProbabilities, topP } from '../lib/probability.js';

const svgNamespace = 'http://www.w3.org/2000/svg';

function createSvgElement(name, attributes = {}) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [attribute, value] of Object.entries(attributes)) {
    element.setAttribute(attribute, String(value));
  }
  return element;
}

class DecodingExplorerElement extends HTMLElement {
  #description;
  #distribution;
  #keptCount;
  #probabilities;
  #temperatureInput;
  #temperatureValue;
  #tokens;
  #topPInput;
  #topPValue;

  connectedCallback() {
    if (this.#distribution) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }

    this.#tokens = (this.getAttribute('tokens') ?? '的,模型,学习').split(',');
    this.#probabilities = (this.getAttribute('probabilities') ?? '0.2,0.5,0.3')
      .split(',')
      .map(Number);

    this.innerHTML = `
      <figure class="interactive-widget decoding-widget">
        <div class="interactive-controls">
          <div class="interactive-control">
            <label for="decoding-temperature-input">温度</label>
            <input id="decoding-temperature-input" data-testid="decoding-temperature"
              type="range" min="0.1" max="2" step="0.1">
            <output id="decoding-temperature-value" for="decoding-temperature-input"></output>
          </div>
          <div class="interactive-control">
            <label for="decoding-top-p-input">Top-p</label>
            <input id="decoding-top-p-input" data-testid="decoding-top-p"
              type="range" min="0.1" max="1" step="0.05">
            <output id="decoding-top-p-value" for="decoding-top-p-input"></output>
          </div>
        </div>
        <p class="decoding-summary">保留 Token 数：
          <output data-testid="decoding-kept-count"
            for="decoding-temperature-input decoding-top-p-input"></output>
        </p>
        <svg class="interactive-chart" data-testid="decoding-distribution"
          viewBox="0 0 720 350" role="img"
          aria-labelledby="decoding-chart-title" aria-describedby="decoding-chart-desc">
        </svg>
        <figcaption>概率条保持 Token 原顺序；被 Top-p 截断的 Token 仍显示并标记为排除。</figcaption>
      </figure>
    `;

    this.#temperatureInput = this.querySelector('[data-testid="decoding-temperature"]');
    this.#topPInput = this.querySelector('[data-testid="decoding-top-p"]');
    this.#temperatureValue = this.querySelector('#decoding-temperature-value');
    this.#topPValue = this.querySelector('#decoding-top-p-value');
    this.#keptCount = this.querySelector('[data-testid="decoding-kept-count"]');
    this.#distribution = this.querySelector('[data-testid="decoding-distribution"]');
    this.#temperatureInput.value = this.getAttribute('temperature') ?? '1';
    this.#topPInput.value = this.getAttribute('top-p') ?? '1';

    this.#temperatureInput.addEventListener('input', () => this.#update());
    this.#topPInput.addEventListener('input', () => this.#update());
    this.#update();
  }

  #update() {
    const temperature = Number(this.#temperatureInput.value);
    const threshold = Number(this.#topPInput.value);
    const tempered = temperatureProbabilities(this.#probabilities, temperature);
    const result = topP(tempered, threshold);
    const keptTotal = result.kept.filter(Boolean).length;

    this.#temperatureValue.textContent = this.#temperatureInput.value;
    this.#topPValue.textContent = this.#topPInput.value;
    this.#keptCount.textContent = String(keptTotal);
    this.#renderDistribution(temperature, threshold, result);
  }

  #renderDistribution(temperature, threshold, result) {
    const title = createSvgElement('title', { id: 'decoding-chart-title' });
    title.textContent = '温度与 Top-p 解码分布';
    this.#description = createSvgElement('desc', { id: 'decoding-chart-desc' });
    const descriptions = [];

    const groups = this.#tokens.map((token, index) => {
      const kept = result.kept[index];
      const probability = result.probabilities[index];
      const y = 30 + index * 100;
      const group = createSvgElement('g', {
        class: `decoding-token${kept ? '' : ' is-excluded'}`,
        'data-token': token,
        'data-kept': kept,
        'data-probability': probability,
      });
      const label = createSvgElement('text', { x: 20, y: y + 25 });
      label.textContent = token;
      const track = createSvgElement('rect', {
        class: 'decoding-bar-track',
        x: 120,
        y,
        width: 410,
        height: 36,
        rx: 4,
      });
      const bar = createSvgElement('rect', {
        class: 'decoding-bar',
        x: 120,
        y,
        width: 410 * probability,
        height: 36,
        rx: 4,
      });
      const value = createSvgElement('text', { x: 550, y: y + 25 });
      value.textContent = `q = ${probability.toFixed(3)}${kept ? '' : '（排除）'}`;
      descriptions.push(`${token}：${probability.toFixed(3)}${kept ? '' : '（排除）'}`);
      group.append(label, track, bar, value);
      return group;
    });

    this.#description.textContent = `温度 ${temperature}，Top-p ${threshold}，保留 ${result.kept.filter(Boolean).length} 个 Token。${descriptions.join('；')}。`;
    this.#distribution.replaceChildren(title, this.#description, ...groups);
  }
}

export function defineDecodingExplorerElement() {
  if (!customElements.get('llm-decoding-explorer')) {
    customElements.define('llm-decoding-explorer', DecodingExplorerElement);
  }
}
