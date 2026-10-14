import { crossEntropy, perplexity } from '../lib/probability.js';

const chart = {
  bottom: 290,
  height: 270,
  left: 60,
  width: 620,
};

class CrossEntropyElement extends HTMLElement {
  #description;
  #input;
  #loss;
  #maxLoss;
  #perplexity;
  #point;
  #pointLabel;
  #probabilityValue;

  connectedCallback() {
    if (this.#input) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }

    this.innerHTML = `
      <figure class="interactive-widget cross-entropy-widget">
        <div class="interactive-control">
          <label for="ce-probability-input">目标 Token 概率</label>
          <input id="ce-probability-input" data-testid="ce-probability" type="range"
            min="0.01" max="1" step="0.01">
          <output id="ce-probability-value" for="ce-probability-input"></output>
        </div>
        <dl class="interactive-results">
          <div>
            <dt>自然对数损失（nat）</dt>
            <dd><output data-testid="ce-loss" for="ce-probability-input"></output></dd>
          </div>
          <div>
            <dt>单 Token 困惑度</dt>
            <dd><output data-testid="ce-perplexity" for="ce-probability-input"></output></dd>
          </div>
        </dl>
        <svg class="interactive-chart" viewBox="0 0 720 340" role="img"
          aria-labelledby="ce-chart-title" aria-describedby="ce-chart-desc">
          <title id="ce-chart-title">交叉熵损失曲线</title>
          <desc id="ce-chart-desc"></desc>
          <path class="chart-axes" d="M 60 20 V 290 H 680" />
          <text class="chart-label" x="60" y="318" text-anchor="middle">0</text>
          <text class="chart-label" x="370" y="318" text-anchor="middle">0.5</text>
          <text class="chart-label" x="680" y="318" text-anchor="middle">1.0</text>
          <text class="chart-label" x="370" y="338" text-anchor="middle">目标 Token 概率 p</text>
          <text class="chart-label" x="44" y="294" text-anchor="end">0</text>
          <text class="chart-label" x="44" y="25" text-anchor="end">4.605</text>
          <text class="chart-label" x="16" y="155" text-anchor="middle"
            transform="rotate(-90 16 155)">损失 / nat</text>
          <path class="cross-entropy-curve" />
          <circle class="cross-entropy-point" r="6" />
          <text class="cross-entropy-point-label" x="72" y="42"></text>
        </svg>
        <figcaption>目标概率越接近零，交叉熵损失越大；目标概率为一时损失为零。</figcaption>
      </figure>
    `;

    this.#input = this.querySelector('[data-testid="ce-probability"]');
    this.#probabilityValue = this.querySelector('#ce-probability-value');
    this.#loss = this.querySelector('[data-testid="ce-loss"]');
    this.#perplexity = this.querySelector('[data-testid="ce-perplexity"]');
    this.#description = this.querySelector('#ce-chart-desc');
    this.#point = this.querySelector('.cross-entropy-point');
    this.#pointLabel = this.querySelector('.cross-entropy-point-label');
    this.#input.value = this.getAttribute('probability') ?? '0.5';
    this.#maxLoss = crossEntropy(Number(this.#input.min));

    const points = [];
    for (let index = 0; index <= 100; index += 1) {
      const probability = Number(this.#input.min)
        + (index / 100) * (Number(this.#input.max) - Number(this.#input.min));
      const loss = crossEntropy(probability);
      points.push(`${this.#x(probability)},${this.#y(loss)}`);
    }
    this.querySelector('.cross-entropy-curve').setAttribute('d', `M ${points.join(' L ')}`);

    this.#input.addEventListener('input', () => this.#update());
    this.#update();
  }

  #x(probability) {
    return (chart.left + probability * chart.width).toFixed(2);
  }

  #y(loss) {
    return (chart.bottom - (loss / this.#maxLoss) * chart.height).toFixed(2);
  }

  #update() {
    const probability = Number(this.#input.value);
    const loss = crossEntropy(probability);
    const tokenPerplexity = perplexity(probability);

    this.#probabilityValue.textContent = this.#input.value;
    this.#loss.textContent = String(loss);
    this.#perplexity.textContent = String(tokenPerplexity);
    this.#point.setAttribute('cx', this.#x(probability));
    this.#point.setAttribute('cy', this.#y(loss));
    this.#pointLabel.textContent = `p = ${this.#input.value}`;
    this.#description.textContent = `当前目标概率为 ${probability}，自然对数损失为 ${loss} nat，单 Token 困惑度为 ${tokenPerplexity}。曲线显示损失随概率增大而下降。`;
  }
}

export function defineCrossEntropyElement() {
  if (!customElements.get('llm-cross-entropy')) {
    customElements.define('llm-cross-entropy', CrossEntropyElement);
  }
}
