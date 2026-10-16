const transitions = {
  A: [
    { action: 'a₀', next: 'A', reward: 0 },
    { action: 'a₁', next: 'B', reward: 0 },
  ],
  B: [
    { action: 'a₀', next: 'A', reward: 0 },
    { action: 'a₁', next: 'B', reward: 1 },
  ],
};
const states = ['A', 'B'];
const initialValues = { A: 0, B: 0 };
let instanceCount = 0;

function formatNumber(value) {
  if (Math.abs(value) < 1e-12) {
    return '0';
  }
  return String(Number(value.toFixed(6)));
}

function actionValues(values, gamma) {
  return Object.fromEntries(states.map((state) => [
    state,
    transitions[state].map(({ next, reward }) => reward + gamma * values[next]),
  ]));
}

function greedyPolicy(values, gamma) {
  const scores = actionValues(values, gamma);
  return Object.fromEntries(states.map((state) => {
    const best = Math.max(...scores[state]);
    const bestActions = transitions[state]
      .filter((_, index) => Math.abs(scores[state][index] - best) < 1e-12)
      .map(({ action }) => action);
    return [state, bestActions.length === 1 ? bestActions[0] : `${bestActions.join(' 与 ')} 并列`];
  }));
}

function iterate(values, gamma) {
  const scores = actionValues(values, gamma);
  return Object.fromEntries(states.map((state) => [state, Math.max(...scores[state])]));
}

class RlFoundationsElement extends HTMLElement {
  #barA;
  #barB;
  #chartValueA;
  #chartValueB;
  #description;
  #gamma;
  #gammaValue;
  #iteration = 0;
  #iterationOutput;
  #policyA;
  #policyB;
  #valueA;
  #valueB;
  #values = { ...initialValues };

  connectedCallback() {
    if (this.#gamma) {
      return;
    }
    this.style.display = 'block';
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'rl-foundations'}-${instanceCount}`;

    this.innerHTML = `
      <figure class="interactive-widget rl-widget">
        <div class="interactive-controls">
          <div class="interactive-control">
            <label for="${prefix}-gamma">折扣因子 γ</label>
            <input id="${prefix}-gamma" data-testid="rl-gamma"
              type="range" min="0" max="0.9" step="0.05">
            <output id="${prefix}-gamma-value" data-testid="rl-gamma-value" for="${prefix}-gamma"></output>
          </div>
        </div>
        <div class="interactive-actions">
          <button type="button" data-testid="rl-reset">重置价值</button>
          <button type="button" data-testid="rl-next">价值迭代一步</button>
        </div>
        <dl class="interactive-results">
          <div>
            <dt>迭代次数 k</dt>
            <dd><output data-testid="rl-iteration" for="${prefix}-gamma"></output></dd>
          </div>
          <div>
            <dt>状态价值 Vₖ(A)</dt>
            <dd><output data-testid="rl-value-a" for="${prefix}-gamma"></output></dd>
          </div>
          <div>
            <dt>状态价值 Vₖ(B)</dt>
            <dd><output data-testid="rl-value-b" for="${prefix}-gamma"></output></dd>
          </div>
          <div>
            <dt>A 的贪心动作</dt>
            <dd><output data-testid="rl-policy-a" for="${prefix}-gamma"></output></dd>
          </div>
          <div>
            <dt>B 的贪心动作</dt>
            <dd><output data-testid="rl-policy-b" for="${prefix}-gamma"></output></dd>
          </div>
        </dl>
        <svg class="interactive-chart" data-testid="rl-value-chart" viewBox="0 0 720 330"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">两状态 MDP 的价值与贪心策略</title>
          <desc id="${prefix}-chart-desc"></desc>
          <text x="34" y="36" font-size="16">条长按收敛值 V*(B) = 1/(1−γ) 归一化</text>
          <text x="110" y="101" text-anchor="end">Vₖ(A)</text>
          <rect x="130" y="76" width="400" height="36" rx="4" fill="#DDE3E7" />
          <rect data-testid="rl-bar-a" x="130" y="76" width="0" height="36" rx="4" fill="#1F648C" />
          <text data-testid="rl-chart-value-a" x="550" y="101">0</text>
          <text x="110" y="177" text-anchor="end">Vₖ(B)</text>
          <rect x="130" y="152" width="400" height="36" rx="4" fill="#DDE3E7" />
          <rect data-testid="rl-bar-b" x="130" y="152" width="0" height="36" rx="4" fill="#C65D21" />
          <text data-testid="rl-chart-value-b" x="550" y="177">0</text>
          <path d="M 130 222 H 530" stroke="#5E6A70" stroke-width="1.5" />
          <text x="130" y="246" text-anchor="middle" class="chart-label">0</text>
          <text x="530" y="246" text-anchor="middle" class="chart-label">V*(B)</text>
          <text x="34" y="294" font-size="15">实色条表示当前 Vₖ；动作由当前价值的一步前瞻评分决定。</text>
        </svg>
        <figcaption>确定性转移：A 的 a₀/a₁ 分别到 A/B，B 的 a₀/a₁ 分别到 A/B；仅 B 执行 a₁ 得奖励 1。修改 γ 会重新从 V₀ = 0 开始。</figcaption>
      </figure>
    `;

    this.#gamma = this.querySelector('[data-testid="rl-gamma"]');
    this.#gammaValue = this.querySelector('[data-testid="rl-gamma-value"]');
    this.#iterationOutput = this.querySelector('[data-testid="rl-iteration"]');
    this.#valueA = this.querySelector('[data-testid="rl-value-a"]');
    this.#valueB = this.querySelector('[data-testid="rl-value-b"]');
    this.#policyA = this.querySelector('[data-testid="rl-policy-a"]');
    this.#policyB = this.querySelector('[data-testid="rl-policy-b"]');
    this.#barA = this.querySelector('[data-testid="rl-bar-a"]');
    this.#barB = this.querySelector('[data-testid="rl-bar-b"]');
    this.#chartValueA = this.querySelector('[data-testid="rl-chart-value-a"]');
    this.#chartValueB = this.querySelector('[data-testid="rl-chart-value-b"]');
    this.#description = this.querySelector(`#${prefix}-chart-desc`);
    this.#gamma.value = this.getAttribute('gamma') ?? '0.5';

    this.#gamma.addEventListener('input', () => {
      this.#iteration = 0;
      this.#values = { ...initialValues };
      this.#render();
    });
    this.querySelector('[data-testid="rl-next"]').addEventListener('click', () => {
      this.#values = iterate(this.#values, Number(this.#gamma.value));
      this.#iteration += 1;
      this.#render();
    });
    this.querySelector('[data-testid="rl-reset"]').addEventListener('click', () => {
      this.#iteration = 0;
      this.#values = { ...initialValues };
      this.#render();
    });
    this.#render();
  }

  #render() {
    const gamma = Number(this.#gamma.value);
    const policy = greedyPolicy(this.#values, gamma);
    const scale = 1 / (1 - gamma);
    const widthA = Number((400 * this.#values.A / scale).toFixed(6));
    const widthB = Number((400 * this.#values.B / scale).toFixed(6));

    this.#gammaValue.textContent = this.#gamma.value;
    this.#iterationOutput.textContent = String(this.#iteration);
    this.#valueA.textContent = formatNumber(this.#values.A);
    this.#valueB.textContent = formatNumber(this.#values.B);
    this.#policyA.textContent = policy.A;
    this.#policyB.textContent = policy.B;
    this.#barA.setAttribute('width', widthA);
    this.#barB.setAttribute('width', widthB);
    this.#chartValueA.textContent = formatNumber(this.#values.A);
    this.#chartValueB.textContent = formatNumber(this.#values.B);
    this.#description.textContent = `第 ${this.#iteration} 次迭代，折扣因子 γ = ${this.#gamma.value}，V(A) = ${formatNumber(this.#values.A)}，V(B) = ${formatNumber(this.#values.B)}。状态 A 的贪心动作为 ${policy.A}；状态 B 的贪心动作为 ${policy.B}。`;
  }
}

export function defineRlFoundationsElement() {
  if (!customElements.get('llm-rl-foundations')) {
    customElements.define('llm-rl-foundations', RlFoundationsElement);
  }
}
