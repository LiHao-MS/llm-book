const chart = {
  left: 70,
  right: 710,
  top: 35,
  bottom: 300,
  alphaMax: 2,
  yMin: -0.3,
  yMax: 1,
};
let instanceCount = 0;

function formatNumber(value) {
  if (Math.abs(value) < 5e-7) {
    return '0';
  }
  return String(Number(value.toFixed(6)));
}

function softplus(value) {
  if (value > 0) {
    return value + Math.log1p(Math.exp(-value));
  }
  return Math.log1p(Math.exp(value));
}

function expectedReward(alpha) {
  return Math.tanh(alpha);
}

function klDivergence(alpha) {
  return alpha * Math.tanh(alpha) - Math.log(Math.cosh(alpha));
}

function rlhfObjective(alpha, beta) {
  return expectedReward(alpha) - beta * klDivergence(alpha);
}

function dpoLoss(alpha, beta) {
  return softplus(-2 * beta * alpha);
}

function chartX(alpha) {
  return chart.left + (chart.right - chart.left) * alpha / chart.alphaMax;
}

function chartY(value) {
  return chart.top + (chart.yMax - value) / (chart.yMax - chart.yMin)
    * (chart.bottom - chart.top);
}

function curvePath(valueAt) {
  return Array.from({ length: 81 }, (_, index) => {
    const alpha = chart.alphaMax * index / 80;
    return `${index === 0 ? 'M' : 'L'} ${chartX(alpha).toFixed(3)} ${chartY(valueAt(alpha)).toFixed(3)}`;
  }).join(' ');
}

class RlhfAlignmentElement extends HTMLElement {
  #alphaInput;
  #betaInput;
  #alphaValue;
  #betaValue;
  #preferredProbability;
  #expectedReward;
  #kl;
  #klPenalty;
  #objective;
  #dpoLoss;
  #optimalAlpha;
  #description;
  #chart;
  #rewardPath;
  #objectivePath;
  #dpoPath;
  #optimumLine;
  #rewardMarker;
  #objectiveMarker;
  #dpoMarker;

  connectedCallback() {
    if (this.#alphaInput) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    this.style.display = 'block';
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'rlhf-alignment'}-${instanceCount}`;

    this.innerHTML = `
      <figure class="interactive-widget rlhf-widget">
        <div class="interactive-controls">
          <div class="interactive-control">
            <label for="${prefix}-alpha">策略倾斜强度 α</label>
            <input id="${prefix}-alpha" data-testid="rlhf-alpha"
              type="range" min="0" max="2" step="0.05">
            <output id="${prefix}-alpha-value" data-testid="rlhf-alpha-value" for="${prefix}-alpha"></output>
          </div>
          <div class="interactive-control">
            <label for="${prefix}-beta">KL 系数 β</label>
            <input id="${prefix}-beta" data-testid="rlhf-beta"
              type="range" min="0.5" max="2" step="0.25">
            <output id="${prefix}-beta-value" data-testid="rlhf-beta-value" for="${prefix}-beta"></output>
          </div>
        </div>
        <div class="interactive-actions">
          <button type="button" data-testid="rlhf-reset">重置 α 与 β</button>
        </div>
        <dl class="interactive-results">
          <div>
            <dt>优选回答概率 pα</dt>
            <dd><output data-testid="rlhf-preferred-probability" for="${prefix}-alpha ${prefix}-beta"></output></dd>
          </div>
          <div>
            <dt>期望奖励 E[r]</dt>
            <dd><output data-testid="rlhf-expected-reward" for="${prefix}-alpha ${prefix}-beta"></output></dd>
          </div>
          <div>
            <dt>KL(πα‖πref)</dt>
            <dd><output data-testid="rlhf-kl" for="${prefix}-alpha ${prefix}-beta"></output></dd>
          </div>
          <div>
            <dt>KL 惩罚 β KL</dt>
            <dd><output data-testid="rlhf-kl-penalty" for="${prefix}-alpha ${prefix}-beta"></output></dd>
          </div>
          <div>
            <dt>RLHF 目标 Jβ</dt>
            <dd><output data-testid="rlhf-objective-value" for="${prefix}-alpha ${prefix}-beta"></output></dd>
          </div>
          <div>
            <dt>DPO 单对损失</dt>
            <dd><output data-testid="rlhf-dpo-loss" for="${prefix}-alpha ${prefix}-beta"></output></dd>
          </div>
          <div>
            <dt>理论最优 α* = 1/β</dt>
            <dd><output data-testid="rlhf-optimal-alpha" for="${prefix}-beta"></output></dd>
          </div>
        </dl>
        <svg class="interactive-chart" data-testid="rlhf-chart" viewBox="0 0 760 400"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">RLHF 目标与 DPO 偏好损失对比</title>
          <desc id="${prefix}-chart-desc"></desc>
          <g stroke="#DDE3E7" stroke-width="1">
            <line x1="70" y1="35" x2="710" y2="35"/>
            <line x1="70" y1="85.962" x2="710" y2="85.962"/>
            <line x1="70" y1="136.923" x2="710" y2="136.923"/>
            <line x1="70" y1="187.885" x2="710" y2="187.885"/>
            <line x1="70" y1="238.846" x2="710" y2="238.846"/>
            <line x1="70" y1="289.615" x2="710" y2="289.615"/>
            <line x1="230" y1="35" x2="230" y2="300"/>
            <line x1="390" y1="35" x2="390" y2="300"/>
            <line x1="550" y1="35" x2="550" y2="300"/>
          </g>
          <line x1="70" y1="300" x2="710" y2="300" stroke="#172B3A" stroke-width="1.5"/>
          <line x1="70" y1="35" x2="70" y2="300" stroke="#172B3A" stroke-width="1.5"/>
          <line x1="70" y1="238.846" x2="710" y2="238.846" stroke="#5E6A70" stroke-width="1.5"/>
          <g class="chart-label" text-anchor="middle">
            <text x="70" y="322">0</text><text x="230" y="322">0.5</text>
            <text x="390" y="322">1</text><text x="550" y="322">1.5</text><text x="710" y="322">2</text>
          </g>
          <g class="chart-label" text-anchor="end">
            <text x="60" y="39">1</text><text x="60" y="90">0.75</text>
            <text x="60" y="141">0.5</text><text x="60" y="192">0.25</text>
            <text x="60" y="243">0</text><text x="60" y="294">−0.25</text>
          </g>
          <text class="chart-label" x="390" y="344" text-anchor="middle">策略倾斜强度 α</text>
          <line data-testid="rlhf-optimum-line" y1="35" y2="300"
            stroke="#71558A" stroke-width="2" stroke-dasharray="8 6"/>
          <path data-testid="rlhf-reward-path" fill="none" stroke="#1F648C"
            stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
          <path data-testid="rlhf-objective-path" fill="none" stroke="#B55224"
            stroke-width="3" stroke-dasharray="12 7" stroke-linecap="round" stroke-linejoin="round"/>
          <path data-testid="rlhf-dpo-path" fill="none" stroke="#39704A"
            stroke-width="3" stroke-dasharray="2 5" stroke-linecap="round" stroke-linejoin="round"/>
          <circle data-testid="rlhf-reward-marker" r="6" fill="#1F648C" stroke="#FFFFFF" stroke-width="2"/>
          <circle data-testid="rlhf-objective-marker" r="6" fill="#B55224" stroke="#FFFFFF" stroke-width="2"/>
          <circle data-testid="rlhf-dpo-marker" r="6" fill="#39704A" stroke="#FFFFFF" stroke-width="2"/>
          <g class="chart-label" font-size="13">
            <line x1="85" y1="371" x2="120" y2="371" stroke="#1F648C" stroke-width="3"/>
            <text x="130" y="375">期望奖励</text>
            <line x1="285" y1="371" x2="320" y2="371" stroke="#B55224" stroke-width="3" stroke-dasharray="12 7"/>
            <text x="330" y="375">RLHF 目标</text>
            <line x1="485" y1="371" x2="520" y2="371" stroke="#39704A" stroke-width="3" stroke-dasharray="2 5"/>
            <text x="530" y="375">DPO 损失</text>
          </g>
        </svg>
        <figcaption>两个回答的参考概率均为 1/2、奖励分别为 ±1；三条曲线共用数值坐标，但目标含义不同。紫色竖线是理论最优 α* = 1/β，修改控件只会重算，不自动播放。</figcaption>
      </figure>
    `;

    this.#alphaInput = this.querySelector('[data-testid="rlhf-alpha"]');
    this.#betaInput = this.querySelector('[data-testid="rlhf-beta"]');
    this.#alphaValue = this.querySelector('[data-testid="rlhf-alpha-value"]');
    this.#betaValue = this.querySelector('[data-testid="rlhf-beta-value"]');
    this.#preferredProbability = this.querySelector('[data-testid="rlhf-preferred-probability"]');
    this.#expectedReward = this.querySelector('[data-testid="rlhf-expected-reward"]');
    this.#kl = this.querySelector('[data-testid="rlhf-kl"]');
    this.#klPenalty = this.querySelector('[data-testid="rlhf-kl-penalty"]');
    this.#objective = this.querySelector('[data-testid="rlhf-objective-value"]');
    this.#dpoLoss = this.querySelector('[data-testid="rlhf-dpo-loss"]');
    this.#optimalAlpha = this.querySelector('[data-testid="rlhf-optimal-alpha"]');
    this.#description = this.querySelector(`#${prefix}-chart-desc`);
    this.#chart = this.querySelector('[data-testid="rlhf-chart"]');
    this.#rewardPath = this.querySelector('[data-testid="rlhf-reward-path"]');
    this.#objectivePath = this.querySelector('[data-testid="rlhf-objective-path"]');
    this.#dpoPath = this.querySelector('[data-testid="rlhf-dpo-path"]');
    this.#optimumLine = this.querySelector('[data-testid="rlhf-optimum-line"]');
    this.#rewardMarker = this.querySelector('[data-testid="rlhf-reward-marker"]');
    this.#objectiveMarker = this.querySelector('[data-testid="rlhf-objective-marker"]');
    this.#dpoMarker = this.querySelector('[data-testid="rlhf-dpo-marker"]');

    this.#alphaInput.value = this.getAttribute('alpha') ?? '1';
    this.#betaInput.value = this.getAttribute('beta') ?? '1';
    this.#alphaInput.addEventListener('input', () => this.#update());
    this.#betaInput.addEventListener('input', () => this.#update());
    this.querySelector('[data-testid="rlhf-reset"]').addEventListener('click', () => {
      this.#alphaInput.value = '1';
      this.#betaInput.value = '1';
      this.#update();
    });
    this.#update();
  }

  #update() {
    const alpha = Number(this.#alphaInput.value);
    const beta = Number(this.#betaInput.value);
    const probability = 1 / (1 + Math.exp(-2 * alpha));
    const reward = expectedReward(alpha);
    const kl = klDivergence(alpha);
    const objective = rlhfObjective(alpha, beta);
    const preferenceLoss = dpoLoss(alpha, beta);
    const optimalAlpha = 1 / beta;
    const x = chartX(alpha);
    const optimumX = chartX(optimalAlpha);

    this.#alphaValue.textContent = String(alpha);
    this.#betaValue.textContent = String(beta);
    this.#preferredProbability.textContent = formatNumber(probability);
    this.#expectedReward.textContent = formatNumber(reward);
    this.#kl.textContent = formatNumber(kl);
    this.#klPenalty.textContent = formatNumber(beta * kl);
    this.#objective.textContent = formatNumber(objective);
    this.#dpoLoss.textContent = formatNumber(preferenceLoss);
    this.#optimalAlpha.textContent = formatNumber(optimalAlpha);

    this.#rewardPath.setAttribute('d', curvePath(expectedReward));
    this.#objectivePath.setAttribute('d', curvePath((value) => rlhfObjective(value, beta)));
    this.#dpoPath.setAttribute('d', curvePath((value) => dpoLoss(value, beta)));
    this.#optimumLine.setAttribute('x1', optimumX);
    this.#optimumLine.setAttribute('x2', optimumX);
    this.#rewardMarker.setAttribute('cx', x);
    this.#rewardMarker.setAttribute('cy', chartY(reward));
    this.#objectiveMarker.setAttribute('cx', x);
    this.#objectiveMarker.setAttribute('cy', chartY(objective));
    this.#dpoMarker.setAttribute('cx', x);
    this.#dpoMarker.setAttribute('cy', chartY(preferenceLoss));
    this.#chart.dataset.signature = `${alpha}|${beta}`;
    this.#description.textContent = `当前 α 为 ${alpha}，β 为 ${beta}。优选回答概率 ${formatNumber(probability)}，期望奖励 ${formatNumber(reward)}，KL 为 ${formatNumber(kl)}，KL 惩罚为 ${formatNumber(beta * kl)}，RLHF 目标为 ${formatNumber(objective)}，DPO 单对损失为 ${formatNumber(preferenceLoss)}；理论最优 α 为 ${formatNumber(optimalAlpha)}。实线为期望奖励，长虚线为 RLHF 目标，点线为 DPO 损失。`;
  }
}

export function defineRlhfAlignmentElement() {
  if (!customElements.get('llm-rlhf-alignment')) {
    customElements.define('llm-rlhf-alignment', RlhfAlignmentElement);
  }
}
