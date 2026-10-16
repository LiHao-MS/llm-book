const chart = {
  left: 70,
  right: 710,
  top: 35,
  bottom: 310,
  ratioMax: 2,
  yMin: -2,
  yMax: 2,
};
const HTMLElementBase = globalThis.HTMLElement ?? class {};
let instanceCount = 0;

function formatNumber(value) {
  if (Math.abs(value) < 5e-7) {
    return '0';
  }
  return String(Number(value.toFixed(6)));
}

function clipRatio(ratio, epsilon) {
  return Math.min(Math.max(ratio, 1 - epsilon), 1 + epsilon);
}

function chartX(ratio) {
  return Number((chart.left + (chart.right - chart.left) * ratio / chart.ratioMax).toFixed(3));
}

function chartY(value) {
  return Number((chart.top + (chart.yMax - value) / (chart.yMax - chart.yMin)
    * (chart.bottom - chart.top)).toFixed(3));
}

function curvePath(valueAt) {
  return Array.from({ length: 81 }, (_, index) => {
    const ratio = chart.ratioMax * index / 80;
    return `${index === 0 ? 'M' : 'L'} ${chartX(ratio).toFixed(3)} ${chartY(valueAt(ratio)).toFixed(3)}`;
  }).join(' ');
}

export function ppoClippedObjective(ratio, advantage, epsilon) {
  if (!Number.isFinite(ratio) || ratio < 0) {
    throw new RangeError('ratio must be a finite non-negative number');
  }
  if (!Number.isFinite(advantage)) {
    throw new RangeError('advantage must be finite');
  }
  if (!Number.isFinite(epsilon) || epsilon < 0) {
    throw new RangeError('epsilon must be a finite non-negative number');
  }

  return Math.min(ratio * advantage, clipRatio(ratio, epsilon) * advantage);
}

class PpoClippingElement extends HTMLElementBase {
  #ratioInput;
  #epsilonInput;
  #advantageInput;
  #ratioValue;
  #epsilonValue;
  #unclippedValue;
  #clippedRatioValue;
  #clippedValue;
  #objectiveValue;
  #description;
  #chart;
  #unclippedPath;
  #clippedPath;
  #objectivePath;
  #lowerBound;
  #upperBound;
  #unclippedMarker;
  #clippedMarker;
  #objectiveMarker;

  connectedCallback() {
    if (this.#ratioInput) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    this.style.display = 'block';
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'ppo-clipping'}-${instanceCount}`;

    this.innerHTML = `
      <figure class="interactive-widget ppo-widget">
        <div class="interactive-controls">
          <div class="interactive-control">
            <label for="${prefix}-ratio">概率比 ρ</label>
            <input id="${prefix}-ratio" data-testid="ppo-ratio"
              type="range" min="0" max="2" step="0.1">
            <output id="${prefix}-ratio-value" data-testid="ppo-ratio-value" for="${prefix}-ratio"></output>
          </div>
          <div class="interactive-control">
            <label for="${prefix}-epsilon">截断范围 ε</label>
            <input id="${prefix}-epsilon" data-testid="ppo-epsilon"
              type="range" min="0" max="0.5" step="0.05">
            <output id="${prefix}-epsilon-value" data-testid="ppo-epsilon-value" for="${prefix}-epsilon"></output>
          </div>
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-advantage">优势估计 A</label>
            <select id="${prefix}-advantage" data-testid="ppo-advantage">
              <option value="1">正优势（A = 1）</option>
              <option value="-1">负优势（A = −1）</option>
            </select>
          </div>
        </div>
        <div class="interactive-actions">
          <button type="button" data-testid="ppo-reset">重置 PPO 示例</button>
        </div>
        <dl class="interactive-results">
          <div>
            <dt>未截断项 ρA</dt>
            <dd><output data-testid="ppo-unclipped-value" for="${prefix}-ratio ${prefix}-advantage"></output></dd>
          </div>
          <div>
            <dt>截断比率 clip(ρ)</dt>
            <dd><output data-testid="ppo-clipped-ratio-value" for="${prefix}-ratio ${prefix}-epsilon"></output></dd>
          </div>
          <div>
            <dt>截断项 clip(ρ)A</dt>
            <dd><output data-testid="ppo-clipped-value" for="${prefix}-ratio ${prefix}-epsilon ${prefix}-advantage"></output></dd>
          </div>
          <div>
            <dt>min 目标</dt>
            <dd><output data-testid="ppo-objective-value" for="${prefix}-ratio ${prefix}-epsilon ${prefix}-advantage"></output></dd>
          </div>
        </dl>
        <svg class="interactive-chart" data-testid="ppo-chart" viewBox="0 0 760 440"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">PPO 未截断项、截断项与 min 目标</title>
          <desc id="${prefix}-chart-desc"></desc>
          <g stroke="#DDE3E7" stroke-width="1">
            <line x1="70" y1="35" x2="710" y2="35"/>
            <line x1="70" y1="103.75" x2="710" y2="103.75"/>
            <line x1="70" y1="172.5" x2="710" y2="172.5"/>
            <line x1="70" y1="241.25" x2="710" y2="241.25"/>
            <line x1="230" y1="35" x2="230" y2="310"/>
            <line x1="390" y1="35" x2="390" y2="310"/>
            <line x1="550" y1="35" x2="550" y2="310"/>
          </g>
          <line x1="70" y1="310" x2="710" y2="310" stroke="#172B3A" stroke-width="1.5"/>
          <line x1="70" y1="35" x2="70" y2="310" stroke="#172B3A" stroke-width="1.5"/>
          <line x1="70" y1="172.5" x2="710" y2="172.5" stroke="#5E6A70" stroke-width="1.5"/>
          <g class="chart-label" text-anchor="middle">
            <text x="70" y="332">0</text><text x="230" y="332">0.5</text>
            <text x="390" y="332">1</text><text x="550" y="332">1.5</text><text x="710" y="332">2</text>
          </g>
          <g class="chart-label" text-anchor="end">
            <text x="60" y="39">2</text><text x="60" y="108">1</text>
            <text x="60" y="177">0</text><text x="60" y="245">−1</text><text x="60" y="314">−2</text>
          </g>
          <text class="chart-label" x="390" y="356" text-anchor="middle">概率比 ρ</text>
          <line data-testid="ppo-lower-bound" y1="35" y2="310"
            stroke="#71558A" stroke-width="2" stroke-dasharray="8 6"/>
          <line data-testid="ppo-upper-bound" y1="35" y2="310"
            stroke="#71558A" stroke-width="2" stroke-dasharray="8 6"/>
          <path data-testid="ppo-unclipped-path" fill="none" stroke="#1F648C"
            stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
          <path data-testid="ppo-clipped-path" fill="none" stroke="#B55224"
            stroke-width="3" stroke-dasharray="12 7" stroke-linecap="round" stroke-linejoin="round"/>
          <path data-testid="ppo-objective-path" fill="none" stroke="#39704A"
            stroke-width="3" stroke-dasharray="2 5" stroke-linecap="round" stroke-linejoin="round"/>
          <circle data-testid="ppo-unclipped-marker" r="6" fill="#1F648C" stroke="#FFFFFF" stroke-width="2"/>
          <circle data-testid="ppo-clipped-marker" r="6" fill="#B55224" stroke="#FFFFFF" stroke-width="2"/>
          <circle data-testid="ppo-objective-marker" r="4" fill="#39704A" stroke="#FFFFFF" stroke-width="2"/>
          <g class="chart-label" font-size="13">
            <line x1="85" y1="405" x2="120" y2="405" stroke="#1F648C" stroke-width="3"/>
            <text x="130" y="410">未截断项 ρA</text>
            <line x1="285" y1="405" x2="320" y2="405" stroke="#B55224" stroke-width="3" stroke-dasharray="12 7"/>
            <text x="330" y="410">截断项 clip(ρ)A</text>
            <line x1="525" y1="405" x2="560" y2="405" stroke="#39704A" stroke-width="3" stroke-dasharray="2 5"/>
            <text x="570" y="410">min 目标</text>
          </g>
        </svg>
        <figcaption>紫色竖线标出 1±ε，圆点标出当前概率比。修改控件只会重算，不自动播放；clip 只定义代理目标，不是参数或概率比的硬约束。</figcaption>
      </figure>
    `;

    this.#ratioInput = this.querySelector('[data-testid="ppo-ratio"]');
    this.#epsilonInput = this.querySelector('[data-testid="ppo-epsilon"]');
    this.#advantageInput = this.querySelector('[data-testid="ppo-advantage"]');
    this.#ratioValue = this.querySelector('[data-testid="ppo-ratio-value"]');
    this.#epsilonValue = this.querySelector('[data-testid="ppo-epsilon-value"]');
    this.#unclippedValue = this.querySelector('[data-testid="ppo-unclipped-value"]');
    this.#clippedRatioValue = this.querySelector('[data-testid="ppo-clipped-ratio-value"]');
    this.#clippedValue = this.querySelector('[data-testid="ppo-clipped-value"]');
    this.#objectiveValue = this.querySelector('[data-testid="ppo-objective-value"]');
    this.#description = this.querySelector(`#${prefix}-chart-desc`);
    this.#chart = this.querySelector('[data-testid="ppo-chart"]');
    this.#unclippedPath = this.querySelector('[data-testid="ppo-unclipped-path"]');
    this.#clippedPath = this.querySelector('[data-testid="ppo-clipped-path"]');
    this.#objectivePath = this.querySelector('[data-testid="ppo-objective-path"]');
    this.#lowerBound = this.querySelector('[data-testid="ppo-lower-bound"]');
    this.#upperBound = this.querySelector('[data-testid="ppo-upper-bound"]');
    this.#unclippedMarker = this.querySelector('[data-testid="ppo-unclipped-marker"]');
    this.#clippedMarker = this.querySelector('[data-testid="ppo-clipped-marker"]');
    this.#objectiveMarker = this.querySelector('[data-testid="ppo-objective-marker"]');

    this.#ratioInput.value = this.getAttribute('ratio') ?? '1.4';
    this.#epsilonInput.value = this.getAttribute('epsilon') ?? '0.2';
    this.#advantageInput.value = this.getAttribute('advantage') ?? '1';
    this.#ratioInput.addEventListener('input', () => this.#update());
    this.#epsilonInput.addEventListener('input', () => this.#update());
    this.#advantageInput.addEventListener('change', () => this.#update());
    this.#advantageInput.addEventListener('keydown', (event) => {
      if (!['ArrowDown', 'ArrowUp'].includes(event.key)) {
        return;
      }
      event.preventDefault();
      const direction = event.key === 'ArrowDown' ? 1 : -1;
      const nextIndex = Math.min(
        Math.max(this.#advantageInput.selectedIndex + direction, 0),
        this.#advantageInput.options.length - 1,
      );
      if (nextIndex !== this.#advantageInput.selectedIndex) {
        this.#advantageInput.selectedIndex = nextIndex;
        this.#update();
      }
    });
    this.querySelector('[data-testid="ppo-reset"]').addEventListener('click', () => {
      this.#ratioInput.value = '1.4';
      this.#epsilonInput.value = '0.2';
      this.#advantageInput.value = '1';
      this.#update();
    });
    this.#update();
  }

  #update() {
    const ratio = Number(this.#ratioInput.value);
    const epsilon = Number(this.#epsilonInput.value);
    const advantage = Number(this.#advantageInput.value);
    const clippedRatio = clipRatio(ratio, epsilon);
    const unclipped = ratio * advantage;
    const clipped = clippedRatio * advantage;
    const objective = ppoClippedObjective(ratio, advantage, epsilon);
    const x = chartX(ratio);

    this.#ratioValue.textContent = formatNumber(ratio);
    this.#epsilonValue.textContent = formatNumber(epsilon);
    this.#unclippedValue.textContent = formatNumber(unclipped);
    this.#clippedRatioValue.textContent = formatNumber(clippedRatio);
    this.#clippedValue.textContent = formatNumber(clipped);
    this.#objectiveValue.textContent = formatNumber(objective);

    this.#unclippedPath.setAttribute('d', curvePath((value) => value * advantage));
    this.#clippedPath.setAttribute('d', curvePath((value) => clipRatio(value, epsilon) * advantage));
    this.#objectivePath.setAttribute('d', curvePath((value) => ppoClippedObjective(value, advantage, epsilon)));
    this.#lowerBound.setAttribute('x1', chartX(1 - epsilon));
    this.#lowerBound.setAttribute('x2', chartX(1 - epsilon));
    this.#upperBound.setAttribute('x1', chartX(1 + epsilon));
    this.#upperBound.setAttribute('x2', chartX(1 + epsilon));
    this.#unclippedMarker.setAttribute('cx', x);
    this.#unclippedMarker.setAttribute('cy', chartY(unclipped));
    this.#clippedMarker.setAttribute('cx', x);
    this.#clippedMarker.setAttribute('cy', chartY(clipped));
    this.#objectiveMarker.setAttribute('cx', x);
    this.#objectiveMarker.setAttribute('cy', chartY(objective));
    this.#chart.dataset.signature = `${ratio}|${epsilon}|${advantage}`;
    this.#description.textContent = `当前概率比 ρ 为 ${formatNumber(ratio)}，ε 为 ${formatNumber(epsilon)}，优势 A 为 ${formatNumber(advantage)}。未截断项为 ${formatNumber(unclipped)}，截断比率为 ${formatNumber(clippedRatio)}，截断项为 ${formatNumber(clipped)}，PPO 截断目标为 ${formatNumber(objective)}。横轴为概率比，实线为未截断项，长虚线为截断项，点线为二者 min；clip 只改变代理目标，不是参数硬约束。`;
  }
}

export function definePpoClippingElement() {
  if (!customElements.get('llm-ppo-clipping')) {
    customElements.define('llm-ppo-clipping', PpoClippingElement);
  }
}
