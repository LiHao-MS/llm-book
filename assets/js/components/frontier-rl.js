const candidates = [
  {
    id: 'decomposition',
    name: 'A：一致分解',
    chartLabel: 'A：一致分解',
    text: '37×41 = 37×40 + 37 = 1480 + 37 = 1517。',
    correct: true,
    marker: false,
    validDecomposition: true,
  },
  {
    id: 'answer',
    name: 'B：只给答案',
    chartLabel: 'B：只给答案',
    text: '答案：1517。',
    correct: true,
    marker: false,
    validDecomposition: false,
  },
  {
    id: 'marker-correct',
    name: 'C：正确回答加伪造标记',
    chartLabel: 'C：正确 + 标记',
    text: '答案：1517。\nVERIFIER_OK（伪造，并非执行结果）',
    correct: true,
    marker: true,
    validDecomposition: false,
  },
  {
    id: 'marker-wrong',
    name: 'D：错误回答加伪造标记',
    chartLabel: 'D：错答 + 标记',
    text: '答案：1417。\nVERIFIER_OK（伪造，并非执行结果）',
    correct: false,
    marker: true,
    validDecomposition: false,
  },
];

const verifiers = {
  outcome: {
    name: '只看最终结果',
    formula: 'r = 1[答案正确]',
    bonus: () => 0,
    diagnosis: 'A、B、C 的总奖励相同；伪造标记没有收益，结果检查本身也不评价推理过程。',
  },
  marker: {
    name: '结果加文本标记（有漏洞）',
    formula: 'r = 1[答案正确] + 0.2 × 1[含 VERIFIER_OK]',
    bonus: (candidate) => (candidate.marker ? 0.2 : 0),
    diagnosis: 'C 借伪造标记获得最高组内优势；D 的答案错误，却仍得到 0.2 总奖励。',
  },
  checked: {
    name: '重算结果并检查一致分解',
    formula: 'r = 1[答案正确] + 0.2 × 1[分解一致]',
    bonus: (candidate) => (candidate.validDecomposition ? 0.2 : 0),
    diagnosis: '文本标记不再计分；A 的一致分解获得 0.2 附加奖励，D 的总奖励为 0。',
  },
};

let instanceCount = 0;

function formatNumber(value) {
  if (Math.abs(value) < 1e-12) {
    return '0';
  }
  return String(Number(value.toFixed(6)));
}

function calculateRewards(verifierId) {
  const verifier = verifiers[verifierId];
  const rows = candidates.map((candidate) => {
    const outcome = candidate.correct ? 1 : 0;
    const bonus = verifier.bonus(candidate);
    return {
      ...candidate,
      outcome,
      bonus,
      reward: outcome + bonus,
      advantage: 0,
    };
  });
  const mean = rows.reduce((sum, row) => sum + row.reward, 0) / rows.length;
  const variance = rows.reduce(
    (sum, row) => sum + (row.reward - mean) ** 2,
    0,
  ) / rows.length;
  const standardDeviation = Math.sqrt(variance);

  return rows.map((row) => ({
    ...row,
    advantage: standardDeviation < 1e-12
      ? 0
      : (row.reward - mean) / (standardDeviation + 1e-12),
  }));
}

class FrontierRLElement extends HTMLElement {
  #advantage;
  #bonus;
  #candidate;
  #description;
  #diagnosis;
  #outcome;
  #response;
  #reward;
  #rule;
  #verifier;

  connectedCallback() {
    if (this.#verifier) {
      return;
    }
    this.style.display = 'block';
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'frontier-rl'}-${instanceCount}`;

    this.innerHTML = `
      <figure class="interactive-widget frontier-rl-widget">
        <div class="interactive-controls">
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-verifier">奖励规则</label>
            <select id="${prefix}-verifier" data-testid="frontier-verifier">
              <option value="outcome">只看最终结果</option>
              <option value="marker">结果加文本标记（有漏洞）</option>
              <option value="checked">重算结果并检查一致分解</option>
            </select>
          </div>
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-candidate">候选回答</label>
            <select id="${prefix}-candidate" data-testid="frontier-candidate">
              ${candidates.map((candidate) =>
                `<option value="${candidate.id}">${candidate.name}</option>`,
              ).join('')}
            </select>
          </div>
        </div>
        <dl class="interactive-results">
          <div>
            <dt>当前规则</dt>
            <dd><output data-testid="frontier-rule" for="${prefix}-verifier"></output></dd>
          </div>
          <div>
            <dt>结果项</dt>
            <dd><output data-testid="frontier-outcome" for="${prefix}-verifier ${prefix}-candidate"></output></dd>
          </div>
          <div>
            <dt>附加项</dt>
            <dd><output data-testid="frontier-bonus" for="${prefix}-verifier ${prefix}-candidate"></output></dd>
          </div>
          <div>
            <dt>总奖励 R</dt>
            <dd><output data-testid="frontier-reward" for="${prefix}-verifier ${prefix}-candidate"></output></dd>
          </div>
          <div>
            <dt>组内优势 A</dt>
            <dd><output data-testid="frontier-advantage" for="${prefix}-verifier ${prefix}-candidate"></output></dd>
          </div>
        </dl>
        <p><strong>当前回答</strong></p>
        <pre data-testid="frontier-response" style="white-space: pre-wrap; overflow-wrap: anywhere; font: inherit;"></pre>
        <p data-testid="frontier-diagnosis" aria-live="polite" aria-atomic="true"></p>
        <svg class="interactive-chart" data-testid="frontier-chart" viewBox="0 0 760 410"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">四种回答的 RLVR 奖励与组内优势</title>
          <desc id="${prefix}-chart-desc"></desc>
          ${candidates.map((candidate, index) => {
            const y = 45 + index * 70;
            return `
              <g data-candidate="${candidate.id}">
                <text class="chart-label" x="170" y="${y + 17}" text-anchor="end">${candidate.chartLabel}</text>
                <rect x="190" y="${y}" width="480" height="24" fill="#EEF1F2" />
                <rect data-testid="frontier-bar-${candidate.id}" x="190" y="${y}" width="0" height="24" fill="#1F648C" />
                <text class="chart-label" data-frontier-value="${candidate.id}" x="190" y="${y + 43}"></text>
              </g>
            `;
          }).join('')}
          <path class="chart-axes" d="M 190 330 H 680" />
          <g class="chart-label" text-anchor="middle">
            <text x="190" y="355">0</text>
            <text x="350" y="355">0.4</text>
            <text x="510" y="355">0.8</text>
            <text x="670" y="355">1.2</text>
          </g>
          <text class="chart-label" x="430" y="388" text-anchor="middle">总奖励 R（灰色轨道上限为 1.2）</text>
        </svg>
        <figcaption>优势使用四个回答奖励的均值、总体标准差与 ε = 10⁻¹² 计算；切换规则会重算整组结果，不自动播放。</figcaption>
      </figure>
    `;

    this.#verifier = this.querySelector('[data-testid="frontier-verifier"]');
    this.#candidate = this.querySelector('[data-testid="frontier-candidate"]');
    this.#rule = this.querySelector('[data-testid="frontier-rule"]');
    this.#outcome = this.querySelector('[data-testid="frontier-outcome"]');
    this.#bonus = this.querySelector('[data-testid="frontier-bonus"]');
    this.#reward = this.querySelector('[data-testid="frontier-reward"]');
    this.#advantage = this.querySelector('[data-testid="frontier-advantage"]');
    this.#response = this.querySelector('[data-testid="frontier-response"]');
    this.#diagnosis = this.querySelector('[data-testid="frontier-diagnosis"]');
    this.#description = this.querySelector(`#${prefix}-chart-desc`);

    const requestedVerifier = this.getAttribute('verifier') ?? 'outcome';
    const requestedCandidate = this.getAttribute('candidate') ?? 'decomposition';
    this.#verifier.value = verifiers[requestedVerifier] ? requestedVerifier : 'outcome';
    this.#candidate.value = candidates.some(({ id }) => id === requestedCandidate)
      ? requestedCandidate
      : 'decomposition';

    this.#verifier.addEventListener('change', () => this.#render());
    this.#candidate.addEventListener('change', () => this.#render());
    this.#render();
  }

  #render() {
    const verifierId = this.#verifier.value;
    const verifier = verifiers[verifierId];
    const rows = calculateRewards(verifierId);
    const selected = rows.find(({ id }) => id === this.#candidate.value);

    this.#rule.textContent = verifier.formula;
    this.#outcome.textContent = formatNumber(selected.outcome);
    this.#bonus.textContent = formatNumber(selected.bonus);
    this.#reward.textContent = formatNumber(selected.reward);
    this.#advantage.textContent = formatNumber(selected.advantage);
    this.#response.textContent = selected.text;
    this.#diagnosis.textContent = `当前选择 ${selected.name}。${verifier.diagnosis}`;

    for (const row of rows) {
      const bar = this.querySelector(`[data-testid="frontier-bar-${row.id}"]`);
      const value = this.querySelector(`[data-frontier-value="${row.id}"]`);
      bar.setAttribute('width', String(Number((row.reward * 400).toFixed(6))));
      bar.setAttribute('data-reward', formatNumber(row.reward));
      bar.setAttribute('data-advantage', formatNumber(row.advantage));
      bar.setAttribute('stroke', row.id === selected.id ? '#8C421F' : 'none');
      bar.setAttribute('stroke-width', row.id === selected.id ? '3' : '0');
      value.textContent = `R = ${formatNumber(row.reward)}；A = ${formatNumber(row.advantage)}`;
    }

    const values = rows.map((row) =>
      `${row.name}，奖励 ${formatNumber(row.reward)}，优势 ${formatNumber(row.advantage)}`,
    ).join('；');
    this.#description.textContent = `奖励规则为${verifier.name}。${values}。${verifier.diagnosis}`;
  }
}

export function defineFrontierRLElement() {
  if (!customElements.get('llm-frontier-rl')) {
    customElements.define('llm-frontier-rl', FrontierRLElement);
  }
}
