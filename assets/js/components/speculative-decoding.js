const NORMALIZATION_TOLERANCE = Number.EPSILON;

function validateDistributions(p, q) {
  if (!Array.isArray(p) || !Array.isArray(q) || p.length === 0 || p.length !== q.length) {
    throw new RangeError('目标分布与草稿分布必须是等长的非空数组。');
  }

  for (const [name, distribution] of [['目标', p], ['草稿', q]]) {
    let sum = 0;
    for (const probability of distribution) {
      if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
        throw new RangeError(`${name}分布必须只含 0 到 1 之间的有限概率。`);
      }
      sum += probability;
    }
    if (Math.abs(sum - 1) > NORMALIZATION_TOLERANCE * distribution.length) {
      throw new RangeError(`${name}分布的概率之和必须为 1。`);
    }
  }
}

export function speculativeAcceptance(p, q) {
  validateDistributions(p, q);
  const perToken = p.map((targetProbability, index) => {
    const draftProbability = q[index];
    return draftProbability === 0
      ? null
      : Math.min(1, targetProbability / draftProbability);
  });
  const total = p.reduce((sum, targetProbability, index) =>
    sum + Math.min(targetProbability, q[index]), 0);

  return { perToken, total };
}

export function speculativeResidual(p, q) {
  validateDistributions(p, q);
  const rejectedMass = p.map((targetProbability, index) =>
    Math.max(0, targetProbability - q[index]));
  const totalRejectedMass = rejectedMass.reduce((sum, value) => sum + value, 0);

  if (totalRejectedMass === 0) {
    return null;
  }
  return rejectedMass.map((value) => value / totalRejectedMass);
}

const TOKENS = ['A', 'B', 'C'];
const PRESETS = {
  worked: { p: [0.5, 0.3, 0.2], q: [0.4, 0.4, 0.2] },
  unreachable: { p: [0.4, 0.4, 0.2], q: [0.6, 0, 0.4] },
  identical: { p: [0.5, 0.3, 0.2], q: [0.5, 0.3, 0.2] },
};
const svgNamespace = 'http://www.w3.org/2000/svg';
const HTMLElementBase = globalThis.HTMLElement ?? class {};
let instanceCount = 0;

const componentStyles = `
  <style>
    llm-speculative-decoding { display: block; min-inline-size: 0; }
    .speculative-widget { min-inline-size: 0; }
    .speculative-title { margin: 0 0 0.75rem; font-size: 1.05rem; }
    .speculative-controls { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .speculative-controls .interactive-control {
      grid-template-columns: minmax(0, 1fr);
      min-inline-size: 0;
    }
    .speculative-input {
      inline-size: 100%;
      min-inline-size: 0;
      min-block-size: 2.5rem;
      box-sizing: border-box;
      padding: 0.35rem 0.5rem;
      border: 1px solid #8b969c;
      border-radius: 0.25rem;
      background: #ffffff;
      color: #172b3a;
      font: inherit;
    }
    .speculative-input:focus-visible {
      outline: 3px solid #8c421f;
      outline-offset: 3px;
    }
    .speculative-input[aria-invalid='true'] { border-color: #8c421f; }
    .speculative-error {
      margin: 0.5rem 0;
      color: #8c421f;
      font-weight: 600;
      overflow-wrap: anywhere;
    }
    .speculative-summary,
    .speculative-residual-state { overflow-wrap: anywhere; }
    .speculative-chart { inline-size: 100%; block-size: auto; }
    .speculative-data {
      inline-size: 100%;
      table-layout: fixed;
      border-collapse: collapse;
      font-size: 0.78rem;
      font-variant-numeric: tabular-nums;
    }
    .speculative-data caption,
    .speculative-data th,
    .speculative-data td { overflow-wrap: anywhere; }
    .speculative-data caption { padding: 0.4rem 0; text-align: left; }
    .speculative-data th,
    .speculative-data td {
      padding: 0.35rem 0.2rem;
      border-bottom: 1px solid #dde3e7;
      text-align: left;
      vertical-align: top;
    }
    @media (max-width: 48rem) {
      .speculative-controls { grid-template-columns: minmax(0, 1fr); }
    }
  </style>
`;

function formatProbability(value) {
  return String(Number(value.toFixed(6)));
}

function createSvgElement(name, attributes = {}) {
  const element = document.createElementNS(svgNamespace, name);
  for (const [attribute, value] of Object.entries(attributes)) {
    element.setAttribute(attribute, String(value));
  }
  return element;
}

function probabilityControlMarkup(prefix, token, kind) {
  const id = `${prefix}-${kind}-${token}`;
  const symbol = kind === 'target' ? 'p' : 'q';
  const name = kind === 'target' ? '目标' : '草稿';
  return `
    <div class="interactive-control">
      <label for="${id}">${name} ${symbol}(${token})</label>
      <input class="speculative-input" id="${id}"
        data-testid="speculative-input-${kind}-${token}" type="text"
        inputmode="decimal" required aria-invalid="false"
        aria-describedby="${prefix}-error">
    </div>
  `;
}

function resultRowMarkup(token) {
  return `
    <tr data-testid="speculative-row-${token}">
      <th scope="row">${token}</th>
      <td><output data-testid="speculative-target-${token}"></output></td>
      <td><output data-testid="speculative-draft-${token}"></output></td>
      <td><output data-testid="speculative-acceptance-${token}"></output></td>
      <td><output data-testid="speculative-residual-${token}"></output></td>
    </tr>
  `;
}

class SpeculativeDecodingElement extends HTMLElementBase {
  #els;

  connectedCallback() {
    if (this.#els) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'speculative-decoding'}-${instanceCount}`;
    const controls = TOKENS.map((token) => [
      probabilityControlMarkup(prefix, token, 'target'),
      probabilityControlMarkup(prefix, token, 'draft'),
    ].join('')).join('');

    this.innerHTML = `
      ${componentStyles}
      <figure class="interactive-widget speculative-widget">
        <form data-testid="speculative-form" aria-labelledby="${prefix}-title" novalidate>
          <p class="speculative-title" id="${prefix}-title"><strong>投机解码分布交互</strong></p>
          <div class="interactive-controls speculative-controls">
            <div class="interactive-control">
              <label for="${prefix}-preset">分布预设</label>
              <select class="speculative-input" id="${prefix}-preset" data-testid="speculative-preset">
                <option value="worked">默认算例</option>
                <option value="unreachable">草稿不可达 Token</option>
                <option value="identical">相同分布 p = q</option>
                <option value="custom">自定义概率</option>
              </select>
            </div>
            ${controls}
          </div>
          <p class="speculative-error" id="${prefix}-error"
            data-testid="speculative-error" role="alert" hidden></p>
        </form>
        <dl class="interactive-results">
          <div><dt>总接受率</dt><dd><output data-testid="speculative-total-acceptance"></output></dd></div>
        </dl>
        <p class="speculative-residual-state" data-testid="speculative-residual-state"></p>
        <svg class="interactive-chart speculative-chart" data-testid="speculative-chart"
          viewBox="0 0 960 340" role="img"
          aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">投机解码目标与草稿分布</title>
          <desc id="${prefix}-chart-desc"></desc>
          <g data-testid="speculative-probability-bars"></g>
        </svg>
        <table class="speculative-data">
          <caption>当前逐 Token 接受率与拒绝后残差</caption>
          <thead><tr><th scope="col">Token</th><th scope="col">目标 p</th><th scope="col">草稿 q</th><th scope="col">接受率</th><th scope="col">残差</th></tr></thead>
          <tbody>${TOKENS.map(resultRowMarkup).join('')}</tbody>
        </table>
        <p class="speculative-summary" data-testid="speculative-summary"
          role="status" aria-live="polite" aria-atomic="true"></p>
        <figcaption>概率条比较目标 p 与草稿 q；接受—修正保持目标分布并减少严格串行步数，不扩大测试时搜索空间。</figcaption>
      </figure>
    `;

    const byTestId = (testId) => this.querySelector(`[data-testid="${testId}"]`);
    this.#els = {
      form: byTestId('speculative-form'),
      preset: byTestId('speculative-preset'),
      error: byTestId('speculative-error'),
      total: byTestId('speculative-total-acceptance'),
      residualState: byTestId('speculative-residual-state'),
      chart: byTestId('speculative-chart'),
      description: this.querySelector(`#${prefix}-chart-desc`),
      bars: byTestId('speculative-probability-bars'),
      summary: byTestId('speculative-summary'),
      outputs: [...this.querySelectorAll('output[data-testid]')],
      inputs: TOKENS.flatMap((token, index) => ['target', 'draft'].map((kind) => ({
        input: byTestId(`speculative-input-${kind}-${token}`),
        index,
        kind,
      }))),
    };
    const controlIds = [this.#els.preset, ...this.#els.inputs.map(({ input }) => input)]
      .map((control) => control.id).join(' ');
    for (const output of this.#els.outputs) {
      output.setAttribute('for', controlIds);
    }

    this.#els.form.addEventListener('submit', (event) => event.preventDefault());
    this.#els.preset.addEventListener('change', () => {
      if (this.#els.preset.value !== 'custom') {
        this.#applyPreset(this.#els.preset.value);
      }
      this.#update();
    });
    for (const { input } of this.#els.inputs) {
      input.addEventListener('input', () => {
        this.#els.preset.value = 'custom';
        this.#update();
      });
    }
    this.#applyPreset('worked');
    this.#update();
  }

  #applyPreset(name) {
    const preset = PRESETS[name];
    this.#els.preset.value = name;
    for (const { input, index, kind } of this.#els.inputs) {
      input.value = String(preset[kind === 'target' ? 'p' : 'q'][index]);
    }
  }

  #update() {
    let inputsValid = true;
    for (const { input } of this.#els.inputs) {
      const rawValue = input.value.trim();
      const value = Number(rawValue);
      const valid = rawValue !== '' && Number.isFinite(value) && value >= 0 && value <= 1;
      input.setAttribute('aria-invalid', String(!valid));
      inputsValid &&= valid;
    }
    if (!inputsValid) {
      this.#showError('每个概率必须是 0 到 1 之间的有限数值。');
      return;
    }

    const values = this.#els.inputs.map(({ input }) => Number(input.value.trim()));
    const p = TOKENS.map((_, index) => values[index * 2]);
    const q = TOKENS.map((_, index) => values[index * 2 + 1]);
    try {
      const acceptance = speculativeAcceptance(p, q);
      const residual = speculativeResidual(p, q);
      this.#els.form.dataset.valid = 'true';
      this.#els.error.hidden = true;
      this.#els.error.textContent = '';
      for (const { input } of this.#els.inputs) {
        input.setAttribute('aria-invalid', 'false');
      }
      this.#render(p, q, acceptance, residual);
    } catch (error) {
      if (!(error instanceof RangeError)) {
        throw error;
      }
      for (const { input } of this.#els.inputs) {
        input.setAttribute('aria-invalid', 'true');
      }
      this.#showError(error.message);
    }
  }

  #showError(message) {
    this.#els.form.dataset.valid = 'false';
    this.#els.error.textContent = message;
    this.#els.error.hidden = false;
    for (const output of this.#els.outputs) {
      output.textContent = '';
      delete output.dataset.value;
    }
    this.#els.residualState.textContent = '';
    this.#els.summary.textContent = '请先修正输入，再查看分布结果。';
    this.#els.bars.replaceChildren();
    this.#els.chart.removeAttribute('data-signature');
    this.#els.description.textContent = `输入无效：${message}`;
  }

  #render(p, q, acceptance, residual) {
    const residualText = residual
      ? `[${residual.map(formatProbability).join(', ')}]`
      : 'null';
    const residualDescription = residual
      ? `残差分布 ${residualText}`
      : '无拒绝质量，残差分布为 null';

    this.#setNumber(this.#els.total, acceptance.total);
    this.#els.residualState.textContent = residual
      ? `拒绝后归一化残差：${residualText}`
      : '无拒绝质量，残差分布为 null。';

    const tokenDescriptions = TOKENS.map((token, index) => {
      this.#setNumber(this.querySelector(`[data-testid="speculative-target-${token}"]`), p[index]);
      this.#setNumber(this.querySelector(`[data-testid="speculative-draft-${token}"]`), q[index]);
      this.#setNullable(
        this.querySelector(`[data-testid="speculative-acceptance-${token}"]`),
        acceptance.perToken[index],
        'null（q = 0，草稿不可达）',
      );
      this.#setNullable(
        this.querySelector(`[data-testid="speculative-residual-${token}"]`),
        residual?.[index] ?? null,
        '—',
      );
      const acceptanceText = acceptance.perToken[index] === null
        ? 'null（q = 0）'
        : formatProbability(acceptance.perToken[index]);
      const tokenResidual = residual ? formatProbability(residual[index]) : '—';
      return `Token ${token}：p=${formatProbability(p[index])}，q=${formatProbability(q[index])}，接受率=${acceptanceText}，残差=${tokenResidual}`;
    });

    this.#els.description.textContent = `总接受率 ${formatProbability(acceptance.total)}；${residualDescription}。${tokenDescriptions.join('；')}。接受—修正保持目标分布并减少严格串行步数，不是测试时搜索。`;
    this.#els.summary.textContent = `总接受率 ${formatProbability(acceptance.total)}；${residualDescription}。接受—修正保持目标分布，减少严格串行的目标模型步数；这不是测试时搜索。`;
    this.#els.chart.dataset.signature = [
      p.join(','),
      q.join(','),
      acceptance.perToken.map(String).join(','),
      acceptance.total,
      residual?.join(',') ?? 'null',
    ].join('|');
    this.#renderBars(p, q);
  }

  #setNumber(output, value) {
    output.dataset.value = String(value);
    output.textContent = formatProbability(value);
  }

  #setNullable(output, value, nullText) {
    if (value === null) {
      output.textContent = nullText;
      delete output.dataset.value;
      return;
    }
    this.#setNumber(output, value);
  }

  #renderBars(p, q) {
    const groups = TOKENS.map((token, index) => {
      const y = 25 + index * 100;
      const group = createSvgElement('g', { 'data-token': token });
      const label = createSvgElement('text', { x: 20, y: y + 34, 'font-size': 16, 'font-weight': 700 });
      label.textContent = `Token ${token}`;
      const targetTrack = createSvgElement('rect', {
        x: 140, y, width: 570, height: 26, rx: 4, fill: '#eef1f2',
      });
      const targetBar = createSvgElement('rect', {
        class: 'speculative-target-bar', 'data-distribution': 'target', 'data-probability': p[index],
        x: 140, y, width: 570 * p[index], height: 26, rx: 4, fill: '#1f648c',
      });
      const targetValue = createSvgElement('text', { x: 730, y: y + 19, 'font-size': 14 });
      targetValue.textContent = `p = ${formatProbability(p[index])}`;
      const draftTrack = createSvgElement('rect', {
        x: 140, y: y + 40, width: 570, height: 26, rx: 4, fill: '#eef1f2',
      });
      const draftBar = createSvgElement('rect', {
        class: 'speculative-draft-bar', 'data-distribution': 'draft', 'data-probability': q[index],
        x: 140, y: y + 40, width: 570 * q[index], height: 26, rx: 4, fill: '#c65d21',
      });
      const draftValue = createSvgElement('text', { x: 730, y: y + 59, 'font-size': 14 });
      draftValue.textContent = `q = ${formatProbability(q[index])}`;
      group.append(label, targetTrack, targetBar, targetValue, draftTrack, draftBar, draftValue);
      return group;
    });
    this.#els.bars.replaceChildren(...groups);
  }
}

export function defineSpeculativeDecodingElement() {
  if (!customElements.get('llm-speculative-decoding')) {
    customElements.define('llm-speculative-decoding', SpeculativeDecodingElement);
  }
}
