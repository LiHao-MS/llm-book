const WEIGHT_BYTES = 2;
const GRADIENT_BYTES = 2;
const OPTIMIZER_BYTES = 4 + 8;

function assertPositiveInteger(value, name) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new RangeError(`${name} 必须是正整数。`);
  }
}

export function estimateStaticStates({
  parameters,
  layers,
  dataParallel,
  tensorParallel,
  pipelineParallel,
  zeroStage,
}) {
  assertPositiveInteger(parameters, 'parameters');
  assertPositiveInteger(layers, 'layers');
  assertPositiveInteger(dataParallel, 'dataParallel');
  assertPositiveInteger(tensorParallel, 'tensorParallel');
  assertPositiveInteger(pipelineParallel, 'pipelineParallel');
  if (!Number.isSafeInteger(zeroStage) || zeroStage < 0 || zeroStage > 3) {
    throw new RangeError('zeroStage 必须是 0、1、2 或 3。');
  }
  if (layers % pipelineParallel !== 0) {
    throw new RangeError('pipelineParallel 必须均匀切分 layers。');
  }

  const modelParallel = tensorParallel * pipelineParallel;
  assertPositiveInteger(modelParallel, 'tensorParallel × pipelineParallel');
  const maximumStateShards = modelParallel * (zeroStage > 0 ? dataParallel : 1);
  assertPositiveInteger(maximumStateShards, '最大状态切分度');
  if (parameters % maximumStateShards !== 0) {
    throw new RangeError('parameters 必须能被当前使用的状态切分度整除。');
  }

  const weightShards = modelParallel * (zeroStage === 3 ? dataParallel : 1);
  const gradientShards = modelParallel * (zeroStage >= 2 ? dataParallel : 1);
  const optimizerShards = modelParallel * (zeroStage >= 1 ? dataParallel : 1);
  assertPositiveInteger(weightShards, '权重切分度');
  assertPositiveInteger(gradientShards, '梯度切分度');
  assertPositiveInteger(optimizerShards, '优化器切分度');

  const weightBytes = parameters / weightShards * WEIGHT_BYTES;
  const gradientBytes = parameters / gradientShards * GRADIENT_BYTES;
  const optimizerBytes = parameters / optimizerShards * OPTIMIZER_BYTES;
  assertPositiveInteger(weightBytes, 'weightBytes');
  assertPositiveInteger(gradientBytes, 'gradientBytes');
  assertPositiveInteger(optimizerBytes, 'optimizerBytes');
  const totalBytes = weightBytes + gradientBytes + optimizerBytes;
  assertPositiveInteger(totalBytes, 'totalBytes');
  return {
    weightBytes,
    gradientBytes,
    optimizerBytes,
    totalBytes,
  };
}

const TOY_PARAMETERS = 1_000_000_000;
const TOY_LAYERS = 12;
const MAX_STATIC_BYTES = 16_000_000_000;
const BAR = { x: 70, width: 620 };
const PIPELINE_ROWS = [
  { stage: 1, layers: 'L1–L3', forward: 't1–t4：F1→F4', backward: 't11–t14：B4→B1' },
  { stage: 2, layers: 'L4–L6', forward: 't2–t5：F1→F4', backward: 't10–t13：B4→B1' },
  { stage: 3, layers: 'L7–L9', forward: 't3–t6：F1→F4', backward: 't9–t12：B4→B1' },
  { stage: 4, layers: 'L10–L12', forward: 't4–t7：F1→F4', backward: 't8–t11：B4→B1' },
];
const HTMLElementBase = globalThis.HTMLElement ?? class {};
let instanceCount = 0;

const componentStyles = `
  <style>
    llm-distributed-training { display: block; }
    .distributed-controls { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .distributed-controls .interactive-control { grid-template-columns: minmax(0, 1fr); }
    .distributed-controls select {
      inline-size: 100%;
      min-block-size: 2.5rem;
      min-inline-size: 0;
      padding: 0.35rem 0.5rem;
      border: 1px solid #8b969c;
      border-radius: 0.25rem;
      background: #ffffff;
      color: #172b3a;
      font: inherit;
    }
    .distributed-controls select:focus-visible {
      outline: 3px solid #8c421f;
      outline-offset: 3px;
    }
    .distributed-communication { margin: 0.5rem 0 0; }
    .distributed-schedule-text { margin-top: 0.75rem; }
    .distributed-schedule-text summary { min-block-size: 2rem; cursor: pointer; }
    .distributed-schedule-text table { inline-size: 100%; border-collapse: collapse; }
    .distributed-schedule-text th,
    .distributed-schedule-text td {
      padding: 0.4rem 0.5rem;
      border-bottom: 1px solid #dde3e7;
      text-align: left;
      overflow-wrap: anywhere;
    }
    @media (max-width: 36rem) {
      .distributed-controls { grid-template-columns: minmax(0, 1fr); }
    }
  </style>
`;

function formatGb(bytes) {
  return (bytes / 1e9).toFixed(3);
}

function communicationEvents({ dataParallel, tensorParallel, pipelineParallel, zeroStage }) {
  const events = [];
  if (dataParallel > 1) {
    if (zeroStage === 0) {
      events.push(`DP=${dataParallel}：梯度 AllReduce；可在梯度累积边界同步。`);
    } else if (zeroStage === 1) {
      events.push(`DP=${dataParallel}：梯度 AllReduce，优化器分片更新后 AllGather 权重。`);
    } else if (zeroStage === 2) {
      events.push(`DP=${dataParallel}：梯度 ReduceScatter，优化器分片更新后 AllGather 权重。`);
    } else {
      events.push(`DP=${dataParallel}：梯度 ReduceScatter；前向与反向按需 AllGather 权重。`);
    }
  }
  if (tensorParallel > 1) {
    events.push(`TP=${tensorParallel}：同一 microbatch 在层内进行激活 AllReduce 或等价集合通信。`);
  }
  if (pipelineParallel > 1) {
    events.push(`PP=${pipelineParallel}：相邻阶段 Send/Recv 前向激活与反向梯度。`);
  }
  return events.length > 0 ? events : ['单设备配置：没有跨设备通信事件。'];
}

function pipelineCells(row) {
  const backwardStart = 11 - row.stage;
  return Array.from({ length: 14 }, (_, time) => {
    const forwardStart = row.stage - 1;
    const forward = time >= forwardStart && time < forwardStart + 4;
    const backward = time >= backwardStart && time < backwardStart + 4;
    const label = forward
      ? `F${time - forwardStart + 1}`
      : backward ? `B${4 - (time - backwardStart)}` : '';
    const fill = forward ? '#1F648C' : backward ? '#B55224' : '#EEF1F2';
    return `
      <rect x="${170 + time * 38}" y="${270 + (row.stage - 1) * 50}"
        width="34" height="32" rx="3" fill="${fill}" stroke="#FFFFFF"/>
      <text x="${187 + time * 38}" y="${291 + (row.stage - 1) * 50}"
        text-anchor="middle" font-size="11" fill="#FFFFFF">${label}</text>`;
  }).join('');
}

function pipelineMarkup() {
  const timeLabels = Array.from({ length: 14 }, (_, time) =>
    `<text x="${187 + time * 38}" y="254" text-anchor="middle" font-size="10">t${time + 1}</text>`,
  ).join('');
  const rows = PIPELINE_ROWS.map((row) => `
    <text x="20" y="${292 + (row.stage - 1) * 50}" font-size="12">阶段 ${row.stage}（${row.layers}）</text>
    ${pipelineCells(row)}
  `).join('');
  return `${timeLabels}${rows}`;
}

class DistributedTrainingElement extends HTMLElementBase {
  #els;

  connectedCallback() {
    if (this.#els) {
      return;
    }
    if (!this.id && this.dataset.figureId) {
      this.id = this.dataset.figureId;
    }
    instanceCount += 1;
    const prefix = `${this.dataset.figureId || 'distributed-training'}-${instanceCount}`;

    this.innerHTML = `
      ${componentStyles}
      <figure class="interactive-widget distributed-training-widget">
        <div class="interactive-controls distributed-controls">
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-zero">状态切分策略</label>
            <select id="${prefix}-zero" data-testid="distributed-zero-stage">
              <option value="0">ZeRO-0：DP 状态复制</option>
              <option value="1">ZeRO-1：切优化器状态</option>
              <option value="2">ZeRO-2：再切梯度</option>
              <option value="3">ZeRO-3 / FSDP：再切权重</option>
            </select>
          </div>
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-dp">数据并行度 DP</label>
            <select id="${prefix}-dp" data-testid="distributed-data-parallel">
              <option value="1">1</option><option value="2">2</option>
              <option value="4">4</option><option value="8">8</option>
            </select>
          </div>
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-tp">张量并行度 TP</label>
            <select id="${prefix}-tp" data-testid="distributed-tensor-parallel">
              <option value="1">1</option><option value="2">2</option><option value="4">4</option>
            </select>
          </div>
          <div class="interactive-control interactive-select-control">
            <label for="${prefix}-pp">流水线并行度 PP</label>
            <select id="${prefix}-pp" data-testid="distributed-pipeline-parallel">
              <option value="1">1（12 层）</option>
              <option value="2">2（每阶段 6 层）</option>
              <option value="4">4（每阶段 3 层）</option>
            </select>
          </div>
        </div>
        <dl class="interactive-results">
          <div><dt>权重（2 字节/参数）</dt><dd><output data-testid="distributed-weight-bytes"></output></dd></div>
          <div><dt>梯度（2 字节/参数）</dt><dd><output data-testid="distributed-gradient-bytes"></output></dd></div>
          <div><dt>优化器（12 字节/参数）</dt><dd><output data-testid="distributed-optimizer-bytes"></output></dd></div>
          <div><dt>总静态状态</dt><dd><output data-testid="distributed-total-bytes"></output></dd></div>
          <div><dt>PP 每阶段层数</dt><dd><output data-testid="distributed-stage-layers"></output></dd></div>
        </dl>
        <p><strong>当前通信事件</strong></p>
        <ul class="distributed-communication" data-testid="distributed-communication"></ul>
        <svg class="interactive-chart" data-testid="distributed-chart" viewBox="0 0 760 550"
          role="img" aria-labelledby="${prefix}-chart-title" aria-describedby="${prefix}-chart-desc">
          <title id="${prefix}-chart-title">每设备静态状态账目与四阶段流水线概念时间线</title>
          <desc id="${prefix}-chart-desc"></desc>
          <text x="30" y="30" font-size="16" font-weight="700">每设备静态状态（固定 0–16 GB 刻度）</text>
          <g stroke="#DDE3E7" stroke-width="1">
            <line x1="70" y1="52" x2="70" y2="118"/><line x1="225" y1="52" x2="225" y2="118"/>
            <line x1="380" y1="52" x2="380" y2="118"/><line x1="535" y1="52" x2="535" y2="118"/>
            <line x1="690" y1="52" x2="690" y2="118"/>
          </g>
          <rect data-testid="distributed-weight-bar" x="70" y="65" height="36" fill="#1F648C"/>
          <rect data-testid="distributed-gradient-bar" x="70" y="65" height="36" fill="#39704A"/>
          <rect data-testid="distributed-optimizer-bar" x="70" y="65" height="36" fill="#C65D21"/>
          <g font-size="11" fill="#4C5F6A" text-anchor="middle">
            <text x="70" y="118">0</text><text x="225" y="118">4</text><text x="380" y="118">8</text>
            <text x="535" y="118">12</text><text x="690" y="118">16 GB</text>
          </g>
          <text data-testid="distributed-chart-total" x="70" y="145" font-size="13"></text>
          <g font-size="12">
            <rect x="70" y="160" width="14" height="14" fill="#1F648C"/><text x="91" y="172">权重</text>
            <rect x="180" y="160" width="14" height="14" fill="#39704A"/><text x="201" y="172">梯度</text>
            <rect x="290" y="160" width="14" height="14" fill="#C65D21"/><text x="311" y="172">优化器状态</text>
          </g>
          <text x="30" y="222" font-size="16" font-weight="700">GPipe 式概念调度：4 阶段 × 4 microbatch</text>
          ${pipelineMarkup()}
          <g font-size="12">
            <rect x="70" y="478" width="14" height="14" fill="#1F648C"/><text x="91" y="490">前向 F</text>
            <rect x="190" y="478" width="14" height="14" fill="#B55224"/><text x="211" y="490">反向 B</text>
            <rect x="310" y="478" width="14" height="14" fill="#EEF1F2"/><text x="331" y="490">气泡 / 等待</text>
          </g>
          <text x="30" y="520" font-size="12" fill="#4C5F6A">每格等长仅表达依赖顺序；先全部前向再反向，空白为流水线气泡。</text>
          <text x="30" y="541" font-size="12" fill="#4C5F6A">该固定图不模拟 kernel、通信或真实吞吐。</text>
        </svg>
        <details class="distributed-schedule-text">
          <summary>流水线时间线的文字等价表</summary>
          <table>
            <caption>四阶段、四 microbatch 的 GPipe 式概念调度</caption>
            <thead><tr><th scope="col">阶段与层</th><th scope="col">前向</th><th scope="col">反向</th></tr></thead>
            <tbody>
              ${PIPELINE_ROWS.map((row) => `<tr><th scope="row">阶段 ${row.stage}（${row.layers}）</th><td>${row.forward}</td><td>${row.backward}</td></tr>`).join('')}
            </tbody>
          </table>
        </details>
        <figcaption>
          玩具模型固定为 10 亿参数、12 层；GB 均为十进制。堆叠条与结果表只含权重、梯度和优化器静态状态，不含激活、临时缓冲区与通信桶。
          下方时间线始终为四阶段、四 microbatch 的概念图，与当前 PP 控件解耦。
        </figcaption>
      </figure>
    `;

    const byTestId = (testId) => this.querySelector(`[data-testid="${testId}"]`);
    this.#els = {
      zeroStage: byTestId('distributed-zero-stage'),
      dataParallel: byTestId('distributed-data-parallel'),
      tensorParallel: byTestId('distributed-tensor-parallel'),
      pipelineParallel: byTestId('distributed-pipeline-parallel'),
      weight: byTestId('distributed-weight-bytes'),
      gradient: byTestId('distributed-gradient-bytes'),
      optimizer: byTestId('distributed-optimizer-bytes'),
      total: byTestId('distributed-total-bytes'),
      stageLayers: byTestId('distributed-stage-layers'),
      communication: byTestId('distributed-communication'),
      chart: byTestId('distributed-chart'),
      description: this.querySelector(`#${prefix}-chart-desc`),
      chartTotal: byTestId('distributed-chart-total'),
      weightBar: byTestId('distributed-weight-bar'),
      gradientBar: byTestId('distributed-gradient-bar'),
      optimizerBar: byTestId('distributed-optimizer-bar'),
    };
    this.#els.zeroStage.value = this.getAttribute('zero-stage') ?? '0';
    this.#els.dataParallel.value = this.getAttribute('data-parallel') ?? '8';
    this.#els.tensorParallel.value = this.getAttribute('tensor-parallel') ?? '1';
    this.#els.pipelineParallel.value = this.getAttribute('pipeline-parallel') ?? '1';

    const controls = [
      this.#els.zeroStage,
      this.#els.dataParallel,
      this.#els.tensorParallel,
      this.#els.pipelineParallel,
    ];
    const controlIds = controls.map((control) => control.id).join(' ');
    for (const output of this.querySelectorAll('.interactive-results output')) {
      output.setAttribute('for', controlIds);
    }
    for (const control of controls) {
      control.addEventListener('change', () => this.#update());
      control.addEventListener('keydown', (event) => {
        if (!['ArrowDown', 'ArrowUp'].includes(event.key)) {
          return;
        }
        event.preventDefault();
        const direction = event.key === 'ArrowDown' ? 1 : -1;
        const nextIndex = Math.min(
          Math.max(control.selectedIndex + direction, 0),
          control.options.length - 1,
        );
        if (nextIndex !== control.selectedIndex) {
          control.selectedIndex = nextIndex;
          this.#update();
        }
      });
    }
    this.#update();
  }

  #update() {
    const configuration = {
      dataParallel: Number(this.#els.dataParallel.value),
      tensorParallel: Number(this.#els.tensorParallel.value),
      pipelineParallel: Number(this.#els.pipelineParallel.value),
      zeroStage: Number(this.#els.zeroStage.value),
    };
    const account = estimateStaticStates({
      parameters: TOY_PARAMETERS,
      layers: TOY_LAYERS,
      ...configuration,
    });
    const events = communicationEvents(configuration);
    const outputs = [
      [this.#els.weight, account.weightBytes],
      [this.#els.gradient, account.gradientBytes],
      [this.#els.optimizer, account.optimizerBytes],
      [this.#els.total, account.totalBytes],
    ];
    for (const [output, bytes] of outputs) {
      output.dataset.value = String(bytes);
      output.textContent = `${formatGb(bytes)} GB`;
    }
    this.#els.stageLayers.textContent = `12 / ${configuration.pipelineParallel} = ${TOY_LAYERS / configuration.pipelineParallel} 层`;
    this.#els.communication.replaceChildren(...events.map((event) => {
      const item = document.createElement('li');
      item.textContent = event;
      return item;
    }));

    let x = BAR.x;
    for (const [bar, bytes] of [
      [this.#els.weightBar, account.weightBytes],
      [this.#els.gradientBar, account.gradientBytes],
      [this.#els.optimizerBar, account.optimizerBytes],
    ]) {
      const width = BAR.width * bytes / MAX_STATIC_BYTES;
      bar.setAttribute('x', String(x));
      bar.setAttribute('width', String(width));
      x += width;
    }
    const signature = `${configuration.dataParallel}|${configuration.tensorParallel}|${configuration.pipelineParallel}|${configuration.zeroStage}`;
    this.#els.chart.dataset.signature = signature;
    this.#els.chartTotal.textContent = `当前合计 ${formatGb(account.totalBytes)} GB / 设备`;
    this.#els.description.textContent = `玩具模型为 10 亿参数、12 层。当前 DP=${configuration.dataParallel}、TP=${configuration.tensorParallel}、PP=${configuration.pipelineParallel}、ZeRO-${configuration.zeroStage}；每设备权重 ${formatGb(account.weightBytes)} GB、梯度 ${formatGb(account.gradientBytes)} GB、优化器状态 ${formatGb(account.optimizerBytes)} GB，总静态状态 ${formatGb(account.totalBytes)} GB。通信事件：${events.join('；')}。下方固定为四阶段、四 microbatch 的 GPipe 式概念时间线，空白表示气泡，不表示真实吞吐。`;
  }
}

export function defineDistributedTrainingElement() {
  if (!customElements.get('llm-distributed-training')) {
    customElements.define('llm-distributed-training', DistributedTrainingElement);
  }
}
