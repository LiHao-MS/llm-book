const mib = 1024 ** 2;
let instanceCount = 0;

const componentStyles = `
  <style>
    llm-kv-cache-calculator,
    llm-inference-cost-calculator {
      display: block;
    }
    .inference-calculator-title {
      margin: 0 0 0.75rem;
      font-size: 1.05rem;
    }
    .inference-number-input {
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
    .inference-number-input:focus-visible {
      outline: 3px solid #8c421f;
      outline-offset: 3px;
    }
    .inference-number-input[aria-invalid='true'] {
      border-color: #8c421f;
    }
    .inference-calculator-summary {
      margin: 0.75rem 0 0;
      overflow-wrap: anywhere;
    }
  </style>
`;

function controlMarkup(prefix, config) {
  const id = `${prefix}-${config.name}`;
  return `
    <div class="interactive-control">
      <label for="${id}">${config.label}</label>
      <input class="inference-number-input" id="${id}"
        data-testid="${config.testid}" type="number" min="${config.min}"
        ${config.max === undefined ? '' : `max="${config.max}"`}
        step="${config.step ?? 'any'}" required aria-invalid="false"
        aria-describedby="${id}-unit ${id}-error">
      <span id="${id}-unit">${config.unit}</span>
      <p class="interactive-control-error" id="${id}-error"
        data-testid="${config.testid}-error" hidden>${config.error}</p>
    </div>
  `;
}

function formatNumber(value) {
  if (value === 0) {
    return '0';
  }
  if (!Number.isFinite(value)) {
    return '';
  }
  if (Math.abs(value) >= 1e12 || Math.abs(value) < 1e-6) {
    return String(Number(value.toExponential(6)));
  }
  return String(Number(value.toFixed(6)));
}

function setOutput(output, value, unit = '') {
  output.dataset.value = String(value);
  output.textContent = `${formatNumber(value)}${unit}`;
}

function clearOutputs(outputs, summary, message) {
  for (const output of outputs) {
    output.textContent = '';
    delete output.dataset.value;
  }
  summary.textContent = message;
}

function initializeCalculator(host, options) {
  if (host.querySelector('form')) {
    return;
  }

  instanceCount += 1;
  const prefix = `${options.prefix}-${instanceCount}`;
  host.innerHTML = `
    ${componentStyles}
    <form class="interactive-widget inference-calculator-form" data-testid="${options.formTestid}"
      aria-labelledby="${prefix}-title" novalidate>
      <p class="inference-calculator-title" id="${prefix}-title"><strong>${options.title}</strong></p>
      <div class="interactive-controls">
        ${options.controls.map((control) => controlMarkup(prefix, control)).join('')}
      </div>
      ${options.resultsMarkup}
      <p class="inference-calculator-summary" data-testid="${options.summaryTestid}"
        role="status" aria-live="polite" aria-atomic="true"></p>
    </form>
  `;

  const form = host.querySelector('form');
  const summary = host.querySelector(`[data-testid="${options.summaryTestid}"]`);
  const outputs = [...host.querySelectorAll('output[data-testid]')];
  const inputs = options.controls.map((config) => {
    const input = host.querySelector(`[data-testid="${config.testid}"]`);
    input.value = host.getAttribute(config.attribute) ?? config.defaultValue;
    return { config, input };
  });
  const inputIds = inputs.map(({ input }) => input.id).join(' ');
  for (const output of outputs) {
    output.setAttribute('for', inputIds);
  }

  const update = () => {
    let valid = true;
    const values = {};

    for (const { config, input } of inputs) {
      const value = input.valueAsNumber;
      const integerValid = !config.integer || Number.isInteger(value);
      const inputValid = input.validity.valid && Number.isFinite(value) && integerValid;
      input.setAttribute('aria-invalid', String(!inputValid));
      const error = host.querySelector(`#${input.id}-error`);
      error.hidden = inputValid;
      valid &&= inputValid;
      values[config.name] = value;
    }

    form.dataset.valid = String(valid);
    if (!valid) {
      clearOutputs(outputs, summary, '请先修正输入，再查看计算结果。');
      return;
    }

    const metrics = options.calculate(values);
    if (Object.values(metrics).some((value) => !Number.isFinite(value))) {
      form.dataset.valid = 'false';
      clearOutputs(outputs, summary, '输入组合超出可表示的数值范围。');
      return;
    }
    options.render({ host, metrics, outputs, summary });
  };

  form.addEventListener('submit', (event) => event.preventDefault());
  for (const { input } of inputs) {
    input.addEventListener('input', update);
  }
  update();
}

function outputMap(outputs) {
  return new Map(outputs.map((output) => [output.dataset.testid, output]));
}

export function calculateKvCache(values) {
  const elementCount = 2 * values.batchSize * values.layers * values.sequenceLength
    * values.kvHeads * values.headDim;
  const totalBytes = elementCount * values.bytesPerElement;
  const bytesPerRequest = totalBytes / values.batchSize;
  const budgetBytes = values.budgetMib * mib;

  return {
    elementCount,
    totalBytes,
    totalMiB: totalBytes / mib,
    perRequestMiB: bytesPerRequest / mib,
    maxBatchSize: Math.floor(budgetBytes / bytesPerRequest),
  };
}

export function calculateInferenceCost(values) {
  const effectiveThroughput = values.aggregateThroughput * values.utilizationPercent / 100;
  const hourlyCost = values.gpuCount * values.hourlyPrice;
  const costPerMillion = hourlyCost * 1e6 / (3600 * effectiveThroughput);
  const requestsPerSecond = effectiveThroughput / values.outputTokensPerRequest;

  return {
    effectiveThroughput,
    hourlyCost,
    costPerMillion,
    costPerRequest: costPerMillion * values.outputTokensPerRequest / 1e6,
    requestsPerHour: requestsPerSecond * 3600,
  };
}

const kvControls = [
  {
    name: 'layers', attribute: 'layers', testid: 'kv-layers', label: '层数 L',
    unit: '层', defaultValue: '2', min: '1', max: '4096', step: '1', integer: true,
    error: '请输入 1 到 4096 之间的整数。',
  },
  {
    name: 'kvHeads', attribute: 'kv-heads', testid: 'kv-heads', label: 'KV 头数 H_kv',
    unit: '头', defaultValue: '2', min: '1', max: '4096', step: '1', integer: true,
    error: '请输入 1 到 4096 之间的整数。',
  },
  {
    name: 'headDim', attribute: 'head-dim', testid: 'kv-head-dim', label: '头维 d_h',
    unit: '维', defaultValue: '64', min: '1', max: '65536', step: '1', integer: true,
    error: '请输入 1 到 65536 之间的整数。',
  },
  {
    name: 'sequenceLength', attribute: 'sequence-length', testid: 'kv-sequence-length',
    label: '每序列长度 S', unit: 'Token', defaultValue: '512', min: '1', max: '10000000',
    step: '1', integer: true, error: '请输入 1 到 10000000 之间的整数。',
  },
  {
    name: 'batchSize', attribute: 'batch-size', testid: 'kv-batch-size', label: '批大小 B',
    unit: '序列', defaultValue: '4', min: '1', max: '100000', step: '1', integer: true,
    error: '请输入 1 到 100000 之间的整数。',
  },
  {
    name: 'bytesPerElement', attribute: 'bytes-per-element', testid: 'kv-bytes-per-element',
    label: '每元素字节数 b', unit: '字节', defaultValue: '2', min: '0.125', max: '8',
    error: '请输入 0.125 到 8 之间的数值。',
  },
  {
    name: 'budgetMib', attribute: 'budget-mib', testid: 'kv-budget-mib',
    label: '可用 KV 预算', unit: 'MiB', defaultValue: '16', min: '0.000001', max: '1000000000',
    error: '请输入 0.000001 到 1000000000 之间的数值。',
  },
];

const costControls = [
  {
    name: 'gpuCount', attribute: 'gpu-count', testid: 'cost-gpu-count', label: 'GPU 数 N',
    unit: '张', defaultValue: '1', min: '1', max: '100000', step: '1', integer: true,
    error: '请输入 1 到 100000 之间的整数。',
  },
  {
    name: 'hourlyPrice', attribute: 'hourly-price', testid: 'cost-hourly-price',
    label: '每 GPU 小时价格 c', unit: '成本单位/h', defaultValue: '8', min: '0', max: '1000000000',
    error: '请输入 0 到 1000000000 之间的数值。',
  },
  {
    name: 'aggregateThroughput', attribute: 'aggregate-throughput',
    testid: 'cost-aggregate-throughput', label: '整服务输出吞吐 T', unit: 'Token/s',
    defaultValue: '40', min: '0.000001', max: '1000000000',
    error: '请输入 0.000001 到 1000000000 之间的数值。',
  },
  {
    name: 'utilizationPercent', attribute: 'utilization-percent', testid: 'cost-utilization',
    label: '有效利用率 u', unit: '%', defaultValue: '50', min: '0.000001', max: '100',
    error: '请输入大于 0 且不超过 100 的数值。',
  },
  {
    name: 'outputTokensPerRequest', attribute: 'output-tokens-per-request',
    testid: 'cost-output-tokens', label: '每请求输出长度 Y', unit: 'Token',
    defaultValue: '250', min: '1', max: '10000000', step: '1', integer: true,
    error: '请输入 1 到 10000000 之间的整数。',
  },
];

class KvCacheCalculatorElement extends HTMLElement {
  connectedCallback() {
    initializeCalculator(this, {
      prefix: 'kv-calculator',
      formTestid: 'kv-cache-form',
      summaryTestid: 'kv-summary',
      title: 'KV cache 容量计算器',
      controls: kvControls,
      resultsMarkup: `
        <dl class="interactive-results">
          <div><dt>总 K/V 元素数</dt><dd><output data-testid="kv-element-count"></output></dd></div>
          <div><dt>总字节数</dt><dd><output data-testid="kv-total-bytes"></output></dd></div>
          <div><dt>总 KV cache</dt><dd><output data-testid="kv-total-mib"></output></dd></div>
          <div><dt>每序列 KV cache</dt><dd><output data-testid="kv-per-request-mib"></output></dd></div>
          <div><dt>预算内最大等长 batch</dt><dd><output data-testid="kv-max-batch"></output></dd></div>
        </dl>
      `,
      calculate: calculateKvCache,
      render({ metrics, outputs, summary }) {
        const output = outputMap(outputs);
        setOutput(output.get('kv-element-count'), metrics.elementCount, ' 个');
        setOutput(output.get('kv-total-bytes'), metrics.totalBytes, ' 字节');
        setOutput(output.get('kv-total-mib'), metrics.totalMiB, ' MiB');
        setOutput(output.get('kv-per-request-mib'), metrics.perRequestMiB, ' MiB');
        setOutput(output.get('kv-max-batch'), metrics.maxBatchSize, ' 条序列');
        summary.textContent = `按 M_KV = 2BLSH_kvd_hb 计算：总计 ${formatNumber(metrics.totalMiB)} MiB，每序列 ${formatNumber(metrics.perRequestMiB)} MiB；忽略碎片与元数据时，预算内最多容纳 ${formatNumber(metrics.maxBatchSize)} 条等长序列。`;
      },
    });
  }
}

class InferenceCostCalculatorElement extends HTMLElement {
  connectedCallback() {
    initializeCalculator(this, {
      prefix: 'cost-calculator',
      formTestid: 'inference-cost-form',
      summaryTestid: 'cost-summary',
      title: '推理服务成本计算器',
      controls: costControls,
      resultsMarkup: `
        <dl class="interactive-results">
          <div><dt>有效输出吞吐</dt><dd><output data-testid="cost-effective-throughput"></output></dd></div>
          <div><dt>整服务每小时成本</dt><dd><output data-testid="cost-hourly-total"></output></dd></div>
          <div><dt>每百万输出 Token 成本</dt><dd><output data-testid="cost-per-million"></output></dd></div>
          <div><dt>每请求成本</dt><dd><output data-testid="cost-per-request"></output></dd></div>
          <div><dt>稳态服务请求率</dt><dd><output data-testid="cost-requests-per-hour"></output></dd></div>
        </dl>
      `,
      calculate: calculateInferenceCost,
      render({ metrics, outputs, summary }) {
        const output = outputMap(outputs);
        setOutput(output.get('cost-effective-throughput'), metrics.effectiveThroughput, ' Token/s');
        setOutput(output.get('cost-hourly-total'), metrics.hourlyCost, ' 成本单位/h');
        setOutput(output.get('cost-per-million'), metrics.costPerMillion, ' 成本单位');
        setOutput(output.get('cost-per-request'), metrics.costPerRequest, ' 成本单位');
        setOutput(output.get('cost-requests-per-hour'), metrics.requestsPerHour, ' 请求/h');
        summary.textContent = `有效吞吐为 ${formatNumber(metrics.effectiveThroughput)} Token/s；每百万输出 Token 成本为 ${formatNumber(metrics.costPerMillion)}，每请求成本为 ${formatNumber(metrics.costPerRequest)} 成本单位。吞吐若已是实际负载测量值，请将利用率设为 100%，避免重复折算。`;
      },
    });
  }
}

export function defineInferenceCalculatorElement() {
  if (!customElements.get('llm-kv-cache-calculator')) {
    customElements.define('llm-kv-cache-calculator', KvCacheCalculatorElement);
  }
  if (!customElements.get('llm-inference-cost-calculator')) {
    customElements.define('llm-inference-cost-calculator', InferenceCostCalculatorElement);
  }
}
