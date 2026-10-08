(function () {
  const META = window.__META__;
  const FEATURES = META.features;
  const LAYERS = META.layers;
  const DEFAULT_RESOURCE = META.default_resource;

  const LABELS = {
    utilization: "Utilization",
    aspect_ratio: "Aspect Ratio",
    core_margin: "Core Margin",
    density: "Density",
    layer_adj: "Layer Adjustment",
  };
  const UNITS = {
    utilization: "%",
    aspect_ratio: "ratio",
    core_margin: "mm",
    density: "0\u20131",
    layer_adj: "0\u20131",
  };
  const STEP = {
    utilization: 0.1,
    aspect_ratio: 0.01,
    core_margin: 0.01,
    density: 0.01,
    layer_adj: 0.01,
  };

  const container = document.getElementById("param-controls");
  const state = {};
  let lastPredict = null; // { total_congestion_pct, success_probability, congestion_level, layer_congestion_pct }

  FEATURES.forEach((f) => {
    state[f] = 0;
  });

  FEATURES.forEach((f) => {
    const b = META.bounds[f];
    const wrap = document.createElement("label");
    wrap.className = "field";
    wrap.innerHTML = `
      <div class="field-label"><span>${LABELS[f] || f}</span><span class="unit">${UNITS[f]}</span></div>
      <input type="number" min="0" step="${STEP[f]}" value="0" data-feature="${f}" class="field-input">
    `;
    container.appendChild(wrap);
  });

  function syncFromNumber(e) {
    const f = e.target.dataset.feature;
    const v = parseFloat(e.target.value);
    if (Number.isNaN(v)) return;
    state[f] = v;
  }
  container.querySelectorAll(".field-input").forEach((el) => el.addEventListener("change", syncFromNumber));

  function applyValuesToInputs(values) {
    FEATURES.forEach((f) => {
      if (f in values) {
        state[f] = values[f];
        const input = container.querySelector(`.field-input[data-feature="${f}"]`);
        if (input) input.value = state[f];
      }
    });
  }

  // ---- Per-metal-layer resource inputs: real model input, read at Run time ----
  const inputGrid = document.getElementById("layer-input-grid");
  const RESOURCE_BOUNDS = META.resource_bounds;

  LAYERS.forEach((name, idx) => {
    const b = RESOURCE_BOUNDS[name];
    const item = document.createElement("div");
    item.className = "layer-input-item";
    item.innerHTML = `
      <label><span>Metal ${idx + 1}</span><span class="unit">number</span></label>
      <div class="layer-input-control">
        <button type="button" class="step-minus" data-layer="${name}">&minus;</button>
        <input type="number" class="layer-resource-input" data-layer="${name}" value="0" min="0" step="1">
        <button type="button" class="step-plus" data-layer="${name}">&plus;</button>
      </div>
      <div class="layer-input-range">${b.min.toLocaleString()} &ndash; ${b.max.toLocaleString()} in training data</div>
      <div class="layer-input-warning hidden" data-layer="${name}"></div>
    `;
    inputGrid.appendChild(item);
  });

  function currentResource() {
    const resource = {};
    LAYERS.forEach((name) => {
      const input = inputGrid.querySelector(`.layer-resource-input[data-layer="${name}"]`);
      const v = parseFloat(input.value);
      resource[name] = Number.isNaN(v) ? DEFAULT_RESOURCE[name] : v;
    });
    return resource;
  }

  function markOverridden(name) {
    const input = inputGrid.querySelector(`.layer-resource-input[data-layer="${name}"]`);
    input.classList.toggle("overridden", parseFloat(input.value) !== DEFAULT_RESOURCE[name]);
  }

  inputGrid.querySelectorAll(".layer-resource-input").forEach((input) => {
    input.addEventListener("input", () => markOverridden(input.dataset.layer));
  });
  inputGrid.querySelectorAll(".step-minus, .step-plus").forEach((btn) => {
    btn.addEventListener("click", () => {
      const name = btn.dataset.layer;
      const input = inputGrid.querySelector(`.layer-resource-input[data-layer="${name}"]`);
      const cur = parseFloat(input.value) || 0;
      const step = Math.max(1, Math.round(cur * 0.05) || Math.round(DEFAULT_RESOURCE[name] * 0.05) || 10);
      const delta = btn.classList.contains("step-plus") ? step : -step;
      input.value = Math.max(0, cur + delta).toFixed(0);
      markOverridden(name);
    });
  });

  // ---- Turbo-style colormap (blue -> teal -> green -> yellow -> orange -> red) ----
  const STOPS = [
    [0.00, [31, 42, 110]],
    [0.22, [24, 140, 170]],
    [0.45, [70, 180, 90]],
    [0.65, [210, 210, 60]],
    [0.82, [235, 150, 48]],
    [1.00, [220, 71, 55]],
  ];
  function turboColor(t) {
    t = Math.max(0, Math.min(1, t));
    let lo = STOPS[0], hi = STOPS[STOPS.length - 1];
    for (let i = 0; i < STOPS.length - 1; i++) {
      if (t >= STOPS[i][0] && t <= STOPS[i + 1][0]) { lo = STOPS[i]; hi = STOPS[i + 1]; break; }
    }
    const span = hi[0] - lo[0] || 1;
    const f = (t - lo[0]) / span;
    const rgb = lo[1].map((v, i) => Math.round(v + (hi[1][i] - v) * f));
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
  }

  function noise2(a, b) {
    const seed = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453;
    return seed - Math.floor(seed);
  }

  // ---- Top view: full-size heatmap square (50x50 cell resolution) ----
  function drawTopView(canvas, pct) {
    const cells = 50;
    const size = 240;
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d");
    const norm = Math.max(0, Math.min(1, pct / 90));
    const cellSize = size / cells;
    for (let r = 0; r < cells; r++) {
      for (let c = 0; c < cells; c++) {
        const n = noise2(r, c);
        const dx = (c - (cells - 1) / 2) / (cells / 2);
        const dy = (r - (cells - 1) / 2) / (cells / 2);
        const dist = Math.sqrt(dx * dx + dy * dy);
        const centerBoost = Math.max(0, 1 - dist);
        let v = norm * 0.5 + n * 0.35 + centerBoost * norm * 0.55;
        v = Math.max(0, Math.min(1, v));
        ctx.fillStyle = turboColor(v);
        ctx.fillRect(c * cellSize, r * cellSize, cellSize + 0.4, cellSize + 0.4);
      }
    }
  }

  // ---- Layer analysis: compact 2-col list, mini bar + FIX (model output) ----
  const analysisList = document.getElementById("layer-analysis-list");

  LAYERS.forEach((name) => {
    const label = name.replace("metal", "M");
    const row = document.createElement("div");
    row.className = "layer-analysis-row";
    row.id = `analysis-${name}`;
    row.innerHTML = `
      <span class="la-label">${label}</span>
      <span class="la-track"><span class="la-track-full"></span><span class="la-track-mask"></span></span>
      <span class="la-pct mono">&mdash;</span>
      <button class="la-fix-btn" data-layer="${name}">FIX</button>
    `;
    analysisList.appendChild(row);
  });
  analysisList.querySelectorAll(".la-fix-btn").forEach((btn) =>
    btn.addEventListener("click", () => showLayerFix(btn.dataset.layer))
  );

  // ---- Layer viewer: grid of top-view squares + ONE combined side view ----
  const layerGrid = document.getElementById("layer-grid");
  const stackView = document.getElementById("stack-view");

  LAYERS.forEach((name) => {
    const label = name.replace("metal", "M");

    const cell = document.createElement("div");
    cell.className = "layer-cell";
    cell.id = `cell-${name}`;
    cell.innerHTML = `
      <div class="layer-cell-head"><span>${label} &middot; ROUTING</span><span class="la-pct mono">&mdash;</span></div>
      <canvas></canvas>
      <div class="layer-cell-foot">derived pressure distribution</div>
    `;
    layerGrid.appendChild(cell);

    const stackRow = document.createElement("div");
    stackRow.className = "stack-row";
    stackRow.id = `stack-${name}`;
    stackRow.innerHTML = `
      <span class="stack-label">${label}</span>
      <span class="stack-track"><span class="stack-track-full"></span><span class="stack-track-mask"></span></span>
      <span class="stack-pct mono">&mdash;</span>
      <button class="stack-fix-btn" data-layer="${name}">Fix this layer</button>
    `;
    stackView.appendChild(stackRow);
  });
  stackView.querySelectorAll(".stack-fix-btn").forEach((btn) =>
    btn.addEventListener("click", () => showLayerFix(btn.dataset.layer))
  );

  function maskWidth(pct) {
    const shown = Math.max(0, Math.min(100, pct));
    return (100 - shown) + "%";
  }

  function renderLayers(perLayer) {
    LAYERS.forEach((name) => {
      const pct = perLayer[`${name}_usage_pct`];

      const analysisRow = document.getElementById(`analysis-${name}`);
      analysisRow.querySelector(".la-track-mask").style.width = maskWidth(pct);
      analysisRow.querySelector(".la-pct").textContent = pct.toFixed(1) + "%";

      const cell = document.getElementById(`cell-${name}`);
      cell.querySelector(".la-pct").textContent = pct.toFixed(1) + "%";
      drawTopView(cell.querySelector("canvas"), pct);

      const stackRow = document.getElementById(`stack-${name}`);
      stackRow.querySelector(".stack-track-mask").style.width = maskWidth(pct);
      stackRow.querySelector(".stack-pct").textContent = pct.toFixed(1) + "%";
    });
  }

  function renderOutput(data) {
    const pill = document.getElementById("verdict-pill");
    const labels = { low: "ROUTABLE", moderate: "MODERATE CONGESTION", high: "CONGESTION DETECTED", critical: "CONGESTION DETECTED" };
    pill.textContent = labels[data.congestion_level] || data.congestion_level.toUpperCase();
    pill.className = "verdict-pill " + data.congestion_level;

    document.getElementById("congestion-value").textContent = data.total_congestion_pct.toFixed(1) + "%";
    document.getElementById("success-value").textContent = (data.success_probability * 100).toFixed(1) + "%";

    const foot = document.getElementById("congestion-foot");
    if (foot) {
      foot.textContent = `Peak layer: ${data.peak_layer} at ${data.peak_layer_pct.toFixed(1)}%`;
      foot.classList.toggle("peak-warning", data.peak_layer_pct >= 85);
    }

    if (data.thermal) {
      const tPill = document.getElementById("thermal-pill");
      const tLabels = { low: "LOW RISK", moderate: "MODERATE RISK", high: "ELEVATED RISK", critical: "HIGH RISK" };
      tPill.textContent = tLabels[data.thermal.level] || data.thermal.level.toUpperCase();
      tPill.className = "verdict-pill " + data.thermal.level;
      document.getElementById("thermal-value").textContent = data.thermal.risk_index.toFixed(1);
      document.getElementById("thermal-foot").textContent = `Hotspot driver: ${data.thermal.hotspot_layer}`;
    }
  }

  // ---- Recommend panel: manual, on-demand, never auto-applied ----
  function resetRecommendPanel() {
    const title = document.getElementById("recommend-title");
    const body = document.getElementById("recommend-body");
    title.textContent = "Resolve congestion";
    if (!lastPredict) {
      body.innerHTML = `<p class="hint">Run the model to see whether a fix is needed.</p>`;
      return;
    }
    if (lastPredict.total_congestion_pct <= 20) {
      body.innerHTML = `<p class="hint">Congestion is already low &mdash; no fix needed. You can still explore a lower target below.</p>
        <button class="btn btn-ghost" id="suggest-fix-btn" style="margin-top:10px;">Suggest a fix anyway</button>`;
    } else {
      body.innerHTML = `<p class="hint">Predicted usage is ${lastPredict.total_congestion_pct.toFixed(1)}%. Get the exact parameter changes to bring it down.</p>
        <button class="btn btn-ghost" id="suggest-fix-btn" style="margin-top:10px;">Suggest a fix</button>`;
    }
    document.getElementById("suggest-fix-btn").addEventListener("click", () => suggestFix(false));
  }

  async function callRecommend(targetPct, layer, widen) {
    const payload = Object.assign({}, state, { target_pct: targetPct, widen: !!widen, resource: currentResource() });
    if (layer) payload.layer = layer;
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    return res.json();
  }

  function renderRecommendation(result, opts) {
    const title = document.getElementById("recommend-title");
    const body = document.getElementById("recommend-body");
    title.textContent = opts.layer ? `Resolve ${opts.layer.replace("metal", "M")} congestion` : "Resolve congestion";

    if (!result.achieved) {
      body.innerHTML = `
        <p class="rec-empty">${result.message}</p>
        ${result.can_widen ? `<button class="btn btn-ghost" id="widen-btn" style="margin-top:10px;">Search the full parameter space</button>` : ""}
      `;
      const widenBtn = document.getElementById("widen-btn");
      if (widenBtn) widenBtn.addEventListener("click", async () => {
        const r = await callRecommend(opts.targetPct, opts.layer, true);
        renderRecommendation(r, opts);
      });
      return;
    }

    const items = result.changes.length
      ? result.changes.map((c) => `
          <li>
            <span class="rec-param">${LABELS[c.parameter] || c.parameter}</span>
            <span class="rec-values"><span class="from">${c.from}</span><span class="arrow">&rarr;</span><span class="to">${c.to}</span></span>
          </li>`).join("")
      : `<li><span class="rec-param">No change needed</span><span class="rec-values">already under target</span></li>`;

    const beforePct = opts.beforePct.toFixed(1);
    const afterPct = result.predicted_target_metric_pct.toFixed(1);
    const improvement = opts.beforePct > 0 ? (((opts.beforePct - result.predicted_target_metric_pct) / opts.beforePct) * 100).toFixed(1) : "0.0";

    body.innerHTML = `
      <ul class="rec-changes">${items}</ul>
      <div class="rec-callout">
        Predicted usage: <b>${beforePct}% &rarr; ${afterPct}%</b> &middot; improvement <b>${improvement}%</b> &middot;
        implementation success <b>${(result.predicted_success_prob * 100).toFixed(1)}%</b>
      </div>
      <p class="hint" style="margin-top:10px;">Only the 5 design parameters are changed here &mdash; routing resource is treated as a fixed property of your metal stack.</p>
      <button class="btn btn-primary" id="apply-fix-btn" style="margin-top:10px;">Apply these values &amp; re-run</button>
    `;

    document.getElementById("apply-fix-btn").addEventListener("click", () => {
      applyValuesToInputs(result.new_values);
      run();
    });
  }

  async function suggestFix(widen) {
    if (!lastPredict) return;
    const target = Math.max(5, Math.round(lastPredict.total_congestion_pct * 0.7 * 10) / 10);
    const result = await callRecommend(target, null, widen);
    renderRecommendation(result, { beforePct: lastPredict.total_congestion_pct, targetPct: target });
  }

  async function showLayerFix(layer) {
    if (!lastPredict) return;
    const currentPct = lastPredict.layer_congestion_pct[`${layer}_usage_pct`];
    const target = Math.max(1, Math.round(currentPct * 0.7 * 10) / 10);
    const result = await callRecommend(target, layer, false);
    renderRecommendation(result, { beforePct: currentPct, layer, targetPct: target });
    document.getElementById("recommend-panel").scrollIntoView({ behavior: "smooth", block: "center" });
  }

  // ---- Screen navigation ----
  const screenInput = document.getElementById("screen-input");
  const screenResults = document.getElementById("screen-results");
  const readonlyContainer = document.getElementById("param-controls-readonly");
  const specChipRow = document.getElementById("spec-chip-row");

  function showInput() {
    screenResults.classList.add("hidden");
    screenInput.classList.remove("hidden");
    window.scrollTo(0, 0);
  }
  function showResults() {
    screenInput.classList.add("hidden");
    screenResults.classList.remove("hidden");
    window.scrollTo(0, 0);
  }
  document.getElementById("back-btn").addEventListener("click", showInput);
  document.getElementById("back-btn-2").addEventListener("click", showInput);

  function renderReadonlySummary() {
    readonlyContainer.innerHTML = FEATURES.map((f) => `
      <div class="readonly-field">
        <div class="field-label"><span>${LABELS[f] || f}</span><span class="unit">${UNITS[f]}</span></div>
        <div class="field-value">${state[f]}</div>
      </div>
    `).join("");
    specChipRow.innerHTML = FEATURES.map((f) => `<span class="spec-chip">${LABELS[f] || f}: <b>${state[f]}</b></span>`).join("");

    const resource = currentResource();
    const resourceContainer = document.getElementById("resource-readonly");
    resourceContainer.innerHTML = LAYERS.map((name, idx) => `
      <div class="readonly-field">
        <div class="field-label"><span>Metal ${idx + 1}</span></div>
        <div class="field-value">${resource[name].toLocaleString()}</div>
      </div>
    `).join("");
  }

  // ---- Predict: reads the 5 design params AND the per-layer resource inputs ----
  async function run() {
    const payload = Object.assign({}, state, { resource: currentResource() });
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    lastPredict = data;
    applyResourceAdjustments(data.resource_adjusted);
    renderOutput(data);
    renderLayers(data.layer_congestion_pct);
    resetRecommendPanel();
    renderReadonlySummary();
    showResults();
  }

  function applyResourceAdjustments(adjusted) {
    LAYERS.forEach((name) => {
      const warn = inputGrid.querySelector(`.layer-input-warning[data-layer="${name}"]`);
      if (adjusted && adjusted[name]) {
        const { requested, used, realistic_range } = adjusted[name];
        warn.textContent = `${requested.toLocaleString()} is outside the training data's realistic range for this layer (${realistic_range.min.toLocaleString()}\u2013${realistic_range.max.toLocaleString()}) \u2014 used ${used.toLocaleString()} instead so the result stays meaningful.`;
        warn.classList.remove("hidden");
        const input = inputGrid.querySelector(`.layer-resource-input[data-layer="${name}"]`);
        input.value = used.toFixed(0);
      } else if (warn) {
        warn.classList.add("hidden");
        warn.textContent = "";
      }
    });
  }

  document.getElementById("predict-btn").addEventListener("click", run);

  // Draw empty views at 0 so nothing is blank before the first run.
  renderLayers(Object.fromEntries(LAYERS.map((l) => [`${l}_usage_pct`, 0])));
})();
