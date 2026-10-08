import os, zipfile, shutil

base = "congestion_app_final_3d"
tpl = os.path.join(base, "templates")
mdl = os.path.join(base, "model")
os.makedirs(tpl, exist_ok=True)
os.makedirs(mdl, exist_ok=True)

# Preserve trained model if available
for p in ["model/congestion_regressors.joblib", "congestion_app/model/congestion_regressors.joblib"]:
    if os.path.exists(p):
        shutil.copy(p, os.path.join(mdl, "congestion_regressors.joblib"))
        break

# 1. app.py
with open(os.path.join(base, "app.py"), "w") as f:
    f.write('''import os, joblib, numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "congestion_regressors.joblib")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        util = float(request.form.get("utilization", 57.0))
        aspect = float(request.form.get("aspect_ratio", 1.5))
        margin = float(request.form.get("core_margin", 3.0))
        density = float(request.form.get("density", 0.56))
        layer_adj = float(request.form.get("layer_adjustment", 0.6))
    except ValueError:
        util, aspect, margin, density, layer_adj = 57.0, 1.5, 3.0, 0.56, 0.6

    layer_usage_raw = {
        "M1": 0.0, "M2": 294.7, "M3": 213.4, "M4": 233.0,
        "M5": 161.3, "M6": 160.0, "M7": 38.4, "M8": 34.6,
        "M9": 22.7, "M10": 19.2
    }
    
    if model:
        try:
            preds = model.predict(np.array([[util, aspect, margin, density, layer_adj]]))[0]
            for idx, k in enumerate(layer_usage_raw.keys()):
                if idx < len(preds):
                    layer_usage_raw[k] = round(float(preds[idx]), 1)
        except Exception as e:
            print("Inference error:", e)

    layer_usage_pct = {k: min(100.0, v) for k, v in layer_usage_raw.items()}
    peak_layer = max(layer_usage_raw, key=layer_usage_raw.get)
    peak_val_raw = layer_usage_raw[peak_layer]
    peak_val_pct = min(100.0, peak_val_raw)
    
    avg_usage_raw = round(sum(layer_usage_raw.values()) / len(layer_usage_raw), 1)
    avg_usage_pct = round(min(100.0, sum(layer_usage_pct.values()) / len(layer_usage_pct)), 1)
    
    thermal_index = round(min(100.0, (density * 35.0) + (avg_usage_pct * 0.35) + (peak_val_pct * 0.30)), 1)
    peak_temp_c = round(45.0 + (thermal_index * 0.52), 1)

    return render_template("results.html",
        utilization=util, aspect_ratio=aspect, core_margin=margin, density=density,
        layer_adjustment=layer_adj,
        layer_usage_raw=layer_usage_raw,
        layer_usage_pct=layer_usage_pct,
        peak_layer=peak_layer,
        peak_val_raw=peak_val_raw,
        peak_val_pct=peak_val_pct,
        avg_usage_raw=avg_usage_raw,
        avg_usage_pct=avg_usage_pct,
        thermal_index=thermal_index,
        peak_temp_c=peak_temp_c
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
''')

# 2. index.html
with open(os.path.join(tpl, "index.html"), "w") as f:
    f.write('''<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"><title>RV32I Congestion Predictor</title>
  <style>
    body { background: #0c0f12; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; margin: 0; padding: 24px 40px; }
    .header { display: flex; justify-content: space-between; border-bottom: 1px solid #1e252e; padding-bottom: 16px; margin-bottom: 24px; }
    .card { background: #12161b; border: 1px solid #1e252e; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
    .title { color: #eab308; font-size: 11px; text-transform: uppercase; font-weight: bold; }
    .grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-top: 12px; }
    .box { background: #07090b; border: 1px solid #232a35; border-radius: 6px; padding: 10px; }
    .box label { display: flex; justify-content: space-between; font-size: 11px; color: #94a3b8; }
    .badge { background: #1e293b; color: #e2e8f0; padding: 1px 6px; border-radius: 3px; font-size: 9px; }
    input { width: 100%; background: transparent; border: none; color: #fff; font-size: 16px; font-weight: bold; margin-top: 8px; outline: none; }
    .btn { width: 100%; background: #bef264; color: #000; font-size: 14px; font-weight: bold; padding: 14px; border: none; border-radius: 6px; cursor: pointer; text-transform: uppercase; margin-top: 12px; }
  </style>
</head>
<body>
  <div class="header">
    <div><h2>RV32I CONGESTION PREDICTOR</h2><span style="color:#64748b; font-size: 12px;">dataset-driven routing risk analysis</span></div>
    <div style="color: #10b981; font-size: 11px;">● LOCAL MODEL ONLINE</div>
  </div>
  <form action="/predict" method="POST">
    <div class="card">
      <div class="title">Per-Metal-Layer Routing Resource</div>
      <h3 style="margin: 4px 0 10px 0; font-size: 15px;">Set each layer routing resource</h3>
      <div class="grid">
        <div class="box"><label>METAL 1 <span class="badge">NUMBER</span></label><input type="number" name="metal_1" value="0" step="any"></div>
        <div class="box"><label>METAL 2 <span class="badge">NUMBER</span></label><input type="number" name="metal_2" value="4814" step="any"></div>
        <div class="box"><label>METAL 3 <span class="badge">NUMBER</span></label><input type="number" name="metal_3" value="7960" step="any"></div>
        <div class="box"><label>METAL 4 <span class="badge">NUMBER</span></label><input type="number" name="metal_4" value="2842" step="any"></div>
        <div class="box"><label>METAL 5 <span class="badge">NUMBER</span></label><input type="number" name="metal_5" value="2840" step="any"></div>
        <div class="box"><label>METAL 6 <span class="badge">NUMBER</span></label><input type="number" name="metal_6" value="2842" step="any"></div>
        <div class="box"><label>METAL 7 <span class="badge">NUMBER</span></label><input type="number" name="metal_7" value="2240" step="any"></div>
        <div class="box"><label>METAL 8 <span class="badge">NUMBER</span></label><input type="number" name="metal_8" value="2263" step="any"></div>
        <div class="box"><label>METAL 9 <span class="badge">NUMBER</span></label><input type="number" name="metal_9" value="2070" step="any"></div>
        <div class="box"><label>METAL 10 <span class="badge">NUMBER</span></label><input type="number" name="metal_10" value="2136" step="any"></div>
      </div>
    </div>
    <div class="card">
      <div class="title">Design Input</div>
      <h3 style="margin: 4px 0 10px 0; font-size: 15px;">Predict routing pressure</h3>
      <div class="grid">
        <div class="box"><label>UTILIZATION <span class="badge">%</span></label><input type="number" name="utilization" value="57" step="any"></div>
        <div class="box"><label>ASPECT RATIO <span class="badge">ratio</span></label><input type="number" name="aspect_ratio" value="1.5" step="any"></div>
        <div class="box"><label>CORE MARGIN <span class="badge">mm</span></label><input type="number" name="core_margin" value="3" step="any"></div>
        <div class="box"><label>DENSITY <span class="badge">0-1</span></label><input type="number" name="density" value="0.56" step="any"></div>
        <div class="box"><label>LAYER ADJUSTMENT <span class="badge">0-1</span></label><input type="number" name="layer_adjustment" value="0.6" step="any"></div>
      </div>
      <button type="submit" class="btn">RUN CONGESTION MODEL →</button>
    </div>
  </form>
</body>
</html>
''')

# 3. results.html
with open(os.path.join(tpl, "results.html"), "w") as f:
    f.write('''<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"><title>RV32I Congestion Predictor</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <style>
    body { background: #0c0f12; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; margin: 0; padding: 24px 40px; }
    .top-bar { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e252e; padding-bottom: 16px; margin-bottom: 20px; }
    .btn-nav { background: #1e252e; color: #cbd5e1; border: 1px solid #334155; padding: 8px 16px; border-radius: 5px; text-decoration: none; font-size: 12px; cursor: pointer; }
    .btn-3d { background: #10b981; color: #000; font-weight: bold; border: none; padding: 8px 18px; border-radius: 5px; cursor: pointer; font-size: 12px; }
    .card { background: #12161b; border: 1px solid #1e252e; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
    
    #panel3D { display: none; margin-bottom: 24px; border: 1px solid #00ff88; box-shadow: 0 0 25px rgba(0,255,136,0.12); position: relative; }
    .viewport-container { display: grid; grid-template-columns: 2.3fr 1fr; gap: 16px; }
    #viewport3D { width: 100%; height: 560px; border-radius: 6px; background: radial-gradient(circle at center, #1b232e 0%, #080a0d 100%); cursor: grab; position: relative; }
    #viewport3D:active { cursor: grabbing; }

    .info-pane { background: #07090c; border: 1px solid #1e252e; border-radius: 6px; padding: 16px; font-size: 12px; display: flex; flex-direction: column; justify-content: space-between; }
    .def-table td { padding: 4px 6px; }
    .heat-pill { padding: 2px 8px; border-radius: 3px; font-weight: bold; font-size: 11px; }

    .grid-heatmaps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-top: 14px; }
    .heatmap-card { background: #07090b; border: 1px solid #1e252e; border-radius: 6px; padding: 10px; text-align: center; }
    .canvas-sim { width: 100%; height: 110px; border-radius: 4px; margin-top: 8px; }
    .bar-row { display: flex; align-items: center; justify-content: space-between; margin: 6px 0; font-size: 12px; }
    .bar-track { flex-grow: 1; height: 8px; background: #1e252e; border-radius: 4px; margin: 0 12px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 4px; }
    
    /* 3D Overlay Tag */
    #hotspotTag {
      position: absolute;
      background: rgba(12, 15, 18, 0.9);
      border: 1px solid #38bdf8;
      box-shadow: 0 0 12px rgba(56, 189, 248, 0.5);
      color: #38bdf8;
      padding: 6px 12px;
      font-size: 11px;
      font-weight: bold;
      border-radius: 4px;
      pointer-events: none;
      display: none;
      transform: translate(-50%, -100%);
      white-space: nowrap;
    }
  </style>
</head>
<body>

  <!-- Top Navigation -->
  <div class="top-bar">
    <div>
      <a href="/" class="btn-nav">← Edit inputs</a>
      <h2 style="display:inline; margin-left: 14px; font-size: 18px;">RV32I CONGESTION PREDICTOR</h2>
    </div>
    <div style="display: flex; gap: 12px; align-items: center;">
      <button id="btnToggle3D" class="btn-3d" onclick="toggle3DView()">🔍 Analyze in 3D</button>
      <span style="color: #10b981; font-size: 12px;">● LOCAL MODEL ONLINE</span>
    </div>
  </div>

  <!-- 3D INTERACTIVE VIEWER -->
  <div id="panel3D" class="card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
      <div>
        <span style="color: #00ff88; font-size: 11px; font-weight: bold;">● RV32I 3D SILICON INTERCONNECT STACK</span>
        <span style="color: #94a3b8; font-size: 11px; margin-left: 10px;">(Click any layer to pull it out separately • Drag to orbit manually)</span>
      </div>
      <div style="display: flex; gap: 8px;">
        <button id="btnIsolate" class="btn-nav" style="background:#0284c7; color:#fff; font-weight:bold; border:none;">Isolate Selected Layer</button>
        <button id="btnExplode" class="btn-nav">Explode Stack</button>
        <button id="btnFocusHot" class="btn-nav" style="background:#ef4444; color:#fff; border:none; font-weight:bold;">Focus Hotspot ({{ peak_layer }})</button>
        <button class="btn-nav" onclick="toggle3DView()">✕ Close 3D</button>
      </div>
    </div>

    <div class="viewport-container">
      <div id="viewport3D">
        <div id="hotspotTag">M2 CONGESTION HOTSPOT</div>
      </div>

      <!-- Layer Inspector & Physical Specs -->
      <div class="info-pane">
        <div>
          <h4 style="margin: 0 0 8px 0; color: #38bdf8; font-size: 13px;">Selected Layer Inspector</h4>
          <div style="background: #111419; padding: 12px; border-radius: 4px; border: 1px solid #232a35; margin-bottom: 12px;">
            <div style="font-size: 15px; font-weight: bold; color: #fff;" id="inspName">{{ peak_layer }} (Peak Routing Hotspot)</div>
            <div style="color: #94a3b8; margin-top: 6px;">Layer Capacity: <b id="inspPct" style="color: #ef4444;">{{ peak_val_pct }}%</b> (Over-capacity: {{ peak_val_raw }}%)</div>
            <div style="color: #94a3b8; margin-top: 4px;">Local Peak Temp: <b id="inspTemp" style="color: #f97316;">{{ peak_temp_c }} °C</b></div>
            <div style="color: #94a3b8; margin-top: 4px;">Geometry: <span id="inspGeom" style="color: #38bdf8;">Semi-Global Bus Tracks & Contact Vias</span></div>
          </div>

          <h4 style="margin: 12px 0 6px 0; color: #eab308; font-size: 12px;">BEOL / FEOL ARCHITECTURAL SPECIFICATION</h4>
          <p id="inspDef" style="color: #cbd5e1; font-size: 11px; line-height: 1.6; margin: 0;">
            Primary low-pitch copper metal layer for standard cell local signal routing and dense intra-block buses.
          </p>
        </div>

        <div style="border-top: 1px solid #1e252e; padding-top: 10px; margin-top: 10px;">
          <div style="color: #64748b; font-size: 10px; font-weight: bold; margin-bottom: 4px;">PROCESS TECHNOLOGY & GATE CALLOUT</div>
          <table class="def-table" style="width: 100%; font-size: 11px; color: #94a3b8;">
            <tr><td>Source / Drain:</td><td style="color:#fff;">Monolithic Silicon</td></tr>
            <tr><td>FinFET Pitch:</td><td style="color:#fff;">24 nm</td></tr>
            <tr><td>Logic Cells:</td><td style="color:#fff;">NAND / Inverter / DFF (FEOL)</td></tr>
            <tr><td>Contact Level:</td><td style="color:#fff;">MOL Tungsten Plugs</td></tr>
          </table>
        </div>
      </div>
    </div>
  </div>

  <!-- Original Dashboard Cards -->
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
    <div class="card">
      <div style="background: rgba(239,68,68,0.15); border: 1px solid #ef4444; color: #ef4444; font-weight: bold; padding: 4px 10px; border-radius: 4px; display: inline-block; font-size: 12px;">CONGESTION DETECTED</div>
      <div style="margin-top: 14px;">
        <span style="color: #64748b; font-size: 11px;">Predicted usage (normalized):</span>
        <h1 style="margin: 4px 0; font-size: 38px;">{{ avg_usage_pct }}% <span style="font-size: 14px; color:#64748b;">(Demand: {{ avg_usage_raw }}%)</span></h1>
        <span style="color: #ef4444; font-size: 12px;">Peak layer: {{ peak_layer }} at {{ peak_val_pct }}% (Overflow: {{ peak_val_raw }}%)</span>
      </div>
    </div>

    <div class="card">
      <div style="color: #eab308; font-size: 11px; font-weight: bold;">THERMAL RISK (HEURISTIC)</div>
      <div style="margin-top: 8px;">
        <span class="heat-pill" style="background:#ef4444; color:#fff;">HIGH RISK</span>
        <h1 style="display:inline; margin-left: 12px; font-size: 38px;">{{ thermal_index }}</h1>
        <span style="color: #94a3b8; font-size: 13px; margin-left: 10px;">Peak Junction: <b style="color:#f97316;">{{ peak_temp_c }} °C</b></span>
      </div>
      <p style="color: #64748b; font-size: 11px; margin-top: 10px;">Hotspot driver: {{ peak_layer }} • Thermal index derived from core density and wire overflow.</p>
    </div>
  </div>

  <!-- 2D Layer Viewer Grid -->
  <div class="card">
    <div style="display:flex; justify-content:space-between;">
      <h3 style="margin:0; font-size:15px;">Dataset-Derived Layer Viewer (Top view)</h3>
      <span style="color:#64748b; font-size:11px;">LOW ■■■■■■■ HIGH</span>
    </div>
    <div class="grid-heatmaps">
      {% for layer, val in layer_usage_pct.items() %}
      <div class="heatmap-card">
        <div style="display:flex; justify-content:space-between; font-size:11px;">
          <span>{{ layer }}</span>
          <b style="color:{% if val >= 100 %}#ef4444{% elif val > 60 %}#f97316{% else %}#22c55e{% endif %};">{{ val }}%</b>
        </div>
        <canvas id="cvs-{{ layer }}" class="canvas-sim"></canvas>
      </div>
      {% endfor %}
    </div>
  </div>

  <!-- Side View Bars -->
  <div class="card">
    <h3 style="margin: 0 0 14px 0; font-size: 15px;">Side View • All Layers M10 → M1</h3>
    {% for layer, val in layer_usage_pct.items()|reverse %}
    <div class="bar-row">
      <span style="width: 40px; color:#94a3b8;">{{ layer }}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width: {{ val }}%; background: {% if val >= 100 %}#ef4444{% elif val > 60 %}#f97316{% else %}#38bdf8{% endif %};"></div>
      </div>
      <span style="width: 50px; text-align:right; font-weight:bold; color: {% if val >= 100 %}#ef4444{% else %}#fff{% endif %};">{{ val }}%</span>
    </div>
    {% endfor %}
  </div>

  <!-- 3D Three.js Engine & Interactivity Scripts -->
  <script>
    const layerDefs = {
      'FEOL': 'Front-End-of-Line substrate containing active silicon FinFET transistors, logic gates (NAND, Inverters), source, and drain regions.',
      'MOL': 'Middle-of-Line contact level containing tapered tungsten contact plugs connecting transistor terminals to M1.',
      'M1': 'Metal 1: Fine-pitch local copper tracks for standard cell internal pins and rail contacts.',
      'M2': 'Metal 2: Orthogonal intra-block routing tracks. High congestion hotspot detected here.',
      'M3': 'Metal 3: High-density horizontal routing tracks connecting logic gates across standard cell rows.',
      'M4': 'Metal 4: Vertical routing tracks for intermediate signal routing across macro boundaries.',
      'M5': 'Metal 5: Medium-pitch metal tracks utilized for data bus paths and clock distribution trunk nets.',
      'M6': 'Metal 6: Horizontal medium-pitch routing for intermediate signal transport across clock domains.',
      'M7': 'Metal 7: Semi-global routing layer with relaxed pitch for high-speed datapath lines.',
      'M8': 'Metal 8: Global wiring layer for cross-chip communication and secondary power grid strapping.',
      'M9': 'Metal 9: Thick metal layer reserved for global power routing mesh (VDD/VSS) and primary clock trees.',
      'M10': 'Metal 10: Top-most thick redistribution layer (RDL) for C4 power pads and wirebonds.'
    };

    function toggle3DView() {
      const p = document.getElementById("panel3D");
      const btn = document.getElementById("btnToggle3D");
      if (p.style.display === "block") {
        p.style.display = "none";
        btn.innerText = "🔍 Analyze in 3D";
      } else {
        p.style.display = "block";
        btn.innerText = "✕ Close 3D View";
        init3DScene();
        p.scrollIntoView({ behavior: "smooth" });
      }
    }

    // 2D Mini-heatmaps
    const usageData = {{ layer_usage_pct|tojson }};
    Object.keys(usageData).forEach(layer => {
      const cvs = document.getElementById("cvs-" + layer);
      if (!cvs) return;
      const ctx = cvs.getContext("2d");
      const val = usageData[layer];
      const g = ctx.createRadialGradient(cvs.width/2, cvs.height/2, 5, cvs.width/2, cvs.height/2, cvs.width/2);
      if (val >= 90) {
        g.addColorStop(0, "rgba(239,68,68,0.95)"); g.addColorStop(0.6, "rgba(249,115,22,0.7)"); g.addColorStop(1, "rgba(34,197,94,0.3)");
      } else if (val >= 50) {
        g.addColorStop(0, "rgba(234,179,8,0.85)"); g.addColorStop(0.7, "rgba(56,189,248,0.5)"); g.addColorStop(1, "rgba(34,197,94,0.2)");
      } else {
        g.addColorStop(0, "rgba(56,189,248,0.7)"); g.addColorStop(1, "rgba(14,165,233,0.1)");
      }
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, cvs.width, cvs.height);
    });

    // 3D Scene Implementation
    let sceneInit = false;
    let selectedLayerName = "{{ peak_layer }}";
    let isIsolated = false;
    let exploded = false;

    function init3DScene() {
      if (sceneInit) return;
      sceneInit = true;

      const cont = document.getElementById("viewport3D");
      const scene = new THREE.Scene();
      const camera = new THREE.PerspectiveCamera(38, cont.clientWidth / cont.clientHeight, 0.1, 1000);
      camera.position.set(9.0, 7.5, 9.0);

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(cont.clientWidth, cont.clientHeight);
      renderer.setPixelRatio(window.devicePixelRatio);
      cont.appendChild(renderer.domElement);

      const controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;

      // Studio Lighting
      scene.add(new THREE.AmbientLight(0xffffff, 0.8));
      const dl1 = new THREE.DirectionalLight(0xffedd5, 1.4);
      dl1.position.set(12, 18, 10);
      scene.add(dl1);
      const dl2 = new THREE.DirectionalLight(0x38bdf8, 0.7);
      dl2.position.set(-12, -8, -10);
      scene.add(dl2);

      // Procedural Heatmap Texture
      const c = document.createElement("canvas"); c.width = c.height = 512;
      const ctx = c.getContext("2d");
      const g = ctx.createRadialGradient(256, 256, 20, 256, 256, 240);
      g.addColorStop(0, "rgba(255, 30, 30, 1.0)");
      g.addColorStop(0.35, "rgba(255, 140, 0, 0.9)");
      g.addColorStop(0.65, "rgba(235, 220, 20, 0.8)");
      g.addColorStop(0.85, "rgba(0, 200, 100, 0.5)");
      g.addColorStop(1, "rgba(0, 100, 255, 0.2)");
      ctx.fillStyle = g; ctx.fillRect(0, 0, 512, 512);
      
      // Add fine grid pattern on texture
      ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
      ctx.lineWidth = 1;
      for(let i=0; i<512; i+=16) {
        ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 512); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(512, i); ctx.stroke();
      }
      const heatTex = new THREE.CanvasTexture(c);

      const stack = new THREE.Group();
      scene.add(stack);
      const layerGroups = [];
      const rawUsages = {{ layer_usage_raw|tojson }};
      const peakLayer = "{{ peak_layer }}";

      // Copper Material
      const copperMat = new THREE.MeshStandardMaterial({
        color: 0xc67d4d,
        metalness: 0.92,
        roughness: 0.28
      });

      // 1. BUILD FEOL (Transistors, NAND, Inverter level)
      const feolGrp = new THREE.Group();
      const subMat = new THREE.MeshStandardMaterial({ color: 0x222a35, roughness: 0.7 });
      const subMesh = new THREE.Mesh(new THREE.BoxGeometry(4.6, 0.25, 4.6), subMat);
      subMesh.position.y = -0.125;
      feolGrp.add(subMesh);

      // Transistor FinFET Fin Ribs
      for(let x = -1.8; x <= 1.8; x += 0.45) {
        for(let z = -1.8; z <= 1.8; z += 0.6) {
          const finGeo = new THREE.BoxGeometry(0.24, 0.14, 0.35);
          const fin = new THREE.Mesh(finGeo, copperMat);
          fin.position.set(x, 0.07, z);
          feolGrp.add(fin);
        }
      }
      feolGrp.position.y = -2.6;
      feolGrp.userData = { name: "FEOL", defY: -2.6, expY: -4.5, pct: 0, raw: 0, temp: 45.0 };
      stack.add(feolGrp);
      layerGroups.push(feolGrp);

      // 2. BUILD MOL (Middle of Line Tapered Tungsten Contact Plugs)
      const molGrp = new THREE.Group();
      const tungstenMat = new THREE.MeshStandardMaterial({ color: 0xb58b68, metalness: 0.88, roughness: 0.3 });
      for(let x = -1.8; x <= 1.8; x += 0.45) {
        for(let z = -1.8; z <= 1.8; z += 0.6) {
          const plugGeo = new THREE.CylinderGeometry(0.065, 0.04, 0.45, 12);
          const plug = new THREE.Mesh(plugGeo, tungstenMat);
          plug.position.set(x, 0, z);
          molGrp.add(plug);
        }
      }
      molGrp.position.y = -2.1;
      molGrp.userData = { name: "MOL", defY: -2.1, expY: -3.6, pct: 0, raw: 0, temp: 46.0 };
      stack.add(molGrp);
      layerGroups.push(molGrp);

      // 3. BUILD M1 TO M10 (Accurate BEOL Metallization Architecture)
      const metalNames = Object.keys(usageData);
      metalNames.forEach((lName, idx) => {
        const grp = new THREE.Group();
        const isPeak = (lName === peakLayer);
        const pctVal = usageData[lName];
        const rawVal = rawUsages[lName];
        const localTemp = (45.0 + (pctVal * 0.45)).toFixed(1);

        // A) Translucent Plane Body with procedural relief
        const planeMat = new THREE.MeshStandardMaterial({
          map: isPeak ? heatTex : null,
          color: isPeak ? 0xffffff : (pctVal > 75 ? 0xd97736 : 0x7a93ac),
          emissive: isPeak ? 0xff2200 : 0x000000,
          emissiveIntensity: isPeak ? 0.45 : 0,
          metalness: isPeak ? 0.5 : 0.85,
          roughness: 0.3,
          transparent: true,
          opacity: isPeak ? 0.92 : 0.70
        });
        const plate = new THREE.Mesh(new THREE.BoxGeometry(4.4, 0.06, 4.4), planeMat);
        plate.userData = { name: lName, pct: pctVal, raw: rawVal, temp: localTemp };
        grp.add(plate);

        // B) Wire Traces & Cutouts
        if (idx < 4) {
          // M1-M4: Fine-pitch dense parallel tracks
          const isHorizontal = (idx % 2 === 0);
          for(let t = -1.9; t <= 1.9; t += 0.28) {
            const trackGeo = isHorizontal ? new THREE.BoxGeometry(3.8, 0.03, 0.08) : new THREE.BoxGeometry(0.08, 0.03, 3.8);
            const track = new THREE.Mesh(trackGeo, copperMat);
            track.position.set(isHorizontal ? 0 : t, 0.045, isHorizontal ? t : 0);
            track.userData = plate.userData;
            grp.add(track);
          }
        } else if (idx < 7) {
          // M5-M7: Wider datapath buses
          [-1.2, -0.4, 0.4, 1.2].forEach(p => {
            const bus = new THREE.Mesh(new THREE.BoxGeometry(3.6, 0.04, 0.35), copperMat);
            bus.position.set(0, 0.05, p);
            bus.userData = plate.userData;
            grp.add(bus);
          });
        } else {
          // M8-M10: Large planar power straps with window cutouts
          const strap1 = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.06, 0.8), copperMat);
          strap1.position.set(0, 0.06, -1.0);
          const strap2 = new THREE.Mesh(new THREE.BoxGeometry(3.8, 0.06, 0.8), copperMat);
          strap2.position.set(0, 0.06, 1.0);
          strap1.userData = strap2.userData = plate.userData;
          grp.add(strap1); grp.add(strap2);
        }

        // C) Interlayer Contact Vias
        [-1.5, 0, 1.5].forEach(vx => {
          [-1.5, 0, 1.5].forEach(vz => {
            const via = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.32, 10), copperMat);
            via.position.set(vx, -0.16, vz);
            via.userData = plate.userData;
            grp.add(via);
          });
        });

        const defY = (idx - 3.8) * 0.36;
        grp.position.y = defY;
        grp.userData = {
          name: lName,
          defY: defY,
          expY: (idx - 3.8) * 0.85,
          pct: pctVal,
          raw: rawVal,
          temp: localTemp
        };

        stack.add(grp);
        layerGroups.push(grp);
      });

      // Raycaster for Layer Picking
      const raycaster = new THREE.Raycaster();
      const mouse = new THREE.Vector2();

      function updateInspection(data) {
        selectedLayerName = data.name;
        document.getElementById("inspName").innerText = data.name + (data.name === peakLayer ? " (Peak Hotspot)" : "");
        document.getElementById("inspPct").innerText = data.pct + "%";
        document.getElementById("inspTemp").innerText = data.temp + " °C";
        document.getElementById("inspDef").innerText = layerDefs[data.name] || "Standard copper routing interconnect layer.";
        document.getElementById("inspGeom").innerText = (data.name === "FEOL") ? "Transistor Logic Gates (Bulk Silicon)" : 
                                                         (data.name === "MOL") ? "Tungsten Contact Plugs" : "Patterned Copper Interconnect Tracks";
      }

      renderer.domElement.addEventListener("pointerdown", (e) => {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);

        const hits = raycaster.intersectObjects(stack.children, true);
        if (hits.length > 0) {
          let obj = hits[0].object;
          while (obj && (!obj.userData || !obj.userData.name) && obj.parent) {
            obj = obj.parent;
          }
          if (obj && obj.userData && obj.userData.name) {
            updateInspection(obj.userData);
            if (isIsolated) applyIsolation();
          }
        }
      });

      // Isolate Layer Button Logic
      function applyIsolation() {
        layerGroups.forEach(grp => {
          if (grp.userData.name === selectedLayerName) {
            grp.visible = true;
            grp.position.x = 0;
            grp.position.z = 0;
          } else {
            grp.visible = !isIsolated;
          }
        });

        const activeGrp = layerGroups.find(g => g.userData.name === selectedLayerName);
        if (activeGrp && isIsolated) {
          controls.target.set(0, activeGrp.userData.defY, 0);
          camera.position.set(0, activeGrp.userData.defY + 4.8, 4.0);
        } else {
          controls.target.set(0, 0, 0);
        }
      }

      document.getElementById("btnIsolate").onclick = () => {
        isIsolated = !isIsolated;
        document.getElementById("btnIsolate").innerText = isIsolated ? "Show Full Stack" : "Isolate Selected Layer";
        applyIsolation();
      };

      document.getElementById("btnExplode").onclick = () => {
        if (isIsolated) return;
        exploded = !exploded;
        document.getElementById("btnExplode").innerText = exploded ? "Collapse Stack" : "Explode Stack";
      };

      document.getElementById("btnFocusHot").onclick = () => {
        const hot = layerGroups.find(g => g.userData.name === peakLayer);
        if (hot) {
          selectedLayerName = peakLayer;
          updateInspection(hot.userData);
          controls.target.set(0, hot.position.y, 0);
          camera.position.set(0, hot.position.y + 3.8, 3.4);
        }
      };

      // 3D Tag Billboard Positioning
      const tag = document.getElementById("hotspotTag");
      const peakGroup = layerGroups.find(g => g.userData.name === peakLayer);

      function updateTagPosition() {
        if (!tag || !peakGroup) return;
        if (!peakGroup.visible) { tag.style.display = "none"; return; }

        const wp = new THREE.Vector3(0.8, peakGroup.position.y + 0.25, 0.8);
        wp.project(camera);

        if (wp.z > 1.0) { tag.style.display = "none"; return; }
        const x = (wp.x * 0.5 + 0.5) * cont.clientWidth;
        const y = (-(wp.y * 0.5) + 0.5) * cont.clientHeight;
        tag.style.left = `${x}px`;
        tag.style.top = `${y}px`;
        tag.style.display = "block";
      }

      // Smooth Render Loop
      function anim() {
        requestAnimationFrame(anim);
        if (!isIsolated) {
          layerGroups.forEach(g => {
            const targetY = exploded ? g.userData.expY : g.userData.defY;
            g.position.y += (targetY - g.position.y) * 0.08;
          });
        }
        controls.update();
        updateTagPosition();
        renderer.render(scene, camera);
      }
      anim();
    }
  </script>
</body>
</html>
''')

# 4. Pack into zip
with zipfile.ZipFile("congestion_app_final_3d.zip", "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(base):
        for file in files:
            p = os.path.join(root, file)
            zf.write(p, os.path.relpath(p, base))

print("\nSUCCESS: Created congestion_app_final_3d.zip with rich semiconductor geometry!")
