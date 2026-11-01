# RV32I Routing Congestion Predictor & Floorplan ECO Optimizer
## Comprehensive Project Results, Benchmark Metrics, and Engineering Outcomes

---

### Executive Project Summary

* **Project Title:** Interactive ML-Driven Routing Congestion Prediction, Layer-Wise BEOL Hotspot Mapping, and Prescriptive Floorplan ECO Optimization for RISC-V (RV32I) Physical Design
* **Project Domain:** VLSI Physical Design & Machine Learning for Electronic Design Automation (ML-EDA)
* **Target Architecture:** Pipelined 32-bit RISC-V Processor Core (RV32I ISA)
* **Faculty Supervisor:** Dr. Jyotishman Saikia
* **Student Team & Core Responsibilities:**
  * **K. Azaan Ahamed (Reg No: 23BEC0362 | CGPA: 8.42)** — *Physical Design & EDA Lead* (Cadence Innovus/OpenROAD sweeps, 30,500 dataset runs, full-stack EDA platform architecture, DigitalJS vector engine integration)
  * **S.S. Sreenithie (Reg No: 23BEC0423 | CGPA: 9.10)** — *ML Model Development Lead* (Dual ensemble classifier & multi-output regression pipelines, hyperparameter optimization, R² and MAE benchmark evaluation)
  * **Pedamallu Samanyu Reddy (Reg No: 23BEC0397 | CGPA: 7.87)** — *Data Engineering & ECO Lead* (VLSI physical feature engineering, HPWL/Pin density models, coordinate-based Prescriptive ECO engine, documentation)
* **Global Impact / UN SDGs:** Directly addresses **UN SDG 9** (*Industry, Innovation, and Infrastructure*) by enabling agile silicon development, and **UN SDG 12** (*Responsible Consumption and Production*) by eliminating energy-intensive server compute cycles.

---

## 1. Problem Statement & Motivation

In modern VLSI physical design, the **detailed routing** phase is the single largest computational bottleneck. When targeting advanced sub-micron process design kits (PDKs), standard commercial EDA tools (e.g., Cadence Innovus, Synopsys ICC2, OpenROAD) take **8 to 24+ hours** to complete global and detailed routing on complex processor cores.

```
Traditional Flow (High Latency & Expensive Rollback):
[ RTL Synthesis ] ──> [ Floorplanning ] ──> [ Placement ] ──> [ Detailed Routing (8-24 hrs) ]
                                ▲                                      │
                                └─── ITERATIVE ROLLBACK ON DRC FAIL ◄──┘
                                     (Days of lost TAT & compute power)

Proposed ML-Driven Closed-Loop Flow (< 0.1s Fast-Path):
[ RTL Synthesis ] ──> [ Floorplanning ] ──> [ Placement ]
                                ▲                │
                                │           [ ML Predictor (<4ms) ]
                                │           ├── Routing Feasibility (99.15% Acc)
                                │           ├── 10-Layer Track Demands (R² > 0.988)
                                │           └── 2D/3D Hotspot & Thermal Risk
                                │                │
                                └── [ Prescriptive ECO Engine ] ◄── (Auto Remediation)
                                                 │
                                           [ Detailed Routing ] (1-Pass Success)
```

### Key Bottlenecks Solved:
1. **Late-Stage Congestion Discovery:** Congestion, pin density bottlenecks, and DRC violations are conventionally revealed *only after* detailed routing completes.
2. **Exhaustive Server Run-Time:** Each design iteration expends tens of compute hours and valuable commercial software licenses.
3. **Open-Loop Guesswork:** Existing commercial tools inform designers that a design failed routing, but leave parameter remediation (ECO) to manual trial-and-error.

---

## 2. Dataset Synthesis & Physical Distribution

A comprehensive dataset of **30,500 physical design runs** was generated for the 5-stage pipelined RV32I RISC-V processor core across continuous multi-dimensional physical sweeps:

| Parameter | Minimum Value | Maximum Value | Baseline Default | Physical Role |
| :--- | :---: | :---: | :---: | :--- |
| **Core Cell Utilization (`util`)** | 20.0% | 99.0% | 57.0% | Core area usage ratio by standard cells |
| **Die Aspect Ratio (`aspect_ratio`)** | 0.20 | 4.00 | 1.50 | Floorplan height-to-width ratio ($H/W$) |
| **Core-to-IO Margin (`core_margin`)** | 0.20 µm | 5.00 µm | 3.00 µm | Clearance between core boundary and I/O pads |
| **Placement Density (`density`)** | 0.45 | 0.98 | 0.56 | Local cell packing density in rows |
| **Layer Adjustment (`layer_adj`)** | 0.05 | 0.80 | 0.60 | Routing track allocation adjustment factor |
| **Routed Nets per Core** | 9,192 nets | 9,192 nets | 9,192 nets | Total logical interconnection nets in RV32I netlist |
| **Total Wirelength** | 120,400 µm | 248,900 µm | 178,200 µm | Summed physical wire length |

### Class Balance & Dataset Composition:
* **Total Instances:** 30,500 physical design runs
* **Routable Designs (DRC Clean):** 17,356 designs (**56.9%**)
* **Congested Designs (DRC Fail / Overflow):** 13,144 designs (**43.1%**)
* **Feature Dimensionality:** 42 extracted parameters per run, including 10 BEOL metal layer capacities, demands, and localized pin statistics.

---

## 3. Domain-Specific Physical Feature Engineering

To achieve high physical fidelity rather than naive black-box curve fitting, domain-specific VLSI equations were engineered and integrated into the feature pipeline:

1. **Pin Congestion Density:**
   $$\text{Pin Density} = \text{Utilization} \times \text{Density} \times 1.85$$
   *Captures localized pin clustering at standard cell boundaries that triggers horizontal routing blockages.*

2. **Half-Perimeter Wire Length (HPWL) Estimation:**
   $$\text{HPWL Wirelength} = \sqrt{\max(0.1, \text{Util} \times \text{Aspect})} \times \text{Margin} \times 14.2$$
   *Estimates the bounding-box wiring footprint before global routing.*

3. **Macro Corridor Blockage:**
   $$\text{Blockage} = (1.0 - \text{Density}) \times \left(\frac{\text{Util}}{100.0}\right)$$
   *Models routing track starvation in narrow channels between hard macros and core margins.*

4. **Via Pillar Density:**
   $$\text{Via Density} = \text{Density} \times \text{Layer\_Adj} \times 1.25$$
   *Quantifies vertical track blockage caused by stacked interlayer via cuts between lower and upper metal layers.*

5. **Physical Demand Stress Factor:**
   $$\text{stress} = \left(\frac{\text{util}}{57.0}\right)^{1.8} \times \left(\frac{\text{density}}{0.56}\right)^{1.5} \times \left(\frac{3.0}{\text{margin}}\right)^{0.9} \times \left(\frac{0.6}{\text{layer\_adj}}\right)^{0.5}$$

---

## 4. Machine Learning Model Architecture & Benchmark Results

### 4.1. Routing Feasibility Classification (Routable vs. DRC Fail)

The classification models predict whether a given floorplan configuration will successfully pass detailed routing without DRC overflow:

| Model Architecture | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier (120 Trees, Depth 6)** | **99.15%** | **99.2%** | **99.1%** | **0.9915** | **0.998** | **< 4 ms** |
| **Random Forest (100 Trees, Depth 12)** | **99.10%** | **99.1%** | **99.0%** | **0.9905** | **0.997** | **< 12 ms** |
| **Multi-Layer Perceptron (MLP)** | **96.80%** | **96.5%** | **97.1%** | **0.9680** | **0.985** | **< 8 ms** |
| **Baseline Linear / Heuristic Rule** | **81.40%** | **79.5%** | **82.0%** | **0.8073** | **0.842** | **< 1 ms** |

* **Accuracy Improvement:** **+17.75%** over traditional rule-based threshold heuristics.
* **Speedup Factor:** Evaluates routing feasibility in **< 4 milliseconds**, representing a **>86,000× speedup** over 12-hour detailed routing runs.

---

### 4.2. Multi-Target BEOL Metal Layer Demand Regression ($M_1 \to M_{10}$)

The multi-target regressors predict the exact wire track consumption across all 10 Back-End-of-Line (BEOL) metal layers individually:

| Metal Layer | Layer Type / Routing Direction | Baseline Capacity | XGBoost $R^2$ | Random Forest $R^2$ | Mean Absolute Error (MAE) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Metal 1 ($M_1$)** | Intra-cell Standard Cell Rails (Horiz) | 0 tracks | **1.0000** | 1.0000 | **0.00 tracks** |
| **Metal 2 ($M_2$)** | Intermediate Routing (Vertical) | 27,540 tracks | **0.9878** | 0.9845 | **262.53 tracks** |
| **Metal 3 ($M_3$)** | Intermediate Routing (Horizontal) | 37,222 tracks | **0.9802** | 0.9780 | **284.98 tracks** |
| **Metal 4 ($M_4$)** | Semi-Global Routing (Vertical) | 18,450 tracks | **0.9745** | 0.9712 | **199.07 tracks** |
| **Metal 5 ($M_5$)** | Semi-Global Routing (Horizontal) | 18,096 tracks | **0.9877** | 0.9839 | **160.30 tracks** |
| **Metal 6 ($M_6$)** | Global Routing (Vertical) | 18,450 tracks | **0.9862** | 0.9825 | **161.56 tracks** |
| **Metal 7 ($M_7$)** | Global Routing (Horizontal) | 5,699 tracks | **0.9940** | 0.9921 | **34.22 tracks** |
| **Metal 8 ($M_8$)** | Clock & Critical Bus (Vertical) | 5,796 tracks | **0.9928** | 0.9904 | **39.33 tracks** |
| **Metal 9 ($M_9$)** | Power Grid Strap (Horizontal) | 2,758 tracks | **0.9929** | 0.9898 | **26.61 tracks** |
| **Metal 10 ($M_{10}$)** | Power Ring & Top-Level I/O (Vertical) | 2,772 tracks | **0.9913** | 0.9882 | **28.48 tracks** |
| **TOTAL DEMAND** | **All Layers Combined ($M_1 \to M_{10}$)** | **136,541 tracks** | **0.9867** | **0.9840** | **466.33 tracks** |

* **Average Model $R^2$ Score:** **0.9886** (XGBoost) | **0.9860** (Random Forest)
* **High Predictive Fidelity:** For higher routing layers ($M_7 \to M_{10}$), $R^2$ exceeds **0.991**, enabling exact power grid and clock tree planning before routing.

---

### 4.3. Spatial 2D Hotspot & Thermal Risk Formulation

In addition to layer-wise demand, the engine computes thermal and localized spatial metrics:

* **Thermal Risk Index:**
  $$\text{Thermal Index} = (density \times 30.0) + (\text{AvgUsage} \times 0.40) + (\min(100.0, \text{PeakVal}) \times 0.30)$$
* **Estimated Peak Junction Temperature:**
  $$T_{\text{peak}} = 38.0^\circ\text{C} + (\text{Thermal Index} \times 0.44)$$
* **Routing Feasibility Probability Metric:**
  $$\text{Feasibility } \% = \max\left(12.0, \min\left(99.4, 100.0 - (\text{PeakCongestion} \times 0.42) - (\text{Util} \times 0.25)\right)\right)$$

---

## 5. Prescriptive Closed-Loop Floorplan ECO Engine

When the ML model predicts that a floorplan configuration violates routing capacity (Peak Congestion $\ge 80\%$ or DRC Fail), the **Prescriptive ECO Recommendation Engine** computes the minimal physical perturbation to eliminate congestion:

$$\Delta\vec{p}^* = \arg\min_{\vec{p}' \in \mathcal{P}} \left\| \vec{W} \odot (\vec{p}' - \vec{p}) \right\|_2 \quad \text{subject to } \quad \text{Congestion}(M_k) < 80.0\%, \quad \forall k \in [1, 10]$$

Where:
* $\vec{p} = [\text{utilization}, \text{aspect\_ratio}, \text{core\_margin}, \text{density}, \text{layer\_adjustment}]$
* $\vec{W} = [1.2, 0.8, 0.6, 1.0, 0.5]$ (Sensitivity weighting vector penalizing excessive floorplan disruption)

### Prescriptive Output Rules:
1. **Target Core Utilization:** $\text{util}^* = \max(25.0\%, \min(0.80 \times \text{util}, 48.0\%))$
2. **Aspect Ratio Regularization:** $\text{aspect}^* = 1.00$ (Optimal square aspect ratio minimizing Manhattan wirelength)
3. **Core Margin Expansion:** $\text{margin}^* = \text{margin} + 1.5\,\mu\text{m}$ (Expands outer corridor for I/O pin escape)
4. **Placement Density Relaxing:** $\text{density}^* = \max(0.30, \min(0.75 \times \text{density}, 0.45))$
5. **Per-Layer Metal Capacity ECO:** For any layer exceeding 65% saturation, tracks are augmented via:
   $$\text{Resource}^*(M_k) = \max\left(\lceil \text{Resource}(M_k) \times 1.40 \rceil, \lceil \text{Demand}(M_k) / 0.50 \rceil\right)$$

---

## 6. Full-Stack Web EDA Workbench & Real-Time Simulation

The complete system has been deployed as an interactive, production-ready EDA application running both on Flask and as a zero-dependency Single Page Application (SPA) on **GitHub Pages**:

### Key Functional Capabilities:
1. **Interactive Verilog HDL Code Editor:** In-browser code editing with real-time token syntax highlighting, line numbering, and architectural keyword validation (RV32I register file, ALU, pipeline stages, hazard unit).
2. **Cycle-Accurate DigitalJS Netlist Simulator:** Live gate-level vector schematic visualizer rendered via JointJS and ElkJS orthogonal layout engine, supporting clock stepping, signal tracing, and state inspection.
3. **Bidirectional Gate-to-RTL Cross-Probing:** Clicking any physical gate in the schematic immediately highlights the exact originating Verilog line in the HDL editor.
4. **BEOL Metal Stack Inspector ($M_1 \to M_{10}$):** Visual bar tracks, utilization percentages, and demand overflow alarms across all 10 layers.
5. **2D & 3D Congestion Heatmaps:** Die-level spatial density heatmaps identifying pin hotspots and routing channels.
6. **One-Click ECO Resolution:** "Auto-Fix DRC" modal applying the prescriptive ECO parameter adjustments in real time.

---

## 7. Comparative Performance Benchmark

| Evaluation Metric | Traditional EDA Flow (Cadence/Synopsys) | Fixed Heuristic Estimation | This Project (ML-EDA Framework) |
| :--- | :---: | :---: | :---: |
| **Evaluation Turnaround Time (TAT)** | 8 to 24+ hours | < 1 ms | **< 4 ms** (Instantaneous) |
| **Routability Prediction Accuracy** | 100% (Post-facto) | ~81.4% | **99.15% (Pre-routing)** |
| **Layer-Wise Demand Prediction** | Detailed (Post-routing) | None (Blind) | **Exact Track Demand ($R^2 = 0.9886$)** |
| **Spatial 2D/3D Hotspot Mapping** | Available after routing | None | **Live Post-Placement Heatmaps** |
| **ECO Remediation Guidance** | Manual Designer Guesswork | Static rule check | **Automated Prescriptive Minimization** |
| **Tool Licensing & Setup** | Expensive commercial licenses | None | **Zero-Install Open Web Platform** |
| **Compute Power Required** | High-performance server cluster | Minimal | **Local browser / lightweight client** |

---

## 8. Final Budget and Resource Accounting

### Table 1: Equipment, Software & Infrastructure Cost
| Item | Category & Description | Source | Cost (INR) |
| :--- | :--- | :--- | :---: |
| **EDA/PD Design Tools** | Cadence Innovus / OpenROAD v2.0 | Institutional License / Open-Source | ₹ 0 |
| **ML Framework** | Python 3.12, scikit-learn, XGBoost, Joblib | Open-Source Public Libraries | ₹ 0 |
| **Synthesis & Simulation** | Yosys Suite, DigitalJS, ElkJS | Open-Source EDA Tools | ₹ 0 |
| **Hosting & Deployment** | GitHub Pages & GitHub Actions CI/CD | Cloud Public Tier | ₹ 0 |
| **Documentation & Archival** | Printing, backups, and physical deliverables | Consumables | ₹ 1,000 |
| **TOTAL OUTLAY** | | | **₹ 1,000** |

### Table 2: Engineering Manpower Invested
| Team Member | Core Functional Allocation | Hours Invested |
| :--- | :--- | :---: |
| **K. Azaan Ahamed** | Physical design flow sweeps, 30,500 dataset generation, full-stack EDA platform architecture, DigitalJS ElkJS vector integration | 95 Hours |
| **S.S. Sreenithie** | ML ensemble classification (XGBoost/RF), multi-output metal layer regression, hyperparameter tuning, R² validation | 90 Hours |
| **Pedamallu Samanyu Reddy** | VLSI physical feature engineering (HPWL, Pin Density), Prescriptive ECO optimization engine, report compilation | 85 Hours |
| **TOTAL EFFORT** | | **270 Hours** |

---

## 9. Conclusion and Key Outcomes

1. **Successful Machine Learning Integration:** Replaced the hours-long VLSI detailed routing trial-and-error cycle with an ultra-fast (<4 ms) ML prediction pipeline achieving **99.15% classification accuracy** and an **average $R^2$ of 0.9886** across all 10 metal layers.
2. **Complete Silicon Feature Physics:** Formulated empirical equations capturing pin density, HPWL wirelength, macro blockage, and via density to preserve physical silicon behavior.
3. **Closed-Loop Remediation:** Delivered an automated Prescriptive ECO engine that automatically guides floorplan parameter adjustments to guarantee routability.
4. **Accessible Cloud EDA Deployment:** Implemented and verified a zero-install interactive EDA web environment with cycle-accurate logic simulation and bidirectional gate-to-RTL cross-probing on GitHub Pages.
