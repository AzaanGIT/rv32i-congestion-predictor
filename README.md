# RV32I RISC-V Core Congestion-Aware Floorplanner & DRC Predictor

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen?logo=github)](https://azaangit.github.io/rv32i-congestion-predictor/)
[![Render](https://img.shields.io/badge/Render-Cloud%20Deployment-blue?logo=render)](https://rv32icongestionpred.onrender.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-purple?logo=python)](https://www.python.org/)
[![XGBoost Engine](https://img.shields.io/badge/AI%20Engine-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![RISC-V](https://img.shields.io/badge/Architecture-RISC--V%2032--Bit-red)](https://riscv.org/)

An advanced machine-learning EDA tool and interactive 3D web platform for predicting post-route DRC violations, interconnect track pressures, and thermal hotspot risks in **RV32I RISC-V Integer Core** silicon macro floorplans.

---

## 🌐 Live Interactive Applications

| Environment | Platform | Link | Purpose |
| :--- | :--- | :--- | :--- |
| **GitHub Pages** | Client-Side Web App | [https://azaangit.github.io/rv32i-congestion-predictor/](https://azaangit.github.io/rv32i-congestion-predictor/) | Instant interactive web app with 3D Silicon Router Viewer & DRC Optimizations |
| **Render Cloud** | Flask Python API Server | [https://rv32icongestionpred.onrender.com](https://rv32icongestionpred.onrender.com) | Full cloud web server with live Python ML model inference |

---

## 🚀 Key System Features

1. **⚡ Physical VLSI Feature Matrix**
   - Calculates domain-specific IC metrics from pre-placement parameters:
     - **Pin Density Index:** $Utilization \times Density \times 1.85$ ($pins/mm^2$)
     - **HPWL Estimated Wirelength:** $\sqrt{Util \times Aspect} \times Margin \times 14.2$ ($\mu m$)
     - **Macro Blockage Ratio:** $(1 - Density) \times Util$ (%)
     - **Via Pillar Density:** $Density \times LayerAdjustment \times 1.25$ ($\mu m^{-2}$)

2. **🤖 High-Precision XGBoost Inference Engine**
   - **DRC Violation Classifier:** Predicts routing feasibility ($99.15\%$ accuracy).
   - **Per-Layer Demand Regressors:** Predicts exact wire track demands across Metal 1 to Metal 10 ($R^2 = 0.9886$, MAE = 466.3 tracks).

3. **🗺️ Spatial 2D Hotspot CNN Predictor**
   - Maps macro floorplan geometry into $50 \times 50$ cellular wire density heatmaps for visual spatial routing breakdown across all BEOL metal layers.

4. **🏢 3D Commercial Silicon Router Viewer**
   - Interactive Three.js 3D visualizer rendering silicon substrate layers, tungsten contact pillars, and copper interconnects with exploded stack view and single-layer isolation.

5. **🛠️ Multi-Objective DRC Optimization Engine**
   - Automatically calculates design rule fixes to bring peak wire track density below the $80\%$ DRC congestion threshold.

---

## 📂 Repository Structure

```
rv32i-congestion-predictor/
├── docs/                         # GitHub Pages static interactive web app
│   └── index.html                # Standalone SPA for GitHub Pages deployment
├── congestion_app/               # Flask Web Application backend
│   ├── app.py                    # Flask server & model inference routes
│   ├── recommend.py              # Multi-objective DRC optimization engine
│   ├── requirements.txt          # Dependencies (xgboost, scikit-learn, gunicorn)
│   ├── model/                    # Trained model binaries
│   │   ├── xgboost_classifier.joblib
│   │   ├── xgboost_regressors.joblib
│   │   ├── spatial_hotspot_model.joblib
│   │   └── meta.json
│   └── templates/                # Flask templates (index.html, results.html)
├── train_model.py                # End-to-end ML model training pipeline
├── congestion_dataset_full.csv   # 30,500+ design point VLSI routing dataset
├── render.yaml                   # Render 1-click cloud deployment spec
└── README.md                     # Documentation
```

---

## 💻 Local Setup & Running Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/AzaanGIT/rv32i-congestion-predictor.git
cd rv32i-congestion-predictor
```

### 2. Install Dependencies
```bash
pip install -r congestion_app/requirements.txt
```

### 3. Run the Flask Web Application
```bash
python3 congestion_app/app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

### 4. (Optional) Retrain ML Models
```bash
python3 train_model.py
```

---

## 📊 Dataset & Model Performance Overview

| Metric | Random Forest Baseline | XGBoost + Feature Matrix |
| :--- | :--- | :--- |
| **Classification Accuracy** | 99.10% | **99.15%** |
| **Regression $R^2$ Score** | 0.9860 | **0.9886** |
| **Total Track Demand MAE** | 524.7 tracks | **466.3 tracks (11% reduction)** |
| **Inference Time** | ~160ms | **< 25ms** |

---

## 📜 Dataset Reference

The dataset contains 30,500+ placement and routing exploration runs for the **RV32I RISC-V 32-bit Integer Processor Core**:
* **Input Features:** `utilization`, `aspect_ratio`, `core_margin`, `density`, `layer_adj`, `metal1_resource` ... `metal10_resource`.
* **Targets:** `success` (DRC clean status), `metal1_demand` ... `metal10_demand`, `total_wirelength_um`.
