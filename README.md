# RV32I Core with Congestion-Aware Floorplanning Dataset

This repository hosts the machine-learning floorplanning and routing congestion dataset for the **RV32I Core** design implemented across varied physical design parameters.

## Dataset Overview

The dataset contains physical design exploration runs across diverse floorplan parameters, measuring utilization, aspect ratio, margins, layer usage percentages, and routed net metrics.

- **File**: `congestion_dataset_full.csv`
- **Total Samples**: 30,500+ design point evaluations
- **Design Under Test (DUT)**: RV32I RISC-V 32-bit Integer Processor Core

### Feature Columns:
1. `run_id`: Unique design exploration identifier
2. `utilization`: Core target cell utilization (%)
3. `aspect_ratio`: Floorplan height-to-width aspect ratio
4. `core_margin`: Die boundary clearance / margins (µm)
5. `density`: Target placement density
6. `layer_adj`: Metal layer routing adjustment factors
7. `success`: Placement and routing completion status (`True`/`False`)
8. `metal1_resource` ... `metal10_resource`: Routing tracks and pitch capacity per metal layer
9. `metal1_demand` ... `metal10_demand`: Track demand per metal layer
10. `metal1_usage_pct` ... `metal10_usage_pct`: Local layer congestion utilization (%)
11. `Total_resource`, `Total_demand`, `Total_usage_pct`: Aggregate chip routing congestion metric
12. `total_wirelength_um`: Post-route total wirelength (µm)
13. `routed_nets`: Total successfully routed signal nets

## Applications
- Training graph neural networks (GNNs), random forests, and gradient-boosted trees (XGBoost/LightGBM) to predict post-route DRC violations and track congestion directly from pre-placement floorplan parameters.
- Congestion-aware floorplanning parameter optimization.
