# SPOT: Self-Play Online Transfer — Submission Package

This package contains all code, pre-trained models, and results needed to
reproduce every figure and table in the paper.

---
[![DOI](https://zenodo.org/badge/latestdoi/1202476012.svg)](https://doi.org/10.5281/zenodo.19479428)


## Directory Structure

```
submission_package/
├── envs/                          # JSBSim environment wrappers
├── sac_agent/                     # SAC agent (actor, critic, replay buffer)
├── ppo_agent/                     # PPO agent (actor, critic, rollout buffer)
├── scripts/
│   ├── 20260109_SAC_Self_Play.py          # Stage 1: SAC self-play training
│   ├── 20260226_PPO_Self_Play.py          # Stage 1: PPO self-play (baseline)
│   ├── 20260113_SAC_Return.py             # Stage 2: Return mission (SAC scratch)
│   ├── 20260210_SAC_Return_FineTune.py    # Stage 2: Return mission (SAC FT)
│   ├── 20260226_PPO_Return_Scratch.py     # Stage 2: Return mission (PPO scratch)
│   ├── 20260226_PPO_Return_FineTune.py    # Stage 2: Return mission (PPO FT)
│   ├── 20260219_SAC_SAM_FineTune.py       # Stage 2: SAM mission — SPOT (proposed)
│   ├── 20260219_SAC_SAM_Scratch.py        # Stage 2: SAM mission (SAC scratch)
│   ├── 20260306_SAC_SAM_FineTune_From_Return.py  # SAM FT from Return (ablation)
│   ├── 20260310_PPO_SAM_FineTune.py       # SAM mission (PPO FT)
│   ├── 20260311_PPO_SAM_Scratch.py        # SAM mission (PPO scratch)
│   ├── 20260304_SAM_Robustness_Test.py    # Robustness evaluation script
│   ├── 20260320_SAC_Ablation_CkptSelection.py  # Ablation: checkpoint selection
│   ├── run_ablation_weekend.py            # Ablation batch runner
│   └── paper/
│       ├── generate_paper_figures.py      # Regenerate ALL paper figures from CSVs
│       ├── eval_final.py                  # Final evaluation (Return + SAM missions)
│       ├── collect_eval_trajectories.py   # Collect trajectory data for plots
│       ├── eval_trajectory.py             # SAM trajectory evaluation
│       ├── eval_return_trajectory.py      # Return trajectory evaluation
│       ├── eval_selfplay_trajectory.py    # Self-play trajectory evaluation
│       ├── plot_trajectory.py             # Plot SAM trajectories
│       ├── plot_return_trajectory.py      # Plot Return trajectories
│       ├── plot_return_figures.py         # Plot Return learning curves
│       ├── replot_trajectories.py         # Re-render saved trajectory data
│       └── run_trajectory_pipeline.py     # Full trajectory pipeline (collect + plot)
├── paper/
│   ├── results/                   # Pre-computed CSV results (used by figure scripts)
│   └── paper_figures/             # Generated figures as used in the paper
├── runs/
│   └── 20260219_SAC_SAM_FineTune/ # Training logs (episode_metrics.csv) for F2 ablation
└── models/                        # Trained model checkpoints
    ├── 20260109_SAC_Self_Play/             selfplay_epi_00900.pth
    ├── 20260226_PPO_Self_Play/             selfplay_epi_00960.pth
    ├── 20260113_SAC_Return_20260119_1057/  epi_00499.pth
    ├── 20260210_SAC_Return_FineTune_20260212_1249/  epi_00499.pth
    ├── 20260219_SAC_SAM_FineTune_20260306_1730/     epi_00499.pth  <- SPOT
    ├── 20260219_SAC_SAM_Scratch_20260226_1638/      epi_00499.pth
    ├── 20260226_PPO_Return_FineTune_20260316_0927/  epi_00499.pth
    ├── 20260226_PPO_Return_Scratch_20260225_2153/   epi_00499.pth
    ├── 20260306_SAC_SAM_FineTune_From_Return_20260310_1052/  epi_00499.pth
    ├── 20260310_PPO_SAM_FineTune_20260311_0901/     epi_00499.pth
    └── 20260311_PPO_SAM_Scratch_20260312_0902/      epi_00499.pth
```

---

## Requirements

```bash
pip install -r requirements.txt
```

JSBSim must be installed separately. See: https://github.com/JSBSim-Team/jsbsim

---

## Reproducing Paper Figures

All figures (A2, B1, B2, C1, C2, D, F2) and trajectory plots can be regenerated
from the pre-computed CSVs in `paper/results/` without re-running any simulation.

**Regenerate all figures (A2–F2):**
```bash
python scripts/paper/generate_paper_figures.py
# Output: paper/paper_figures/
```

**Regenerate trajectory figures:**
```bash
python scripts/paper/run_trajectory_pipeline.py
# Output: paper/paper_figures/traj_*.png
```

---

## Re-running Evaluations

To re-evaluate from model checkpoints (requires JSBSim):

**Final performance evaluation (Return + SAM missions):**
```bash
python scripts/paper/eval_final.py
# Output: paper/results/eval_return_final.csv, eval_sam_final.csv
```

**Robustness evaluation (Table D in paper):**
```bash
# Threat scaling
python scripts/20260304_SAM_Robustness_Test.py --mode threat_scaling --model ft_self

# Initial perturbation
python scripts/20260304_SAM_Robustness_Test.py --mode init_perturb --model ft_self

# Pop-up SAM
python scripts/20260304_SAM_Robustness_Test.py --mode popup_sam --model ft_self
```

---

## Training from Scratch

Stage 1 — Self-Play:
```bash
python scripts/20260109_SAC_Self_Play.py
```

Stage 2a — Return Mission (proposed SPOT transfer):
```bash
python scripts/20260210_SAC_Return_FineTune.py
```

Stage 2b — SAM Mission (proposed SPOT transfer):
```bash
python scripts/20260219_SAC_SAM_FineTune.py
```

---

## Model Naming Convention

| Model directory | Description |
|---|---|
| `20260219_SAC_SAM_FineTune_20260306_1730` | **SPOT** (proposed method) |
| `20260219_SAC_SAM_Scratch_20260226_1638` | SAC trained from scratch |
| `20260306_SAC_SAM_FineTune_From_Return_20260310_1052` | SAC fine-tuned from Return pretrain |
| `20260310_PPO_SAM_FineTune_20260311_0901` | PPO fine-tuned (self-play pretrain) |
| `20260311_PPO_SAM_Scratch_20260312_0902` | PPO trained from scratch |
