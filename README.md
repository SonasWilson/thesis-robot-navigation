# Curriculum Reinforcement Learning for Robot Navigation

A curriculum-based SAC training pipeline for 2D robot navigation with static and dynamic obstacles in PyBullet. The project includes baseline failure analysis, safety-aware reward shaping, curriculum learning across multiple stages, and generalization assessment across out-of-distribution environments.

## Overview

This repository contains the code used for the thesis **“Analysis and Improvement of Deep Reinforcement Learning-Based Mobile Robot Navigation in Dynamic Environments.”**  
The main goal of the project is to improve navigation safety, stability, and generalization for a mobile robot operating in dynamic environments.

The repository includes:
- custom PyBullet navigation environments,
- training scripts for DDPG, TD3, and SAC,
- curriculum-based training across six stages,
- evaluation scripts for in-distribution and out-of-distribution tests,
- visualization scripts for thesis figures,
- trained models,
- and supplementary evidence videos.

## Project Structure

```text
.
├── envs/          # Custom Gymnasium/PyBullet environments
│   ├── static_2.py
│   ├── dynamic_2.py
│   ├── v5.py              # Improved model (Safety v6)
│   └── sphere_nav_env.py
├── training/              # Training scripts
│   ├── train_gap1.py
│   ├── curriculum_train.py
│   ├── static_2_sac.py
│   ├── static_2_td3.py
│   ├── dynamic_2_sac.py
│   ├── dynamic_2_td3.py
│   ├── v5_sac.py
│   └── v5_td3.py
├── evaluation/            # Evaluation scripts
│   ├── curriculum_eval.py
│   ├── eval_sac_vs_td3.py
│   ├── generalization_assessment.py
│   ├── final_eval_compare.py
│   ├── final_eval_multiseed.py
│   ├── final_eval_static_dynamic_sac.py
│   ├── final_eval_static_dynamic_td3.py
│   └── final_eval_td3.py
├── visualization/         # Plotting and figure generation scripts
│   ├── plot_curriculum_results.py
│   ├── plot_curriculum_stage_progression.py
│   ├── plot_curriculum_training_metrics.py
│   ├── plot_generalization_figures.py
│   ├── plot_generalization_study.py
│   ├── plot_sac_vs_td3.py
│   ├── plot_stage1_anomaly.py
│   ├── plot_test_configs_overview.py
│   ├── plot_training_metrics.py
│   └── screenshot_envs.py
├── plots/         # Plots generated
```

## Curriculum Stages

| Stage | Arena | Static Obstacles | Dynamic Obstacles | Speed | Motion |
|-------|-------|------------------|-------------------|-------|--------|
| 1 | 10×10 m | 2 | 1 | 0.45 | Sinusoidal |
| 2 | 20×20 m | 2 | 2 | 0.65 | Sinusoidal |
| 3 | 30×30 m | 2 | 3 | 0.85 | Sinusoidal |
| 4 | 40×40 m | 3 | 3 | 1.00 | Sinusoidal |
| 5 | 50×50 m | 3 | 4 | 1.20 | Sinusoidal |
| 6 | 50×50 m | 4 | 5 | 1.40 | 50% Orbital |

## Requirements

```text
gymnasium
pybullet
stable-baselines3
numpy
pandas
matplotlib
opencv-python
torch
```

## Quick Start

### 1. Train curriculum stages
```bash
python training/curriculum_train.py --stage_start 1 --stage_end 6 --enforce_stage_gates --resume_from_checkpoint
```

### 2. Evaluate all curriculum models
```bash
python evaluation/curriculum_eval.py --stage_start 1 --stage_end 6 --episodes 100
```

### 3. Run generalization assessment
```bash
python evaluation/generalization_assessment.py --mode all --source_stage 6
```

### 4. Generate figures
```bash
python visualization/plot_curriculum_results.py
python visualization/plot_generalization_figures.py --out_dir ./figures
python visualization/plot_curriculum_stage_progression.py --out_dir ./figures
```




