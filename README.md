# Counterfactual Explanations for Continuous Action Reinforcement Learning

This repository provides code and data needed to reproduce the experiments presented in our paper: **Counterfactual Explanations for Continuous Action Reinforcement Learning (IJCAI 2025)**.

The project involves two case studies:
- A diabetes management scenario based on a Type 1 Diabetes (T1D) patient simulator
- A physics-based Lunar Lander environment

Each case study includes steps to:
- Train baseline policies using PPO
- Generate counterfactual explanations using TD3
- Post-process results for quantitative analysis and visualization

---

## Contents

- [Diabetes Case Study](#diabetes-case-study)
  - [Step 1: Train Baseline PPO Agent](#step-1-train-baseline-ppo-agent)
  - [Step 2: Run TD3-Based Experiments with Original Traces](#step-2-run-td3-based-experiments-with-original-traces)
  - [Step 3: Run Experiments with New Traces](#step-3-run-experiments-with-new-traces)
  - [Step 4: Postprocess Results and Generate Tables](#step-4-postprocess-results-and-generate-tables)
  - [Step 5: Generate Final Figures](#step-5-generate-final-figures)

- [Lunar Lander Case Study](#lunar-lander-case-study)
  - [Step 1: Train Baseline PPO Agent](#step-1-train-baseline-ppo-agent-1)
  - [Step 2: Run TD3-Based Experiments with Original Traces](#step-2-run-td3-based-experiments-with-original-traces-1)
  - [Step 3: Run Experiments with New Traces](#step-3-run-experiments-with-new-traces-1)
  - [Step 4: Postprocess Results and Generate Tables](#step-4-postprocess-results-and-generate-tables-1)
  - [Step 5: Generate Final Figures](#step-5-generate-final-figures-1)

---

## Diabetes Case Study

This case study corresponds to the experiments reported in **Section 5.1** of the paper. It includes:

- Training a PPO agent to generate original traces
- Running TD3-based counterfactual experiments
- Post-processing experimental results

---

### Step 1: Train Baseline PPO Agent

We use **Stable-Baselines3** to train a PPO agent for generating original traces, serving as the baseline model.

#### 1.1 Setup Conda Environment

```bash
conda create -n CF_diabetic_train_ppo python=3.8
conda activate CF_diabetic_train_ppo

pip install stable-baselines3==1.7.0
pip install gym==0.21.0
pip install torch==2.0.0
pip install pandas==1.5.3
pip install tensorboard
```

#### 1.2 Overwrite Default Packages

Replace the default packages with the provided ones:

```bash
cd yourpath/.conda/envs/CF_diabetic_train_ppo/lib/python3.8/site-packages
```

Upload and replace files in `train_ppo/package`.

#### 1.3 Run PPO Training

- Single Environment (Adult Patient #7):

```bash
python diabetic_case_train_ppo.py \
  -arg_patient_type adult \
  -arg_patient_id 7 \
  -arg_cuda 0 \
  -arg_train_step 100000 \
  -arg_callback_step 100000
```

- Multi Environment (Train: Patients 4–6, Test: Patients 7–9):

```bash
python diabetic_case_train_ppo_generalize.py \
  -arg_exp_id 1 \
  -arg_cuda 0 \
  -arg_train_step_each_env 3000 \
  -arg_callback_step 3000 \
  -arg_train_round 179 \
  -arg_lr 0.0001 \
  -arg_test_epochs_each_env 1 \
  -arg_max_test_time_each_env 3000
```

> The trained models are saved under `train_ppo/trained_model`.

---

### Step 2: Run TD3-Based Experiments with Original Traces

#### 2.1 Setup Environment

```bash
conda create -n CF_diabetic python=3.8
conda activate CF_diabetic

pip install stable-baselines3==1.7.0
pip install gym==0.21.0
pip install torch==2.0.0
pip install pandas==1.5.3
```

Replace packages as before:

```bash
cd yourpath/.conda/envs/CF_diabetic/lib/python3.8/site-packages
```

Upload files from `train_td3/package`.

#### 2.2 Trace Filtering and Grouping

The code automatically filters trace segments with:

1. Maximum reward (20)
2. Initial CGM < 65
3. All-zero actions

Remaining segments are grouped into:

- (65, 100]
- (100, 150]
- (150, 260]

Then, 6 segments per group are selected for training and 6 for testing.

#### 2.3 Run Batch Experiments with SLURM

- P1 (Baseline Reward):

```bash
# Single-environment
sbatch run_diabetic_exp1.sh

# Multi-environment
sbatch run_diabetic_exp2.sh
```

- P2-base (State Threshold):

```bash
# Single-environment
sbatch run_diabetic_exp3_State_RP2.sh with diabetic_experiments_exp1_state_RP2_UET11.txt

# Multi-environment
sbatch run_diabetic_exp3_State_RP2.sh with diabetic_experiments_exp3_state_RP2_UET6.txt
```

- P2-fixed (State Threshold + Fixed Action):

```bash
# Single-environment
sbatch run_diabetic_exp3_State_RP3.sh with diabetic_experiments_exp1_state_RP3_UET11.txt

# Multi-environment
sbatch run_diabetic_exp3_State_RP3.sh with diabetic_experiments_exp3_state_RP3_UET6.txt
```

---

### Step 3: Run Experiments with New Traces

To run on newly generated original traces:

1. Use `script_diabetic_exp_multiple_trace.py` to generate parameter `.txt` files
2. Apply them in the above `.sh` SLURM scripts

Example scripts are available in:

```
train_td3/code/P1/generate_new_trace/
```

---

### Step 4: Postprocess Results and Generate Tables

#### 4.1 Collect Trials

- Rename `trial_1` to `trial_X` (X = 1 to 7)
- Move `trial_2` ~ `trial_7` into `td3_cf_results` inside `trial_1`

#### 4.2 Run Aggregation

Edit `data_postprocess.py`:

```python
case_name = 'diabetic'  # Line 3059
```

Then run the block under:

```python
# basic postprocess to get all metric data
```

> Ensure experiment parameters and file paths are consistent with `.txt` configs.

#### 4.3 Output File Examples

- `all_metric_aveg_best_baseline_test_0_RP1.csv` — P1, no constraints
- `all_metric_aveg_best_baseline_test_100_RP2.csv` — P2-base, threshold 100
- `all_metric_aveg_best_baseline_test_100-0.03_RP3.csv` — P2-fixed, threshold 100 & fixed action 0.03

- `P_hr` column corresponds to rho_plus
- `compare_count_perc` column corresponds to rho_adv

---

### Step 5: Generate Final Figures

1. Create a folder `across_rp`
2. Move all `*_RP*.csv` results into it
3. Run code under:

```python
# draw final learning curve
```

- Set `metric_name='rho_plus'` for Figure 2
- Set `metric_name='rho_adv'` for Figure 3

> Final data and figures are located in:
```
Supplementary/code_and_data/data_postprocess/Diabetes_case_study/across_rp/final_data
```


## Lunar Lander Case Study

This case study corresponds to the experiments reported in **Section 5.2** of the paper.

---

### Step 1: Train Baseline PPO Agent

#### 1.1 Setup Conda Environment

```bash
conda create -n CF_LunarLander_train_ppo_generalize python=3.8
conda activate CF_LunarLander_train_ppo_generalize

pip install stable-baselines3==1.7.0
pip install gym==0.21.0
pip install torch==2.0.0
pip install pandas==1.5.3
pip install tensorboard
```

Replace packages:

```bash
cd yourpath/.conda/envs/CF_LunarLander_train_ppo_generalize/lib/python3.8/site-packages
```

Upload files in `train_ppo/package`.

#### 1.2 Run PPO Training

- Single Environment (gravity = -9):

```bash
python openai_case_train_ppo_generalize.py \
  -arg_exp_id 1 \
  -arg_cuda 0 \
  -arg_train_step_each_env 500 \
  -arg_callback_step 500 \
  -arg_train_round 5 \
  -arg_lr 0.0001 \
  -arg_test_epochs_each_env 1 \
  -arg_max_test_time_each_env 1800 \
  -arg_if_train_personalize 1 \
  -arg_assigned_gravity -9.0
```

- Multi Environment (Train: g = -11, -9, -5; Test: g = -10, -8, -6):

```bash
python openai_case_train_ppo_generalize.py \
  -arg_exp_id 1 \
  -arg_cuda 0 \
  -arg_train_step_each_env 500 \
  -arg_callback_step 500 \
  -arg_train_round 3 \
  -arg_lr 0.0001 \
  -arg_test_epochs_each_env 1 \
  -arg_max_test_time_each_env 1800
```

> The trained models are saved under `train_ppo/trained_model`.

---

### Step 2: Run TD3-Based Experiments with Original Traces

#### 2.1 Setup Conda Environment

```bash
conda create -n CF_lunarlander python=3.8
conda activate CF_lunarlander

pip install stable-baselines3==1.7.0
conda install swig
pip install box2d-py
pip install gym==0.21.0
pip install Box2D
pip install torch==2.0.0
pip install pandas==1.5.3
pip install tensorflow==2.13.0
```

Replace packages:

```bash
cd yourpath/.conda/envs/CF_lunarlander/lib/python3.8/site-packages
```

Upload files from `train_td3/package`.

#### 2.2 Run Batch Experiments with SLURM

- P1:

```bash
# Single-environment
sbatch run_LL_exp1_double_test.sh

# Multi-environment
sbatch run_LL_exp3_double_test.sh
```

- P2-base:

```bash
# Single-environment
sbatch run_LL_RP2_exp1_S_double_test.sh

# Multi-environment
sbatch run_LL_RP2_exp3_S_double_test.sh
```

- P2-fixed:

```bash
# Single-environment
sbatch run_LL_RP3_exp1_S_double_test.sh

# Multi-environment
sbatch run_LL_RP3_exp3_S_double_test.sh
```

---

### Step 3: Run Experiments with New Traces

Use `script_LL_exp.py` to generate new `.txt` configs, and apply them in SLURM scripts above.

Example scripts:
```
train_td3/code/P1/generate_new_trace/
```

---

### Step 4: Postprocess Results and Generate Tables

Follow the same steps as in the Diabetes Case Study.
Set `case_name = 'lunar_lander'` in `data_postprocess.py`, and execute the block under:

```python
# basic postprocess to get all metric data
```

Example result files:

- `all_metric_aveg_best_baseline_test_0.0_RP1.csv`
- `all_metric_aveg_best_baseline_test_0.18_RP2.csv`
- `all_metric_aveg_best_baseline_test_0.18-0_0_RP3.csv`

---

### Step 5: Generate Final Figures

Move all results into `across_rp/` and run:

```python
# draw final learning curve
```

- `metric_name='rho_plus'` for Figure 4
- `metric_name='rho_adv'` for Figure 5

> Final data and figures:
```
data_postprocess/lunar_lander/across_rp/final_data
```

## 📌 Citation & Attribution

This repository was originally developed by **Shuyang Dong** as part of the research project:

> **Counterfactual Explanations for Continuous Action Reinforcement Learning**  
> Presented at the 34th International Joint Conference on Artificial Intelligence (IJCAI), 2025

If you use this code or build upon it (including via fork or mirror), **please cite the original work**:

```bibtex
@inproceedings{dong2025counterfactual,
  title={Counterfactual Explanations for Continuous Action Reinforcement Learning},
  author={Shuyang Dong and Shangtong Zhang and Lu Feng},
  booktitle={Proceedings of the 34th International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2025}
}

