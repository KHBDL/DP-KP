# DP-KP: Differentially Private Katz Propagation with Private Pooling

This repository contains a compact implementation of DP-KP, also referred to as PGP in the code. The package keeps only the core functions needed for graph classification and one privacy attack experiment.

## What Is Included

- `fast_tune_pgp.py`: main entry for DP-KP graph classification.
- `Untils/`: data loading, perturbation, privacy accounting, and model modules.
- `attack_experiments/run_same_graph_shadow_node_mia.py`: shadow node membership inference attack.
- `dataset/`: included TU-format graph classification datasets.
- `results/`: default output folder for generated CSV files.

## Environment

Python 3.8 or newer is recommended. Install PyTorch and PyG according to your CUDA version first, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The original experiments used a conda environment named `env-GNN`, but the code can run in any environment with compatible `torch`, `torch_geometric`, `torch_scatter`, `numpy`, `scipy`, `networkx`, `scikit-learn`, and `tqdm`.

## Graph Classification

Run a node-level DP-KP classification experiment:

```bash
python fast_tune_pgp.py \
  --dataset PROTEINS \
  --level node \
  --device cuda:0 \
  --aggregation_method katz \
  --pooling_method diffpool \
  --epsilon_list 1,2,4,8 \
  --Hop_1 2 \
  --katz_beta 0.15 \
  --katz_mu_mode inverse_dmax \
  --D_max 25 \
  --private_adj_mode raw \
  --adj_mode exact \
  --epoch_2 20 \
  --Batch_size 128
```

Run an edge-level DP-KP classification experiment:

```bash
python fast_tune_pgp.py \
  --dataset PROTEINS \
  --level edge \
  --device cuda:0 \
  --aggregation_method katz \
  --pooling_method diffpool \
  --epsilon_list 1,2,4,8 \
  --Hop_1 2 \
  --katz_beta 0.15 \
  --katz_mu_mode inverse_dmax \
  --D_max 25 \
  --private_adj_mode raw \
  --adj_mode exact \
  --epoch_2 20 \
  --Batch_size 128
```

## Privacy Options

- `--level node`: node-level DP. The code uses a public degree bound `--D_max` and applies PageRank-based clipping before private release.
- `--level edge`: edge-level DP.
- `--epsilon_list`: total privacy budgets used in the experiment.
- `--private_adj_mode raw`: perturbs the raw adjacency matrix.
- `--private_adj_mode elementwise`: perturbs the element-wise bounded adjacency matrix.
- `--katz_mu_mode inverse_dmax`: uses `mu = 1 / D_max` for Katz sensitivity.

The private inputs are generated once before training. Subsequent pooling and classification are treated as post-processing and do not consume additional privacy budget.

## Attack Experiment

Run a lightweight shadow node membership inference attack:

```bash
python attack_experiments/run_same_graph_shadow_node_mia.py \
  --dataset PROTEINS \
  --device cuda:0 \
  --epsilon_list 1,8 \
  --target_mode dp \
  --shadow_mode dp \
  --aggregation_method katz \
  --pooling_method diffpool \
  --Hop_1 2 \
  --katz_beta 0.15 \
  --katz_mu_mode inverse_dmax \
  --D_max 25 \
  --private_adj_mode raw \
  --adj_mode exact \
  --epoch_2 10 \
  --Batch_size 128 \
  --max_train_graphs 200 \
  --max_test_graphs 80 \
  --max_attack_graphs 20 \
  --max_shadow_attack_graphs 20 \
  --nodes_per_graph 4 \
  --result_file results/demo_shadow_node_mia.csv
```

The attack reports AUC, accuracy, F1, and average precision. Lower AUC indicates stronger privacy protection.

## Outputs

Classification results are saved as CSV files under `results/` by default. You can specify a file explicitly:

```bash
--result_file results/my_experiment.csv
```

Attack results are also saved to the path provided by `--result_file`.

## Repository Notes

This package is intentionally minimal. Historical notebooks, large datasets, old logs, and full experiment result folders are not included. Add a license file before making the repository public.
