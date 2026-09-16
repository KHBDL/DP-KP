#!/usr/bin/env bash
set -euo pipefail

python attack_experiments/run_same_graph_shadow_node_mia.py \
  --dataset PROTEINS \
  --device "${DEVICE:-cuda:0}" \
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
