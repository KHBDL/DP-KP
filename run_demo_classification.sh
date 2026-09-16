#!/usr/bin/env bash
set -euo pipefail

python fast_tune_pgp.py \
  --dataset PROTEINS \
  --level node \
  --device "${DEVICE:-cuda:0}" \
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
  --Batch_size 128 \
  --result_file results/demo_classification.csv
