#!/bin/bash
# Qwen3-30B-A3B MoE MFU baseline on 8x MI355X
# Config: gbs=8, mbs=1, seq=8192, EP=8, no grad accumulation
# Expected: ~300 TFLOP/s/GPU, ~740ms/iter
#
# Usage:
#   # Build image (one-time):
#   docker build -f docker/Dockerfile.mi355x-mfu -t primus-mi355x-mfu .
#
#   # Launch container:
#   docker run --rm -it --network host --device /dev/kfd --device /dev/dri \
#     --group-add video --shm-size 128g \
#     -v $(pwd):/workspace/primus_train/Primus \
#     primus-mi355x-mfu bash scripts/run_qwen3_30b_mfu_baseline.sh
#
#   # Or run directly inside an existing container:
#   bash scripts/run_qwen3_30b_mfu_baseline.sh [--profile]

set -euo pipefail

PROFILE_FLAGS=""
if [[ "${1:-}" == "--profile" ]]; then
    PROFILE_FLAGS="--profile True --use_pytorch_profiler True --profile_step_start 6 --profile_step_end 8"
    echo ">>> Profiling enabled (steps 6-8)"
fi

cd /workspace/primus_train/Primus

./primus-cli direct \
  -- train pretrain --config examples/megatron/configs/MI355X/qwen3_30B_A3B-BF16-pretrain.yaml \
  --train_iters 10 \
  --micro_batch_size 1 \
  --global_batch_size 8 \
  --seq_length 8192 \
  --max_position_embeddings 8192 \
  --expert_model_parallel_size 8 \
  --mock_data True \
  --disable_last_saving True \
  --moe_use_legacy_grouped_gemm True \
  --use_turbo_grouped_mlp True \
  --use_turbo_attention True \
  --enable_primus_turbo True \
  --use_turbo_deepep True \
  --turbo_deepep_num_cu 80 \
  --turbo_sync_free_moe_stage 1 \
  --enable_experimental True \
  --apply_rope_fusion True \
  --cross_entropy_fusion_impl te \
  --cross_entropy_loss_fusion True \
  --use_precision_aware_optimizer True \
  --main_grads_dtype bf16 \
  --exp_avg_dtype bf16 \
  --exp_avg_sq_dtype bf16 \
  --recompute_num_layers 5 \
  --recompute_granularity full \
  --recompute_method block \
  --disable_wandb True \
  --disable_tensorboard True \
  $PROFILE_FLAGS
