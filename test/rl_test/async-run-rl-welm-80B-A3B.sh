#!/bin/bash

set -xe

# For H20
# hugepage
echo 'never' > /sys/kernel/mm/transparent_hugepage/enabled

# pin memory
ulimit -l unlimited
sync && echo 3 > /proc/sys/vm/drop_caches
echo 1 > /proc/sys/vm/compact_memory

unset http_proxy
unset https_proxy

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export SLIME_HOST_IP=${__POD_IP__:-"127.0.0.1"}
export SGLANG_HOST_IP=${__POD_IP__:-"127.0.0.1"}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export CUDA_LAUNCH_BLOCKING=1

if [[ -z "$VIRTUAL_ENV" ]]; then
     source /envs/train/bin/activate
fi


# export http_proxy=http://hk-mmhttpproxy.woa.com:11113
# export https_proxy=http://hk-mmhttpproxy.woa.com:11113
# export no_proxy="11.152.212.103,wandb.woa.com,wandb.dev.woa.com:8080,wandb.dev1.woa.com:8080,wandb.dev2.woa.com:8080,wandb.dev3.woa.com:8080,wandb.dev4.woa.com:8080"
# SLIME_PATH=/mnt/cephfs/cilei/welm-v4-rl/slime-e24fc0c-welm-v4.tar.zst
# cp $SLIME_PATH /root/slime.tar.zst
# pushd /root
# zstd -cd slime.tar.zst | tar -xf -
# pip install ./slime
# popd
#
# unset http_proxy
# unset https_proxy

MMQ_CONFIG_PATH=${MMQ_CONFIG_PATH:-/mnt/cephfs/cilei/welm-v4-rl/welm-80B-A3B.yaml}
HF_CKPT_PATH=${HF_CKPT_PATH:-/mnt/cephfs/chappyzhou/workspace/2025-11-27-mmq-post-train/checkpoints/80a3_seq_64k_gbs_32_lr_4e-5_adam/epoch_003_step_0000555_hf}
PROMPT_DATA_PATH=${PROMPT_DATA_PATH:-/mnt/cephfs/cilei/hf-data/dapo-math-17k/dapo-math-17k.jsonl}
WANDB_PROJECT=${WANDB_PROJECT:-"cilei-welm_v4-rl-test"}
WANDB_GROUP=${WANDB_GROUP:-"WeLM_80B_A3B"}
WANDB_HOST=${WANDB_HOST:-"http://wandb.dev4.woa.com:8080"}
WANDB_KEY=${WANDB_KEY:-local-7288cbd9a5c07c8d2d537153711d0237dff1fa9a}
ROLLOUR_NUM_GPUS_PER_ENGINE=${ROLLOUR_NUM_GPUS_PER_ENGINE:-"4"}

DEBUG_ARGS=()

MODEL_ARGS=(
    --config ${MMQ_CONFIG_PATH}
)

CKPT_ARGS=(
    --hf-checkpoint ${HF_CKPT_PATH}
)

ROLLOUT_ARGS=(
    --prompt-data ${PROMPT_DATA_PATH}
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type deepscaler
    --num-rollout 3000
    --rollout-batch-size 32
    --n-samples-per-prompt 8
    # --rollout-batch-size 2
    # --n-samples-per-prompt 2
    --rollout-max-response-len ${ROLLOUT_MAX_SEQ_LEN:-4096}
    --rollout-temperature 0.8
)

EVAL_ARGS=(
    # --eval-interval 20
    # --eval-prompt-data aime /mnt/cephfs/omergao/github/slime/datas/aime-2024/aime-2024.jsonl
    # --n-samples-per-eval-prompt 16
    # --eval-max-response-len 16384
    # --eval-top-p 0.7
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --entropy-coef 0.00
    --eps-clip 0.2
    --eps-clip-high 0.28
)


WANDB_ARGS=(
    --use-wandb
    --wandb-project ${WANDB_PROJECT}
    --wandb-group ${WANDB_GROUP}
    --wandb-host ${WANDB_HOST}
    --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine ${ROLLOUR_NUM_GPUS_PER_ENGINE}
    # --sglang-expert-parallel-size ${ROLLOUR_NUM_GPUS_PER_ENGINE}
    --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
    --sglang-attention-backend fa3
    --sglang-disable-overlap-schedule
    --sglang-enable-over-encoding
    --sglang-served-model-name welmv4
)
if [[ "${SGLANG_DUMMY_LOAD}" == "1" ]]; then
    SGLANG_ARGS+=(--sglang-load-format dummy)
fi
    # --save-debug-rollout-data /mnt/cephfs/cilei/welm-v4-rl/debug-rollout-data/{rollout_id}.pt \
    # --load-debug-rollout-data /mnt/cephfs/cilei/welm-v4-rl/debug-rollout-data/{rollout_id}.pt \
# if [[ "$DEBUG_ROLLOUT_ONLY" == "1" ]]; then
#     DEBUG_ARGS+=(--debug-rollout-only)
#     SGLANG_ARGS+=(--sglang-mem-fraction-static 0.9)
# else
#     SGLANG_ARGS+=(--sglang-load-format dummy)
#     SGLANG_ARGS+=(--sglang-mem-fraction-static 0.8)
# fi
# if [[ "$DEBUG_TRAIN_ONLY" == "1" ]]; then
#     export NCCL_DEBUG=INFO
#     DEBUG_ARGS+=(
#         --debug-train-only
#         --load-debug-rollout-data /mnt/cephfs/cilei/welm-v4-rl/debug-rollout-data/{rollout_id}.pt
#     )
# else
#     DEBUG_ARGS+=(
#         --save-debug-rollout-data /mnt/cephfs/cilei/welm-v4-rl/debug-rollout-data/{rollout_id}.pt
#     )
# fi

DATA_ARGS=(
    --global-batch-size 256
    --micro-batch-size 1
)

# MISC_ARGS=(
#     # default dropout in megatron is 0.1
#     --attention-dropout 0.0
#     --hidden-dropout 0.0
#     # should be good for model performance
#     --accumulate-allreduce-grads-in-fp32
#     --attention-softmax-in-fp32
#     # need to comment this when using model with MLA
#     --attention-backend flash
# )

export USE_TCCL=${USE_TCCL:-0}
if [[ "$USE_TCCL" == "1" ]]; then
    TORCH_PATH=$(dirname $(python3 -c 'import torch;print(torch.__file__)'))
    TORCH_NCCL_LIB="${TORCH_PATH}/../nvidia/nccl/lib/libnccl.so.2"
    export NCCL_PRIMS_PROFILE_ENABLE=1
    export NCCL_PRIMS_PROFILE_VERSION=1
    update-alternatives --install\
      "${TORCH_NCCL_LIB}" \
      libnccl.so \
      /usr/local/lib/libtccl.so \
      10
fi

SKIP_RAY_START=${SKIP_RAY_START:-0}

if [[ "${SKIP_RAY_START}" == "0" ]]; then
    if [ $RANK -eq 0 ]; then
        ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8080
        sleep 5
    else
        ray start --address=${MASTER_ADDR}:6379 --node-ip-address ${__POD_IP__} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8080
        sleep infinity
    fi
fi

# Build the runtime environment JSON with proper variable substitution
export RUNTIME_ENV_JSON="{
  \"env_vars\": {
     \"NCCL_DEBUG\": \"${NCCL_DEBUG:-WARN}\",
     \"NCCL_PRIMS_PROFILE_VERSION\": \"${NCCL_PRIMS_PROFILE_VERSION:-0}\",
     \"NCCL_PRIMS_PROFILE_ENABLE\": \"${NCCL_PRIMS_PROFILE_ENABLE:-0}\",
     \"CUDA_LAUNCH_BLOCKING\": \"0\",
     \"MMQ_FUSE_QUANT_AND_MUL_PROBS\": \"1\",
     \"MMQ_FUSE_RMS_NORM_AND_RESIDUAL\": \"1\",
     \"MMQ_MUL_PROBS_CAPTURE_FP8\": \"1\",
     \"MMQ_MUL_PROBS_CAPTURE_OUTPUT\": \"1\",
     \"MMQ_PERMUTE_CAPTURE_FP8\": \"1\",
     \"MMQ_RECOMPUTE_QKV_PROJECTION_QK_NORM_AND_ROPE\": \"1\",
     \"MMQ_SWIGLU_CAPTURE_FP8\": \"1\",
     \"MMQ_FUSE_SWIGLU_AND_MUL_PROBS\": \"1\",
     \"MMQ_RECOMPUTE_RMS_NORM_OUTPUT\": \"1\",
     \"MMQ_OUT_PROJ_CAPTURE_FP8\": \"1\",
     \"MMQ_USE_TRITON_ROUTER_DW\": \"0\",
     \"MMQ_FUSE_ROUTER_TOPK\": \"1\",
     \"MMQ_FUSE_SWIGLU_AND_MUL_PROBS_WITH_JIT\": \"1\",
     \"MMQ_MEMORY_EFFICIENT_RMS_NORM\": \"0\",
     \"MMQ_TESTING_BLOCK_V2\": \"1\",
     \"MMQ_SKIP_OPTIMIZER_STEP\": \"0\",
     \"MMQ_USE_TORCH_ROUTER\": \"0\",
     \"MMQ_TESTING_BLOCK_V2\": \"1\",
     \"MMQ_OPTIMIZER_V2\": \"1\",
     \"MMQ_SLIME_PARAM_UPDATE_USE_LEGACY_UPDATE\": \"1\",
     \"no_proxy\": \"${no_proxy}\",
     \"_MMQ_MEMORY_DUMP_FILE\": \"/mnt/cephfs/omergao/playground/debug/oom/oom\",
     \"CUDA_LAUNCH_BLOCKING\": \"1\"
  }
}"


SKIP_RAY_SUBMIT=${SKIP_RAY_SUBMIT:-0}

if [[ "${SKIP_RAY_SUBMIT}" == "1" ]]; then
    sleep infinity
fi

ray job submit --address="http://127.0.0.1:8080" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- slime_train_mmq_async \
    --actor-num-nodes $((${WORLD_SIZE}/2)) \
    --actor-num-gpus-per-node 8 \
    --rollout-num-gpus $((${WORLD_SIZE}/2*8)) \
    ${DATA_ARGS[@]} \
    ${CKPT_ARGS[@]} \
    ${MODEL_ARGS[@]} \
    ${ROLLOUT_ARGS[@]} \
    ${GRPO_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${EVAL_ARGS[@]} \
    ${SGLANG_ARGS[@]} \
    ${WANDB_ARGS[@]} \
    ${DEBUG_ARGS[@]} \
    "$@"
