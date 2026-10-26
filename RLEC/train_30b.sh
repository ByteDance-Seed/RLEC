#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VERL_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}

# Keep W&B's mutable run state on node-local POSIX storage. Set WANDB_API_KEY
# externally or run `wandb login` before launching.
export WANDB_DIR=${WANDB_DIR:-/tmp/rlec_wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-/tmp/rlec_wandb_cache}
export WANDB_DATA_DIR=${WANDB_DATA_DIR:-/tmp/rlec_wandb_data}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-/tmp/rlec_wandb_config}

PROJECT_NAME=${PROJECT_NAME:-RLEC}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_30b_a3b_rlec}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B-Base}
DATA_DIR=${DATA_DIR:-"${SCRIPT_DIR}/data/processed"}
TRAIN_FILES=${TRAIN_FILES:-"${DATA_DIR}/train.parquet"}
VAL_FILES=${VAL_FILES:-"${DATA_DIR}/test.parquet"}
REWARD_FILE=${REWARD_FILE:-"${SCRIPT_DIR}/reward.py"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/tmp/rlec_checkpoints/${EXPERIMENT_NAME}"}
LOG_FILE=${LOG_FILE:-"${SCRIPT_DIR}/logs/${EXPERIMENT_NAME}.log"}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
ROLLOUT_N=${ROLLOUT_N:-8}

# The released mixed dataset contains 150,000 rows, or about 585 global
# batches. Reward workers process roughly one global batch per schedule step.
export WARMUP_STEPS=${WARMUP_STEPS:-585}
export CALLS_PER_STEP=${CALLS_PER_STEP:-256}

mkdir -p "${CHECKPOINT_DIR}" "$(dirname "${LOG_FILE}")"

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.kl_ctrl.kl_coef=0.0
)

DATA=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.prompt_key=prompt
    data.return_raw_chat=True
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.truncation=left
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=150000
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.2
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2
    actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=2
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=2
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=1
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.expert_parallel_size=4
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=150000
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.max_num_seqs=1024
    actor_rollout_ref.rollout.max_num_batched_tokens=10240
    actor_rollout_ref.rollout.max_model_len=10240
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.n=1
)

REFERENCE=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=150000
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=2
    actor_rollout_ref.ref.megatron.virtual_pipeline_model_parallel_size=2
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=2
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.ref.megatron.context_parallel_size=1
    actor_rollout_ref.ref.megatron.param_offload=True
    actor_rollout_ref.ref.megatron.use_mbridge=True
)

REWARD=(
    reward.custom_reward_function.path="${REWARD_FILE}"
    reward.custom_reward_function.name=compute_score
    reward.reward_manager.name=naive
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=117
    trainer.test_freq=5
    trainer.total_epochs=1
    trainer.resume_mode=auto
    trainer.val_before_train=True
    trainer.log_val_generations=1
    trainer.default_local_dir="${CHECKPOINT_DIR}"
)

RAY_ENV=(
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DIR="${WANDB_DIR}"
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_CACHE_DIR="${WANDB_CACHE_DIR}"
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_DATA_DIR="${WANDB_DATA_DIR}"
    +ray_kwargs.ray_init.runtime_env.env_vars.WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR}"
    +ray_kwargs.ray_init.runtime_env.env_vars.WARMUP_STEPS="${WARMUP_STEPS}"
    +ray_kwargs.ray_init.runtime_env.env_vars.CALLS_PER_STEP="${CALLS_PER_STEP}"
)

cd "${VERL_ROOT}"
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REFERENCE[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${RAY_ENV[@]}" \
    model_engine=megatron \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    "$@" 2>&1 | tee "${LOG_FILE}"
